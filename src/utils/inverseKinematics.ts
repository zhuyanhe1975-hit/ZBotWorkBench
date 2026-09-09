import { Quaternion, Vector3 } from 'three';
import type { ZbotConfiguration } from '../types/zbot';
import { validateConfiguration } from './configuration';
import { forwardKinematics, type KinematicResult } from './kinematics';

export interface EndEffectorTarget {
  position: [number, number, number];
  /** World orientation, xyzw. */
  quaternion: [number, number, number, number];
}

function poseFromFK(fk: KinematicResult): EndEffectorTarget {
  return { position: [...fk.tip], quaternion: new Quaternion().setFromRotationMatrix(fk.parts[fk.parts.length - 1].b).normalize().toArray() };
}

/** q uses degrees; the tool frame is the last module's output docking face. */
export function getEndEffectorPose(config: ZbotConfiguration, q?: number[]): EndEffectorTarget {
  return poseFromFK(forwardKinematics(config, q));
}

/** Recognizes geometry rather than the editable category/name; requires a fixed base. */
export function isOrthogonalArm(config: ZbotConfiguration): boolean {
  if (validateConfiguration(config).length || config.modules.length !== 6 ||
    (config.baseMode ?? (config.category === 'arm' ? 'fixed' : 'free')) !== 'fixed') return false;
  const standard = new Vector3(0, -1, 1).normalize();
  const equivalent = (a: number, b: number) => Math.abs(Math.sin((a - b) * Math.PI / 360)) < 1e-8;
  return config.modules.every((m, i) => {
    if (new Vector3(...m.jointAxis).normalize().distanceTo(standard) > 1e-8) return false;
    const dock = m.customEuler ?? [0, 0, m.dockAngle];
    return i === 0 || equivalent(dock[0], 0) && equivalent(dock[1], 0) && equivalent(dock[2], 180);
  });
}

function errorVector(current: EndEffectorTarget, target: EndEffectorTarget): number[] {
  const delta = new Quaternion(...target.quaternion).normalize().multiply(new Quaternion(...current.quaternion).invert()).normalize();
  // q and -q represent the same orientation: always follow the shorter rotation.
  if (delta.w < 0) delta.set(-delta.x, -delta.y, -delta.z, -delta.w);
  const sine = Math.hypot(delta.x, delta.y, delta.z);
  const scale = sine < 1e-10 ? 2 : 2 * Math.atan2(sine, delta.w) / sine;
  return [...target.position.map((v, i) => v - current.position[i]), delta.x * scale, delta.y * scale, delta.z * scale];
}

/** Gaussian elimination with partial pivoting, for the damped 6×6 normal system. */
function solveSystem(matrix: number[][], rhs: number[]): number[] {
  const a = matrix.map((row, i) => [...row, rhs[i]]);
  for (let col = 0; col < rhs.length; col++) {
    let pivot = col;
    for (let row = col + 1; row < rhs.length; row++) if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
    [a[col], a[pivot]] = [a[pivot], a[col]];
    if (Math.abs(a[col][col]) < 1e-15) return rhs.map(() => 0);
    const divisor = a[col][col];
    for (let j = col; j <= rhs.length; j++) a[col][j] /= divisor;
    for (let row = 0; row < rhs.length; row++) {
      if (row === col) continue;
      const factor = a[row][col];
      for (let j = col; j <= rhs.length; j++) a[row][j] -= factor * a[col][j];
    }
  }
  return a.map(row => row[rhs.length]);
}

/** Bounded local pose IK. Nonconverged angles are diagnostic, not safe controller commands. */
export function solveInverseKinematics(config: ZbotConfiguration, target: EndEffectorTarget, seed: number[]): {
  angles: number[]; converged: boolean; positionError: number; orientationError: number; iterations: number;
} {
  if (!isOrthogonalArm(config)) throw new Error('末端控制仅支持固定基座的六模块经典正交构型');
  if (seed.length !== 6 || seed.some(v => !Number.isFinite(v))) throw new Error('IK 初值必须包含六个有限角度');
  if (target.position.length !== 3 || target.quaternion.length !== 4 ||
    [...target.position, ...target.quaternion].some(v => !Number.isFinite(v)) || Math.hypot(...target.quaternion) < 1e-10) throw new Error('末端目标必须是有限位置和非零四元数');
  const clamp = (q: number, i: number) => Math.max(config.modules[i].jointRange[0], Math.min(config.modules[i].jointRange[1], q));
  let angles = seed.map(clamp);
  const orientationWeight = 0.15;
  const weights = [1, 1, 1, orientationWeight, orientationWeight, orientationWeight];
  const cost = (e: number[]) => e.reduce((sum, v, i) => sum + (v * weights[i]) ** 2, 0);
  let damping = 0.003;
  let iterations = 0;
  for (; iterations < 100; iterations++) {
    const fk = forwardKinematics(config, angles);
    const error = errorVector(poseFromFK(fk), target);
    if (Math.hypot(...error.slice(0, 3)) <= .0005 && Math.hypot(...error.slice(3)) <= .01) break;
    const tip = new Vector3(...fk.tip);
    const columns = fk.joints.map(j => {
      const axis = new Vector3(...j.axis);
      const linear = axis.clone().cross(tip.clone().sub(new Vector3(...j.position)));
      return [...linear.toArray(), ...axis.toArray()].map((v, i) => v * weights[i]);
    });
    const matrix = Array.from({ length: 6 }, (_, i) => Array.from({ length: 6 }, (_, j) =>
      columns.reduce((sum, c) => sum + c[i] * c[j], 0) + (i === j ? damping ** 2 : 0)));
    const taskStep = solveSystem(matrix, error.map((v, i) => v * weights[i]));
    const jointStep = columns.map(c => c.reduce((sum, v, i) => sum + v * taskStep[i], 0) * 180 / Math.PI);
    if (jointStep.some(v => !Number.isFinite(v))) break;
    const stepScale = Math.min(1, 12 / Math.max(...jointStep.map(Math.abs), 1e-10));
    let accepted = false;
    for (let scale = stepScale; scale >= stepScale / 32; scale /= 2) {
      const next = angles.map((q, i) => clamp(q + jointStep[i] * scale, i));
      if (cost(errorVector(getEndEffectorPose(config, next), target)) < cost(error) - 1e-14) {
        angles = next; damping = Math.max(.0003, damping * .7); accepted = true; break;
      }
    }
    if (!accepted) { damping = Math.min(.3, damping * 3); if (damping === .3) break; }
  }
  const error = errorVector(getEndEffectorPose(config, angles), target);
  const positionError = Math.hypot(...error.slice(0, 3));
  const orientationError = Math.hypot(...error.slice(3));
  return { angles, converged: positionError <= .0005 && orientationError <= .01, positionError, orientationError, iterations };
}
