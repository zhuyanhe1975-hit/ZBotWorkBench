import { Vector3 } from 'three';
import { PRESET_CONFIGURATIONS } from '../data/presets';
import { withSerialModules } from './configuration';
import { forwardKinematics } from './kinematics';

/** Discrete design, with the same physical mounting frame for both chain lengths. */
export function designConfiguration(sigma: number[]) {
  const base = structuredClone(PRESET_CONFIGURATIONS[1]);
  const config = withSerialModules(base, Array.from({ length: sigma.length + 1 }, (_, i) => ({
    ...structuredClone(base.modules[0]), id: `mod_${i}`, name: `模块 ${i + 1}`,
    dockAngle: i ? sigma[i - 1] : 0, initialAngle: 0,
  })));
  config.id = `design_${sigma.length + 1}_${sigma.map(v => v / 90).join('')}`;
  config.name = `${sigma.length + 1}模块候选 ${sigma.join('/')}`;
  config.description = '离散构型搜索候选；接口角度、负载、刚度及连续无碰撞路径尚需实物验证。';
  config.hypothesis = '固定基座通用空间操作候选，不代表综合性能已最优。';
  config.rootPos = [0, 0, .35]; config.rootEuler = [0, 0, 0];
  config.defaultGait.manualAngles = Object.fromEntries(config.modules.map((_, i) => [`joint_${i}`, 0]));
  return config;
}

/** Deterministic Halton samples; identical first six coordinates across six/seven joints. */
export function jointSample(index: number, count: number): number[] {
  return [2, 3, 5, 7, 11, 13, 17].slice(0, count).map(base => {
    let n = index + 1, fraction = 1, value = 0;
    while (n > 0) { fraction /= base; value += (n % base) * fraction; n = Math.floor(n / base); }
    return 360 * value - 180;
  });
}

/** Eigenvalues of a small real symmetric matrix, using Jacobi rotations. */
export function symmetricEigenvalues(input: number[][]): number[] {
  const a = input.map(r => [...r]), n = a.length;
  for (let step = 0; step < 100 * n * n; step++) {
    let p = 0, q = 1, largest = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++)
      if (Math.abs(a[i][j]) > largest) { largest = Math.abs(a[i][j]); p = i; q = j; }
    if (largest < 1e-12) break;
    const angle = .5 * Math.atan2(2 * a[p][q], a[q][q] - a[p][p]);
    const c = Math.cos(angle), s = Math.sin(angle), pp = a[p][p], qq = a[q][q], pq = a[p][q];
    for (let k = 0; k < n; k++) if (k !== p && k !== q) {
      const kp = a[k][p], kq = a[k][q];
      a[k][p] = a[p][k] = c * kp - s * kq;
      a[k][q] = a[q][k] = s * kp + c * kq;
    }
    a[p][p] = c*c*pp - 2*s*c*pq + s*s*qq;
    a[q][q] = s*s*pp + 2*s*c*pq + c*c*qq;
    a[p][q] = a[q][p] = 0;
  }
  return a.map((r, i) => r[i]).sort((a, b) => a - b);
}

/** Fixed characteristic length .3 m; never normalize separately by chain length. */
export function poseQuality(fk: ReturnType<typeof forwardKinematics>) {
  const tip = new Vector3(...fk.tip);
  const columns = fk.joints.map(j => {
    const axis = new Vector3(...j.axis);
    const linear = axis.clone().cross(tip.clone().sub(new Vector3(...j.position)));
    return [...linear.toArray().map(v => v / .3), ...axis.toArray()];
  });
  const gram = Array.from({length: 6}, (_, i) => Array.from({length: 6}, (_, j) => columns.reduce((sum, c) => sum + c[i]*c[j], 0)));
  const values = symmetricEigenvalues(gram);
  const minimum = Math.sqrt(Math.max(0, values[0]));
  // Additional torque per kg at the docking face, downward gravity, no tool offset/self-weight.
  const payloadTorquePerKg = columns.map(c => -9.81 * c[2] * .3);
  return { minimum, inverseCondition: minimum / Math.sqrt(Math.max(1e-15, values[5])), payloadTorquePerKg,
    maxPayloadTorquePerKg: Math.max(...payloadTorquePerKg.map(Math.abs)) };
}

export function percentile(values: number[], fraction: number): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((a,b) => a-b);
  return sorted[Math.floor((sorted.length - 1) * fraction)];
}
