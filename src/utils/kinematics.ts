import { Euler, Matrix4, Vector3 } from 'three';
import { ZbotConfiguration } from '../types/zbot';
import { validateConfiguration } from './configuration';
import { moduleMount } from './moduleMount';

export type Vec3 = [number, number, number];
export interface KinematicJoint { id: string; position: Vec3; axis: Vec3 }
export interface KinematicResult {
  joints: KinematicJoint[];
  endpoints: Vec3[];
  tip: Vec3;
  parts: { moduleId: string; a: Matrix4; b: Matrix4 }[];
}
const rad = Math.PI / 180;
/** World-space FK. q is degrees in module order; omitted q uses initialAngle then manualAngles.
 * Tin → Tmb = Tin Th Raxis(q) T−h → Tnext = Tmb Tz(.106) Rz(sigma).
 * Euler XYZ matches MJCF's intrinsic xyz convention; no mesh scaling or invented links.
 */
export function forwardKinematics(config: ZbotConfiguration, q?: number[]): KinematicResult {
  const errors = validateConfiguration(config);
  if (errors.length) throw new Error(errors.join('；'));
  if (q && (q.length !== config.modules.length || q.some(v => !Number.isFinite(v)))) throw new Error('q 必须为每个模块提供有限角度');
  const root = new Matrix4().makeRotationFromEuler(new Euler(...config.rootEuler.map(v => v * rad) as Vec3, 'XYZ'));
  root.setPosition(...config.rootPos);
  const result: KinematicResult = { joints: [], endpoints: [new Vector3().setFromMatrixPosition(root).toArray() as Vec3], tip: [0, 0, 0], parts: [] };
  config.modules.forEach((m, i) => {
    const parent = m.parentId === null ? root : result.parts.find(p => p.moduleId === m.parentId)!.b;
    const input = parent.clone().multiply(moduleMount(config, m));
    const axis = new Vector3(...m.jointAxis).normalize();
    result.joints.push({ id: m.id, position: new Vector3(0, 0, .053).applyMatrix4(input).toArray() as Vec3,
      axis: axis.clone().transformDirection(input).toArray() as Vec3 });
    const angle = q?.[i] ?? m.initialAngle ?? config.defaultGait.manualAngles[`joint_${i}`] ?? 0;
    const b = input.clone().multiply(new Matrix4().makeTranslation(0, 0, .053))
      .multiply(new Matrix4().makeRotationAxis(axis, angle * rad)).multiply(new Matrix4().makeTranslation(0, 0, -.053));
    result.parts.push({ moduleId: m.id, a: input.clone(), b });
    result.endpoints.push(new Vector3(0, 0, .106).applyMatrix4(b).toArray() as Vec3);
  });
  result.tip = result.endpoints[result.endpoints.length - 1];
  return result;
}
