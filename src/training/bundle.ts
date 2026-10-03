import type { TrainingReplayBundle } from './types';
import type { ReplayProfile } from '../rl/profiles';
import { loadCheckpoint } from '../rl/checkpoint';
import type { MujocoEngine } from '../mujoco/MujocoEngine';
import { PolicyReplay } from '../rl/replay';

const invalid = (detail: string): never => { throw new Error(`训练结果包无效：${detail}`); };
export function parseTrainingBundle(value: unknown): TrainingReplayBundle {
  const b = value as TrainingReplayBundle;
  if (!b || b.schemaVersion !== 1 || b.source !== 'mjlab') invalid('仅支持 mjlab v1 格式');
  if (typeof b.taskId !== 'string' || typeof b.name !== 'string' || !b.name.length || b.name.length > 200) invalid('缺少任务名称');
  if (typeof b.xml !== 'string' || !b.xml.includes('<mujoco') || b.xml.length > 12 * 1024 * 1024) invalid('模型 XML 无效或过大');
  if (!b.assets || typeof b.assets !== 'object' || Array.isArray(b.assets) || Object.keys(b.assets).length > 64) invalid('资源列表无效');
  let size = 0;
  for (const [name, text] of Object.entries(b.assets)) {
    if (!/^training_[A-Za-z0-9_.-]+$/.test(name) || name.includes('..') || typeof text !== 'string') invalid('资源必须是独立的 training_ 文本文件');
    size += text.length;
  }
  if (size > 32 * 1024 * 1024) invalid('资源超过 32 MiB');
  if (/<include\b/i.test(b.xml)) invalid('模型必须内联，不能引用其他 XML');
  for (const match of b.xml.matchAll(/\bfile\s*=\s*["']([^"']+)["']/g)) {
    if (!Object.hasOwn(b.assets, match[1])) invalid(`模型引用缺失资源 ${match[1]}`);
  }
  const p = b.profile;
  if (!p || !['quaternion', 'mjlab-zbot'].includes(p.observation) || ![26, 31].includes(p.inputSize) || !Array.isArray(p.jointNames) || p.jointNames.length !== 6
    || new Set(p.jointNames).size !== 6 || !p.jointNames.every(n => typeof n === 'string' && n.length > 0 && n.length <= 128)
    || !Array.isArray(p.defaultAngles) || p.defaultAngles.length !== 6 || !p.defaultAngles.every(Number.isFinite)) invalid('只支持六自由度26维四元数任务');
  if (p.observation === 'mjlab-zbot' && p.inputSize !== 31) invalid('mjlab zbot 观测维度无效');
  if (p.policyFeatures !== undefined && p.policyFeatures !== 'quat-gravity-heading-v1') invalid('不支持此策略特征变换');
  const ratio = p.controlDt / p.physicsDt;
  if (!(p.physicsDt >= .00001 && p.physicsDt <= .05 && p.controlDt <= .1 && ratio >= 1 && ratio <= 1000)
    || Math.abs(ratio - Math.round(ratio)) > 1e-6 || !(p.jointSpeedLimit > 0 && p.jointSpeedLimit <= 5)) invalid('时间步或积分速度参数无效');
  if (typeof b.checkpointBase64 !== 'string' || b.checkpointBase64.length > 48 * 1024 * 1024
    || b.checkpointBase64.length % 4 || !/^[A-Za-z0-9+/]+={0,2}$/.test(b.checkpointBase64)) invalid('权重编码无效或过大');
  if (!b.versions || typeof b.versions !== 'object' || !Object.values(b.versions).every(v => typeof v === 'string')) invalid('缺少软件版本');
  return b;
}

export function trainingBundleProfile(bundle: TrainingReplayBundle): ReplayProfile {
  const p = bundle.profile;
  return { id: `training:${bundle.taskId}`, label: bundle.name, model: '', checkpoint: '', origin: 'mjlab',
    jointNames: [...p.jointNames], defaultAngles: [...p.defaultAngles], observation: p.observation, inputSize: p.inputSize,
    ...(p.policyFeatures ? { policyFeatures: p.policyFeatures } : {}),
    ...(p.observation === 'mjlab-zbot' ? { actionMode: 'position', actionScale: 0.25, stepFrequency: 0.5 } : {}),
    physicsDt: p.physicsDt, controlDt: p.controlDt, jointSpeedLimit: p.jointSpeedLimit, motion: 'walking' };
}

export function loadTrainingBundle(engine: MujocoEngine, value: unknown): { bundle: TrainingReplayBundle; profile: ReplayProfile; replay: PolicyReplay } {
  const bundle = parseTrainingBundle(value);
  const bytes = Uint8Array.from(atob(bundle.checkpointBase64), c => c.charCodeAt(0));
  const policy = loadCheckpoint(bytes.buffer, 'elu');
  const networkInputSize = bundle.profile.inputSize;
  if (policy.inputSize !== networkInputSize || policy.outputSize !== 6) invalid('权重维度与任务不符');
  if (bundle.profile.policyFeatures && policy.normalization) invalid('四元数特征策略不支持观测归一化');
  const profile = trainingBundleProfile(bundle);
  engine.installTrainingAssets(bundle.assets);
  engine.loadModelFromXml(bundle.xml);
  const replay = new PolicyReplay(engine, profile, policy);
  return { bundle, profile, replay };
}
