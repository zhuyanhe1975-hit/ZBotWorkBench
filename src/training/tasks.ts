import type { TrainingConfig, TrainingResources, TrainingTask, TrainingTaskCard, TrainingTaskCardSettings } from './types';

export const WALKING_FINETUNE_TASK = 'Mjlab-Zbot-6dof-Walking-Finetune';
export const QUASISTATIC_TASK = 'Mjlab-Zbot-6dof-Quasistatic-Walking';
export const IN_PLACE_TASK = 'Mjlab-Zbot-6dof-InPlace-Stepping';
const rewards: Pick<TrainingTaskCard, 'stage1Rewards' | 'stage2Rewards' | 'terminatedRewardPenalty'> = {
  stage1Rewards: { feet_downward: -1, feet_forward: -.5, base_heading_x: -1, feet_force_diff: 2, feet_force_sum: -.1 },
  stage2Rewards: { base_vel_forward: 5, feet_downward: -1, feet_forward: -.5, similar_to_default: -.5, base_heading_x: -1, base_heading_x_sum: -1, step_length: 5, small_step: -5, airtime_balance: -15, airtime_sum: 2, action_rate: -.1, body_shake: -.5, torques: -1, energy_consumption: -.01, feet_slide: -10, base_pos_y_err: -.1 },
  terminatedRewardPenalty: 20,
};
const card = (initialStage: 1 | 2, jointSpeedRange: [number, number], stage2Rewards = rewards.stage2Rewards): TrainingTaskCard => ({ physicsHz: 240, controlHz: 30, contactHistory: 8, initialStage, jointSpeedRange, ...rewards, stage2Rewards });
const quasistaticRewards = { base_vel_forward: 0, slow_speed_tracking: 2, feet_downward: -2, feet_forward: -1, similar_to_default: -1.5, base_heading_x: -2, base_heading_x_sum: -1, support_stability: 4, base_tilt: -2, step_length: .5, small_step: -2, step_cadence: -1, airtime_balance: -3, airtime_sum: -.5, double_flight: -10, action_rate: -2, body_shake: -5, joint_velocity: -.5, joint_acceleration: -.2, torques: -.2, energy_consumption: -.05, feet_slide: -20, base_pos_y_err: -2 };
const inPlaceRewards = { support_phase: 4, com_phase: 2, heading: -2, joint_pose: -.2, body_velocity: -.5, upright: -1, action_rate: -.05, single_support: 1 };
export const TRAINING_TASKS: TrainingTask[] = [
  { id: WALKING_FINETUNE_TASK, label: '6DOF 已有行走策略微调（推荐）', description: '从参考工程 model_801 初始化：240 Hz 物理 / 30 Hz 控制、第二阶段奖励。奖励权重可在训练前或运行中修改。', referenceBundleUrl: `${import.meta.env?.BASE_URL ?? '/'}rl/training/walking-reference.json`, taskCard: card(2, [.8, 1.2]) },
  { id: QUASISTATIC_TASK, label: '6DOF 准静态慢走（实验）', description: '从model_801初始化，目标速度0.04 m/s，强调支撑稳定、低抖动、低能耗、低滑移与慢步频。', referenceBundleUrl: `${import.meta.env?.BASE_URL ?? '/'}rl/training/walking-reference.json`, taskCard: card(2, [.2, .4], quasistaticRewards) },
  { id: 'Mjlab-Zbot-6dof-Bipedal-Walking', label: '6DOF 从零训练（240 Hz）', description: '从随机权重开始：240 Hz物理 / 30 Hz控制、8步接触历史和两阶段课程。GPU建议4096环境、至少300次更新。', taskCard: card(1, [.2, 2]) },
  { id: IN_PLACE_TASK, label: '6DOF 原地换脚踏步（固定 1 Hz）', description: '按固定 1 Hz 正弦规律切换左右脚支撑与重心，关节速度上限固定为 2π rad/s，抑制前进和横向漂移。', taskCard: { ...card(1, [2, 2], inPlaceRewards), stage1Rewards: {}, stage2Rewards: inPlaceRewards } },
];

export const copyTaskCardSettings = (value: TrainingTaskCardSettings): TrainingTaskCardSettings => ({ stage1Rewards: { ...value.stage1Rewards }, stage2Rewards: { ...value.stage2Rewards }, terminatedRewardPenalty: value.terminatedRewardPenalty });

/** Apply a task deliberately; resource refreshes must not overwrite user choices. */
export function taskDefaults(taskId: string, resources: TrainingResources, device?: string): TrainingConfig {
  const finetune = taskId === WALKING_FINETUNE_TASK;
  const quasistatic = taskId === QUASISTATIC_TASK;
  const inPlace = taskId === IN_PLACE_TASK;
  const selectedDevice = device ?? (resources.runtime.cudaBuild !== false && resources.gpus.length ? `cuda:${resources.gpus[0].id}` : 'cpu');
  const task = TRAINING_TASKS.find(task => task.id === taskId);
  return { taskId, device: selectedDevice, cpuThreads: Math.max(1, Math.min(4, resources.cpu.availableThreads)),
    numEnvs: selectedDevice === 'cpu' ? 4 : finetune ? 256 : quasistatic ? 1024 : inPlace ? 512 : 4096,
    iterations: finetune ? 100 : quasistatic ? 1000 : inPlace ? 1000 : 300, saveInterval: finetune ? 10 : 50, seed: 42, maxSeconds: 3600,
    ...(task ? { taskCard: copyTaskCardSettings(task.taskCard) } : {}) };
}
