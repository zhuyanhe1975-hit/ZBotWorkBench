import type { TrainingConfig, TrainingResources, TrainingTask, TrainingTaskCard, TrainingTaskCardSettings } from './types';

export const DIRECT_WALKING_TASK = 'Mjlab-Zbot-6dof-Walking';
// Kept for component compatibility; this task is no longer offered in the UI.
export const IN_PLACE_TASK = 'Mjlab-Zbot-6dof-InPlace-Stepping';
const rewards: Pick<TrainingTaskCard, 'stage1Rewards' | 'stage2Rewards' | 'terminatedRewardPenalty'> = {
  stage1Rewards: { feet_downward: -1, feet_forward: -.5, base_heading_x: -1, feet_force_diff: 2, feet_force_sum: -.1 },
  stage2Rewards: { base_vel_forward: 5, feet_downward: -1, feet_forward: -.5, similar_to_default: -.5, base_heading_x: -1, base_heading_x_sum: -1, step_length: 5, small_step: -5, airtime_balance: -15, airtime_sum: 2, action_rate: -.1, body_shake: -.5, torques: -1, energy_consumption: -.01, feet_slide: -10, base_pos_y_err: -.1 },
  terminatedRewardPenalty: 20,
};
const zbotTrainRewards = {
  frequency_tracking: 3, swing_clearance: .5, forward_velocity: 5, lateral_velocity: -0,
  upright: .2, yaw_drift: -.2, support_foot_slip: -10, soft_landing: -.02,
  action_rate: -.05, joint_limits: -.1,
};
const card = (initialStage: 1 | 2, jointSpeedRange: [number, number], stage2Rewards = rewards.stage2Rewards): TrainingTaskCard => ({ physicsHz: 240, controlHz: 30, contactHistory: 8, initialStage, jointSpeedRange, ...rewards, stage2Rewards });
export const TRAINING_TASKS: TrainingTask[] = [
  { id: DIRECT_WALKING_TASK, label: '6DOF Walking（zbot train.sh）', description: '对应 zbot_rl_mjlab/train.sh：无阶段课程，从随机权重开始，0.25～1.0 Hz 步频、8192 环境、600 次更新。', taskCard: { ...card(1, [.2, 2]), stage1Rewards: {}, stage2Rewards: zbotTrainRewards } },
];

export const copyTaskCardSettings = (value: TrainingTaskCardSettings): TrainingTaskCardSettings => ({ stage1Rewards: { ...value.stage1Rewards }, stage2Rewards: { ...value.stage2Rewards }, terminatedRewardPenalty: value.terminatedRewardPenalty });

/** Apply a task deliberately; resource refreshes must not overwrite user choices. */
export function taskDefaults(taskId: string, resources: TrainingResources, device?: string): TrainingConfig {
  const selectedDevice = device ?? (resources.runtime.cudaBuild !== false && resources.gpus.length ? `cuda:${resources.gpus[0].id}` : 'cpu');
  const task = TRAINING_TASKS.find(task => task.id === taskId);
  return { taskId, device: selectedDevice, cpuThreads: Math.max(1, Math.min(4, resources.cpu.availableThreads)),
    numEnvs: selectedDevice === 'cpu' ? 4 : 8192,
    iterations: 600, saveInterval: 50, seed: 42, maxSeconds: 3600,
    ...(task ? { taskCard: copyTaskCardSettings(task.taskCard) } : {}) };
}
