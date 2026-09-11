export interface ReplayProfile {
  id: string;
  label: string;
  model: string;
  checkpoint: string;
  observation: 'bipedal' | 'snake' | 'quaternion' | 'velocity' | 'imu' | 'run';
  inputSize: number;
  jointNames: string[];
  defaultAngles: number[];
  actionScale?: number;
  deltaLimit?: number;
  targetLimit?: number;
  commands?: [number, number, number];
  phaseFrequency?: number;
  mujocoCompatible?: boolean;
  motion?: 'walking' | 'snake' | 'transition' | 'wheel' | 'velocity' | 'run';
  note?: string;
  controlDt?: number;
  physicsDt?: number;
  jointSpeedLimit?: number;
  origin?: 'mjlab';
  policyFeatures?: 'quat-gravity-heading-v1';
}

const sixAngles = [.312, .837, -2.02, 2.02, -.837, -.312];
const eightJoints = Array.from({ length: 8 }, (_, i) => `joint${i}`);
const sixJoints = Array.from({ length: 6 }, (_, i) => `joint${i + 1}`);
const radians = (degrees: number[]) => degrees.map(v => v * Math.PI / 180);
const humanV1 = [-.08, 0, ...radians([25, -125, 125, -25]), 0, .08];
// These contracts match the supplied checkpoints and the named task implementations.
// Equal input dimensions alone do not identify a task's observation semantics.
export const REPLAY_PROFILES: ReplayProfile[] = [
  { id: 'Zbot-Direct-8dof-bipedal-v0', label: '8DOF 双足 v0', model: 'zbot_8s_human', checkpoint: 'model_4050.pt', observation: 'bipedal', inputSize: 30, jointNames: eightJoints, defaultAngles: [0, ...sixAngles, 0] },
  { id: 'Zbot-Direct-8dof-snake-v0', label: '8DOF 蛇形 v0', model: 'zbot_8s_snake_v0', checkpoint: 'model_12050.pt', observation: 'snake', inputSize: 30, jointNames: eightJoints, defaultAngles: Array(8).fill(0) },
  { id: 'Zbot-Direct-6dof-bipedal-quat-v0', label: '6DOF 双足 · 四元数观测', model: 'zbot_6s_new', checkpoint: 'model_3850.pt', observation: 'quaternion', inputSize: 26, jointNames: sixJoints, defaultAngles: sixAngles },
  { id: 'Zbot-Direct-6dof-bipedal-to-snake-v0', label: '6DOF 双足到蛇形 v0', model: 'zbot_6s_new', checkpoint: 'model_1000.pt', observation: 'quaternion', inputSize: 26, jointNames: sixJoints, defaultAngles: sixAngles },
  { id: 'Zbot-Direct-6dof-bipedal-to-snake-v1', label: '6DOF 双足到蛇形 v1', model: 'zbot_6s_new', checkpoint: 'model_3400.pt', observation: 'quaternion', inputSize: 26, jointNames: sixJoints, defaultAngles: sixAngles },
  { id: 'Zbot-Direct-6dof-bipedal-v0', label: '6DOF 双足 v0', model: 'zbot_6s_new', checkpoint: 'model_8350.pt', observation: 'bipedal', inputSize: 24, jointNames: sixJoints, defaultAngles: sixAngles, motion: 'walking' },
  { id: 'Zbot-Direct-8dof-bipedal-v1', label: '8DOF 双足 v1', model: 'zbot_8s_human_v1', checkpoint: 'model_4850.pt', observation: 'bipedal', inputSize: 30, jointNames: eightJoints, defaultAngles: humanV1, motion: 'walking' },
  { id: 'Zbot-Direct-8dof-bipedal-v2', label: '8DOF 双足 v2', model: 'zbot_8s_human_v2', checkpoint: 'model_1550.pt', observation: 'bipedal', inputSize: 30, jointNames: eightJoints, defaultAngles: radians([0, 10, -10, 140, -140, 10, -10, 0]), motion: 'walking' },
  { id: 'Zbot-Direct-8dof-bipedal-v3', label: '8DOF 双足 v3（实验）', model: 'zbot_8s_human_v3', checkpoint: 'model_1700.pt', observation: 'bipedal', inputSize: 30, jointNames: eightJoints, defaultAngles: radians([15, 10, 0, 120, -120, 0, -10, -15]), motion: 'walking', note: '该历史权重在原生 Isaac 的当前姿态及历史初始朝向下也会倒地；此处提供原样回放，不代表已学会稳定行走。' },
  { id: 'Zbot-Direct-8dof-bird-v0', label: '8DOF 鸟形', model: 'zbot_8s_bird', checkpoint: 'model_13900.pt', observation: 'bipedal', inputSize: 30, jointNames: eightJoints, defaultAngles: [0, ...sixAngles, 0], motion: 'walking' },
  { id: 'Zbot-Direct-8dof-wheel-v0', label: '8DOF 轮式', model: 'zbot_8s_wheel', checkpoint: 'model_800.pt', observation: 'snake', inputSize: 30, jointNames: eightJoints, defaultAngles: [1.2, -1.2, 1.8, -1.2, 1.8, -1.2, 1.2, -1.2], deltaLimit: Math.PI / 2, motion: 'wheel' },
  { id: 'Zbot-Direct-6dof-bipedal-velocity-v0', label: '6DOF 全向速度（实验）', model: 'zbot_6s_new', checkpoint: 'model_latest.pt', observation: 'velocity', inputSize: 33, jointNames: sixJoints, defaultAngles: sixAngles, commands: [.2, 0, 0], phaseFrequency: 1.2, mujocoCompatible: false, motion: 'velocity', note: '该历史速度权重的跟踪稳定性有限：固定前进 0.2 m/s 的原生测试中也会失稳。可用速度命令比较其原有行为。' },
  { id: 'Zbot-Direct-8dof-bipedal-velocity-v0', label: '8DOF 全向速度（实验）', model: 'zbot_8s_human_v1', checkpoint: 'model_latest.pt', observation: 'velocity', inputSize: 39, jointNames: eightJoints, defaultAngles: humanV1, targetLimit: Math.PI / 2, commands: [.2, 0, 0], phaseFrequency: 1.2, mujocoCompatible: false, motion: 'velocity', note: '该历史速度权重在固定前进 0.2 m/s 的原生测试中也会失稳；保持原权重，未把跌倒隐藏为自动重置。' },
  { id: 'Zbot-Direct-6dof-bipedal-velocity-imu-v0', label: '6DOF IMU 学生策略', model: 'zbot_6s_new', checkpoint: 'model_latest.pt', observation: 'imu', inputSize: 34, jointNames: sixJoints, defaultAngles: sixAngles, commands: [.2, 0, 0], phaseFrequency: 1.2, mujocoCompatible: false, motion: 'velocity' },
  { id: 'Zbot-Direct-8dof-bipedal-run-v0', label: '8DOF 跑步', model: 'zbot_8s_run', checkpoint: 'model_latest.pt', observation: 'run', inputSize: 40, jointNames: eightJoints, defaultAngles: [0, ...sixAngles, 0], actionScale: 1.2, mujocoCompatible: false, motion: 'run' },
];

export const CONTROL_DT = 1 / 30;
// Ten solver subdivisions per original 60 Hz PhysX step keep MuJoCo contacts stable.
export const PHYSICS_DT = 1 / 600;
export const JOINT_SPEED_LIMIT = 2;
