import { Quaternion, Vector3 } from 'three';
import { MujocoEngine } from '../mujoco/MujocoEngine';
import { inferPolicy, PolicyNetwork } from './checkpoint';
import { CONTROL_DT, PHYSICS_DT, JOINT_SPEED_LIMIT, ReplayProfile } from './profiles';

export interface PolicyState {
  quaternion: number[]; // base body WORLD quaternion, wxyz (not the foot/root)
  angularVelocity: number[]; // base body WORLD angular velocity
  positions: number[];
  velocities: number[];
  linearVelocity?: number[]; // middle base COM velocity in world coordinates
  rootQuaternion?: number[]; // foot_0, used by the velocity tasks' gravity sensor
  basePosition?: number[];
  footContactForces?: number[]; // mean world-Z force over five physics samples
  footAirTimes?: number[];
}

/** A policy controls either the training-compatible PhysX scene or the MuJoCo comparison. */
export interface ReplayDynamics {
  reset(): void;
  step(targets: ArrayLike<number>): void;
  state(): PolicyState;
}

export interface ObservationContext {
  commands?: [number, number, number];
  time?: number;
  filteredLinear?: ArrayLike<number>;
  filteredAngular?: ArrayLike<number>;
}

export function velocityObservationRaw(state: PolicyState): { linear: Float32Array; angular: Float32Array } {
  if (!state.linearVelocity) throw new Error('速度策略缺少机身线速度');
  const [w, x, y, z] = state.quaternion;
  const inverse = new Quaternion(x, y, z, w).invert();
  const gravity = new Vector3(0, 0, -1).applyQuaternion(inverse);
  const forward = gravity.cross(new Vector3(0, 0, 1)).normalize();
  const velocity = new Vector3(...state.linearVelocity as [number, number, number]).applyQuaternion(inverse);
  const angular = new Vector3(...state.angularVelocity as [number, number, number]).applyQuaternion(inverse);
  // The source's side direction is local Z, so its last two linear terms coincide.
  return { linear: new Float32Array([velocity.dot(forward), velocity.z, velocity.z]),
    angular: new Float32Array([angular.x, angular.y, state.angularVelocity[2]]) };
}

export function buildObservation(profile: ReplayProfile, state: PolicyState, previousActions: ArrayLike<number>, context: ObservationContext = {}): Float32Array {
  const [w, x, y, z] = state.quaternion;
  const quat = new Quaternion(x, y, z, w);
  let head: number[];
  const jointSpeedLimit = profile.jointSpeedLimit ?? JOINT_SPEED_LIMIT;
  let suffix: number[] = [jointSpeedLimit];
  if (profile.observation === 'velocity' || profile.observation === 'imu') {
    if (!state.rootQuaternion) throw new Error('速度策略缺少根节点重力观测');
    const raw = velocityObservationRaw(state);
    const [rw, rx, ry, rz] = state.rootQuaternion;
    const gravity = new Vector3(0, 0, -1).applyQuaternion(new Quaternion(rx, ry, rz, rw).invert());
    const frequency = profile.phaseFrequency ?? 1.2;
    const phase = 2 * Math.PI * frequency * (context.time ?? 0);
    head = [...(profile.observation === 'imu' ? state.quaternion : Array.from(context.filteredLinear ?? raw.linear)),
      ...Array.from(context.filteredAngular ?? raw.angular), ...gravity.toArray(), ...(context.commands ?? profile.commands ?? [0, 0, 0]),
      Math.sin(phase), Math.cos(phase), 2 * (frequency - .8) / .8 - 1];
    suffix = [];
  } else if (profile.observation === 'run') {
    if (!state.linearVelocity || !state.basePosition || !state.footContactForces || !state.footAirTimes) throw new Error('跑步策略缺少速度、机身高度或足端接触观测');
    const inverse = quat.clone().invert();
    const gravity = new Vector3(0, 0, -1).applyQuaternion(inverse);
    const forward = gravity.clone().cross(new Vector3(0, 0, 1));
    const velocity = new Vector3(...state.linearVelocity as [number, number, number]).applyQuaternion(inverse);
    const angular = new Vector3(...state.angularVelocity as [number, number, number]).applyQuaternion(inverse);
    head = [velocity.dot(forward), velocity.z, state.linearVelocity[2], angular.dot(forward), angular.z, state.angularVelocity[2],
      ...gravity.toArray(), -forward.clone().applyQuaternion(quat).y, state.basePosition[2]];
    suffix = [...state.footContactForces.map(force => force > 5 ? 1 : 0), ...state.footAirTimes, jointSpeedLimit];
  } else if (profile.observation === 'quaternion') {
    head = [...state.quaternion, ...state.angularVelocity];
  } else {
    const gravity = new Vector3(0, 0, -1).applyQuaternion(quat.clone().invert());
    const forward = profile.observation === 'snake'
      ? new Vector3(0, 0, 1).applyQuaternion(quat)
      : gravity.clone().cross(new Vector3(0, 0, 1)).applyQuaternion(quat);
    const heading = profile.observation === 'snake' ? Math.hypot(forward.y, forward.z) : -forward.y;
    head = [state.angularVelocity[2], ...gravity.toArray(), heading];
  }
  const obs = new Float32Array([...head, ...state.positions.map((q, i) => profile.observation === 'run' ? q : q - profile.defaultAngles[i]),
    ...state.velocities, ...Array.from(previousActions), ...suffix]);
  if (obs.length !== profile.inputSize || !obs.every(Number.isFinite)) throw new Error('观测维度或数值无效，已停止策略');
  return obs;
}

/** Exact QuatFeatureMLP input: preserve the raw observation and append walking_frame.
 * Both quaternion rotation and gravity-cross-Z forward remain unnormalized, as in mjlab.
 */
export function policyFeatures(profile: ReplayProfile, observation: Float32Array): Float32Array {
  if (profile.policyFeatures === undefined) return observation;
  if (profile.policyFeatures !== 'quat-gravity-heading-v1' || profile.origin !== 'mjlab'
    || profile.observation !== 'quaternion' || profile.inputSize !== 26 || observation.length !== 26
    || !observation.every(Number.isFinite)) throw new Error('不支持此策略特征变换');
  const [w, x, y, z] = observation;
  const quat = new Quaternion(x, y, z, w);
  const gravity = new Vector3(0, 0, -1).applyQuaternion(quat.clone().invert());
  const forward = gravity.clone().cross(new Vector3(0, 0, 1)).applyQuaternion(quat);
  return new Float32Array([...observation, ...gravity.toArray(), -forward.y]);
}

/** Exactly the training environment's tanh + integrated position-delta controller. */
export function integrateActions(raw: ArrayLike<number>, delta: Float64Array, defaults: number[], options: Pick<ReplayProfile, 'deltaLimit' | 'actionScale' | 'targetLimit' | 'controlDt' | 'jointSpeedLimit'> = {}): { actions: Float32Array; targets: Float64Array } {
  if (raw.length !== delta.length || raw.length !== defaults.length) throw new Error('动作维度不匹配');
  const actions = Float32Array.from(raw, value => Math.tanh(value));
  if (!actions.every(Number.isFinite)) throw new Error('策略输出包含非有限数值');
  const targets = new Float64Array(delta.length);
  for (let i = 0; i < delta.length; i++) {
    const limit = options.deltaLimit ?? Math.PI;
    delta[i] = Math.max(-limit, Math.min(limit, delta[i] + Math.PI * (options.jointSpeedLimit ?? JOINT_SPEED_LIMIT) * (options.actionScale ?? 1) * actions[i] * (options.controlDt ?? CONTROL_DT)));
    targets[i] = defaults[i] + delta[i];
    if (options.targetLimit !== undefined) {
      targets[i] = Math.max(defaults[i] - options.targetLimit, Math.min(defaults[i] + options.targetLimit, targets[i]));
      delta[i] = targets[i] - defaults[i];
    }
  }
  return { actions, targets };
}

export class PolicyReplay {
  private previousActions: Float32Array;
  private delta: Float64Array;
  private joints: { qpos: number; dof: number; actuator: number }[];
  private sensorAddresses: Record<string, number> = {};
  public lastObservation = new Float32Array();
  public lastActions = new Float32Array();
  public controlSteps = 0;
  private commands: [number, number, number];
  private filterStep = -1;
  private filteredLinear: Float32Array | null = null;
  private filteredAngular: Float32Array | null = null;
  public readonly controlDt: number;
  private readonly physicsDt: number;

  constructor(private engine: MujocoEngine, public profile: ReplayProfile, private policy: PolicyNetwork, private dynamics?: ReplayDynamics) {
    this.commands = [...(profile.commands ?? [0, 0, 0])];
    this.controlDt = profile.controlDt ?? CONTROL_DT;
    this.physicsDt = profile.physicsDt ?? PHYSICS_DT;
    const ratio = this.controlDt / this.physicsDt;
    if (!Number.isFinite(ratio) || this.physicsDt <= 0 || ratio < 1 || ratio > 1000 || Math.abs(ratio - Math.round(ratio)) > 1e-6) throw new Error('策略控制周期必须是物理周期的整数倍');
    if (!dynamics && profile.mujocoCompatible === false) throw new Error('此任务需要 PhysX 的完整状态与接触观测，请选择 PhysX 引擎');
    // Validate the feature contract before touching the simulation model.
    const networkInputSize = policyFeatures(profile, new Float32Array(profile.inputSize)).length;
    if (profile.policyFeatures && policy.normalization) throw new Error('四元数特征策略不支持观测归一化');
    if (policy.inputSize !== networkInputSize || policy.outputSize !== profile.jointNames.length) {
      throw new Error(`网络为 ${policy.inputSize}→${policy.outputSize}，所选任务需要 ${networkInputSize}→${profile.jointNames.length}；请选择匹配的训练任务。`);
    }
    const model = engine.getModel();
    if (!model || model.nu !== profile.jointNames.length || !dynamics && Math.abs(model.opt.timestep - this.physicsDt) > 1e-9) throw new Error('仿真模型与策略控制周期不匹配');
    const name = (address: number) => {
      let result = '';
      for (let i = address; i >= 0 && i < model.names.length && model.names[i]; i++) result += String.fromCharCode(model.names[i]);
      return result;
    };
    this.joints = profile.jointNames.map(jointName => {
      const joint = Array.from(model.name_jntadr as Int32Array).findIndex(a => name(a) === jointName);
      const actuator = Array.from({ length: model.nu }, (_, i) => i).find(i => model.actuator_trnid[2 * i] === joint);
      if (joint < 0 || actuator === undefined) throw new Error(`策略模型缺少关节 ${jointName}`);
      return { qpos: model.jnt_qposadr[joint], dof: model.jnt_dofadr[joint], actuator };
    });
    for (let i = 0; i < model.nsensor; i++) this.sensorAddresses[name(model.name_sensoradr[i])] = model.sensor_adr[i];
    for (const s of ['rl_base_quat', 'rl_base_angvel']) if (this.sensorAddresses[s] === undefined) throw new Error(`策略模型缺少传感器 ${s}`);
    this.delta = new Float64Array(profile.jointNames.length);
    this.previousActions = new Float32Array(profile.jointNames.length);
    this.reset();
  }

  public reset(): void {
    if (this.dynamics) this.dynamics.reset();
    else this.engine.resetPolicyPose();
    this.delta.fill(0); this.previousActions.fill(0); this.controlSteps = 0;
    this.filterStep = -1; this.filteredLinear = this.filteredAngular = null;
    this.lastActions = new Float32Array(this.previousActions);
    this.lastObservation = this.observe();
  }

  public observe(): Float32Array {
    if (this.dynamics) {
      const state = this.dynamics.state();
      if ((this.profile.observation === 'velocity' || this.profile.observation === 'imu') && this.filterStep !== this.controlSteps) {
        const raw = velocityObservationRaw(state);
        if (!this.filteredLinear || !this.filteredAngular) {
          this.filteredLinear = raw.linear; this.filteredAngular = raw.angular;
        } else {
          const alpha = 1 - Math.exp(-this.controlDt / .08);
          for (let i = 0; i < 3; i++) {
            this.filteredLinear[i] += alpha * (raw.linear[i] - this.filteredLinear[i]);
            this.filteredAngular[i] += alpha * (raw.angular[i] - this.filteredAngular[i]);
          }
        }
        this.filterStep = this.controlSteps;
      }
      return buildObservation(this.profile, state, this.previousActions, { commands: this.commands, time: this.controlSteps * this.controlDt,
        filteredLinear: this.filteredLinear ?? undefined, filteredAngular: this.filteredAngular ?? undefined });
    }
    const data = this.engine.getData();
    const sensor = (key: string, n: number) => Array.from(data.sensordata.slice(this.sensorAddresses[key], this.sensorAddresses[key] + n)) as number[];
    return buildObservation(this.profile, { quaternion: sensor('rl_base_quat', 4), angularVelocity: sensor('rl_base_angvel', 3),
      positions: this.joints.map(j => data.qpos[j.qpos]), velocities: this.joints.map(j => data.qvel[j.dof]) }, this.previousActions);
  }

  public setCommands(commands: [number, number, number]): void {
    if (!this.profile.commands || !commands.every(Number.isFinite) || Math.abs(commands[0]) > .4 || Math.abs(commands[1]) > .4 || Math.abs(commands[2]) > 1) throw new Error('速度命令超出训练范围');
    this.commands = [...commands];
  }

  public step(): void {
    this.lastObservation = this.observe();
    const { actions, targets } = integrateActions(inferPolicy(this.policy, policyFeatures(this.profile, this.lastObservation)), this.delta, this.profile.defaultAngles, this.profile);
    const data = this.engine.getData();
    for (let i = 0; i < this.joints.length; i++) data.ctrl[this.joints[i].actuator] = targets[i];
    this.previousActions = actions; this.lastActions = actions;
    const before = data.time;
    if (this.dynamics) this.dynamics.step(targets);
    else {
      this.engine.step(Math.round(this.controlDt / this.physicsDt));
      this.engine.forward();
      for (let i = 0; i < data.warning.size(); i++) {
        if (data.warning.get(i).number) throw new Error('MuJoCo 报告数值或容量警告，已停止回放，请重置');
      }
    }
    if (Math.abs(data.time - before - this.controlDt) > 1e-7) throw new Error('物理引擎时间异常，已停止回放，请重置');
    if (!Array.from(data.qpos as Float64Array).every(Number.isFinite) || !Array.from(data.qvel as Float64Array).every(Number.isFinite)) throw new Error('仿真状态无效，请重置回放');
    this.controlSteps++;
  }
}
