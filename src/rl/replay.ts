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
  bodyComPositions?: number[][];
  bodyMasses?: number[];
  extremityContactForces?: [number[], number[]];
  extremityPositions?: [number[], number[]];
  legacyBodies?: { position: number[]; quaternion: number[]; linearVelocity: number[]; angularVelocity: number[]; mass: number }[];
}

/** A policy controls either the training-compatible PhysX scene or the MuJoCo comparison. */
export interface ReplayDynamics {
  reset(elapsedTime?: number, targets?: ArrayLike<number>): void;
  step(targets: ArrayLike<number>): void;
  resetAsync?(elapsedTime?: number, targets?: ArrayLike<number>): Promise<void>;
  stepAsync?(targets: ArrayLike<number>): Promise<void>;
  state(): PolicyState;
}

export interface ObservationContext {
  commands?: [number, number, number];
  time?: number;
  phase?: number;
  frequency?: number;
  filteredLinear?: ArrayLike<number>;
  filteredAngular?: ArrayLike<number>;
}

function periodicWalkingObservation(profile: ReplayProfile, state: PolicyState, previousActions: ArrayLike<number>, context: ObservationContext): Float32Array {
  if (!state.linearVelocity || !state.bodyComPositions || !state.bodyMasses
    || !state.extremityPositions || !state.extremityContactForces
    || state.bodyComPositions.length !== state.bodyMasses.length || previousActions.length !== 6) {
    throw new Error('周期行走策略缺少机身、质心或足端接触观测');
  }
  const [w, x, y, z] = state.quaternion;
  const inverse = new Quaternion(x, y, z, w).invert();
  const local = (vector: number[]) => new Vector3(...vector as [number, number, number]).applyQuaternion(inverse).toArray();
  const mass = state.bodyMasses.reduce((sum, value) => sum + value, 0);
  if (!(mass > 0)) throw new Error('周期行走策略刚体质量无效');
  const com = [0, 1, 2].map(axis => state.bodyComPositions!.reduce(
    (sum, position, index) => sum + position[axis] * state.bodyMasses![index], 0) / mass);
  const feet = state.extremityPositions;
  const phase = context.phase ?? ((context.time ?? 0) * (context.frequency ?? profile.phaseFrequency ?? 1)) % 1;
  const frequency = context.frequency ?? profile.phaseFrequency ?? 1;
  const side = phase < .5 ? 1 : -1;
  const distances = feet.map(foot => Math.hypot(com[0] - foot[0], com[1] - foot[1]));
  const ideal = side > 0 ? distances[0] : distances[1];
  const other = side > 0 ? distances[1] : distances[0];
  const heading = Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  const observation = new Float32Array([
    ...local(state.linearVelocity), ...local(state.angularVelocity), ...local([0, 0, -1]),
    ...state.positions.map((position, index) => position - profile.defaultAngles[index]),
    ...state.velocities, ...Array.from(previousActions),
    ...state.extremityContactForces.flatMap(force => local(force).map(value => value * .01)),
    ...feet.flatMap(foot => local(com.map((value, axis) => value - foot[axis]))),
    frequency, side, Math.sin(2 * Math.PI * phase), Math.cos(2 * Math.PI * phase),
    (ideal - other) / Math.max(ideal + other, 1e-6),
    Math.atan2(Math.sin(-heading), Math.cos(-heading)),
  ]);
  if (observation.length !== 45 || !observation.every(Number.isFinite)) throw new Error('周期行走观测维度或数值无效');
  return observation;
}

function buildLegacyObservation(profile: ReplayProfile, state: PolicyState, jointSpeedLimit: number): number[] {
  if (!profile.legacyObservation || !state.legacyBodies || !state.bodyComPositions || !state.bodyMasses
    || !state.extremityContactForces || !state.extremityPositions) {
    throw new Error('旧 Isaac Gym 策略缺少刚体、质心或端部接触观测');
  }
  const signs = profile.jointSigns ?? profile.jointNames.map(() => 1);
  const positions = state.positions.map((value, i) => signs[i] * value / Math.PI);
  const velocities = state.velocities.map((value, i) => signs[i] * value * .2);
  const base = state.legacyBodies[Math.floor(state.legacyBodies.length / 2)];
  const bodyIndex = profile.legacyBodyIndex ?? (['body-joints', 'posture', 'scalar-posture'].includes(profile.legacyObservation)
    ? Math.min(Math.floor(state.legacyBodies.length / 2) + 1, state.legacyBodies.length - 1)
    : Math.floor(state.legacyBodies.length / 2));
  const observedBody = state.legacyBodies[bodyIndex];
  const bodyState = (body: typeof observedBody) => [...body.position, body.quaternion[1], body.quaternion[2], body.quaternion[3], body.quaternion[0],
    ...body.linearVelocity, ...body.angularVelocity];
  const totalMass = state.bodyMasses.reduce((sum, mass) => sum + mass, 0);
  if (!(totalMass > 0) || state.bodyComPositions.length !== state.bodyMasses.length) throw new Error('旧 Isaac Gym 质量数据无效');
  const com = [0, 1, 2].map(axis => state.bodyComPositions!.reduce((sum, position, i) => sum + position[axis] * state.bodyMasses![i], 0) / totalMass);
  const zAxis = (quaternion: number[]) => new Vector3(0, 0, 1).applyQuaternion(new Quaternion(quaternion[1], quaternion[2], quaternion[3], quaternion[0]));
  const footZ = zAxis(state.legacyBodies[0].quaternion), headZ = zAxis(state.legacyBodies.at(-1)!.quaternion);
  const standup = base.position[2] > .25 ? 1 : 0;
  const feetDirection = -Math.abs(footZ.z - 1) - Math.abs(headZ.z + 1);
  const shape = -Math.hypot(...state.positions.map((value, i) => value - profile.defaultAngles[i]));
  const contacts = [...state.extremityContactForces[0], ...state.extremityContactForces[1]];
  const posture = [...bodyState(observedBody), ...com, standup, feetDirection, shape, ...positions, ...velocities];
  switch (profile.legacyObservation) {
    case 'footdown': return [...positions, ...velocities, ...com, footZ.z, headZ.z];
    case 'body-joints': return [...bodyState(observedBody), ...positions, ...velocities];
    case 'speed-body-joints': return [jointSpeedLimit, ...bodyState(base), ...positions, ...velocities];
    case 'body-com-joints': return [...bodyState(base), ...com, ...positions, ...velocities];
    case 'command-body-com-joints': return [profile.legacyCommand ?? 1, ...bodyState(base), ...com, ...positions, ...velocities];
    case 'posture': return posture;
    case 'scalar-posture': return [jointSpeedLimit, ...posture];
    case 'body-com-joints-contact': return [...bodyState(base), ...com, ...positions, ...velocities, ...contacts];
    case 'scalar-posture-contact': return [jointSpeedLimit, ...posture, ...contacts];
    case 'scalar-posture-contact-step': return [jointSpeedLimit, ...posture, ...contacts, 0, 0];
    case 'kinematic-contact': {
      const heading = zAxis(base.quaternion);
      const distance = (position: number[]) => Math.hypot(position[0] - com[0], position[1] - com[1]);
      return [jointSpeedLimit, ...base.position, ...heading.toArray(), ...base.linearVelocity, ...base.angularVelocity,
        ...com, distance(state.extremityPositions[0]), distance(state.extremityPositions[1]), ...positions, ...velocities, ...contacts];
    }
  }
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
  if (profile.observation === 'isaaclab-periodic') return periodicWalkingObservation(profile, state, previousActions, context);
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
  } else if (profile.observation === 'isaacgym') {
    head = buildLegacyObservation(profile, state, jointSpeedLimit);
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
  const policyState = profile.observation === 'isaacgym' ? [] : [
    ...state.positions.map((q, i) => profile.observation === 'run' ? q : q - profile.defaultAngles[i]),
    ...state.velocities, ...Array.from(previousActions),
  ];
  const obs = new Float32Array([...head, ...policyState, ...suffix]);
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

/** Training-compatible action transform plus integrated position-delta controller. */
export function integrateActions(raw: ArrayLike<number>, delta: Float64Array, defaults: number[], options: Pick<ReplayProfile, 'deltaLimit' | 'actionScale' | 'targetLimit' | 'controlDt' | 'jointSpeedLimit' | 'jointSigns' | 'actionTransform'> = {}): { actions: Float32Array; targets: Float64Array } {
  if (raw.length !== delta.length || raw.length !== defaults.length) throw new Error('动作维度不匹配');
  const actions = Float32Array.from(raw, value => options.actionTransform === 'clamp'
    ? Math.max(-1, Math.min(1, value)) : Math.tanh(value));
  if (!actions.every(Number.isFinite)) throw new Error('策略输出包含非有限数值');
  const targets = new Float64Array(delta.length);
  for (let i = 0; i < delta.length; i++) {
    const limit = options.deltaLimit ?? Math.PI;
    const direction = options.jointSigns?.[i] ?? 1;
    delta[i] = Math.max(-limit, Math.min(limit, delta[i] + direction * Math.PI * (options.jointSpeedLimit ?? JOINT_SPEED_LIMIT) * (options.actionScale ?? 1) * actions[i] * (options.controlDt ?? CONTROL_DT)));
    targets[i] = defaults[i] + delta[i];
    if (options.targetLimit !== undefined) {
      targets[i] = Math.max(defaults[i] - options.targetLimit, Math.min(defaults[i] + options.targetLimit, targets[i]));
      delta[i] = targets[i] - defaults[i];
    }
  }
  return { actions, targets };
}

export function integrateLegacyActions(raw: ArrayLike<number>, delta: Float64Array, defaults: number[], profile: ReplayProfile, time: number,
  loopStep = 1): { actions: Float32Array; targets: Float64Array } {
  const controller = profile.legacyController;
  if (!controller || raw.length !== (profile.policyOutputSize ?? defaults.length) || delta.length !== defaults.length) throw new Error('旧 Isaac Gym 动作契约不匹配');
  const actions = Float32Array.from(raw, value => Math.max(-1, Math.min(1, value)));
  if (!actions.every(Number.isFinite)) throw new Error('策略输出包含非有限数值');
  const stride = controller.kind === 'direct' ? 1 : controller.kind === 'cpg3' ? 3 : 4;
  if (actions.length !== defaults.length * stride) throw new Error('旧 Isaac Gym 动作维度与控制器不匹配');
  const targets = new Float64Array(defaults.length), dt = profile.controlDt ?? CONTROL_DT;
  for (let i = 0; i < defaults.length; i++) {
    let velocity: number;
    const decay = controller.loopDecayRate === undefined ? 1
      : Math.max(controller.loopDecayMinimum ?? 0, Math.min(1, 2 - 2 / (1 + Math.exp(-controller.loopDecayRate * loopStep))));
    const velocityScale = controller.velocityScale * decay * (controller.usesJointSpeed ? profile.jointSpeedLimit ?? JOINT_SPEED_LIMIT : 1);
    if (controller.kind === 'direct') velocity = actions[i] * velocityScale;
    else {
      const offset = actions[i * stride], amplitude = (1 - Math.abs(offset)) * actions[i * stride + 1];
      const phase = actions[i * stride + 2] * 2 * Math.PI;
      const omega = controller.kind === 'cpg3' ? controller.omegaScale ?? 2 * Math.PI
        : (actions[i * stride + 3] + (controller.omegaBias ?? 1)) * (controller.omegaScale ?? Math.PI);
      velocity = velocityScale * (offset + amplitude * Math.sin(omega * time + phase));
    }
    const limit = profile.deltaLimit ?? Math.PI;
    delta[i] = Math.max(-limit, Math.min(limit, delta[i] + (profile.jointSigns?.[i] ?? 1) * velocity * dt));
    targets[i] = defaults[i] + delta[i];
  }
  return { actions, targets };
}

export class PolicyReplay {
  private previousActions: Float32Array;
  private delta: Float64Array;
  private joints: { qpos: number; dof: number; actuator: number; sign: number }[];
  private sensorAddresses: Record<string, number> = {};
  public lastObservation = new Float32Array();
  public lastActions = new Float32Array();
  public controlSteps = 0;
  private commands: [number, number, number];
  private filterStep = -1;
  private filteredLinear: Float32Array | null = null;
  private filteredAngular: Float32Array | null = null;
  private jointSpeedLimit: number;
  private periodicFrequency: number;
  private periodicPhase = 0;
  public readonly controlDt: number;
  private readonly physicsDt: number;
  private pendingZeroObservation = false;
  private pendingObservationOverride: Float32Array | null = null;
  private legacyResetPending = false;
  private previousBaseHeight: number | null = null;
  private legacySimulationSteps = 0;

  constructor(private engine: MujocoEngine, public profile: ReplayProfile, private policy: PolicyNetwork, private dynamics?: ReplayDynamics) {
    this.commands = [...(profile.commands ?? [0, 0, 0])];
    this.jointSpeedLimit = profile.jointSpeedLimit ?? JOINT_SPEED_LIMIT;
    this.periodicFrequency = profile.phaseFrequency ?? 1;
    this.controlDt = profile.controlDt ?? CONTROL_DT;
    this.physicsDt = profile.physicsDt ?? PHYSICS_DT;
    const ratio = this.controlDt / this.physicsDt;
    if (!Number.isFinite(ratio) || this.physicsDt <= 0 || ratio < 1 || ratio > 1000 || Math.abs(ratio - Math.round(ratio)) > 1e-6) throw new Error('策略控制周期必须是物理周期的整数倍');
    if (!dynamics && profile.mujocoCompatible === false) throw new Error('此任务需要 PhysX 的完整状态与接触观测，请选择 PhysX 引擎');
    // Validate the feature contract before touching the simulation model.
    const networkInputSize = policyFeatures(profile, new Float32Array(profile.inputSize)).length;
    if (profile.policyFeatures && policy.normalization) throw new Error('四元数特征策略不支持观测归一化');
    if (profile.jointSigns && (profile.jointSigns.length !== profile.jointNames.length || profile.jointSigns.some(sign => sign !== 1 && sign !== -1))) {
      throw new Error('策略关节方向映射无效');
    }
    if (profile.displayJointNames && profile.displayJointNames.length !== profile.jointNames.length
      || profile.displayJointSigns && (profile.displayJointSigns.length !== profile.jointNames.length
        || profile.displayJointSigns.some(sign => sign !== 1 && sign !== -1))) throw new Error('显示关节映射无效');
    if (profile.bootstrapResetObservations && (!profile.bootstrapResetObservations.length
      || profile.bootstrapResetObservations.some(observation => observation.length !== profile.inputSize
        || !observation.every(Number.isFinite)))) throw new Error('启动重置观测无效');
    if (profile.legacyBodyIndex !== undefined && (!Number.isInteger(profile.legacyBodyIndex) || profile.legacyBodyIndex < 0)) {
      throw new Error('旧任务观测刚体索引无效');
    }
    if (profile.legacyAutoReset && (!Object.values(profile.legacyAutoReset).some(value => value !== undefined)
      || Object.values(profile.legacyAutoReset).some(value => value !== undefined && (!Number.isFinite(value) || value <= 0)))) {
      throw new Error('旧任务自动重置参数无效');
    }
    if (profile.legacyForcedResetSteps && (!profile.legacyForcedResetSteps.length
      || profile.legacyForcedResetSteps.some((step, i) => !Number.isInteger(step) || step < 0
        || i > 0 && step <= profile.legacyForcedResetSteps![i - 1]))) throw new Error('旧任务固定重置序列无效');
    const expectedOutput = profile.policyOutputSize ?? profile.jointNames.length;
    if (policy.inputSize !== networkInputSize || policy.outputSize !== expectedOutput) {
      throw new Error(`网络为 ${policy.inputSize}→${policy.outputSize}，所选任务需要 ${networkInputSize}→${expectedOutput}；请选择匹配的训练任务。`);
    }
    const model = engine.getModel();
    if (!model || model.nu !== profile.jointNames.length || !dynamics && Math.abs(model.opt.timestep - this.physicsDt) > 1e-9) throw new Error('仿真模型与策略控制周期不匹配');
    const name = (address: number) => {
      let result = '';
      for (let i = address; i >= 0 && i < model.names.length && model.names[i]; i++) result += String.fromCharCode(model.names[i]);
      return result;
    };
    const displayJointNames = profile.displayJointNames ?? profile.jointNames;
    const displayJointSigns = profile.displayJointSigns ?? profile.jointNames.map(() => 1);
    this.joints = displayJointNames.map((jointName, index) => {
      const joint = Array.from(model.name_jntadr as Int32Array).findIndex(a => name(a) === jointName);
      const actuator = Array.from({ length: model.nu }, (_, i) => i).find(i => model.actuator_trnid[2 * i] === joint);
      if (joint < 0 || actuator === undefined) throw new Error(`策略模型缺少关节 ${jointName}`);
      return { qpos: model.jnt_qposadr[joint], dof: model.jnt_dofadr[joint], actuator, sign: displayJointSigns[index] };
    });
    for (let i = 0; i < model.nsensor; i++) this.sensorAddresses[name(model.name_sensoradr[i])] = model.sensor_adr[i];
    for (const s of ['rl_base_quat', 'rl_base_angvel']) if (this.sensorAddresses[s] === undefined) throw new Error(`策略模型缺少传感器 ${s}`);
    this.delta = new Float64Array(profile.jointNames.length);
    this.previousActions = new Float32Array(expectedOutput);
    this.reset();
  }

  public reset(): void {
    if (this.dynamics) this.dynamics.reset();
    else this.engine.resetPolicyPose();
    this.delta.fill(0); this.previousActions.fill(0); this.controlSteps = 0;
    this.periodicPhase = 0;
    this.filterStep = -1; this.filteredLinear = this.filteredAngular = null;
    this.lastActions = new Float32Array(this.previousActions);
    this.pendingZeroObservation = this.profile.zeroInitialObservation === true;
    this.pendingObservationOverride = null;
    this.legacyResetPending = false;
    this.previousBaseHeight = this.dynamics?.state().basePosition?.[2] ?? null;
    this.legacySimulationSteps = 0;
    this.lastObservation = this.pendingZeroObservation ? new Float32Array(this.profile.inputSize) : this.observe();
  }

  public observe(): Float32Array {
    if (this.pendingObservationOverride) return new Float32Array(this.pendingObservationOverride);
    const observationProfile = { ...this.profile, jointSpeedLimit: this.jointSpeedLimit };
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
      return buildObservation(observationProfile, state, this.previousActions, { commands: this.commands, time: this.controlSteps * this.controlDt,
        phase: this.periodicPhase, frequency: this.periodicFrequency,
        filteredLinear: this.filteredLinear ?? undefined, filteredAngular: this.filteredAngular ?? undefined });
    }
    const data = this.engine.getData();
    const sensor = (key: string, n: number) => Array.from(data.sensordata.slice(this.sensorAddresses[key], this.sensorAddresses[key] + n)) as number[];
    return buildObservation(observationProfile, { quaternion: sensor('rl_base_quat', 4), angularVelocity: sensor('rl_base_angvel', 3),
      positions: this.joints.map(j => data.qpos[j.qpos]), velocities: this.joints.map(j => data.qvel[j.dof]) }, this.previousActions);
  }

  public setCommands(commands: [number, number, number]): void {
    if (!this.profile.commands || !commands.every(Number.isFinite) || Math.abs(commands[0]) > .4 || Math.abs(commands[1]) > .4 || Math.abs(commands[2]) > 1) throw new Error('速度命令超出训练范围');
    this.commands = [...commands];
  }

  /** Runtime policy parameter used by both the observation and action integrator. */
  public setJointSpeedLimit(value: number): void {
    if (!Number.isFinite(value) || value < .1 || value > 5) throw new Error('关节速度参数必须在 0.1–5 之间');
    this.jointSpeedLimit = value;
  }

  public getJointSpeedLimit(): number { return this.jointSpeedLimit; }

  public setPeriodicFrequency(value: number): void {
    if (this.profile.observation !== 'isaaclab-periodic' || !Number.isFinite(value) || value < .5 || value > 2) {
      throw new Error('周期频率必须在 0.5–2.0 Hz 之间');
    }
    this.periodicFrequency = value;
    this.lastObservation = this.observe();
  }

  public step(): void {
    if (this.dynamics?.stepAsync) throw new Error('异步 PhysX 必须使用 stepAsync');
    this.lastObservation = this.pendingZeroObservation ? new Float32Array(this.profile.inputSize) : this.observe();
    this.pendingZeroObservation = false;
    this.pendingObservationOverride = null;
    const runtimeProfile = { ...this.profile, jointSpeedLimit: this.jointSpeedLimit };
    const raw = inferPolicy(this.policy, policyFeatures(this.profile, this.lastObservation));
    const { actions, targets } = this.profile.legacyController
      ? integrateLegacyActions(raw, this.delta, this.profile.defaultAngles, runtimeProfile,
        (this.legacySimulationSteps + (this.profile.legacyController.timeOffsetSteps ?? 0)) * this.controlDt, this.controlSteps + 1)
      : integrateActions(raw, this.delta, this.profile.defaultAngles, runtimeProfile);
    const data = this.engine.getData();
    for (let i = 0; i < this.joints.length; i++) data.ctrl[this.joints[i].actuator] = this.joints[i].sign * targets[i];
    this.previousActions = this.profile.observation === 'isaaclab-periodic' ? Float32Array.from(raw) : actions;
    this.lastActions = actions;
    const before = data.time;
    if (this.dynamics) this.dynamics.step(targets);
    else {
      this.engine.step(Math.round(this.controlDt / this.physicsDt));
      this.engine.forward();
      for (let i = 0; i < data.warning.size(); i++) {
        if (data.warning.get(i).number) throw new Error('MuJoCo 报告数值或容量警告，已停止回放，请重置');
      }
    }
    const bootstrapObservation = this.profile.bootstrapResetObservations?.[this.controlSteps];
    let resetThisStep = false;
    if (this.dynamics && bootstrapObservation) {
      const defaults = this.profile.bootstrapResetTargets === 'defaults';
      if (defaults) this.delta.fill(0);
      this.dynamics.reset((this.controlSteps + 1) * this.controlDt, defaults ? this.profile.defaultAngles : targets);
      this.pendingObservationOverride = Float32Array.from(bootstrapObservation);
      resetThisStep = true;
    } else if (this.dynamics && (this.legacyResetPending || this.profile.legacyForcedResetSteps?.includes(this.controlSteps))) {
      const preResetObservation = this.observe();
      this.delta.fill(0);
      this.dynamics.reset((this.controlSteps + 1) * this.controlDt, this.profile.defaultAngles);
      this.pendingObservationOverride = preResetObservation;
      this.legacyResetPending = false;
      resetThisStep = true;
    }
    if (this.dynamics && this.profile.legacyAutoReset) {
      const state = this.dynamics.state();
      if (!resetThisStep && state.basePosition && state.extremityPositions) {
        const distance = Math.hypot(state.extremityPositions[0][0] - state.extremityPositions[1][0],
          state.extremityPositions[0][1] - state.extremityPositions[1][1]);
        const reset = this.profile.legacyAutoReset;
        this.legacyResetPending = reset.fallingBaseHeight !== undefined
          && (this.previousBaseHeight ?? state.basePosition[2]) > reset.fallingBaseHeight && state.basePosition[2] < reset.fallingBaseHeight
          || reset.minimumBaseHeight !== undefined && state.basePosition[2] < reset.minimumBaseHeight
          || reset.maximumBaseHeight !== undefined && state.basePosition[2] > reset.maximumBaseHeight
          || reset.minimumExtremityDistance !== undefined && distance < reset.minimumExtremityDistance
          || reset.maximumAbsBaseX !== undefined && Math.abs(state.basePosition[0]) > reset.maximumAbsBaseX
          || reset.minimumLastExtremityHeight !== undefined && state.extremityPositions[1][2] < reset.minimumLastExtremityHeight
          || reset.maximumAbsFirstExtremityY !== undefined && Math.abs(state.extremityPositions[0][1]) > reset.maximumAbsFirstExtremityY;
      }
      this.previousBaseHeight = state.basePosition?.[2] ?? null;
    }
    this.legacySimulationSteps = resetThisStep ? 0 : this.legacySimulationSteps + 1;
    if (Math.abs(data.time - before - this.controlDt) > 1e-7) throw new Error('物理引擎时间异常，已停止回放，请重置');
    if (!Array.from(data.qpos as Float64Array).every(Number.isFinite) || !Array.from(data.qvel as Float64Array).every(Number.isFinite)) throw new Error('仿真状态无效，请重置回放');
    if (this.profile.observation === 'isaaclab-periodic') this.periodicPhase = (this.periodicPhase + this.periodicFrequency * this.controlDt) % 1;
    this.controlSteps++;
  }

  public async resetAsync(): Promise<void> {
    if (!this.dynamics?.resetAsync) { this.reset(); return; }
    await this.dynamics.resetAsync();
    this.delta.fill(0); this.previousActions.fill(0); this.controlSteps = 0;
    this.periodicPhase = 0;
    this.filterStep = -1; this.filteredLinear = this.filteredAngular = null;
    this.lastActions = new Float32Array(this.previousActions);
    this.pendingZeroObservation = this.profile.zeroInitialObservation === true;
    this.pendingObservationOverride = null; this.legacyResetPending = false;
    this.previousBaseHeight = this.dynamics.state().basePosition?.[2] ?? null;
    this.legacySimulationSteps = 0;
    this.lastObservation = this.pendingZeroObservation ? new Float32Array(this.profile.inputSize) : this.observe();
  }

  public async stepAsync(): Promise<void> {
    if (!this.dynamics?.stepAsync) { this.step(); return; }
    this.lastObservation = this.pendingZeroObservation ? new Float32Array(this.profile.inputSize) : this.observe();
    this.pendingZeroObservation = false; this.pendingObservationOverride = null;
    const runtimeProfile = { ...this.profile, jointSpeedLimit: this.jointSpeedLimit };
    const raw = inferPolicy(this.policy, policyFeatures(this.profile, this.lastObservation));
    const { actions, targets } = this.profile.legacyController
      ? integrateLegacyActions(raw, this.delta, this.profile.defaultAngles, runtimeProfile,
        (this.legacySimulationSteps + (this.profile.legacyController.timeOffsetSteps ?? 0)) * this.controlDt, this.controlSteps + 1)
      : integrateActions(raw, this.delta, this.profile.defaultAngles, runtimeProfile);
    const data = this.engine.getData();
    for (let i = 0; i < this.joints.length; i++) data.ctrl[this.joints[i].actuator] = this.joints[i].sign * targets[i];
    this.previousActions = this.profile.observation === 'isaaclab-periodic' ? Float32Array.from(raw) : actions;
    this.lastActions = actions;
    const before = data.time;
    await this.dynamics.stepAsync(targets);
    const bootstrapObservation = this.profile.bootstrapResetObservations?.[this.controlSteps];
    let resetThisStep = false;
    if (bootstrapObservation) {
      const defaults = this.profile.bootstrapResetTargets === 'defaults';
      if (defaults) this.delta.fill(0);
      await this.dynamics.resetAsync!((this.controlSteps + 1) * this.controlDt, defaults ? this.profile.defaultAngles : targets);
      this.pendingObservationOverride = Float32Array.from(bootstrapObservation); resetThisStep = true;
    } else if (this.legacyResetPending || this.profile.legacyForcedResetSteps?.includes(this.controlSteps)) {
      const preResetObservation = this.observe(); this.delta.fill(0);
      await this.dynamics.resetAsync!((this.controlSteps + 1) * this.controlDt, this.profile.defaultAngles);
      this.pendingObservationOverride = preResetObservation; this.legacyResetPending = false; resetThisStep = true;
    }
    if (this.profile.legacyAutoReset) {
      const state = this.dynamics.state();
      if (!resetThisStep && state.basePosition && state.extremityPositions) {
        const distance = Math.hypot(state.extremityPositions[0][0] - state.extremityPositions[1][0], state.extremityPositions[0][1] - state.extremityPositions[1][1]);
        const reset = this.profile.legacyAutoReset;
        this.legacyResetPending = reset.fallingBaseHeight !== undefined && (this.previousBaseHeight ?? state.basePosition[2]) > reset.fallingBaseHeight && state.basePosition[2] < reset.fallingBaseHeight
          || reset.minimumBaseHeight !== undefined && state.basePosition[2] < reset.minimumBaseHeight
          || reset.maximumBaseHeight !== undefined && state.basePosition[2] > reset.maximumBaseHeight
          || reset.minimumExtremityDistance !== undefined && distance < reset.minimumExtremityDistance
          || reset.maximumAbsBaseX !== undefined && Math.abs(state.basePosition[0]) > reset.maximumAbsBaseX
          || reset.minimumLastExtremityHeight !== undefined && state.extremityPositions[1][2] < reset.minimumLastExtremityHeight
          || reset.maximumAbsFirstExtremityY !== undefined && Math.abs(state.extremityPositions[0][1]) > reset.maximumAbsFirstExtremityY;
      }
      this.previousBaseHeight = state.basePosition?.[2] ?? null;
    }
    this.legacySimulationSteps = resetThisStep ? 0 : this.legacySimulationSteps + 1;
    if (Math.abs(data.time - before - this.controlDt) > 1e-7) throw new Error('物理引擎时间异常，已停止回放，请重置');
    if (!Array.from(data.qpos as Float64Array).every(Number.isFinite) || !Array.from(data.qvel as Float64Array).every(Number.isFinite)) throw new Error('仿真状态无效，请重置回放');
    if (this.profile.observation === 'isaaclab-periodic') this.periodicPhase = (this.periodicPhase + this.periodicFrequency * this.controlDt) % 1;
    this.controlSteps++;
  }
}
