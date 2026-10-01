import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { before, test } from 'node:test';
import loadMujoco from '@mujoco/mujoco';
import { MujocoEngine } from '../src/mujoco/MujocoEngine';
import { compiledMeshGeometry } from '../src/utils/compiledMeshGeometry';
import { loadCheckpoint, inferPolicy, type PolicyNetwork } from '../src/rl/checkpoint';
import { buildObservation, integrateActions, integrateLegacyActions, policyFeatures, PolicyReplay, velocityObservationRaw, type PolicyState, type ReplayDynamics } from '../src/rl/replay';
import { CONTROL_DT, REPLAY_PROFILES, type ReplayProfile } from '../src/rl/profiles';

let mujoco: Awaited<ReturnType<typeof loadMujoco>>;
before(async () => { mujoco = await loadMujoco(); });
const human = REPLAY_PROFILES.find(p => p.id === 'Zbot-Direct-8dof-bipedal-v0')!;
const snake = REPLAY_PROFILES.find(p => p.id === 'Zbot-Direct-8dof-snake-v0')!;
const quaternion = REPLAY_PROFILES.find(p => p.id === 'Zbot-Direct-6dof-bipedal-quat-v0')!;
const periodic = REPLAY_PROFILES.find(p => p.id === 'ZbotRlIsaaclab-6DOF-Periodic-Walking')!;
const xmlFor = (profile: ReplayProfile) => readFileSync(new URL(`../public/rl/models/${profile.model}.xml`, import.meta.url), 'utf8');
function engine(xml: string) {
  const result = new MujocoEngine();
  (result as any).mujoco = mujoco;
  result.loadModelFromXml(xml);
  return result;
}
function close(actual: number, expected: number, tolerance = 1e-6) {
  assert.ok(Math.abs(actual - expected) < tolerance, `${actual} != ${expected}`);
}
function constantPolicy(profile: ReplayProfile): PolicyNetwork {
  const outputSize = profile.jointNames.length;
  return { inputSize: profile.inputSize, outputSize, activation: 'elu', layers: [{
    inputSize: profile.inputSize, outputSize,
    weights: new Float32Array(profile.inputSize * outputSize),
    bias: Float32Array.from({ length: outputSize }, (_, i) => (i + 1) / 10),
  }] };
}
function nameAt(model: any, address: number): string {
  let result = '';
  for (let i = address; i >= 0 && i < model.names.length && model.names[i]; i++) result += String.fromCharCode(model.names[i]);
  return result;
}

test('observation uses base world angular velocity and task-specific gravity/heading semantics', () => {
  // A +90° Y rotation maps local forward +Z to world +X and world gravity to local +X.
  const rotation = [Math.SQRT1_2, 0, Math.SQRT1_2, 0];
  for (const profile of [human, snake]) {
    const previous = profile.jointNames.map((_, i) => i / 10);
    const obs = buildObservation(profile, { quaternion: rotation, angularVelocity: [2, 3, 7],
      positions: profile.defaultAngles.map(q => q + 0.25), velocities: profile.jointNames.map(() => -0.5) }, previous);
    close(obs[0], 7); // world angular Z; rotating it into the body would incorrectly yield -2
    close(obs[1], 1); close(obs[2], 0); close(obs[3], 0);
    close(obs[4], profile === human ? 1 : 0);
    const n = profile.jointNames.length;
    for (let i = 0; i < n; i++) { close(obs[5 + i], 0.25); close(obs[5 + n + i], -0.5); close(obs[5 + 2 * n + i], previous[i]); }
    close(obs.at(-1)!, 2);
  }
  const obs = buildObservation(quaternion, { quaternion: rotation, angularVelocity: [2, 3, 7],
    positions: quaternion.defaultAngles, velocities: Array(6).fill(0) }, new Float32Array(6));
  for (let i = 0; i < 7; i++) close(obs[i], [...rotation, 2, 3, 7][i]);
  assert.equal(obs.length, 26);
  assert.throws(() => buildObservation(human, { quaternion: [NaN, 0, 0, 1], angularVelocity: [0, 0, 0], positions: human.defaultAngles, velocities: Array(8).fill(0) }, new Float32Array(8)), /观测/);
});

test('IsaacLab periodic walking observation preserves its 45-field training order and phase direction', () => {
  const state: PolicyState = {
    quaternion: [1, 0, 0, 0], angularVelocity: [.1, .2, .3], linearVelocity: [1, 2, 3],
    positions: [...periodic.defaultAngles], velocities: Array(6).fill(0),
    bodyComPositions: Array.from({ length: 12 }, () => [0, -.05, 1]), bodyMasses: Array(12).fill(1),
    extremityPositions: [[0, -.1, 0], [0, .1, 0]],
    extremityContactForces: [[0, 0, 80], [0, 0, 20]],
  };
  const previous = [.1, .2, .3, .4, .5, .6];
  const first = buildObservation(periodic, state, previous, { frequency: 1.5, phase: .25 });
  assert.equal(first.length, 45);
  [1, 2, 3, .1, .2, .3, 0, 0, -1].forEach((value, index) => close(first[index], value));
  previous.forEach((value, index) => close(first[21 + index], value));
  [0, 0, .8, 0, 0, .2].forEach((value, index) => close(first[27 + index], value));
  [0, .05, 1, 0, -.15, 1].forEach((value, index) => close(first[33 + index], value));
  [1.5, 1, 1, 0, -.5, 0].forEach((value, index) => close(first[39 + index], value));
  const other = buildObservation(periodic, state, previous, { frequency: 1.5, phase: .75 });
  [1.5, -1, -1, 0, .5, 0].forEach((value, index) => close(other[39 + index], value));
});

test('actions apply tanh, integrate clamped deltas, and preserve default-plus-delta beyond pi', () => {
  const delta = new Float64Array([0, Math.PI - 0.01, -Math.PI + 0.01]);
  const defaults = [0.4, 2.02, -2.02];
  const result = integrateActions([0.5, 100, -100], delta, defaults);
  close(result.actions[0], Math.tanh(0.5));
  close(delta[0], Math.PI * 2 * Math.tanh(0.5) * CONTROL_DT);
  close(delta[1], Math.PI); close(delta[2], -Math.PI);
  close(result.targets[1], 2.02 + Math.PI); close(result.targets[2], -2.02 - Math.PI);
  assert.ok(result.targets[1] > Math.PI && result.targets[2] < -Math.PI);
  assert.throws(() => integrateActions([0], new Float64Array(2), [0, 0]), /维度/);
  assert.throws(() => integrateActions([NaN], new Float64Array(1), [0]), /非有限/);
});

test('named base sensors, joint coordinates and reordered actuators determine replay state and targets', () => {
  const xml = xmlFor(human).replace(/<actuator>([\s\S]*?)<\/actuator>/, (_, body: string) =>
    `<actuator>${body.match(/<position[^>]+\/>/g)!.reverse().join('\n')}</actuator>`);
  const e = engine(xml);
  try {
    const replay = new PolicyReplay(e, human, constantPolicy(human));
    const model = e.getModel(), data = e.getData();
    // Middle body base is rotated by the articulated initial pose, while free root is identity.
    assert.ok(Math.abs(replay.lastObservation[1]) + Math.abs(replay.lastObservation[2]) > 0.1);
    for (let i = 0; i < human.jointNames.length; i++) data.qpos[7 + i] = human.defaultAngles[i] + (i + 1) * 0.01;
    e.forward();
    const obs = replay.observe();
    for (let i = 0; i < 8; i++) close(obs[5 + i], (i + 1) * 0.01);
    replay.step();
    for (let a = 0; a < model.nu; a++) {
      const name = nameAt(model, model.name_jntadr[model.actuator_trnid[2 * a]]);
      const i = human.jointNames.indexOf(name);
      close(data.ctrl[a], human.defaultAngles[i] + Math.PI * 2 * Math.tanh((i + 1) / 10) * CONTROL_DT);
    }
    const next = replay.observe();
    for (let i = 0; i < 8; i++) close(next[21 + i], Math.tanh((i + 1) / 10));
    assert.throws(() => new PolicyReplay(e, { ...human, jointNames: ['missing', ...human.jointNames.slice(1)] }, constantPolicy(human)), /缺少关节/);
    assert.throws(() => new PolicyReplay(e, human, { ...constantPolicy(human), inputSize: 26 }), /网络为/);
  } finally { e.destroy(); }
  const withoutSensors = engine(xmlFor(human).replace(/<sensor>[\s\S]*?<\/sensor>/, ''));
  try { assert.throws(() => new PolicyReplay(withoutSensors, human, constantPolicy(human)), /缺少传感器/); }
  finally { withoutSensors.destroy(); }
  const wrongTimestep = engine(xmlFor(human).replace(/timestep="[^"]+"/, 'timestep="0.0166666666667"'));
  try { assert.throws(() => new PolicyReplay(wrongTimestep, human, constantPolicy(human)), /控制周期/); }
  finally { wrongTimestep.destroy(); }
});

const baselineIds = new Set(['Zbot-Direct-8dof-bipedal-v0', 'Zbot-Direct-8dof-snake-v0',
  'Zbot-Direct-6dof-bipedal-quat-v0', 'Zbot-Direct-6dof-bipedal-to-snake-v0', 'Zbot-Direct-6dof-bipedal-to-snake-v1']);
for (const profile of REPLAY_PROFILES.filter(p => baselineIds.has(p.id))) {
  test(`${profile.id}: bundled checkpoint drives CPU WASM and resets deterministically`, () => {
    const bytes = readFileSync(new URL(`../public/rl/checkpoints/${profile.id}/${profile.checkpoint}`, import.meta.url));
    const policy = loadCheckpoint(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);
    assert.equal(policy.inputSize, profile.inputSize);
    assert.equal(policy.outputSize, profile.jointNames.length);
    const e = engine(xmlFor(profile));
    try {
      const replay = new PolicyReplay(e, profile, policy);
      const initial = Array.from(e.getData().qpos as Float64Array);
      const initialObs = Array.from(replay.lastObservation);
      for (let i = 0; i < 35; i++) replay.step();
      const first35 = Array.from(e.getData().qpos as Float64Array);
      for (let i = 35; i < 600; i++) replay.step();
      const moved = Array.from(e.getData().qpos as Float64Array);
      assert.ok(moved.every(Number.isFinite));
      assert.ok(Array.from(e.getData().qvel as Float64Array).every(Number.isFinite));
      assert.ok(moved.slice(7).some((q, i) => Math.abs(q - initial[7 + i]) > 1e-3), 'policy must move articulated joints');
      assert.equal(replay.controlSteps, 600);
      close(e.getTime(), 20);
      for (let warning = 0; warning < e.getData().warning.size(); warning++) assert.equal(e.getData().warning.get(warning).number, 0, `MuJoCo warning ${warning}`);
      replay.reset();
      assert.equal(replay.controlSteps, 0);
      assert.equal(e.getTime(), 0);
      assert.deepEqual(Array.from(e.getData().qpos), initial);
      assert.deepEqual(Array.from(replay.lastObservation), initialObs);
      assert.ok(replay.lastActions.every(a => a === 0));
      for (let i = 0; i < 35; i++) replay.step();
      assert.deepEqual(Array.from(e.getData().qpos), first35);
    } finally { e.destroy(); }
  });
}

test('static model inertias, sensors and visual meshes survive WASM compilation', () => {
  for (const profile of [human, snake, quaternion]) {
    const e = engine(xmlFor(profile));
    try {
      const model = e.getModel();
      close(model.opt.timestep, 1 / 600, 1e-12);
      const mass = Array.from(model.body_mass as Float64Array).reduce((a, b) => a + b, 0);
      close(mass, profile.jointNames.length === 8 ? 4.006720066 : 3.005040050, 1e-7);
      const expected = ['rl_base_quat', 'rl_base_angvel', 'rl_base_pos', 'rl_base_linvel'];
      for (let i = 0; i < expected.length; i++) {
        assert.equal(nameAt(model, model.name_sensoradr[i]), expected[i]);
        assert.equal(nameAt(model, model.name_bodyadr[model.sensor_objid[i]]), 'base');
      }
      let visualCount = 0;
      for (let g = 0; g < model.ngeom; g++) {
        if (model.geom_group[g] !== 1) continue;
        visualCount++;
        const id = model.geom_dataid[g];
        const geometry = compiledMeshGeometry(model, id);
        try {
          assert.equal(geometry.getAttribute('position').count, model.mesh_vertnum[id]);
          assert.equal(geometry.index!.count, model.mesh_facenum[id] * 3);
          assert.ok(Array.from(geometry.getAttribute('position').array).every(Number.isFinite));
          geometry.computeBoundingBox();
          const bounds = geometry.boundingBox!;
          assert.ok(bounds.max.distanceTo(bounds.min) > 0.05);
          assert.ok(bounds.max.distanceTo(bounds.min) < 0.25, 'metre scale module geometry');
        } finally { geometry.dispose(); }
      }
      assert.equal(visualCount, profile.jointNames.length * 2);
    } finally { e.destroy(); }
  }
});

const velocity6 = REPLAY_PROFILES.find(p => p.id === 'Zbot-Direct-6dof-bipedal-velocity-v0')!;
const imu = REPLAY_PROFILES.find(p => p.observation === 'imu')!;
const run = REPLAY_PROFILES.find(p => p.observation === 'run')!;
const legacy = REPLAY_PROFILES.find(p => p.id === 'IsaacGym-ZBotBipedalWalking_20250527')!;
function syntheticState(profile: ReplayProfile): PolicyState {
  return { quaternion: [Math.SQRT1_2, 0, Math.SQRT1_2, 0], rootQuaternion: [1, 0, 0, 0],
    angularVelocity: [2, 3, 7], linearVelocity: [1, 2, 3], basePosition: [0, 0, .35],
    positions: profile.defaultAngles.map(q => q + .1), velocities: profile.jointNames.map(() => -.2),
    footContactForces: [5, 6], footAirTimes: [.1, 0] };
}
test('velocity and IMU observations distinguish base/body frames, root gravity, commands and phase', () => {
  const raw = velocityObservationRaw(syntheticState(velocity6));
  // +90deg about Y: local linear [-3,2,1], angular [-7,3,2]; local forward -Y.
  [...raw.linear].forEach((v, i) => close(v, [-2, 1, 1][i]));
  [...raw.angular].forEach((v, i) => close(v, [-7, 3, 7][i]));
  const tilted = { ...syntheticState(velocity6), quaternion: [Math.cos(Math.PI / 12), 0, Math.sin(Math.PI / 12), 0] };
  close(velocityObservationRaw(tilted).linear[0], -2); // velocity tasks normalize forward direction
  for (const profile of [velocity6, imu]) {
    const obs = buildObservation(profile, syntheticState(profile), new Float32Array(6), {
      filteredLinear: [9, 8, 7], filteredAngular: [6, 5, 4], commands: [.2, -.1, .3], time: 1 / (4 * 1.2),
    });
    const head = profile === imu ? 4 : 3;
    const first = profile === imu ? syntheticState(profile).quaternion : [9, 8, 7];
    first.forEach((v, i) => close(obs[i], v));
    [6, 5, 4, 0, 0, -1, .2, -.1, .3, 1, 0, 0].forEach((v, i) => close(obs[head + i], v));
    assert.equal(obs.length, profile.inputSize);
    const noContext = buildObservation(profile, syntheticState(profile), new Float32Array(6));
    close(noContext[head + 6], .2); close(noContext[head + 9], 0); close(noContext[head + 10], 1);
  }
  assert.throws(() => buildObservation(velocity6, { ...syntheticState(velocity6), rootQuaternion: undefined }, new Float32Array(6)), /根节点/);
});
test('run observation includes absolute joints, unnormalized direction, contact flags and air times', () => {
  const state = syntheticState(run), obs = buildObservation(run, state, new Float32Array(8));
  assert.equal(obs.length, 40);
  [-2, 1, 3, -3, 2, 7, 1, 0, 0, 1, .35].forEach((v, i) => close(obs[i], v));
  state.positions.forEach((v, i) => close(obs[11 + i], v));
  [0, 1, .1, 0, 2].forEach((v, i) => close(obs[35 + i], v));
  const tilted = buildObservation(run, { ...state, quaternion: [Math.cos(Math.PI / 12), 0, Math.sin(Math.PI / 12), 0] }, new Float32Array(8));
  close(tilted[0], -1); close(tilted[3], -1.5); close(tilted[9], .5); // run keeps forward magnitude sin(30°)
  assert.throws(() => buildObservation(run, { ...state, footContactForces: undefined }, new Float32Array(8)), /接触/);
});
test('legacy Isaac Gym observation preserves its 36-value world-state contract', () => {
  const state: PolicyState = {
    quaternion: [1, 0, 0, 0], angularVelocity: [4, 5, 6], linearVelocity: [1, 2, 3], basePosition: [.1, .2, .3],
    positions: legacy.defaultAngles.map((_, i) => (i + 1) * .1), velocities: legacy.jointNames.map((_, i) => i + 1),
    bodyComPositions: [[0, 0, 0], [2, 0, 0]], bodyMasses: [1, 3],
    extremityPositions: [[0, 1, 0], [2, 0, 0]], extremityContactForces: [[7, 8, 9], [10, 11, 12]],
    legacyBodies: Array.from({ length: 7 }, () => ({ position: [.1, .2, .3], quaternion: [1, 0, 0, 0],
      linearVelocity: [1, 2, 3], angularVelocity: [4, 5, 6], mass: 1 })),
  };
  const obs = buildObservation(legacy, state, new Float32Array(6));
  assert.equal(obs.length, 36);
  [legacy.jointSpeedLimit, .1, .2, .3, 0, 0, 1, 1, 2, 3, 4, 5, 6, 1.5, 0, 0,
    Math.hypot(1.5, -1), .5].forEach((value, i) => close(obs[i], value!));
  state.positions.forEach((value, i) => close(obs[18 + i], legacy.jointSigns![i] * value / Math.PI));
  state.velocities.forEach((value, i) => close(obs[24 + i], legacy.jointSigns![i] * value * .2));
  [7, 8, 9, 10, 11, 12].forEach((value, i) => close(obs[30 + i], value));
  assert.throws(() => buildObservation(legacy, { ...state, bodyMasses: undefined }, new Float32Array(6)), /质心/);
});
test('converted legacy checkpoint matches rl_games CPU inference', () => {
  const bytes = readFileSync(new URL(`../public/rl/checkpoints/${legacy.id}/${legacy.checkpoint}`, import.meta.url));
  const policy = loadCheckpoint(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);
  const observation = Float32Array.from({ length: 36 }, (_, i) => -.5 + i / 35);
  const expected = [-3.9074974060058594, -3.5163440704345703, -1.797161340713501,
    -8.801865577697754, -.38750961422920227, 1.27094566822052];
  inferPolicy(policy, observation).forEach((value, i) => close(value, expected[i], 5e-6));
  assert.equal(policy.normalization?.clip, 5);
});
test('wheel delta, velocity target bounds and run action scaling match task controllers', () => {
  const wheel = REPLAY_PROFILES.find(p => p.motion === 'wheel')!;
  const velocity8 = REPLAY_PROFILES.find(p => p.id === 'Zbot-Direct-8dof-bipedal-velocity-v0')!;
  for (const profile of [wheel, velocity8]) {
    const delta = Float64Array.from(profile.defaultAngles, (_, i) => i % 2 ? -Math.PI : Math.PI);
    const result = integrateActions(profile.defaultAngles.map((_, i) => i % 2 ? -100 : 100), delta, profile.defaultAngles, profile);
    delta.forEach((v, i) => close(v, i % 2 ? -Math.PI / 2 : Math.PI / 2));
    result.targets.forEach((v, i) => close(v, profile.defaultAngles[i] + (i % 2 ? -Math.PI / 2 : Math.PI / 2)));
  }
  const result = integrateActions([.5], new Float64Array(1), [.2], run);
  close(result.targets[0], .2 + Math.PI * 2 * 1.2 * Math.tanh(.5) * CONTROL_DT);
});
test('legacy direct, three-parameter and four-parameter controllers produce bounded joint targets', () => {
  for (const [id, time, tuple] of [
    ['IsaacGym-ZBotBipedalWalking_20250527', 0, [2]],
    ['IsaacGym-zbot_rolling', 0, [0, 1, .25]],
    ['IsaacGym-ZBotFootDown', .5, [0, 1, 0, 0]],
  ] as const) {
    const profile = REPLAY_PROFILES.find(value => value.id === id)!;
    const raw = Float32Array.from({ length: profile.policyOutputSize! }, (_, i) => tuple[i % tuple.length]);
    const result = integrateLegacyActions(raw, new Float64Array(profile.jointNames.length),
      Array(profile.jointNames.length).fill(0), profile, time);
    assert.equal(result.actions.length, profile.policyOutputSize);
    assert.equal(result.targets.length, profile.jointNames.length);
    assert.ok(result.targets.every(Number.isFinite));
    assert.ok(result.targets.some(value => Math.abs(value) > 1e-6));
    assert.ok(result.actions.every(value => value >= -1 && value <= 1));
    assert.ok(result.targets.every(value => Math.abs(value) <= profile.deltaLimit! + 1e-9));
  }
  assert.equal(REPLAY_PROFILES.find(value => value.id === 'IsaacGym-zbot_rolling')!.deltaLimit, Math.PI / 2);
  assert.equal(REPLAY_PROFILES.find(value => value.id === 'IsaacGym-ZBotFootDown')!.deltaLimit, .75 * Math.PI);
  assert.equal(REPLAY_PROFILES.find(value => value.id === 'IsaacGym-ZBotBipedalWalking_20250527')!.deltaLimit, Math.PI);
  assert.deepEqual(REPLAY_PROFILES.find(value => value.id === 'IsaacGym-ZBotSingleLeg')!.defaultAngles, [0, 0, 0, 0, 0, 0]);
  const rolling = REPLAY_PROFILES.find(value => value.id === 'IsaacGym-zbot_rolling')!;
  assert.deepEqual(rolling.defaultAngles, [0, 0, 0, 0, 0, 0]);
  assert.equal(rolling.model, 'legacy_zbot_rolling');
  assert.equal(rolling.initialRootPosition, undefined);
  assert.equal(rolling.initialRootQuaternion, undefined);
  assert.equal(rolling.bootstrapResetObservations?.[0].length, 25);
  assert.deepEqual(REPLAY_PROFILES.find(value => value.id === 'IsaacGym-ZBotBipedalWalking_BigFoot')!.defaultAngles,
    [0, -Math.PI / 4, Math.PI / 2, -Math.PI / 2, Math.PI / 4, 0]);
});
test('runtime joint speed parameter changes the policy observation and action integration', () => {
  const e = engine(xmlFor(quaternion));
  try {
    const replay = new PolicyReplay(e, quaternion, constantPolicy(quaternion));
    assert.equal(replay.lastObservation.at(-1), 2);
    replay.setJointSpeedLimit(.7);
    close(replay.observe().at(-1)!, .7);
    replay.step();
    close(replay.lastObservation.at(-1)!, .7);
    close(e.getData().ctrl[0], quaternion.defaultAngles[0] + Math.PI * .7 * Math.tanh(.1) * CONTROL_DT);
    assert.equal(replay.getJointSpeedLimit(), .7);
    assert.throws(() => replay.setJointSpeedLimit(0), /0.1–5/);
    assert.throws(() => replay.setJointSpeedLimit(Infinity), /0.1–5/);
  } finally { e.destroy(); }
});
test('velocity filter updates once per control step, resets deterministically and accepts bounded commands', () => {
  const e = engine(xmlFor(velocity6));
  let state = syntheticState(velocity6);
  const initial = structuredClone(state);
  const dynamics: ReplayDynamics = {
    reset() { state = structuredClone(initial); e.resetPolicyPose(); },
    state() { return structuredClone(state); },
    step() { state.linearVelocity = [3, 4, 5]; state.angularVelocity = [4, 5, 9]; e.getData().time += CONTROL_DT; },
  };
  try {
    const replay = new PolicyReplay(e, velocity6, constantPolicy(velocity6), dynamics);
    const initialObs = [...replay.lastObservation];
    replay.step();
    const obs = replay.observe(), raw = velocityObservationRaw(state), alpha = 1 - Math.exp(-CONTROL_DT / .08);
    for (let i = 0; i < 3; i++) close(obs[i], initialObs[i] + alpha * (raw.linear[i] - initialObs[i]));
    assert.deepEqual([...replay.observe()], [...obs], 'observing twice must not advance EMA');
    replay.setCommands([.3, -.2, .5]);
    [.3, -.2, .5].forEach((v, i) => close(replay.observe()[9 + i], v));
    assert.throws(() => replay.setCommands([.5, 0, 0]), /范围/);
    assert.throws(() => replay.setCommands([0, NaN, 0]), /范围/);
    replay.setCommands([.2, 0, 0]); replay.reset();
    assert.deepEqual([...replay.lastObservation], initialObs);
    assert.throws(() => new PolicyReplay(e, velocity6, constantPolicy(velocity6)), /PhysX/);
  } finally { e.destroy(); }
});


test('mjlab quaternion features and ELU inference match the original Python model fixture', () => {
  const fixtures = JSON.parse(readFileSync('tests/fixtures/quatFeatures.json', 'utf8'));
  const bytes = readFileSync('tests/fixtures/quatFeatures.pt');
  const policy = loadCheckpoint(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength), 'elu');
  const profile: ReplayProfile = { ...quaternion, origin: 'mjlab', policyFeatures: 'quat-gravity-heading-v1' };
  assert.equal(policy.inputSize, 30);
  for (const sample of fixtures.samples) {
    const obs = new Float32Array(sample.observation);
    const features = policyFeatures(profile, obs);
    assert.equal(features.length, 30);
    assert.deepEqual(features.slice(0, 26), obs);
    features.forEach((v, i) => close(v, sample.features[i], 3e-7));
    inferPolicy(policy, features).forEach((v, i) => close(v, sample.actions[i], 3e-7));
    assert.equal(policyFeatures(quaternion, obs), obs);
  }
  assert.throws(() => policyFeatures({ ...profile, origin: undefined }, new Float32Array(26)), /特征/);
  assert.throws(() => policyFeatures(profile, new Float32Array(30)), /特征/);
  const e = engine(xmlFor(quaternion));
  try {
    const replay = new PolicyReplay(e, profile, policy);
    const obs = replay.observe();
    const expected = inferPolicy(policy, policyFeatures(profile, obs));
    replay.step();
    assert.equal(replay.lastObservation.length, 26);
    replay.lastActions.forEach((v, i) => close(v, Math.tanh(expected[i])));
    assert.throws(() => new PolicyReplay(e, quaternion, policy), /网络为/);
    assert.throws(() => new PolicyReplay(e, profile, constantPolicy(quaternion)), /网络为/);
    assert.throws(() => new PolicyReplay(e, profile, { ...policy, normalization: { mean: new Float32Array(30), std: new Float32Array(30).fill(1), epsilon: .01 } }), /归一化/);
  } finally { e.destroy(); }
});
