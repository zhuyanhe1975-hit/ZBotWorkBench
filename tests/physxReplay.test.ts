import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { before, test } from 'node:test';
import loadMujoco from '@mujoco/mujoco';
import loadPhysx from 'physx-js-webidl';
import { MujocoEngine } from '../src/mujoco/MujocoEngine';
import { PhysxSimulation, type PhysxModel } from '../src/rl/physx';
import { loadCheckpoint } from '../src/rl/checkpoint';
import { PolicyReplay } from '../src/rl/replay';
import { REPLAY_PROFILES, type ReplayProfile } from '../src/rl/profiles';

let mujoco: Awaited<ReturnType<typeof loadMujoco>>;
let physx: any;
const rollingNative = JSON.parse(readFileSync(new URL('./fixtures/rollingNative.json', import.meta.url), 'utf8'));
const bipedalNativeById: Record<string, any> = Object.fromEntries([
  ['IsaacGym-ZBotBipedalWalking', '../results/legacy-bipedal-walking/native.json'],
  ['IsaacGym-ZBotBipedalWalking_20250514', '../results/legacy-bipedal-walking/ZBotBipedalWalking_20250514.json'],
  ['IsaacGym-ZBotBipedalWalking_20250527', '../results/legacy-bipedal-walking/ZBotBipedalWalking_20250527.json'],
].map(([id, path]) => [id, JSON.parse(readFileSync(new URL(path, import.meta.url), 'utf8'))]));
const runShNativeById: Record<string, any> = Object.fromEntries([
  ['IsaacGym-zbot_side_moving', 'side-moving.json'], ['IsaacGym-zbot_forward_moving', 'forward-moving.json'],
  ['IsaacGym-ZBotStandUp', 'stand-up.json'], ['IsaacGym-last_ZBotFootDown_ep_2500_rew_970.54364', 'foot-down.json'],
  ['IsaacGym-ZBotSingleLeg', 'single-leg.json'], ['IsaacGym-ZBotSnakeAndBipedal', 'snake-and-bipedal.json'],
  ['IsaacGym-ZBotBipedalWalking_BigFoot', 'bipedal-bigfoot.json'],
  ['IsaacGym-ZBotBipedalWalking_BigFoot_8DOF', 'bipedal-bigfoot-8dof.json'],
].map(([id, file]) => [id, JSON.parse(readFileSync(new URL(`../results/run-sh/${file}`, import.meta.url), 'utf8'))]));
before(async () => {
  mujoco = await loadMujoco();
  mujoco.FS.writeFile('ma.obj', readFileSync(new URL('../public/assets/ma.obj', import.meta.url)));
  mujoco.FS.writeFile('mb.obj', readFileSync(new URL('../public/assets/mb.obj', import.meta.url)));
  const wasmBinary = readFileSync(new URL('../node_modules/physx-js-webidl/physx-js-webidl.wasm', import.meta.url));
  // This browser-only upstream loader otherwise assumes CommonJS require in Node.
  // Supply the binary and select its browser branch without fetching any resource.
  const originalProcess = globalThis.process;
  const hadWindow = Object.hasOwn(globalThis, 'window');
  const originalWindow = (globalThis as any).window;
  try {
    (globalThis as any).process = undefined;
    (globalThis as any).window = {};
    physx = await (loadPhysx as any)({ wasmBinary });
  } finally {
    globalThis.process = originalProcess;
    if (hadWindow) (globalThis as any).window = originalWindow;
    else delete (globalThis as any).window;
  }
});

function modelFor(profile: ReplayProfile): PhysxModel {
  return JSON.parse(readFileSync(new URL(`../public/rl/physx/${profile.model}.json`, import.meta.url), 'utf8'));
}
function displayFor(profile: ReplayProfile) {
  const display = new MujocoEngine();
  (display as any).mujoco = mujoco;
  display.loadModelFromXml(readFileSync(new URL(`../public/rl/models/${profile.displayModel ?? profile.model}.xml`, import.meta.url), 'utf8'));
  return display;
}
function fixture(profile: ReplayProfile) {
  const display = displayFor(profile);
  const model = modelFor(profile);
  model.simulation = { ...model.simulation, ...profile.physxSimulation };
  if (profile.origin === 'isaacgym') model.defaultQ = [...profile.defaultAngles];
  model.displayJointNames = profile.displayJointNames;
  model.displayJointSigns = profile.displayJointSigns;
  model.displayRootQuaternionOffset = profile.displayRootQuaternionOffset;
  const dynamics = new PhysxSimulation(physx, model, display,
    { physicsDt: profile.physicsDt, controlDt: profile.controlDt,
      rootPosition: profile.initialRootPosition, rootQuaternion: profile.initialRootQuaternion });
  const bytes = readFileSync(new URL(`../public/rl/checkpoints/${profile.id}/${profile.checkpoint}`, import.meta.url));
  const policy = loadCheckpoint(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);
  const replay = new PolicyReplay(display, profile, policy, dynamics);
  return { display, dynamics, replay, dispose: () => { dynamics.dispose(); display.destroy(); } };
}
function near(actual: number, expected: number, tolerance = 2e-6) {
  assert.ok(Math.abs(actual - expected) < tolerance, `${actual} differs from ${expected}`);
}
function multiplyQuaternion(a: number[], b: number[]) {
  return [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
    a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
    a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]];
}
function assertDisplayMatchesNative(display: MujocoEngine, dynamics: PhysxSimulation, profile: ReplayProfile) {
  const state = dynamics.state();
  const data = display.getData(), model = display.getModel();
  for (let i = 0; i < state.positions.length; i++) {
    const sign = profile.displayJointSigns?.[i] ?? 1;
    near(data.qpos[7 + i], sign * state.positions[i], 1e-12);
    near(data.qvel[6 + i], sign * state.velocities[i], 1e-12);
  }
  if (profile.displayRootQuaternionOffset) {
    const expected = multiplyQuaternion(state.rootQuaternion!, profile.displayRootQuaternionOffset);
    expected.forEach((value, i) => near(data.qpos[3 + i], value, 2e-6));
  }
  if (profile.displayModel) return;
  const base = dynamics.getBasePosition();
  // PhysX transforms are Float32. Captured legacy importer frames also retain
  // small GPU-settling offsets that the source display MJCF cannot encode.
  for (let i = 0; i < 3; i++) near(data.sensordata[model.sensor_adr[2] + i], base[i], Math.max(5e-3, Math.abs(base[i]) * 2 ** -22));
  const similarity = state.quaternion.reduce((s, q, i) => s + q * data.sensordata[i], 0);
  // The archived BigFoot 8DOF display MJCF and the reconstructed PhysX
  // principal-inertia frames differ by a few 1e-5 in quaternion dot product.
  near(Math.abs(similarity), 1, 5e-5);
}

for (const profile of REPLAY_PROFILES) {
  test(`${profile.id}: actual checkpoint completes its PhysX validation horizon and resets deterministically`, () => {
    const f = fixture(profile);
    try {
      const initial = f.dynamics.getBasePosition();
      const initialState = f.dynamics.state();
      const renderBodies = f.dynamics.getRenderBodies();
      assert.ok(renderBodies.length > 0, `${profile.id}: missing physical render bodies`);
      assert.equal(new Set(renderBodies.map(body => body.name)).size, renderBodies.length, profile.id);
      for (const body of renderBodies) {
        assert.ok(body.position.every(Number.isFinite), `${profile.id}: invalid physical render position`);
        near(Math.hypot(...body.quaternion), 1, 2e-5);
        assert.ok(body.hulls.length > 0 && body.hulls.every(hull => hull.length >= 4), `${profile.id}: missing physical hull`);
      }
      if (profile.id === 'IsaacGym-zbot_rolling') {
        const expected = rollingNative.clean_reset_reference.body_state_4_13;
        initial.forEach((value, i) => near(value, expected[i], 2e-5));
        initialState.legacyBodies![4].position.forEach((value, i) => near(value, expected[i], 2e-5));
        assert.ok(initial[2] < .15, `rolling must start prone, base height ${initial[2]}`);
      }
      const bipedalNative = bipedalNativeById[profile.id];
      if (bipedalNative) {
        const expected = bipedalNative.clean_reset.body_poses.body_3.p;
        initial.forEach((value, i) => near(value, expected[i], 2e-5));
        initialState.legacyBodies![3].position.forEach((value, i) => near(value, expected[i], 2e-5));
        assert.ok(initial[2] > .24, `bipedal walking must start upright, base height ${initial[2]}`);
      }
      const initialObs = Array.from(f.replay.lastObservation);
      if (profile.id === 'IsaacGym-zbot_rolling') {
        assert.deepEqual(initialObs, rollingNative.initial_env_reset_return_obs_25);
        const first = rollingNative.effective_deterministic_steps[0];
        f.replay.step();
        f.replay.lastActions.forEach((value, i) => near(value, first.clipped_action_18[i], 2e-6));
        Array.from(f.display.getData().ctrl as Float64Array).forEach((value, i) => near(value, first.pos_d_after_pre_physics_step_6[i], 2e-6));
        assert.deepEqual(Array.from(f.replay.observe()), first.obs_returned_after_env_step_25);
        const second = rollingNative.effective_deterministic_steps[1];
        f.replay.step();
        f.replay.lastActions.forEach((value, i) => near(value, second.clipped_action_18[i], 2e-6));
        Array.from(f.display.getData().ctrl as Float64Array).forEach((value, i) => near(value, second.pos_d_after_pre_physics_step_6[i], 2e-6));
        const secondObservation = Array.from(f.replay.observe());
        secondObservation.slice(0, 7).forEach((value, i) => near(value, second.obs_returned_after_env_step_25[i], 3e-3));
        secondObservation.slice(13, 19).forEach((value, i) => near(value, second.obs_returned_after_env_step_25[i + 13], 5e-3));
        const rms = Math.sqrt(secondObservation.reduce((sum, value, i) => sum + (value - second.obs_returned_after_env_step_25[i]) ** 2, 0) / secondObservation.length);
        assert.ok(rms < .07, `rolling second-step observation RMS ${rms}`);
        f.replay.reset();
      }
      if (bipedalNative) {
        assert.deepEqual(initialObs, bipedalNative.initial_env_reset_obs);
        for (let step = 0; step < 3; step++) {
          const expected = bipedalNative.effective_steps[step];
          f.replay.step();
          f.replay.lastActions.forEach((value, i) => near(value, expected.actions[i], 2e-6));
          Array.from(f.display.getData().ctrl as Float64Array).forEach((value, i) => near(value, expected.pos_d[i], 2e-6));
          const observation = Array.from(f.replay.observe());
          if (step < 2) assert.deepEqual(observation, expected.obs_after);
          else {
            observation.slice(1, 7).forEach((value, i) => near(value, expected.obs_after[i + 1], 1e-2));
            const rms = Math.sqrt(observation.reduce((sum, value, i) => sum + (value - expected.obs_after[i]) ** 2, 0) / observation.length);
            assert.ok(rms < .085, `${profile.id} third-step observation RMS ${rms}`);
          }
        }
        f.replay.reset();
      }
      const runShNative = runShNativeById[profile.id];
      if (runShNative) {
        assert.deepEqual(initialObs, runShNative.initial_env_reset_obs);
        const exactSteps = profile.bootstrapResetObservations!.length + 1;
        for (let step = 0; step < exactSteps; step++) {
          const expected = runShNative.effective_steps[step];
          f.replay.step();
          f.replay.lastActions.forEach((value, i) => near(value, expected.actions[i], 2e-6));
          Array.from(f.display.getData().ctrl as Float64Array).forEach((value, i) =>
            near(value, (profile.displayJointSigns?.[i] ?? 1) * expected.pos_d[i], 5e-5));
          if (step < profile.bootstrapResetObservations!.length) {
            assert.deepEqual(Array.from(f.replay.observe()), expected.obs_after);
          }
        }
        f.replay.reset();
      }
      let minimumHeight = Infinity, maximumHeight = -Infinity;
      let first30: ReturnType<PhysxSimulation['state']>;
      const duration = profile.origin === 'isaacgym' ? 2 : 20;
      const stepCount = Math.round(duration / (profile.controlDt ?? 1 / 30));
      for (let i = 0; i < stepCount; i++) {
        f.replay.step();
        const pos = f.dynamics.getBasePosition();
        minimumHeight = Math.min(minimumHeight, pos[2]);
        maximumHeight = Math.max(maximumHeight, pos[2]);
        if (i === 29) first30 = f.dynamics.state();
        if (i % 30 === 0) assertDisplayMatchesNative(f.display, f.dynamics, profile);
      }
      const final = f.dynamics.getBasePosition();
      const displacement = Math.hypot(final[0] - initial[0], final[1] - initial[1]);
      near(f.display.getTime(), stepCount * (profile.controlDt ?? 1 / 30), 1e-9);
      assert.equal(f.replay.controlSteps, stepCount);
      if (profile.id === 'Zbot-Direct-8dof-bipedal-v0') {
        assert.ok(minimumHeight > .30, `human fell: minimum height ${minimumHeight}`);
        assert.ok(final[0] - initial[0] < -5, `human forward travel ${final[0] - initial[0]} m`);
      } else if (profile.id === 'Zbot-Direct-6dof-bipedal-quat-v0') {
        assert.ok(minimumHeight > .22, `six-DOF biped fell: minimum height ${minimumHeight}`);
        assert.ok(final[0] - initial[0] > 2.5, `biped forward travel ${final[0] - initial[0]} m`);
      } else if (profile.id === 'Zbot-Direct-8dof-snake-v0') {
        assert.ok(displacement > 2, `snake displacement ${displacement} m`);
        assert.ok(minimumHeight > .025 && maximumHeight < .12, `snake height range ${minimumHeight}–${maximumHeight}`);
      } else if (profile.id.startsWith('Zbot-Direct-6dof-bipedal-to-snake-')) {
        assert.ok(initial[2] > .22, 'transition starts upright');
        assert.ok(final[2] > .025 && final[2] < .12, `transition ends low: ${final[2]} m`);
        if (profile.id.endsWith('-v1')) assert.ok(displacement > 1, `transition v1 displacement ${displacement} m`);
      } else if (profile.id === 'Zbot-Direct-6dof-bipedal-v0') {
        assert.ok(minimumHeight > .22 && final[0] - initial[0] > 1, `6DOF min=${minimumHeight}, x=${final[0] - initial[0]}`);
      } else if (['Zbot-Direct-8dof-bipedal-v1', 'Zbot-Direct-8dof-bipedal-run-v0'].includes(profile.id)) {
        assert.ok(minimumHeight > .30 && final[0] - initial[0] < -5, `${profile.id}: min=${minimumHeight}, x=${final[0] - initial[0]}`);
      } else if (profile.id === 'Zbot-Direct-8dof-bipedal-v2') {
        assert.ok(minimumHeight > .30 && final[0] - initial[0] > 2, `v2 min=${minimumHeight}, x=${final[0] - initial[0]}`);
      } else if (profile.id === 'Zbot-Direct-8dof-bird-v0') {
        assert.ok(minimumHeight > .25 && final[0] - initial[0] < -5, `bird min=${minimumHeight}, x=${final[0] - initial[0]}`);
      } else if (profile.id === 'Zbot-Direct-8dof-wheel-v0') {
        assert.ok(displacement > 2 && minimumHeight > .02 && maximumHeight < .5, `wheel xy=${displacement}, height=${minimumHeight}–${maximumHeight}`);
      } else if (profile.id === 'Zbot-Direct-6dof-bipedal-velocity-imu-v0') {
        assert.ok(minimumHeight > .22 && displacement > .5, `IMU min=${minimumHeight}, xy=${displacement}`);
      } else if (profile.id === 'ZbotRlIsaaclab-6DOF-Periodic-Walking') {
        assert.ok(minimumHeight > .22 && final[0] - initial[0] > 2,
          `periodic walking min=${minimumHeight}, x=${final[0] - initial[0]}`);
      } else if (profile.id === 'IsaacGym-ZBotSingleLeg') {
        assert.ok(minimumHeight > .18 && displacement > .8, `SingleLeg min=${minimumHeight}, xy=${displacement}`);
      } else if (profile.origin === 'isaacgym') {
        assert.notDeepEqual(f.dynamics.state().positions, initialState.positions, 'legacy policy must drive the joints');
        assert.ok(f.replay.lastActions.every(Number.isFinite));
      } else {
        assert.ok(['Zbot-Direct-8dof-bipedal-v3', 'Zbot-Direct-6dof-bipedal-velocity-v0', 'Zbot-Direct-8dof-bipedal-velocity-v0'].includes(profile.id));
        assert.ok(profile.note, 'Experimental source-policy limitations must remain visible');
      }
      f.replay.reset();
      assert.equal(f.replay.controlSteps, 0);
      assert.equal(f.display.getTime(), 0);
      assert.deepEqual(f.dynamics.state(), initialState);
      assert.deepEqual(Array.from(f.replay.lastObservation), initialObs);
      for (let i = 0; i < 30; i++) f.replay.step();
      assert.deepEqual(f.dynamics.state(), first30!);
      assertDisplayMatchesNative(f.display, f.dynamics, profile);
    } finally { f.dispose(); }
  });
}

test('PhysX rejects mismatched joint names before creating a simulation', () => {
  const profile = REPLAY_PROFILES[0];
  const display = displayFor(profile);
  try {
    const model = modelFor(profile);
    model.jointNames[0] = 'missing_joint';
    assert.throws(() => new PhysxSimulation(physx, model, display), /关节/);
  } finally { display.destroy(); }
});

test('repeated PhysX scene load/reset/dispose releases resources and preserves memory capacity', () => {
  const capacities: number[] = [];
  for (let i = 0; i < 8; i++) {
    const f = fixture(REPLAY_PROFILES[i % REPLAY_PROFILES.length]);
    for (let step = 0; step < 10; step++) f.replay.step();
    f.replay.reset();
    f.dynamics.dispose();
    f.dynamics.dispose();
    assert.throws(() => f.dynamics.state(), /未就绪/);
    f.display.destroy();
    capacities.push(physx.HEAPU8.buffer.byteLength);
  }
  // WASM memory cannot shrink, but repeated release/recreate must not keep growing.
  assert.ok(capacities.at(-1)! - capacities[2] <= 16 * 1024 * 1024, `heap capacities: ${capacities}`);
});
