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
before(async () => {
  mujoco = await loadMujoco();
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
  display.loadModelFromXml(readFileSync(new URL(`../public/rl/models/${profile.model}.xml`, import.meta.url), 'utf8'));
  return display;
}
function fixture(profile: ReplayProfile) {
  const display = displayFor(profile);
  const dynamics = new PhysxSimulation(physx, modelFor(profile), display);
  const bytes = readFileSync(new URL(`../public/rl/checkpoints/${profile.id}/${profile.checkpoint}`, import.meta.url));
  const policy = loadCheckpoint(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);
  const replay = new PolicyReplay(display, profile, policy, dynamics);
  return { display, dynamics, replay, dispose: () => { dynamics.dispose(); display.destroy(); } };
}
function near(actual: number, expected: number, tolerance = 2e-6) {
  assert.ok(Math.abs(actual - expected) < tolerance, `${actual} differs from ${expected}`);
}
function assertDisplayMatchesNative(display: MujocoEngine, dynamics: PhysxSimulation) {
  const state = dynamics.state();
  const data = display.getData(), model = display.getModel();
  for (let i = 0; i < state.positions.length; i++) {
    near(data.qpos[7 + i], state.positions[i], 1e-12);
    near(data.qvel[6 + i], state.velocities[i], 1e-12);
  }
  const base = dynamics.getBasePosition();
  // PhysX transforms are Float32; long wheel travel exceeds the resolution of a fixed 2µm bound.
  for (let i = 0; i < 3; i++) near(data.sensordata[model.sensor_adr[2] + i], base[i], Math.max(2e-6, Math.abs(base[i]) * 2 ** -22));
  const similarity = state.quaternion.reduce((s, q, i) => s + q * data.sensordata[i], 0);
  near(Math.abs(similarity), 1);
}

for (const profile of REPLAY_PROFILES) {
  test(`${profile.id}: actual checkpoint completes 20 seconds in PhysX and resets deterministically`, () => {
    const f = fixture(profile);
    try {
      const initial = f.dynamics.getBasePosition();
      const initialState = f.dynamics.state();
      const initialObs = Array.from(f.replay.lastObservation);
      let minimumHeight = Infinity, maximumHeight = -Infinity;
      let first30: ReturnType<PhysxSimulation['state']>;
      for (let i = 0; i < 600; i++) {
        f.replay.step();
        const pos = f.dynamics.getBasePosition();
        minimumHeight = Math.min(minimumHeight, pos[2]);
        maximumHeight = Math.max(maximumHeight, pos[2]);
        if (i === 29) first30 = f.dynamics.state();
        if (i % 30 === 0) assertDisplayMatchesNative(f.display, f.dynamics);
      }
      const final = f.dynamics.getBasePosition();
      const displacement = Math.hypot(final[0] - initial[0], final[1] - initial[1]);
      near(f.display.getTime(), 20, 1e-9);
      assert.equal(f.replay.controlSteps, 600);
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
      assertDisplayMatchesNative(f.display, f.dynamics);
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
