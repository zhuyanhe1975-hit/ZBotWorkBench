import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { before, test } from 'node:test';
import loadMujoco from '@mujoco/mujoco';
import loadPhysx from 'physx-js-webidl';
import { MujocoEngine } from '../src/mujoco/MujocoEngine';
import { PhysxSimulation, type PhysxModel } from '../src/rl/physx';
import { loadCheckpoint } from '../src/rl/checkpoint';
import { PolicyReplay } from '../src/rl/replay';
import { REPLAY_PROFILES } from '../src/rl/profiles';

let mj: any, P: any;
before(async () => {
  mj = await loadMujoco();
  const process = globalThis.process, window = (globalThis as any).window;
  try {
    (globalThis as any).process = undefined; (globalThis as any).window = {};
    P = await (loadPhysx as any)({ wasmBinary: readFileSync(new URL('../node_modules/physx-js-webidl/physx-js-webidl.wasm', import.meta.url)) });
  } finally {
    globalThis.process = process;
    if (window === undefined) delete (globalThis as any).window; else (globalThis as any).window = window;
  }
});
function fixture(contactObservations: boolean, lift = 0, footNames?: [string, string], modelKey = 'zbot_8s_human') {
  const data: PhysxModel = JSON.parse(readFileSync(new URL(`../public/rl/physx/${modelKey}.json`, import.meta.url), 'utf8'));
  data.simulation = { ...data.simulation, contactObservations, ...(footNames ? { footNames } : {}) };
  for (const body of data.bodies) body.pos[2] += lift;
  const display = new MujocoEngine(); (display as any).mujoco = mj;
  display.loadModelFromXml(readFileSync(new URL(`../public/rl/models/${modelKey}.xml`, import.meta.url), 'utf8'));
  const sim = new PhysxSimulation(P, data, display);
  return { sim, display, targets: data.defaultQ, dispose: () => { sim.dispose(); display.destroy(); } };
}

test('contact impulse sensor reports upward support force and clears history on reset', () => {
  const f = fixture(true);
  try {
    assert.deepEqual(f.sim.state().footContactForces, [0, 0]);
    assert.deepEqual(f.sim.state().footAirTimes, [0, 0]);
    const samples: number[] = [];
    for (let i = 0; i < 90; i++) {
      f.sim.step(f.targets);
      const state = f.sim.state();
      const force = state.footContactForces!.reduce((a, b) => a + b, 0);
      if (i > 30) samples.push(force);
    }
    const support = samples.reduce((a, b) => a + b, 0) / samples.length;
    assert.ok(support > 20 && support < 70, `expected upward ground support near robot weight, got ${support} N`);
    assert.ok(f.sim.state().footAirTimes!.some(time => time === 0), 'grounded foot has zero air timer');
    f.sim.reset();
    assert.deepEqual(f.sim.state().footContactForces, [0, 0]);
    assert.deepEqual(f.sim.state().footAirTimes, [0, 0]);
    f.sim.step(f.targets);
    assert.ok(f.sim.state().footContactForces!.some(force => force > 0));
  } finally { f.dispose(); }
});

test('viewer diagnostics expose PhysX contacts, forces and mass-weighted center without policy contact observations', () => {
  const f = fixture(false);
  try {
    assert.deepEqual(f.sim.getDiagnostics().forceLines, []);
    f.sim.setDebugVisualization(true);
    for (let i = 0; i < 90; i++) f.sim.step(f.targets);
    const diagnostics = f.sim.getDiagnostics();
    assert.ok(diagnostics.contacts.length > 0, 'settled robot should report viewer contacts');
    for (const contact of diagnostics.contacts) {
      assert.ok(contact.position.every(Number.isFinite));
      assert.ok(contact.force.every(Number.isFinite));
      assert.ok(Number.isFinite(contact.magnitude) && contact.magnitude >= 0);
      assert.ok(Math.abs(contact.magnitude - Math.hypot(...contact.force)) < 1e-9);
    }
    assert.ok(diagnostics.contacts.some(contact => contact.magnitude > 1));
    assert.ok(diagnostics.centerOfMass.every(Number.isFinite));
    assert.ok(diagnostics.centerOfMass[2] > 0);
    assert.ok(diagnostics.contacts.some(contact => contact.force[2] > 1), 'ground point force should support the robot upward');
    assert.ok(diagnostics.forceLines.some(line => line.kind === 'normal'), 'native normal impulse lines should be available');
    assert.ok(diagnostics.forceLines.some(line => line.kind === 'friction'), 'native friction impulse lines should be available');
    for (const line of diagnostics.forceLines) assert.ok([...line.start, ...line.end].every(Number.isFinite));
    assert.ok(diagnostics.groundResultant, 'ground impulse lines should produce one foot-contact resultant');
    assert.ok(Math.abs(diagnostics.groundResultant.start[2]) < .02, 'resultant starts at the ground center of pressure');
    for (let axis = 0; axis < 3; axis++) {
      const expected = diagnostics.forceLines.filter(line => Math.abs(line.start[2]) < .02)
        .reduce((sum, line) => sum + line.end[axis] - line.start[axis], 0);
      assert.ok(Math.abs(diagnostics.groundResultant.end[axis] - diagnostics.groundResultant.start[axis] - expected) < 1e-8);
    }
    assert.deepEqual(f.sim.state().footContactForces, [0, 0]);
  } finally { f.dispose(); }
});

test('air timer advances without contact and resets from actual landing impulses', () => {
  const f = fixture(true, 1);
  try {
    for (let i = 0; i < 3; i++) f.sim.step(f.targets);
    for (const air of f.sim.state().footAirTimes!) assert.ok(Math.abs(air - .1) < 1e-10);
    assert.deepEqual(f.sim.state().footContactForces, [0, 0]);
    assert.ok(f.sim.state().linearVelocity![2] < 0, 'base COM falls under gravity');
    let landed = false;
    for (let i = 0; i < 60; i++) {
      f.sim.step(f.targets);
      const state = f.sim.state();
      if (state.footAirTimes!.some(time => time === 0) && state.footContactForces!.some(force => force > 1)) landed = true;
    }
    assert.ok(landed, 'landing impulses reset at least one foot timer');
  } finally { f.dispose(); }
});

test('enabling contact reports preserves native dynamics and exposes separate base/root frames', () => {
  const baseline = fixture(false), reporting = fixture(true);
  try {
    for (let i = 0; i < 30; i++) {
      baseline.sim.step(baseline.targets); reporting.sim.step(reporting.targets);
      const a = baseline.sim.state(), b = reporting.sim.state();
      for (const field of ['quaternion', 'angularVelocity', 'positions', 'velocities', 'linearVelocity', 'rootQuaternion', 'basePosition'] as const) {
        for (let j = 0; j < a[field]!.length; j++) assert.ok(Math.abs(a[field]![j] - b[field]![j]) < 1e-6, `${field}[${j}] changed with reporting`);
      }
    }
    const state = reporting.sim.state();
    assert.ok(state.quaternion.some((q, i) => Math.abs(q - state.rootQuaternion![i]) > .1));
    assert.deepEqual(state.basePosition, reporting.sim.getBasePosition());
    assert.deepEqual(baseline.sim.state().footContactForces, [0, 0]);
    reporting.sim.dispose(); reporting.sim.dispose();
  } finally { baseline.dispose(); reporting.dispose(); }
});


test('explicit sensor foot order changes only observation ordering', () => {
  const normal = fixture(true), reversed = fixture(true, 0, ['foot_1', 'foot_0']);
  try {
    for (let i = 0; i < 20; i++) {
      normal.sim.step(normal.targets); reversed.sim.step(reversed.targets);
      const a = normal.sim.state(), b = reversed.sim.state();
      assert.deepEqual(b.footContactForces, [...a.footContactForces!].reverse());
      assert.deepEqual(b.footAirTimes, [...a.footAirTimes!].reverse());
      assert.deepEqual(b.positions, a.positions);
    }
  } finally { normal.dispose(); reversed.dispose(); }
});


test('run checkpoint uses native sensor foot order and stays upright for 20 seconds', () => {
  const profile = REPLAY_PROFILES.find(item => item.id === 'Zbot-Direct-8dof-bipedal-run-v0')!;
  const f = fixture(true, 0, undefined, profile.model);
  try {
    const bytes = readFileSync(new URL(`../public/rl/checkpoints/${profile.id}/${profile.checkpoint}`, import.meta.url));
    const policy = loadCheckpoint(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);
    const replay = new PolicyReplay(f.display, profile, policy, f.sim);
    replay.step();
    // Native reference: foot_1 is in the air, foot_0 bears the initial ground load.
    assert.deepEqual(f.sim.state().footContactForces!.map(force => Number(force > 5)), [0, 1]);
    assert.ok(Math.abs(f.sim.state().footAirTimes![0] - 1 / 30) < 1e-8);
    assert.equal(f.sim.state().footAirTimes![1], 0);
    let minimumHeight = f.sim.getBasePosition()[2];
    for (let i = 1; i < 600; i++) {
      replay.step();
      minimumHeight = Math.min(minimumHeight, f.sim.getBasePosition()[2]);
    }
    assert.ok(minimumHeight > .30, `run fell with height ${minimumHeight}`);
    assert.ok(f.sim.getBasePosition()[0] < -5, 'run moves forward instead of merely holding posture');
  } finally { f.dispose(); }
});
