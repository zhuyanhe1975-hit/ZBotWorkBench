import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import loadMujoco from '@mujoco/mujoco';
import { MujocoEngine } from '../src/mujoco/MujocoEngine';
import { loadTrainingBundle, parseTrainingBundle, trainingBundleProfile } from '../src/training/bundle';
import { integrateActions } from '../src/rl/replay';
import type { TrainingReplayBundle } from '../src/training/types';

// Existing compatible geometry/weights are data fixtures, not evidence of training.
function bundle(): TrainingReplayBundle {
  return { schemaVersion: 1, source: 'mjlab', taskId: 'fixture', name: 'Training bundle fixture',
    xml: readFileSync('public/rl/models/zbot_6s_new.xml', 'utf8'), assets: {},
    checkpointBase64: readFileSync('public/rl/checkpoints/Zbot-Direct-6dof-bipedal-quat-v0/model_3850.pt').toString('base64'),
    profile: { jointNames: ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
      defaultAngles: [.312, .837, -2.02, 2.02, -.837, -.312], observation: 'quaternion', inputSize: 26,
      physicsDt: 1 / 600, controlDt: 1 / 30, jointSpeedLimit: .8 }, versions: { mjlab: 'fixture' } };
}

test('training replay restores its clock and action speed rather than Isaac defaults', async () => {
  const engine = new MujocoEngine();
  (engine as any).mujoco = await loadMujoco();
  try {
    const value = bundle();
    const { replay, profile } = loadTrainingBundle(engine, value);
    assert.equal(profile.origin, 'mjlab');
    assert.equal(profile.jointSpeedLimit, .8);
    assert.ok(Math.abs(replay.lastObservation.at(-1)! - .8) < 1e-6);
    replay.step();
    assert.ok(Math.abs(engine.getTime() - value.profile.controlDt) < 1e-8);
    replay.reset();
    assert.equal(engine.getTime(), 0);
    assert.equal(replay.controlSteps, 0);
    const delta = new Float64Array(1);
    integrateActions([.5], delta, [0], { controlDt: .02, jointSpeedLimit: .8 });
    assert.ok(Math.abs(delta[0] - Math.PI * .8 * Math.tanh(.5) * .02) < 1e-8);
  } finally { engine.destroy(); }
});

test('training bundle rejects mismatched schemas, clocks, networks and escaping VFS paths', () => {
  const original = bundle();
  assert.equal(trainingBundleProfile(parseTrainingBundle(original)).inputSize, 26);
  const extra = structuredClone(original) as any;
  extra.profile.origin = 'isaac'; extra.profile.actionScale = 999;
  assert.equal(trainingBundleProfile(parseTrainingBundle(extra)).origin, 'mjlab');
  assert.equal(trainingBundleProfile(parseTrainingBundle(extra)).actionScale, undefined);
  for (const changed of [
    { ...original, source: 'isaac' },
    { ...original, profile: { ...original.profile, policyFeatures: 'arbitrary-code' } },
    { ...original, profile: { ...original.profile, inputSize: 30 } },
    { ...original, profile: { ...original.profile, physicsDt: 0 } },
    { ...original, profile: { ...original.profile, controlDt: .031 } },
    { ...original, profile: { ...original.profile, jointSpeedLimit: Infinity } },
    { ...original, checkpointBase64: '!invalid!' },
    { ...original, assets: { '../ma.obj': 'mesh' } },
    { ...original, assets: { 'ma.obj': 'mesh' } },
    { ...original, xml: '<mujoco><include file="training_extra.xml"/></mujoco>' },
    { ...original, xml: '<mujoco><asset><mesh file="training_missing.obj"/></asset></mujoco>' },
  ]) assert.throws(() => parseTrainingBundle(changed), /训练结果包无效/);
});

test('training scenes with terrain bodies before the robot still track the actual free root', async () => {
  const engine = new MujocoEngine();
  (engine as any).mujoco = await loadMujoco();
  try {
    const value = bundle();
    value.xml = value.xml.replace('<worldbody>', '<worldbody><body name="decoration" pos="5 4 3"><geom type="sphere" size=".1" contype="0" conaffinity="0" group="0"/></body>');
    loadTrainingBundle(engine, value);
    assert.equal(engine.getRootBodyId(), 2);
    const before = engine.getMetrics();
    assert.ok(Math.abs(before.rootPos[0]) < 1e-10);
    assert.ok(Math.abs(before.rootPos[1] + .06) < 1e-10);
    engine.applyImpulse(.2, 0, 0);
    assert.equal(engine.getMetrics().rootVelocity[0], .2);
  } finally { engine.destroy(); }
});


test('feature bundles explicitly require a 30-input MLP while retaining 26 external observations', async () => {
  const value = bundle();
  value.profile.policyFeatures = 'quat-gravity-heading-v1';
  assert.equal(trainingBundleProfile(parseTrainingBundle(value)).policyFeatures, 'quat-gravity-heading-v1');
  // Shape failure must precede engine loading.
  assert.throws(() => loadTrainingBundle({} as MujocoEngine, value), /权重维度/);
  value.checkpointBase64 = readFileSync('tests/fixtures/quatFeatures.pt').toString('base64');
  const ordinary = structuredClone(value);
  delete ordinary.profile.policyFeatures;
  assert.throws(() => loadTrainingBundle({} as MujocoEngine, ordinary), /权重维度/);
  const engine = new MujocoEngine();
  (engine as any).mujoco = await loadMujoco();
  try {
    const { replay } = loadTrainingBundle(engine, value);
    replay.step();
    assert.equal(replay.lastObservation.length, 26);
    assert.ok(replay.lastActions.every(Number.isFinite));
  } finally { engine.destroy(); }
});
