import test from 'node:test';
import assert from 'node:assert/strict';
import { Quaternion, Vector3 } from 'three';
import type { ZbotConfiguration } from '../src/types/zbot';
import { PRESET_CONFIGURATIONS } from '../src/data/presets';
import { getEndEffectorPose, supportsEndEffectorControl, solveInverseKinematics } from '../src/utils/inverseKinematics';

const arm = (): ZbotConfiguration => ({ ...structuredClone(PRESET_CONFIGURATIONS[1]), baseMode: 'fixed' });
const initial = [0, 55, -55, -55, 55, 0];

test('end effector control accepts custom fixed serial geometry rather than labels', () => {
  const c = arm(); assert.ok(supportsEndEffectorControl(c));
  c.category = 'custom'; assert.ok(supportsEndEffectorControl(c));
  c.modules[2].customEuler = [0, 0, -180]; assert.ok(supportsEndEffectorControl(c));
  c.modules[2].customEuler = [15, 0, 180]; assert.ok(supportsEndEffectorControl(c));
  delete c.modules[2].customEuler;
  c.baseMode = 'free'; assert.equal(supportsEndEffectorControl(c), false);
});

test('independent world translations and rotations reach target while holding the other pose component', () => {
  const c = arm(); const seed = [20, 70, -30, -75, 35, 25];
  for (let i = 0; i < 6; i++) {
    const target = getEndEffectorPose(c, seed);
    if (i < 3) target.position[i] += .003;
    else {
      const axis = new Vector3(); axis.setComponent(i - 3, 1);
      target.quaternion = new Quaternion().setFromAxisAngle(axis, .03)
        .multiply(new Quaternion(...target.quaternion)).toArray();
    }
    const result = solveInverseKinematics(c, target, seed);
    assert.ok(result.converged, `axis ${i}: ${JSON.stringify(result)}`);
  }
});

test('local six DOF IK reaches positions and orientations in several base frames', () => {
  for (const rootEuler of [[0, 0, 0], [25, -40, 75]] as [number, number, number][]) {
    const c = arm(); c.rootEuler = rootEuler; c.rootPos = [.2, -.1, .3];
    for (const q of [[5, 60, -48, -60, 50, 8], [-10, 45, -65, -48, 63, -12], [20, 70, -30, -75, 35, 25]]) {
      const result = solveInverseKinematics(c, getEndEffectorPose(c, q), initial);
      assert.ok(result.converged, JSON.stringify(result));
      assert.ok(result.positionError <= .0005); assert.ok(result.orientationError <= .01);
    }
  }
});

test('quaternion sign equivalence leaves an already reached target unchanged', () => {
  const c = arm(); const target = getEndEffectorPose(c, initial);
  target.quaternion = target.quaternion.map(v => -v) as typeof target.quaternion;
  const result = solveInverseKinematics(c, target, initial);
  assert.ok(result.converged); assert.equal(result.iterations, 0); assert.deepEqual(result.angles, initial);
});

test('unreachable and singular targets stay finite, limited and bounded', () => {
  const c = arm(); c.modules.forEach(m => { m.jointRange = [-80, 80]; });
  for (const seed of [initial, [0, 0, 0, 0, 0, 0], [180, -180, 0, 0, 0, 0]]) {
    const result = solveInverseKinematics(c, { position: [5, 5, 5], quaternion: [0, 0, 0, 1] }, seed);
    assert.equal(result.converged, false); assert.ok(result.iterations <= 100);
    assert.ok(Number.isFinite(result.positionError)); assert.ok(Number.isFinite(result.orientationError));
    assert.ok(result.angles.every(q => Number.isFinite(q) && q >= -80 && q <= 80));
  }
});

test('invalid IK target or seed is rejected before computation', () => {
  const c = arm(); const target = getEndEffectorPose(c, initial);
  assert.throws(() => solveInverseKinematics(c, { ...target, quaternion: [0, 0, 0, 0] }, initial));
  assert.throws(() => solveInverseKinematics(c, { ...target, position: [NaN, 0, 0] }, initial));
  assert.throws(() => solveInverseKinematics(c, target, [0]));
});


test('extreme finite positions cannot stall the bounded line search', () => {
  const result = solveInverseKinematics(arm(), { position: [1e308, 0, 0], quaternion: [0,0,0,1] }, initial);
  assert.equal(result.converged, false);
  assert.ok(result.iterations <= 100);
  assert.ok(result.angles.every(Number.isFinite));
});

for (const preset of PRESET_CONFIGURATIONS) {
  test(`pose and position IK for ${preset.id}`, () => {
    const c = structuredClone(preset);
    c.baseMode = 'fixed';
    assert.ok(supportsEndEffectorControl(c));
    const seed = c.modules.map(m => m.initialAngle ?? 0);
    const goal = seed.map((v, i) => v + (i % 2 ? -3 : 3));
    for (const positionOnly of [false, true]) {
      const target = getEndEffectorPose(c, goal);
      if (positionOnly) target.quaternion = [1, 0, 0, 0];
      const result = solveInverseKinematics(c, target, seed, positionOnly);
      assert.ok(result.converged, JSON.stringify(result));
      assert.ok(result.positionError <= .0005);
    }
    const unreachable = solveInverseKinematics(c, { position: [5, 5, 5], quaternion: [0, 0, 0, 1] }, seed, true);
    assert.equal(unreachable.converged, false);
    assert.ok(unreachable.angles.every((q, i) => Number.isFinite(q) && q >= c.modules[i].jointRange[0] && q <= c.modules[i].jointRange[1]));
  });
}

test('IK supports short chains and custom axes and docking rotations', () => {
  for (const count of [1, 3, 6]) {
    const c = arm(); c.modules = c.modules.slice(0, count);
    c.modules.forEach((m, i) => { m.jointAxis = [1, 0, 0]; if (i) m.customEuler = [20, 30, 45]; });
    c.defaultGait.manualAngles = Object.fromEntries(c.modules.map((m, i) => [`joint_${i}`, 0]));
    const seed = c.modules.map(() => 0);
    const result = solveInverseKinematics(c, getEndEffectorPose(c, seed.map(() => 4)), seed);
    assert.ok(result.converged, JSON.stringify(result));
    assert.equal(result.angles.length, count);
  }
});
