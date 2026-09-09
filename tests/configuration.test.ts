import test from 'node:test';
import assert from 'node:assert/strict';
import { PRESET_CONFIGURATIONS } from '../src/data/presets';
import { parseConfiguration, validateConfiguration, withSerialModules } from '../src/utils/configuration';
import { forwardKinematics } from '../src/utils/kinematics';

const clone = () => structuredClone(PRESET_CONFIGURATIONS[1]);
test('six reference candidates have exact q, sigma and separate categories', () => {
  const sigmas = [[0,0,0,0,0], [180,180,180,180,180], [90,90,90,90,90], [90,270,90,270,90], [180,180,0,180,0], [0,180,180,90,0]];
  const qs = [[0,35,35,35,35,0], [0,55,-55,-55,55,0], [55,55,55,55,55,55], [35,55,-35,-55,35,55], [0,80,0,-120,0,50], [0,0,90,90,60,60]];
  assert.equal(PRESET_CONFIGURATIONS.length, 6);
  assert.equal(new Set(PRESET_CONFIGURATIONS.map(p => p.category)).size, 6);
  PRESET_CONFIGURATIONS.forEach((p, i) => {
    assert.deepEqual(validateConfiguration(p), []);
    assert.deepEqual(p.modules.slice(1).map(m => m.dockAngle), sigmas[i]);
    assert.deepEqual(p.modules.map(m => m.initialAngle), qs[i]);
    assert.equal(p.defaultGait.type, 'manual');
    assert.deepEqual(parseConfiguration(JSON.stringify(p)), p);
  });
});
test('unknown imports reject malformed values, branch/cycle and unsupported ranges', () => {
  for (const v of [null, {}, [], { modules: [null] }]) assert.ok(validateConfiguration(v).length);
  for (const mutate of [
    (c: ReturnType<typeof clone>) => { c.modules[2].parentId = c.modules[0].id; },
    (c: ReturnType<typeof clone>) => { c.modules[0].parentId = c.modules[5].id; },
    (c: ReturnType<typeof clone>) => { c.modules[1].id = c.modules[0].id; },
    (c: ReturnType<typeof clone>) => { c.modules[0].jointAxis = [0,0,0]; },
    (c: ReturnType<typeof clone>) => { c.rootPos[0] = Infinity; },
    (c: ReturnType<typeof clone>) => { c.modules[0].initialAngle = 181; },
    (c: ReturnType<typeof clone>) => { c.defaultGait.manualAngles.joint_90 = 0; },
  ]) { const c = clone(); mutate(c); assert.ok(validateConfiguration(c).length); assert.throws(() => parseConfiguration(JSON.stringify(c))); }
});
test('deletion and reorder retain module identity and manual targets', () => {
  const c = clone(); c.defaultGait.manualAngles.joint_3 = 72;
  const next = withSerialModules(c, [c.modules[3], c.modules[1], c.modules[5]]);
  assert.deepEqual(next.modules.map(m => m.initialAngle), [-55,55,0]);
  assert.deepEqual(next.defaultGait.manualAngles, { joint_0:72, joint_1:55, joint_2:0 });
  assert.deepEqual(next.modules.map(m => m.parentId), [null, c.modules[3].id, c.modules[1].id]);
  assert.deepEqual(validateConfiguration(next), []);
  assert.equal(next.id, 'custom');
});
test('FK preserves original zero chain, hinge centers and rotated axis relationship', () => {
  const c = clone(); c.rootPos = [0,0,0]; c.rootEuler = [0,0,0];
  const f = forwardKinematics(c, [0,0,0,0,0,0]);
  f.endpoints.forEach((p, i) => assert.ok(Math.hypot(p[0],p[1],p[2] - .106 * i) < 1e-12));
  assert.ok(Math.abs(f.joints[0].position[2] - .053) < 1e-12);
  for (const p of PRESET_CONFIGURATIONS) {
    const f = forwardKinematics(p);
    for (let i = 0; i < 5; i++) {
      const dot = f.joints[i].axis.reduce((s, x, j) => s + x * f.joints[i + 1].axis[j], 0);
      assert.ok(Math.abs(dot - (1 + Math.cos(p.modules[i + 1].dockAngle * Math.PI / 180)) / 2) < 1e-12);
    }
    assert.equal(f.parts.length, 6); assert.ok(f.tip.every(Number.isFinite));
  }
});

test('legacy JSON imports retain manual pose and base semantics', () => {
  const legacy = structuredClone(PRESET_CONFIGURATIONS[0]);
  delete legacy.baseMode;
  legacy.modules.forEach(m => delete m.initialAngle);
  const restored = parseConfiguration(JSON.stringify(legacy));
  assert.equal(restored.baseMode, 'free');
  assert.deepEqual(restored.modules.map(m => m.initialAngle), [0,35,35,35,35,0]);
});
