import assert from 'node:assert/strict';
import { before, test } from 'node:test';
import loadMujoco from '@mujoco/mujoco';
import { MujocoEngine } from '../src/mujoco/MujocoEngine';
import { createConfiguration, insertModule, parseConfiguration } from '../src/utils/configuration';
import { generateMujocoXML } from '../src/utils/xmlGenerator';
import { generateZbotOBJ } from '../src/utils/zbotMeshGenerator';
import type { ZbotConfiguration } from '../src/types/zbot';

let mujoco: Awaited<ReturnType<typeof loadMujoco>>;
before(async () => {
  mujoco = await loadMujoco();
  mujoco.FS.writeFile('ma.obj', generateZbotOBJ('ma'));
  mujoco.FS.writeFile('mb.obj', generateZbotOBJ('mb'));
});

function fixture(baseMode: 'fixed' | 'free' = 'fixed'): ZbotConfiguration {
  // Serial chain; actuator order is deliberately reversed below.
  return {
    id: 'physics-test', name: 'Physics test', category: 'arm', description: '', baseMode,
    rootPos: [0, 0, 0.1], rootEuler: [0, 0, 0],
    modules: [
      { id: 'root', name: 'Root', parentId: null, dockAngle: 0, jointAxis: [0, 1, 0], jointRange: [-90, 90], initialAngle: 30 },
      { id: 'child', name: 'Child', parentId: 'root', dockAngle: 0, jointAxis: [0, 1, 0], jointRange: [-180, 180], initialAngle: -45 },
    ],
    defaultGait: { type: 'manual', frequency: 1, amplitude: 20, phaseLag: 60, steering: 0, speed: 1, manualAngles: { joint_0: 30, joint_1: -45 } },
  };
}
function engine(config = fixture()) {
  const engine = new MujocoEngine();
  (engine as any).mujoco = mujoco;
  const xml = generateMujocoXML(config).replace(/(<position[^>]+\/>)(\s*)(<position[^>]+\/>)/, '$3$2$1');
  engine.loadModelFromXml(xml);
  engine.resetSimulation(config);
  return engine;
}
function close(actual: number, expected: number, tolerance = 1e-9) {
  assert.ok(Math.abs(actual - expected) < tolerance, `${actual} != ${expected}`);
}

test('compiled radians and name mappings preserve reordered actuator poses and controls', () => {
  const e = engine();
  const model = e.getModel(), data = e.getData();
  close(model.actuator_ctrlrange[3], Math.PI / 2);
  close(data.ctrl[0], -Math.PI / 4);
  close(data.ctrl[1], Math.PI / 6);
  close(e.getMetrics().jointAngles.joint_0, 30);
  close(e.getMetrics().jointAngles.joint_1, -45);
  e.applyGaitControl({ ...fixture().defaultGait, manualAngles: { joint_0: 200, joint_1: 90 } }, 0);
  close(data.ctrl[0], Math.PI / 2);
  close(data.ctrl[1], Math.PI / 2);
  e.destroy();
});

test('reset restores saved pose, paused pose forwards kinematics, torque is actual generalized force', () => {
  const e = engine();
  const initialTip = e.getMetrics().endEffectorPos;
  e.applyPose({ joint_0: -30, joint_1: 40 });
  assert.notDeepEqual(e.getMetrics().endEffectorPos, initialTip);
  e.resetSimulation();
  close(e.getMetrics().jointAngles.joint_0, 30);
  assert.deepEqual(e.getMetrics().endEffectorPos, initialTip);
  e.applyGaitControl({ ...fixture().defaultGait, manualAngles: { joint_0: 50, joint_1: 0 } }, 0);
  e.step(1);
  const jointId = e.getModel().actuator_trnid[2];
  close(e.getMetrics().jointTorques.joint_0, e.getData().qfrc_actuator[e.getModel().jnt_dofadr[jointId]]);
  assert.notEqual(e.getMetrics().jointTorques.joint_0, e.getData().ctrl[1]);
  e.destroy();
});

test('fixed root impulse does not corrupt hinge velocities; free root impulse targets translation', () => {
  const fixed = engine();
  fixed.applyImpulse(1, 2, 3);
  assert.deepEqual(Array.from(fixed.getData().qvel), [0, 0]);
  assert.deepEqual(fixed.getMetrics().rootVelocity, [0, 0, 0]);
  fixed.destroy();
  const free = engine(fixture('free'));
  free.applyImpulse(1, 2, 3);
  assert.deepEqual(free.getMetrics().rootVelocity, [1, 2, 3]);
  close(free.getMetrics().jointAngles.joint_0, 30);
  free.destroy();
});

test('contact diagnostics expose world-space contact positions and solver forces', () => {
  const e = new MujocoEngine();
  (e as any).mujoco = mujoco;
  e.loadModelFromXml('<mujoco><option timestep="0.002"/><worldbody><geom type="plane" size="2 2 .1"/><body pos="0 0 .2"><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>');
  e.step(300);
  const contacts = e.getContactDiagnostics();
  assert.equal(contacts.length, 1);
  assert.ok(contacts[0].position.every(Number.isFinite));
  assert.ok(Math.abs(contacts[0].magnitude - 9.81) < .02);
  assert.ok(Math.abs(contacts[0].force[0]) < 1e-6);
  assert.ok(Math.abs(contacts[0].force[1]) < 1e-6);
  assert.ok(Math.abs(contacts[0].force[2] - 9.81) < .02);
  e.destroy();
});

test('invalid XML leaves previous simulation live', () => {
  const e = engine();
  const xml = e.getCurrentXml();
  assert.throws(() => e.loadModelFromXml('<invalid/>'));
  assert.equal(e.getCurrentXml(), xml);
  assert.equal(e.isReady(), true);
  e.step(2);
  assert.ok(e.getTime() > 0);
  e.destroy();
});

test('gait is applied each substep around the candidate shape', () => {
  const config = fixture();
  const e = engine(config);
  const gait = { ...config.defaultGait, type: 'serpentine' as const, phaseLag: 0 };
  e.step(3, gait);
  close(e.getData().ctrl[1], (30 + 20 * Math.sin(2 * Math.PI * 0.004)) * Math.PI / 180);
  e.destroy();
});

test('XML rejects malformed trees and visual meshes add no duplicate mass', () => {
  const config = fixture();
  config.modules[0].parentId = 'missing';
  assert.throws(() => generateMujocoXML(config));
  const xml = generateMujocoXML(fixture(), { selfCollision: true });
  assert.match(xml, /mass="0"/);
  assert.match(xml, /<exclude body1="body_0" body2="body_1"/);
  const model = mujoco.MjModel.from_xml_string(xml);
  assert.ok(model.nexclude >= 2);
  const collisionOnly = mujoco.MjModel.from_xml_string(xml.replace(/<geom name="visual_[^"]+"[^>]+\/>/g, ''));
  assert.deepEqual(Array.from(model.body_mass), Array.from(collisionOnly.body_mass));
  collisionOnly.delete();
  model.delete();
});

test('all six candidates load and remain finite for two simulated seconds in fixed and free modes', async () => {
  const { PRESET_CONFIGURATIONS } = await import('../src/data/presets');
  assert.equal(PRESET_CONFIGURATIONS.length, 6);
  for (const preset of PRESET_CONFIGURATIONS) {
    for (const baseMode of ['fixed', 'free'] as const) {
      const config = { ...preset, baseMode };
      const e = engine(config);
      for (let i = 0; i < config.modules.length; i++) close(e.getMetrics().jointAngles[`joint_${i}`], config.modules[i].initialAngle ?? 0);
      e.step(1000, config.defaultGait);
      close(e.getTime(), 2, 1e-8);
      assert.ok(Array.from(e.getData().qpos as Float64Array).every(Number.isFinite), `${preset.id} ${baseMode}`);
      assert.ok(e.getMetrics().endEffectorPos?.every(Number.isFinite));
      e.destroy();
    }
  }
});


test('free reset raises an initially buried configuration clear of the floor', () => {
  const config = fixture('free');
  config.rootPos[2] = -1;
  const e = engine(config);
  assert.ok(e.getMetrics().rootPos[2] > 0);
  assert.equal(e.getMetrics().contactCount, 0);
  e.destroy();
});

test('named tip lookup ignores unrelated sites and unsupported actuators reject transactionally', () => {
  const e = engine();
  const tip = e.getMetrics().endEffectorPos;
  const xml = generateMujocoXML(fixture()).replace('</worldbody>', '<site name="decoy" pos="5 5 5"/></worldbody>');
  e.loadModelFromXml(xml);
  e.resetSimulation();
  assert.deepEqual(e.getMetrics().endEffectorPos, tip);
  const model = e.getModel();
  assert.throws(() => e.loadModelFromXml('<mujoco><worldbody><body><joint name="joint_0"/><geom type="sphere" size=".1"/></body></worldbody><actuator><motor joint="joint_0"/></actuator></mujoco>'), /仅支持/);
  assert.equal(e.getModel(), model);
  e.destroy();
});

test('standalone initial keyframe and imported XML restore q, control and free root orientation', () => {
  for (const baseMode of ['fixed', 'free'] as const) {
    const config = fixture(baseMode);
    config.rootEuler = [23, -37, 81];
    config.rootPos = [.1, -.2, 2];
    const xml = generateMujocoXML(config);
    const model = mujoco.MjModel.from_xml_string(xml);
    const data = new mujoco.MjData(model);
    assert.equal(model.nkey, 1);
    mujoco.mj_resetDataKeyframe(model, data, 0);
    for (let i = 0; i < config.modules.length; i++) {
      close(data.qpos[(baseMode === 'free' ? 7 : 0) + i], config.modules[i].initialAngle! * Math.PI / 180);
      close(data.ctrl[i], config.modules[i].initialAngle! * Math.PI / 180);
    }
    const e = new MujocoEngine();
    (e as any).mujoco = mujoco;
    e.loadModelFromXml(xml);
    e.resetSimulation();
    assert.deepEqual(Array.from(e.getData().qpos), Array.from(data.qpos));
    close(e.getMetrics().jointAngles.joint_0, 30);
    e.destroy(); data.delete(); model.delete();
  }
});

test('actual CAD meshes: six candidates, both bases, collision settings and bounded gaits remain finite without numerical resets', async () => {
  const { readFile } = await import('node:fs/promises');
  const { PRESET_CONFIGURATIONS } = await import('../src/data/presets');
  const { forwardKinematics } = await import('../src/utils/kinematics');
  mujoco.FS.writeFile('ma.obj', await readFile('public/assets/ma.obj', 'utf8'));
  mujoco.FS.writeFile('mb.obj', await readFile('public/assets/mb.obj', 'utf8'));
  try {
    for (const preset of PRESET_CONFIGURATIONS) {
      for (const baseMode of ['fixed', 'free'] as const) {
        for (const selfCollision of [false, true]) {
          for (const gaitType of ['manual', 'serpentine'] as const) {
            const config = { ...preset, baseMode };
            const label = `${preset.id}/${baseMode}/collision=${selfCollision}/${gaitType}`;
            const e = new MujocoEngine();
            (e as any).mujoco = mujoco;
            e.loadModelFromXml(generateMujocoXML(config, { selfCollision }));
            e.resetSimulation(config);
            if (baseMode === 'fixed') forwardKinematics(config).tip.forEach((v, i) => close(e.getMetrics().endEffectorPos![i], v));
            const gait = { ...config.defaultGait, type: gaitType, amplitude: 10, frequency: .5 };
            for (let batch = 0; batch < 10; batch++) {
              e.step(100, gait);
              close(e.getTime(), .2 * (batch + 1), 1e-8);
              assert.ok(Array.from(e.getData().qpos as Float64Array).every(Number.isFinite), label);
              // Require every MuJoCo warning counter to remain clear.
              for (let warning = 0; warning < e.getData().warning.size(); warning++) assert.equal(e.getData().warning.get(warning).number, 0, `${label} warning ${warning}`);
            }
            e.destroy();
          }
        }
      }
    }
    const config = structuredClone(PRESET_CONFIGURATIONS[0]);
    config.rootEuler = [31, -48, 72];
    config.rootPos = [.4, -.2, 1.2];
    config.modules[2].customEuler = [27, 42, -63];
    const e = engine(config);
    forwardKinematics(config).tip.forEach((v, i) => close(e.getMetrics().endEffectorPos![i], v));
    e.destroy();
  } finally {
    mujoco.FS.writeFile('ma.obj', generateZbotOBJ('ma'));
    mujoco.FS.writeFile('mb.obj', generateZbotOBJ('mb'));
  }
});


test('single-module creation and saved two-module poses load in real MuJoCo', () => {
  const single = createConfiguration();
  single.modules[0].initialAngle = 35;
  single.defaultGait.manualAngles.joint_0 = 35;
  const initial = engine(single);
  try {
    assert.equal(initial.getModel().nu, 1);
    close(initial.getMetrics().jointAngles.joint_0, 35);
    assert.ok(initial.getMetrics().endEffectorPos!.every(Number.isFinite));
  } finally { initial.destroy(); }
  const pair = insertModule(single, 0);
  pair.modules[1].dockAngle = 90;
  pair.modules[1].initialAngle = -60;
  pair.defaultGait.manualAngles.joint_1 = -60;
  const saved = parseConfiguration(JSON.stringify(pair));
  const loaded = engine(saved);
  try {
    assert.equal(loaded.getModel().nu, 2);
    close(loaded.getMetrics().jointAngles.joint_0, 35);
    close(loaded.getMetrics().jointAngles.joint_1, -60);
    loaded.step(10, saved.defaultGait);
    assert.ok(loaded.getMetrics().endEffectorPos!.every(Number.isFinite));
    loaded.resetSimulation(saved);
    close(loaded.getMetrics().jointAngles.joint_1, -60);
  } finally { loaded.destroy(); }
});
