import assert from 'node:assert/strict';
import { before, test } from 'node:test';
import loadMujoco from '@mujoco/mujoco';
import { Euler, Matrix4, Vector3 } from 'three';
import { appendCubeModule, createCubeConfiguration, createCubeQuadruped, parseConfiguration, removeCubeBranch, validateConfiguration, withTreeModules } from '../src/utils/configuration';
import { CUBE_FACES, moduleMount } from '../src/utils/moduleMount';
import { forwardKinematics } from '../src/utils/kinematics';
import { generateMujocoXML } from '../src/utils/xmlGenerator';
import { generateUrdf } from '../src/utils/urdfGenerator';
import { generateZbotOBJ } from '../src/utils/zbotMeshGenerator';
import { MujocoEngine } from '../src/mujoco/MujocoEngine';
import { supportsEndEffectorControl } from '../src/utils/inverseKinematics';

let mj: Awaited<ReturnType<typeof loadMujoco>>;
before(async () => { mj = await loadMujoco(); mj.FS.writeFile('ma.obj', generateZbotOBJ('ma')); mj.FS.writeFile('mb.obj', generateZbotOBJ('mb')); });
const name = (model: any, adr: number) => { let s = ''; for (let k = adr; model.names[k]; k++) s += String.fromCharCode(model.names[k]); return s; };

test('cube faces have 50 mm center offsets with outward local Z, unique ports and acyclic branches', () => {
  let config = createCubeConfiguration();
  assert.deepEqual(validateConfiguration(config), []);
  for (const face of CUBE_FACES) {
    config = appendCubeModule(config, undefined, face);
    const module = config.modules.at(-1)!;
    const mount = moduleMount(config, module), pos = new Vector3().setFromMatrixPosition(mount);
    assert.ok(Math.abs(pos.length() - .05) < 1e-12);
    assert.ok(pos.clone().normalize().distanceTo(new Vector3(0, 0, 1).transformDirection(mount)) < 1e-12);
    assert.throws(() => appendCubeModule(config, undefined, face), /未占用/);
  }
  assert.deepEqual(validateConfiguration(config), []);
  config.modules[0].parentId = config.modules.at(-1)!.id;
  assert.ok(validateConfiguration(config).length);
});

test('deleting a branch preserves other module identities, poses and controller targets', () => {
  const config = createCubeQuadruped();
  config.defaultGait.manualAngles.joint_4 = 31;
  const next = removeCubeBranch(config, config.modules[0].id);
  assert.equal(next.modules.length, 6);
  assert.equal(next.modules[2].id, config.modules[4].id);
  assert.equal(next.defaultGait.manualAngles.joint_2, 31);
  assert.deepEqual(parseConfiguration(JSON.stringify(next)), next);
  assert.deepEqual(validateConfiguration(next), []);
  assert.equal(supportsEndEffectorControl(next), false);
});

test('four legs are radially symmetric and isolated under a changed joint pose', () => {
  const config = createCubeQuadruped(), fk = forwardKinematics(config);
  const feet = [1, 3, 5, 7].map(i => fk.endpoints[i + 1]);
  feet.forEach(foot => assert.ok(Math.abs(foot[2] - feet[0][2]) < 1e-12));
  const changed = forwardKinematics(config, config.modules.map((m, i) => i === 0 ? 60 : m.initialAngle!));
  for (const i of [2, 3, 4, 5, 6, 7]) assert.deepEqual(changed.parts[i].b.elements, fk.parts[i].b.elements);
});

test('all cube branches and geometry modes compile, match FK and restore named joint poses', () => {
  for (const geometryMode of ['cad', 'envelope', 'mechanical'] as const) for (const baseMode of ['fixed', 'free'] as const) {
    let config = createCubeQuadruped();
    // Deliberately breadth-first order, different from MuJoCo depth-first traversal.
    config = withTreeModules(config, [0, 2, 4, 6, 1, 3, 5, 7].map(i => config.modules[i]));
    config.geometryMode = geometryMode; config.baseMode = baseMode; config.rootEuler = [15, 20, -30]; config.rootPos = [0, 0, .8];
    const xml = generateMujocoXML(config, { selfCollision: true });
    const engine = new MujocoEngine(); (engine as any).mujoco = mj;
    engine.loadModelFromXml(xml); engine.resetSimulation(config);
    const model = engine.getModel(), data = engine.getData(), fk = forwardKinematics(config);
    assert.equal(model.nu, 8);
    for (let i = 0; i < model.nsite; i++) {
      const index = Number(name(model, model.name_siteadr[i]).slice(4));
      assert.ok(new Vector3(...Array.from(data.site_xpos.slice(i * 3, i * 3 + 3)) as [number, number, number]).distanceTo(new Vector3(...fk.endpoints[index + 1])) < 1e-7);
    }
    for (let j = 0; j < model.njnt; j++) {
      const jointName = name(model, model.name_jntadr[j]);
      if (jointName.startsWith('joint_')) assert.ok(Math.abs(data.qpos[model.jnt_qposadr[j]] - config.modules[Number(jointName.slice(6))].initialAngle! * Math.PI / 180) < 1e-12);
    }
    engine.step(100, config.defaultGait);
    assert.ok(Array.from(data.qpos).every(Number.isFinite)); engine.destroy();
  }
});

test('a bare cube compiles and exports without artificial module joints', () => {
  const config = createCubeConfiguration();
  const xml = generateMujocoXML(config);
  assert.doesNotMatch(xml, /<mesh|type="mesh"/);
  const model = mj.MjModel.from_xml_string(xml);
  assert.equal(model.nu, 0); assert.equal(model.njnt, 0); model.delete();
  const urdf = generateUrdf(config);
  assert.match(urdf, /<box size="0.1 0.1 0.1"/); assert.doesNotMatch(urdf, /type="revolute"/);
});

test('URDF attaches each root branch to the cube and follows actual parent IDs', () => {
  const config = createCubeQuadruped();
  const urdf = generateUrdf(config);
  assert.equal((urdf.match(/<parent link="cube_root"/g) ?? []).length, 4);
  for (const i of [1, 3, 5, 7]) assert.match(urdf, new RegExp(`<joint name="dock_${i}"[\\s\\S]*?<parent link="b_${i - 1}"`));
});

test('free cube reset lifts primitive collision shapes clear of the ground', () => {
  const config = createCubeConfiguration(); config.baseMode = 'free'; config.rootPos = [0, 0, -.2]; config.rootEuler = [30, 20, 10];
  const engine = new MujocoEngine(); (engine as any).mujoco = mj;
  engine.loadModelFromXml(generateMujocoXML(config)); engine.resetSimulation(config);
  assert.ok(engine.getData().qpos[2] > .053);
  engine.step(100, config.defaultGait);
  assert.ok(Array.from(engine.getData().qpos).every(Number.isFinite)); engine.destroy();
});

test('URDF branch transforms match every module at mixed root and docking angles', () => {
  const config = createCubeQuadruped(); config.rootEuler = [15, -20, 40];
  config.modules[0].customEuler = [10, 20, -15]; config.modules[3].dockAngle = 90;
  const frames = new Map<string, import('three').Matrix4>();
  const fk = forwardKinematics(config);
  // Walk joints in URDF topological order, independently reconstructing the output link transforms.
  frames.set('world', new Matrix4());
  for (const match of generateUrdf(config).matchAll(/<joint name="([^"]+)" type="([^"]+)">([\s\S]*?)<\/joint>/g)) {
    const parent = /<parent link="([^"]+)"/.exec(match[3])![1];
    const child = /<child link="([^"]+)"/.exec(match[3])![1];
    const origin = /<origin xyz="([^"]+)" rpy="([^"]+)"/.exec(match[3])!;
    const rotation = origin[2].split(' ').map(Number) as [number, number, number];
    const local = new Matrix4().makeRotationFromEuler(new Euler(...rotation, 'ZYX')).setPosition(...origin[1].split(' ').map(Number) as [number, number, number]);
    const frame = frames.get(parent)!.clone().multiply(local);
    if (match[2] === 'revolute') {
      const i = Number(match[1].slice(6));
      const axis = new Vector3(.../<axis xyz="([^"]+)"/.exec(match[3])![1].split(' ').map(Number) as [number, number, number]);
      frame.multiply(new Matrix4().makeRotationAxis(axis, config.modules[i].initialAngle! * Math.PI / 180));
      const output = frame.clone().multiply(new Matrix4().makeTranslation(0, 0, -.053));
      output.elements.forEach((v, j) => assert.ok(Math.abs(v - fk.parts[i].b.elements[j]) < 1e-12));
    }
    frames.set(child, frame);
  }
});

test('quadruped hinges follow the original CAD cut-plane normals and foot outputs point down', () => {
  const config = createCubeQuadruped();
  const fk = forwardKinematics(config);
  const cutNormal = new Vector3(0, -1, 1).normalize();
  assert.equal((generateMujocoXML(config).match(/axis="0 -1 1"/g) ?? []).length, 8);
  const exportedAxis = cutNormal.toArray().join(' ');
  assert.equal(generateUrdf(config).split(`<axis xyz="${exportedAxis}"/>`).length - 1, 8);
  config.modules.forEach((module, i) => {
    assert.deepEqual(module.jointAxis, [0, -1, 1]);
    const aNormal = cutNormal.clone().transformDirection(fk.parts[i].a);
    const bNormal = cutNormal.clone().transformDirection(fk.parts[i].b);
    assert.ok(aNormal.distanceTo(bNormal) < 1e-12, 'A/B mating planes must remain parallel when the joint turns');
    assert.ok(aNormal.distanceTo(new Vector3(...fk.joints[i].axis)) < 1e-12);
    const aCenter = new Vector3(0, 0, .053).applyMatrix4(fk.parts[i].a);
    const bCenter = new Vector3(0, 0, .053).applyMatrix4(fk.parts[i].b);
    assert.ok(aCenter.distanceTo(bCenter) < 1e-12, 'A/B hinge centers must coincide');
    if (i % 2) assert.ok(new Vector3(0, 0, 1).transformDirection(fk.parts[i].b).distanceTo(new Vector3(0, 0, -1)) < 1e-12);
  });
  const engine = new MujocoEngine(); (engine as any).mujoco = mj;
  engine.loadModelFromXml(generateMujocoXML(config)); engine.resetSimulation(config);
  const model = engine.getModel(), data = engine.getData();
  for (let j = 0; j < model.njnt; j++) {
    const jointName = name(model, model.name_jntadr[j]);
    if (!jointName.startsWith('joint_')) continue;
    const index = Number(jointName.slice(6));
    const actual = new Vector3(...Array.from(data.xaxis.slice(j * 3, j * 3 + 3)) as [number, number, number]);
    assert.ok(actual.distanceTo(new Vector3(...fk.joints[index].axis)) < 1e-7);
  }
  engine.destroy();
});
