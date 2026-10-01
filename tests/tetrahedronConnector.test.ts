import assert from 'node:assert/strict';
import { before, test } from 'node:test';
import loadMujoco from '@mujoco/mujoco';
import { Vector3 } from 'three';
import { appendConnectorModule, createTetrahedronConfiguration, parseConfiguration, removeConnectorBranch, validateConfiguration } from '../src/utils/configuration';
import { connectorFaces, moduleMount } from '../src/utils/moduleMount';
import { TETRAHEDRON_EDGE, TETRAHEDRON_FACES, TETRAHEDRON_VERTICES } from '../src/utils/tetrahedronGeometry';
import { forwardKinematics } from '../src/utils/kinematics';
import { generateMujocoXML } from '../src/utils/xmlGenerator';
import { modelExportFiles, createZip } from '../src/utils/modelExport';
import { generateZbotOBJ } from '../src/utils/zbotMeshGenerator';
import { MujocoEngine } from '../src/mujoco/MujocoEngine';
import { spawnSync } from 'node:child_process';
let mj: Awaited<ReturnType<typeof loadMujoco>>;
before(async () => { mj = await loadMujoco(); mj.FS.writeFile('ma.obj', generateZbotOBJ('ma')); mj.FS.writeFile('mb.obj', generateZbotOBJ('mb')); });
const name = (model: any, adr: number) => { let s = ''; for (let k = adr; model.names[k]; k++) s += String.fromCharCode(model.names[k]); return s; };

test('regular tetrahedron has six 180 mm edges and face incircles slightly above 100 mm, centered vertices and outward wound faces', () => {
  const vertices = TETRAHEDRON_VERTICES.map(v => new Vector3(...v));
  assert.ok(vertices.reduce((sum, v) => sum.add(v), new Vector3()).length() < 1e-12);
  for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) assert.ok(Math.abs(vertices[i].distanceTo(vertices[j]) - TETRAHEDRON_EDGE) < 1e-12);
  TETRAHEDRON_FACES.forEach(face => {
    const [a, b, c] = face.map(i => vertices[i]);
    assert.ok(b.clone().sub(a).cross(c.clone().sub(a)).dot(a) > 0);

  });
});

test('four ports lie at triangular face centroids and point outward; cubes faces are rejected', () => {
  let config = createTetrahedronConfiguration();
  connectorFaces('tetrahedron').forEach((face, i) => {
    config = appendConnectorModule(config, undefined, face);
    const mod = config.modules.at(-1)!;
    const mount = moduleMount(config, mod), center = new Vector3().setFromMatrixPosition(mount);
    const expected = TETRAHEDRON_FACES[i].reduce((v, j) => v.add(new Vector3(...TETRAHEDRON_VERTICES[j])), new Vector3()).multiplyScalar(1 / 3);
    assert.ok(center.distanceTo(expected) < 1e-12);
    assert.ok(center.clone().normalize().distanceTo(new Vector3(0, 0, 1).transformDirection(mount)) < 1e-12);
    assert.deepEqual(mod.jointAxis, [0, -1, 1]);
    assert.throws(() => appendConnectorModule(config, undefined, face), /未占用/);
  });
  assert.deepEqual(parseConfiguration(JSON.stringify(config)), config);
  assert.deepEqual(validateConfiguration(config), []);
  assert.throws(() => appendConnectorModule(config, undefined, '+x'), /未占用/);
  config.modules[0].mountFace = '+x'; assert.ok(validateConfiguration(config).length);
});

test('tetrahedron models compile in all geometry/base modes with named joint positions and FK', () => {
  for (const geometryMode of ['cad', 'envelope', 'mechanical'] as const) for (const baseMode of ['fixed', 'free'] as const) {
    let config = createTetrahedronConfiguration();
    for (const face of connectorFaces('tetrahedron')) config = appendConnectorModule(config, undefined, face);
    config = appendConnectorModule(config, config.modules[2].id);
    config.geometryMode = geometryMode; config.baseMode = baseMode; config.rootPos = [0, 0, .7]; config.rootEuler = [20, -15, 35];
    config.modules.forEach((m, i) => { m.initialAngle = -10 * (i + 1); m.dockAngle = 20 * i; });
    const engine = new MujocoEngine(); (engine as any).mujoco = mj;
    engine.loadModelFromXml(generateMujocoXML(config)); engine.resetSimulation(config);
    const model = engine.getModel(), data = engine.getData(), fk = forwardKinematics(config);
    assert.equal(model.nu, 5);
    for (let i = 0; i < model.nsite; i++) {
      const index = Number(name(model, model.name_siteadr[i]).slice(4));
      const actual = new Vector3(...Array.from(data.site_xpos.slice(i * 3, i * 3 + 3)) as [number, number, number]);
      assert.ok(actual.distanceTo(new Vector3(...fk.endpoints[index + 1])) < 1e-7);
    }
    engine.step(100, config.defaultGait); assert.ok(Array.from(data.qpos).every(Number.isFinite)); engine.destroy();
    const pruned = removeConnectorBranch(config, config.modules[2].id);
    assert.equal(pruned.modules.length, 3); assert.deepEqual(validateConfiguration(pruned), []);
  }
});

test('bare tetrahedron compiles and its URDF ZIP includes a valid four-face mesh', () => {
  const config = createTetrahedronConfiguration();
  const model = mj.MjModel.from_xml_string(generateMujocoXML(config)); assert.equal(model.nu, 0); model.delete();
  const files = modelExportFiles(config, 'urdf', { ma: '', mb: '' });
  assert.ok(files['tetrahedron.obj']); assert.equal(files['ma.obj'], undefined);
  assert.match(files['model.urdf'], /tetrahedron_root/); assert.match(files['model.urdf'], /filename="tetrahedron.obj"/);
  const result = spawnSync('python3', ['-c', 'import sys,io,zipfile,xml.etree.ElementTree as E; z=zipfile.ZipFile(io.BytesIO(sys.stdin.buffer.read())); assert z.testzip() is None; r=E.fromstring(z.read("model.urdf")); assert all(m.attrib["filename"] in z.namelist() for m in r.findall(".//mesh")); obj=z.read("tetrahedron.obj").decode().splitlines(); assert len([x for x in obj if x.startswith("v ")])==4; assert len([x for x in obj if x.startswith("f ")])==4'], { input: createZip(files), encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
});


test('legacy 100 mm edge tetrahedron JSON upgrades to fit a 100 mm module', () => {
  const config = createTetrahedronConfiguration();
  const legacy = { ...config, rootConnector: { type: 'tetrahedron', size: .1 } }; 
  assert.ok(validateConfiguration(legacy).length);
  const upgraded = parseConfiguration(JSON.stringify(legacy));
  assert.equal(upgraded.rootConnector!.size, TETRAHEDRON_EDGE);
  assert.deepEqual(validateConfiguration(upgraded), []);
});
