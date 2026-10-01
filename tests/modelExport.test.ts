import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { Euler, Matrix4, Vector3 } from 'three';
import { createConfiguration, insertModule } from '../src/utils/configuration';
import { forwardKinematics } from '../src/utils/kinematics';
import { generateUrdf } from '../src/utils/urdfGenerator';
import { createZip, modelExportFiles } from '../src/utils/modelExport';

test('URDF preserves mixed mounting rotations and off-center hinge kinematics', () => {
  const config = insertModule(createConfiguration(), 0);
  config.rootEuler = [25, -30, 65];
  config.modules[1].customEuler = [35, 20, -70];
  config.modules[0].initialAngle = 40;
  config.modules[1].initialAngle = -55;
  const xml = generateUrdf(config);
  const expected = forwardKinematics(config);
  let transform = new Matrix4();
  for (const match of xml.matchAll(/<joint name="([^"]+)" type="([^"]+)">([\s\S]*?)<\/joint>/g)) {
    const xyz = /<origin xyz="([^"]+)" rpy="([^"]+)"/.exec(match[3])!;
    const angles = xyz[2].split(' ').map(Number) as [number, number, number];
    const origin = new Matrix4().makeRotationFromEuler(new Euler(...angles, 'ZYX'));
    origin.setPosition(...xyz[1].split(' ').map(Number) as [number, number, number]);
    transform.multiply(origin);
    if (match[2] === 'revolute') {
      const index = Number(match[1].slice(6));
      const axis = new Vector3(.../<axis xyz="([^"]+)"/.exec(match[3])![1].split(' ').map(Number) as [number, number, number]);
      assert.ok(Math.abs(axis.length() - 1) < 1e-12);
      assert.ok(new Vector3().setFromMatrixPosition(transform).distanceTo(new Vector3(...expected.joints[index].position)) < 1e-12);
      transform.multiply(new Matrix4().makeRotationAxis(axis, config.modules[index].initialAngle! * Math.PI / 180));
      const b = transform.clone().multiply(new Matrix4().makeTranslation(0, 0, -.053));
      b.elements.forEach((v, i) => assert.ok(Math.abs(v - expected.parts[index].b.elements[i]) < 1e-12));
    }
  }
  assert.equal((xml.match(/type="revolute"/g) ?? []).length, 2);
  assert.match(xml, /lower="-3.141592653589793"/);
});

test('URDF escapes names, handles base and geometry modes, rejects malformed config', () => {
  const config = createConfiguration(); config.name = 'A & "B"'; config.baseMode = 'free';
  assert.match(generateUrdf(config), /name="A &amp; &quot;B&quot;"/);
  assert.match(generateUrdf(config), /type="floating"/);
  for (const mode of ['envelope', 'mechanical'] as const) {
    config.geometryMode = mode;
    const xml = generateUrdf(config);
    assert.match(xml, /<cylinder/); assert.doesNotMatch(xml, /<mesh/);
  }
  config.modules[0].jointAxis = [0, 0, 0];
  assert.throws(() => generateUrdf(config), /关节轴/);
});

test('model bundles include meshes only when needed and preserve initial radians', () => {
  const config = createConfiguration(); config.modules[0].initialAngle = 90;
  const meshes = { ma: 'mesh A', mb: 'mesh B' };
  for (const format of ['mjcf', 'urdf'] as const) {
    const files = modelExportFiles(config, format, meshes, { kp: 123, friction: .7 });
    assert.equal(files['ma.obj'], meshes.ma);
    assert.equal(JSON.parse(files['joint-positions.json']).joint_0, Math.PI / 2);
    if (format === 'mjcf') { assert.match(files['model.xml'], /kp="123"/); assert.match(files['model.xml'], /key name="initial"/); }
    else assert.match(files['model.urdf'], /<robot/);
  }
  config.geometryMode = 'envelope';
  assert.equal(modelExportFiles(config, 'mjcf', meshes)['ma.obj'], undefined);
});

test('ZIP can be independently read with correct UTF-8 contents and CRC', () => {
  const files = { 'model.xml': '<mujoco/>', '说明.txt': '构型导出', 'ma.obj': 'v 0 0 0\n' };
  const result = spawnSync('python3', ['-c', 'import sys,io,zipfile,json; z=zipfile.ZipFile(io.BytesIO(sys.stdin.buffer.read())); assert z.testzip() is None; print(json.dumps({n:z.read(n).decode() for n in z.namelist()},ensure_ascii=False))'], { input: createZip(files), encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(JSON.parse(result.stdout), files);
});

test('both exported XML formats parse as XML with valid link references', () => {
  const config = insertModule(createConfiguration(), 0);
  config.name = '构型 & <试验>';
  for (const format of ['mjcf', 'urdf'] as const) {
    const xml = Object.values(modelExportFiles(config, format, { ma: '', mb: '' }))[0];
    const result = spawnSync('python3', ['-c', 'import sys,xml.etree.ElementTree as E; r=E.fromstring(sys.stdin.read()); links={x.attrib["name"] for x in r.findall("link")}; joints=r.findall("joint"); assert all(j.find("parent").attrib["link"] in links and j.find("child").attrib["link"] in links for j in joints); print(r.tag)'], { input: xml, encoding: 'utf8' });
    assert.equal(result.status, 0, result.stderr);
    assert.equal(result.stdout.trim(), format === 'mjcf' ? 'mujoco' : 'robot');
  }
});
