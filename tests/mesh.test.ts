import assert from 'node:assert/strict';
import { test } from 'node:test';
import * as THREE from 'three';
import loadMujoco from '@mujoco/mujoco';
import { MeshManager, parseObjGeometry } from '../src/utils/meshManager';
import { generateZbotOBJ } from '../src/utils/zbotMeshGenerator';

test('malformed OBJ and SPA responses are rejected without changing saved meshes', () => {
  const manager = new MeshManager();
  const previous = manager.getMeshAText();
  for (const invalid of ['<!doctype html><html></html>', 'v 0 0 0\nf 1 2 3', 'v NaN 0 0\nf 1 1 1', 'v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 9']) {
    assert.throws(() => manager.setCustomMesh('ma', invalid));
    assert.equal(manager.getMeshAText(), previous);
    assert.equal(manager.hasCadMeshA(), false);
  }
});

test('all objects in an uploaded OBJ contribute to rendered geometry', () => {
  const geometry = parseObjGeometry('v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\no a\nf 1 2 3\no b\nf 1 3 4');
  assert.equal(geometry.getAttribute('position').count, 6);
  geometry.dispose();
});

test('inverse compiler mesh transform reconstructs authored vertices in world space', async () => {
  const mujoco = await loadMujoco();
  mujoco.FS.writeFile('viewport-ma.obj', generateZbotOBJ('ma'));
  mujoco.FS.writeFile('viewport.xml', '<mujoco><asset><mesh name="ma" file="viewport-ma.obj"/></asset><worldbody><geom type="mesh" mesh="ma" pos="1 2 3" euler="20 30 40"/></worldbody></mujoco>');
  const model = mujoco.MjModel.from_xml_string(new TextDecoder().decode(mujoco.FS.readFile('viewport.xml')));
  const data = new mujoco.MjData(model);
  mujoco.mj_forward(model, data);
  const p = model.mesh_pos, q = model.mesh_quat;
  const inverse = new THREE.Matrix4().compose(new THREE.Vector3().fromArray(p), new THREE.Quaternion(q[1], q[2], q[3], q[0]), new THREE.Vector3(1, 1, 1)).invert();
  const rotation = data.geom_xmat, position = data.geom_xpos;
  const world = new THREE.Matrix4().set(rotation[0], rotation[1], rotation[2], position[0], rotation[3], rotation[4], rotation[5], position[1], rotation[6], rotation[7], rotation[8], position[2], 0, 0, 0, 1);
  // Source vertex 0 is the authored origin, which must remain at the declared geom position.
  const origin = new THREE.Vector3(0, 0, 0).applyMatrix4(inverse).applyMatrix4(world);
  assert.ok(origin.distanceTo(new THREE.Vector3(1, 2, 3)) < 1e-7);
  // A raw vertex with no correction is measurably displaced by the compiler centering.
  assert.ok(new THREE.Vector3().applyMatrix4(world).distanceTo(origin) > 0.001);
  data.delete(); model.delete();
});


test('HTML returned with HTTP 200 falls back to procedural mesh without CAD flags', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => new Response('<!doctype html><html>SPA shell</html>', { status: 200 });
  try {
    const manager = new MeshManager();
    const result = await manager.init();
    assert.equal(result.maText, generateZbotOBJ('ma'));
    assert.equal(result.mbText, generateZbotOBJ('mb'));
    assert.equal(manager.hasCadMeshA(), false);
    assert.equal(manager.hasCadMeshB(), false);
    const { geomA, geomB } = manager.getGeometries();
    assert.ok(geomA.getAttribute('position').count > 0);
    geomA.dispose(); geomB.dispose();
  } finally { globalThis.fetch = originalFetch; }
});
