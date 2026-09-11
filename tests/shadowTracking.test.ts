import test from 'node:test';
import assert from 'node:assert/strict';
import { DirectionalLight, Object3D, Scene, Texture, Vector3 } from 'three';
import { anchorGroundGrid, trackGroundShadow } from '../src/utils/shadowTracking';

test('moving a finite grid carrier preserves world-anchored half-metre cells', () => {
  const texture = new Texture();
  anchorGroundGrid(texture, 1.25, -2.5);
  assert.deepEqual(texture.offset.toArray(), [2.5, -5]);
  assert.throws(() => anchorGroundGrid(texture, 0, 0, 0), /无效网格坐标/);
});

test('shadow region follows long travel and reset, keeping robot and floor shadow in its frustum', () => {
  const scene = new Scene();
  const light = new DirectionalLight();
  light.position.set(2, -2, 4);
  Object.assign(light.shadow.camera, { left: -2, right: 2, top: 2, bottom: -2, near: .1, far: 10 });
  light.shadow.camera.updateProjectionMatrix();
  const ground = new Object3D();
  const grid = new Object3D(); grid.position.z = .001;
  scene.add(light, light.target, ground, grid);
  for (const [x, y] of [[-10, 0], [4, 0], [-7, 10], [50, -40], [0, 0]]) {
    assert.equal(trackGroundShadow(light, ground, x, y, [grid]), true);
    assert.deepEqual(light.position.clone().sub(light.target.position).toArray(), [2, -2, 4]);
    assert.deepEqual(ground.position.toArray(), [x, y, 0]);
    assert.deepEqual(grid.position.toArray(), [x, y, .001]);
    scene.updateMatrixWorld(true);
    light.shadow.updateMatrices(light);
    for (const z of [0, .4, .8]) {
      const point = new Vector3(x, y, z);
      assert.ok(light.shadow.getFrustum().containsPoint(point), `caster outside shadow at ${point.toArray()}`);
      const projection = new Vector3(x - z / 2, y + z / 2, 0);
      assert.ok(light.shadow.getFrustum().containsPoint(projection), 'ground shadow outside frustum');
      assert.ok(Math.abs(projection.x - ground.position.x) < 15 && Math.abs(projection.y - ground.position.y) < 15);
    }
    assert.equal(trackGroundShadow(light, ground, x, y, [grid]), false, 'camera-only redraw must retain cached shadows');
    grid.position.x = 999;
    assert.equal(trackGroundShadow(light, ground, x, y, [grid]), false, 'overlay repair does not invalidate the shadow map');
    assert.deepEqual(grid.position.toArray(), [x, y, .001]);
  }
});
