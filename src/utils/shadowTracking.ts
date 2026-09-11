import type { DirectionalLight, Object3D, Texture } from 'three';

/** Keep a repeated grid visually fixed in world space while its finite carrier follows. */
export function anchorGroundGrid(texture: Texture, x: number, y: number, cellSize = .5): void {
  if (!(cellSize > 0) || !Number.isFinite(x) || !Number.isFinite(y)) throw new Error('无效网格坐标');
  texture.offset.set(x / cellSize, y / cellSize);
}

/** Move the bounded shadow region without enlarging its map or changing sunlight direction. */
export function trackGroundShadow(light: DirectionalLight, ground: Object3D, x: number, y: number, groundOverlays: Object3D[] = []): boolean {
  const changed = light.target.position.x !== x || light.target.position.y !== y;
  ground.position.set(x, y, 0);
  for (const overlay of groundOverlays) overlay.position.set(x, y, overlay.position.z);
  if (!changed) return false;
  light.target.position.set(x, y, 0);
  light.position.set(x + 2, y - 2, 4);
  return true;
}
