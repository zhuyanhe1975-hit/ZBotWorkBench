import { Vector3 } from 'three';

// Face incircle diameter = edge / sqrt(3) ≈ 103.92 mm.
export const TETRAHEDRON_EDGE = .18;
const edge = TETRAHEDRON_EDGE;
const inradius = edge * Math.sqrt(6) / 12;
const radius = edge / Math.sqrt(3);
/** Centroid at origin; face 0 is the horizontal bottom, apex points along +Z. */
export const TETRAHEDRON_VERTICES: [number, number, number][] = [
  [0, 0, 3 * inradius],
  [radius, 0, -inradius],
  [-radius / 2, edge / 2, -inradius],
  [-radius / 2, -edge / 2, -inradius],
];
export const TETRAHEDRON_FACES = TETRAHEDRON_VERTICES.map((_, opposite) => {
  const indices = [0, 1, 2, 3].filter(i => i !== opposite);
  const [a, b, c] = indices.map(i => new Vector3(...TETRAHEDRON_VERTICES[i]));
  const centroid = a.clone().add(b).add(c).multiplyScalar(1 / 3);
  if (b.clone().sub(a).cross(c.clone().sub(a)).dot(centroid) < 0) [indices[1], indices[2]] = [indices[2], indices[1]];
  return indices;
});
export function tetrahedronObj(): string {
  return ['o tetrahedron', ...TETRAHEDRON_VERTICES.map(v => `v ${v.join(' ')}`), ...TETRAHEDRON_FACES.map(f => `f ${f.map(i => i + 1).join(' ')}`)].join('\n');
}
