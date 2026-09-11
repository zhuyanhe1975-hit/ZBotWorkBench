import { BufferGeometry, Float32BufferAttribute } from 'three';

/** MuJoCo mesh vertices already include the compiler's mesh-frame transform. */
export function compiledMeshGeometry(model: any, meshId: number): BufferGeometry {
  const start = model.mesh_vertadr[meshId];
  const count = model.mesh_vertnum[meshId];
  const faceStart = model.mesh_faceadr[meshId];
  const faceCount = model.mesh_facenum[meshId];
  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new Float32BufferAttribute(
    model.mesh_vert.slice(3 * start, 3 * (start + count)), 3));
  geometry.setIndex(Array.from(model.mesh_face.slice(3 * faceStart, 3 * (faceStart + faceCount))));
  geometry.computeVertexNormals();
  return geometry;
}
