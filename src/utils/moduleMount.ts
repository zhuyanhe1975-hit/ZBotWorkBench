import { Euler, Matrix4, Quaternion, Vector3 } from 'three';
import { CubeFace, ConnectorFace, RootConnectorType, ZbotConfiguration, ZbotModule } from '../types/zbot';

import { TETRAHEDRON_VERTICES } from './tetrahedronGeometry';

export const TETRAHEDRON_PORTS: ConnectorFace[] = ['face_0', 'face_1', 'face_2', 'face_3'];
export const connectorFaces = (type?: RootConnectorType): ConnectorFace[] => type === 'tetrahedron' ? TETRAHEDRON_PORTS : CUBE_FACES;
export const connectorName = (type?: RootConnectorType) => type === 'tetrahedron' ? '正四面体' : '立方体';
export const faceLabel = (face: ConnectorFace) => face.startsWith('face_') ? `面 ${Number(face.slice(5)) + 1}${face === 'face_0' ? '（底面）' : ''}` : face.toUpperCase();

export const CUBE_FACES: CubeFace[] = ['+x', '-x', '+y', '-y', '+z', '-z'];
// Local +Z points out of each face; local +X consistently points downward on side faces.
const faces: Record<CubeFace, { pos: [number, number, number]; euler: [number, number, number] }> = {
  '+x': { pos: [.05, 0, 0], euler: [0, 90, 0] },
  '-x': { pos: [-.05, 0, 0], euler: [0, -90, 180] },
  '+y': { pos: [0, .05, 0], euler: [-90, 0, 90] },
  '-y': { pos: [0, -.05, 0], euler: [90, 0, -90] },
  '+z': { pos: [0, 0, .05], euler: [0, 0, 0] },
  '-z': { pos: [0, 0, -.05], euler: [180, 0, 0] },
};

export function moduleMount(config: ZbotConfiguration, module: ZbotModule): Matrix4 {
  const rotation = (angles: number[]) => new Matrix4().makeRotationFromEuler(new Euler(angles[0] * Math.PI / 180, angles[1] * Math.PI / 180, angles[2] * Math.PI / 180, 'XYZ'));
  const dock = rotation(module.customEuler ?? [0, 0, module.dockAngle]);
  if (module.parentId !== null) return new Matrix4().makeTranslation(0, 0, .106).multiply(dock);
  if (!config.rootConnector) return new Matrix4();
  if (config.rootConnector.type === 'tetrahedron') {
    const index = Number(module.mountFace!.slice(5));
    const pos = new Vector3(...TETRAHEDRON_VERTICES[index]).multiplyScalar(-1 / 3);
    const orientation = new Quaternion().setFromUnitVectors(new Vector3(0, 0, 1), pos.clone().normalize());
    return new Matrix4().compose(pos, orientation, new Vector3(1, 1, 1)).multiply(dock);
  }
  const face = faces[module.mountFace as CubeFace];
  return new Matrix4().makeTranslation(...face.pos).multiply(rotation(face.euler)).multiply(dock);
}
