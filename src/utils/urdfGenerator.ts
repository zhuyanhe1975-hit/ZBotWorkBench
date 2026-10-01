import { Euler, Matrix4, Quaternion, Vector3 } from 'three';
import { ZbotConfiguration } from '../types/zbot';
import { validateConfiguration } from './configuration';
import { mechanicalGeoms } from './mechanicalGeometry';
import { moduleMount } from './moduleMount';

const rad = Math.PI / 180;
const escape = (text: string) => text.replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&apos;' }[c]!));
// MJCF uses intrinsic XYZ; URDF uses fixed-axis roll/pitch/yaw.
function rpy(angles: number[]): string {
  const e = new Euler().setFromQuaternion(new Quaternion().setFromEuler(new Euler(angles[0] * rad, angles[1] * rad, angles[2] * rad, 'XYZ')), 'ZYX');
  return [e.x, e.y, e.z].join(' ');
}

/** Kinematic URDF; no measured inertia or actuator limits are available in the configuration. */
export function generateUrdf(config: ZbotConfiguration): string {
  const errors = validateConfiguration(config);
  if (errors.length) throw new Error(errors.join('；'));
  const lines = ['<?xml version="1.0" encoding="UTF-8"?>', `<robot name="${escape(config.name)}">`,
    '  <!-- Kinematic model. Effort (20 N*m) and velocity (10 rad/s) are nominal, not measured actuator ratings. -->',
    '  <!-- Initial joint positions are supplied separately in joint-positions.json (radians). -->', '  <link name="world"/>'];
  function link(index: number, part: 'a' | 'b') {
    const module = config.modules[index];
    const offset = part === 'b' ? -.053 : 0;
    const color = module[part === 'a' ? 'colorA' : 'colorB'] ?? '#3687b4';
    const rgb = [1, 3, 5].map(i => parseInt(color.slice(i, i + 2), 16) / 255).join(' ');
    const shapes: { xyz: string; rpy: string; geometry: string }[] = [];
    if (config.geometryMode === 'mechanical') {
      for (const geom of mechanicalGeoms(module.jointAxis, part, index).filter(g => g.includes('name="visual_'))) {
        const coords = /fromto="([^"]+)"/.exec(geom)![1].split(' ').map(Number);
        const p = new Vector3(...coords.slice(0, 3) as [number, number, number]);
        const q = new Vector3(...coords.slice(3) as [number, number, number]);
        const axis = q.clone().sub(p);
        const e = new Euler().setFromQuaternion(new Quaternion().setFromUnitVectors(new Vector3(0, 0, 1), axis.clone().normalize()), 'ZYX');
        shapes.push({ xyz: p.add(q).multiplyScalar(.5).add(new Vector3(0, 0, offset)).toArray().join(' '), rpy: [e.x, e.y, e.z].join(' '), geometry: `<cylinder radius="${/size="([^"]+)"/.exec(geom)![1]}" length="${axis.length()}"/>` });
      }
    } else if (config.geometryMode === 'envelope') {
      shapes.push({ xyz: `0 0 ${(part === 'a' ? .0265 : .0795) + offset}`, rpy: '0 0 0', geometry: '<cylinder radius="0.05" length="0.053"/>' });
    } else {
      shapes.push({ xyz: `0 0 ${offset}`, rpy: '0 0 0', geometry: `<mesh filename="m${part}.obj"/>` });
    }
    lines.push(`  <link name="${part}_${index}">`);
    shapes.forEach((shape, j) => {
      for (const tag of ['visual', 'collision']) {
        lines.push(`    <${tag} name="${tag}_${part}_${index}_${j}">`, `      <origin xyz="${shape.xyz}" rpy="${shape.rpy}"/>`, `      <geometry>${shape.geometry}</geometry>`);
        if (tag === 'visual') lines.push(`      <material name="color_${part}_${index}_${j}"><color rgba="${rgb} 1"/></material>`);
        lines.push(`    </${tag}>`);
      }
    });
    lines.push('  </link>');
  }
  if (config.rootConnector) {
    const rootName = `${config.rootConnector.type}_root`;
    const shape = config.rootConnector.type === 'tetrahedron' ? '<mesh filename="tetrahedron.obj"/>' : '<box size="0.1 0.1 0.1"/>';
    lines.push(`  <link name="${rootName}">`, `    <visual><geometry>${shape}</geometry><material name="connector_color"><color rgba="0.85 0.87 0.9 1"/></material></visual>`, `    <collision><geometry>${shape}</geometry></collision>`, '  </link>',
      `  <joint name="base_joint" type="${(config.baseMode ?? (config.category === 'arm' ? 'fixed' : 'free')) === 'fixed' ? 'fixed' : 'floating'}">`, `    <parent link="world"/><child link="${rootName}"/>`, `    <origin xyz="${config.rootPos.join(' ')}" rpy="${rpy(config.rootEuler)}"/>`, '  </joint>');
  }
  config.modules.forEach((module, i) => {
    link(i, 'a'); link(i, 'b');
    const root = module.parentId === null && !config.rootConnector;
    const type = root && (config.baseMode ?? (config.category === 'arm' ? 'fixed' : 'free')) === 'free' ? 'floating' : 'fixed';
    const mount = root ? new Matrix4().makeRotationFromEuler(new Euler(...config.rootEuler.map(v => v * rad) as [number, number, number], 'XYZ')).setPosition(...config.rootPos) : moduleMount(config, module);
    if (module.parentId !== null) mount.premultiply(new Matrix4().makeTranslation(0, 0, -.053));
    const e = new Euler().setFromRotationMatrix(mount, 'ZYX');
    const parent = module.parentId === null ? (config.rootConnector ? `${config.rootConnector.type}_root` : 'world') : `b_${config.modules.findIndex(m => m.id === module.parentId)}`;
    lines.push(`  <joint name="${root ? 'base_joint' : `dock_${i}`}" type="${type}">`,
      `    <parent link="${parent}"/><child link="a_${i}"/>`,
      `    <origin xyz="${new Vector3().setFromMatrixPosition(mount).toArray().join(' ')}" rpy="${[e.x, e.y, e.z].join(' ')}"/>`, '  </joint>',
      `  <joint name="joint_${i}" type="revolute">`, `    <parent link="a_${i}"/><child link="b_${i}"/>`, '    <origin xyz="0 0 0.053" rpy="0 0 0"/>',
      `    <axis xyz="${new Vector3(...module.jointAxis).normalize().toArray().join(' ')}"/>`,
      `    <limit lower="${module.jointRange[0] * rad}" upper="${module.jointRange[1] * rad}" effort="20" velocity="10"/>`, '    <dynamics damping="0.5" friction="0"/>', '  </joint>');
  });
  return [...lines, '</robot>'].join('\n');
}
