import { ZbotConfiguration, ZbotModule } from '../types/zbot';
import { Euler, Quaternion } from 'three';
import { validateConfiguration } from './configuration';

export interface XmlGeneratorOptions {
  kp?: number;
  kv?: number;
  friction?: number;
  gravity?: number;
  damping?: number;
  isFixedBase?: boolean;
  selfCollision?: boolean;
}

/**
 * Generates canonical, high-stability MuJoCo MJCF XML string from a ZbotConfiguration.
 * Follows the authentic 6-DOF Zbot kinematic definition:
 *  - Base body ("base"):
 *      <geom class="visual_a" pos="0 0 0"/>
 *      <geom class="coliision_a" pos="0 0 0"/>
 *      <body name="body_0" pos="0 0 0.0">
 *        <joint name="joint_0" type="hinge" pos="0 0 0.053" axis="0 -1 1" />
 *        <geom class="visual_b" pos="0 0 0"/>
 *        <geom class="coliision_b" pos="0 0 0"/>
 *  - Each subsequent module i connected to parent module p at docking face z = 0.106:
 *      Inside body_p:
 *        <geom name="visual_a_${childIdx}" class="visual_a" pos="0 0 0.106" euler="0 0 ${dockAngle}"/>
 *        <geom class="coliision_a" pos="0 0 0.106" euler="0 0 ${dockAngle}"/>
 *        <body name="body_i" pos="0 0 0.106" euler="0 0 ${dockAngle}">
 *          <joint name="joint_i" type="hinge" pos="0 0 0.053" axis="${axis}" />
 *          <geom class="visual_b" pos="0 0 0"/>
 *          <geom class="coliision_b" pos="0 0 0"/>
 *        </body>
 *  - Robot collision geoms have contype="1" conaffinity="0", floor has contype="1" conaffinity="1".
 *    This completely eliminates internal self-collision instability while ensuring realistic ground contact.
 */
export function generateMujocoXML(
  config: ZbotConfiguration,
  options: XmlGeneratorOptions = {}
): string {
  const errors = validateConfiguration(config);
  if (errors.length) throw new Error(errors.join('；'));
  const kp = options.kp ?? 80;
  const kv = options.kv ?? 8;
  const friction = options.friction ?? 1.2;
  const damping = options.damping ?? 0.5;
  const isFixedBase = options.isFixedBase ?? (config.baseMode ? config.baseMode === 'fixed' : config.category === 'arm');

  const lines: string[] = [];

  lines.push(`<mujoco model="${(config.id || 'zbot').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&apos;' }[c]!))}">`);
  lines.push(`  <compiler inertiafromgeom="true" angle="degree"/>`);
  lines.push(`  <option timestep="0.002" iterations="50" solver="Newton" integrator="implicitfast" gravity="0 0 ${options.gravity ?? -9.81}"/>`);
  lines.push(``);
  lines.push(`  <default>`);
  lines.push(`    <default class="coliision">`);
  lines.push(`      <geom contype="1" conaffinity="${options.selfCollision ? 1 : 0}" group="4" type="mesh" mesh="mb" density="1200" condim="3" friction="${friction} 0.05 0.001" margin="0.0005"/>`);
  lines.push(`      <default class="coliision_a">`);
  lines.push(`        <geom mesh="ma"/>`);
  lines.push(`      </default>`);
  lines.push(`      <default class="coliision_b">`);
  lines.push(`        <geom mesh="mb"/>`);
  lines.push(`      </default>`);
  lines.push(`    </default>`);
  lines.push(`    <default class="visual">`);
  lines.push(`      <geom contype="0" conaffinity="0" group="1" mass="0" type="mesh" mesh="mb"/>`);
  lines.push(`      <default class="visual_a">`);
  lines.push(`        <geom mesh="ma"/>`);
  lines.push(`      </default>`);
  lines.push(`      <default class="visual_b">`);
  lines.push(`        <geom mesh="mb"/>`);
  lines.push(`      </default>`);
  lines.push(`    </default>`);
  lines.push(`    <joint range="-180 180" limited="true" damping="${damping}" armature="0.01"/>`);
  lines.push(`  </default>`);
  lines.push(``);
  lines.push(`  <asset>`);
  lines.push(`    <mesh name="ma" file="ma.obj"/>`);
  lines.push(`    <mesh name="mb" file="mb.obj"/>`);
  lines.push(`    <texture name="plane_tex" type="2d" builtin="checker" rgb1=".22 .26 .32" rgb2=".14 .17 .22" width="512" height="512" mark="cross" markrgb=".7 .7 .7"/>`);
  lines.push(`    <material name="plane_mat" reflectance="0.15" texture="plane_tex" texrepeat="10 10" texuniform="true"/>`);
  lines.push(`  </asset>`);
  lines.push(``);
  lines.push(`  <worldbody>`);
  lines.push(`    <light directional="true" diffuse=".9 .9 .9" specular=".3 .3 .3" pos="0 0 6" dir="0 0 -1"/>`);
  lines.push(`    <geom name="floor" type="plane" size="20 20 0.1" material="plane_mat" condim="3" friction="${friction} 0.05 0.001" contype="1" conaffinity="1"/>`);

  const modules = config.modules;
  if (!modules || modules.length === 0) {
    lines.push(`  </worldbody>`);
    lines.push(`</mujoco>`);
    return lines.join('\n');
  }

  // Map children
  const childrenMap = new Map<string, ZbotModule[]>();
  for (const mod of modules) {
    if (mod.parentId) {
      if (!childrenMap.has(mod.parentId)) {
        childrenMap.set(mod.parentId, []);
      }
      childrenMap.get(mod.parentId)!.push(mod);
    }
  }

  const rootModule = modules.find((m) => m.parentId === null) || modules[0];
  const rootPos = config.rootPos || [0, 0, 0.05];
  const rootEuler = config.rootEuler || [90, 0, 0];

  const posStr = `${rootPos[0]} ${rootPos[1]} ${rootPos[2]}`;
  const eulerStr = `${rootEuler[0]} ${rootEuler[1]} ${rootEuler[2]}`;

  const jointNames: string[] = [];

  // Recursive tree builder matching authentic Zbot MJCF structure
  function buildModuleChildren(parentMod: ZbotModule, indent: string) {
    const children = childrenMap.get(parentMod.id) || [];
    for (const child of children) {
      const childIdx = modules.findIndex((m) => m.id === child.id);
      const childJointName = `joint_${childIdx}`;
      jointNames.push(childJointName);

      const childAxis = child.jointAxis || [0, -1, 1];
      const childAxisStr = `${childAxis[0]} ${childAxis[1]} ${childAxis[2]}`;
      const childRange = child.jointRange || [-180, 180];
      const childEuler = child.customEuler || [0, 0, child.dockAngle || 0];
      const childEulerStr = `${childEuler[0]} ${childEuler[1]} ${childEuler[2]}`;

      lines.push(`${indent}<!-- Module ${childIdx}: Part A fixed to parent docking port at z=0.106 with euler="${childEulerStr}" -->`);
      lines.push(`${indent}<geom name="visual_a_${childIdx}" class="visual_a" pos="0 0 0.106" euler="${childEulerStr}"/>`);
      lines.push(`${indent}<geom class="coliision_a" pos="0 0 0.106" euler="${childEulerStr}"/>`);

      lines.push(`${indent}<!-- Module ${childIdx}: Body & Joint ${childJointName} -->`);
      lines.push(`${indent}<body name="body_${childIdx}" pos="0 0 0.106" euler="${childEulerStr}">`);
      lines.push(`${indent}  <joint name="${childJointName}" type="hinge" pos="0 0 0.053" axis="${childAxisStr}" range="${childRange[0]} ${childRange[1]}"/>`);
      lines.push(`${indent}  <geom name="visual_b_${childIdx}" class="visual_b" pos="0 0 0"/>`);
      lines.push(`${indent}  <geom class="coliision_b" pos="0 0 0"/>`);

      if (!childrenMap.has(child.id)) lines.push(`${indent}  <site name="tip_${childIdx}" pos="0 0 0.106" size="0.008" rgba="0.2 0.9 0.7 1"/>`);
      // Recurse for deeper children
      buildModuleChildren(child, indent + '  ');

      lines.push(`${indent}</body>`);
    }
  }

  // Base Body containing Part A of root module
  const rootModIdx = modules.findIndex((m) => m.id === rootModule.id);
  const rootJointName = `joint_${rootModIdx}`;
  jointNames.push(rootJointName);

  const rootAxis = rootModule.jointAxis || [0, -1, 1];
  const rootAxisStr = `${rootAxis[0]} ${rootAxis[1]} ${rootAxis[2]}`;
  const rootRange = rootModule.jointRange || [-180, 180];

  lines.push(`    <body name="base" pos="${posStr}" euler="${eulerStr}">`);
  if (!isFixedBase) {
    lines.push(`      <freejoint/>`);
  }
  lines.push(`      <!-- Root Module ${rootModIdx} Part A (Base Half) -->`);
  lines.push(`      <geom name="visual_a_${rootModIdx}" class="visual_a" pos="0 0 0"/>`);
  lines.push(`      <geom class="coliision_a" pos="0 0 0"/>`);
  lines.push(``);
  lines.push(`      <!-- Root Module ${rootModIdx} Part B with ${rootJointName} -->`);
  lines.push(`      <body name="body_${rootModIdx}" pos="0 0 0.0">`);
  lines.push(`        <joint name="${rootJointName}" type="hinge" pos="0 0 0.053" axis="${rootAxisStr}" range="${rootRange[0]} ${rootRange[1]}"/>`);
  lines.push(`        <geom name="visual_b_${rootModIdx}" class="visual_b" pos="0 0 0"/>`);
  lines.push(`        <geom class="coliision_b" pos="0 0 0"/>`);

  if (!childrenMap.has(rootModule.id)) lines.push(`        <site name="tip_${rootModIdx}" pos="0 0 0.106" size="0.008"/>`);
  buildModuleChildren(rootModule, '        ');

  lines.push(`      </body>`);
  lines.push(`    </body>`);
  lines.push(`  </worldbody>`);
  lines.push(``);

  // Adjacent docking halves overlap by design; exclude neighboring bodies.
  if (options.selfCollision) {
    lines.push('  <contact>');
    for (const mod of modules) {
      const index = modules.indexOf(mod);
      const parent = mod.parentId === null ? 'base' : `body_${modules.findIndex((m) => m.id === mod.parentId)}`;
      lines.push(`    <exclude body1="${parent}" body2="body_${index}"/>`);
    }
    lines.push('  </contact>');
  }
  // Actuators
  lines.push(`  <actuator>`);
  for (let i = 0; i < jointNames.length; i++) {
    const jName = jointNames[i];
    const index = Number(jName.slice(6));
    const range = modules[index].jointRange;
    lines.push(`    <position name="act_${index}" joint="${jName}" kp="${kp}" kv="${kv}" ctrllimited="true" ctrlrange="${range[0] * Math.PI / 180} ${range[1] * Math.PI / 180}"/>`);
  }
  lines.push(`  </actuator>`);
  const initial = jointNames.map((name) => {
    const index = Number(name.slice(6));
    return (modules[index].initialAngle ?? config.defaultGait.manualAngles[name] ?? 0) * Math.PI / 180;
  });
  const baseQpos: number[] = [];
  if (!isFixedBase) {
    const r = Math.PI / 180;
    const quat = new Quaternion().setFromEuler(new Euler(rootEuler[0] * r, rootEuler[1] * r, rootEuler[2] * r, 'XYZ'));
    baseQpos.push(...rootPos, quat.w, quat.x, quat.y, quat.z);
  }
  lines.push('  <keyframe>');
  lines.push(`    <key name="initial" qpos="${[...baseQpos, ...initial].join(' ')}" ctrl="${initial.join(' ')}"/>`);
  lines.push('  </keyframe>');
  lines.push(`</mujoco>`);

  return lines.join('\n');
}
