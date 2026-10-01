import { ConnectorFace, RootConnectorType, ZbotConfiguration, ZbotModule } from '../types/zbot';

import { connectorFaces } from './moduleMount';
import { TETRAHEDRON_EDGE } from './tetrahedronGeometry';

const finite = (v: unknown, min: number, max: number): v is number => typeof v === 'number' && Number.isFinite(v) && v >= min && v <= max;
const object = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);
const vector = (v: unknown, n: number, min: number, max: number) => Array.isArray(v) && v.length === n && v.every(x => finite(x, min, max));
const label = (v: unknown) => typeof v === 'string' && v.length > 0 && v.length <= 200;

/** Accepts unknown input so malformed imported JSON cannot reach XML generation or recursion. */
export function validateConfiguration(config: unknown): string[] {
  const errors: string[] = [];
  if (!object(config)) return ['构型必须是 JSON 对象'];
  if (!label(config.id) || !label(config.name)) errors.push('构型 ID 与名称必须为 1–200 字符');
  if (typeof config.description !== 'string' || config.description.length > 5000) errors.push('描述必须是最多 5000 字符的文本');
  if (config.hypothesis !== undefined && (typeof config.hypothesis !== 'string' || config.hypothesis.length > 5000)) errors.push('研究假设必须是最多 5000 字符的文本');
  if (typeof config.category !== 'string' || !['planar', 'orthogonal', 'helical', 'alternating', 'leg', 'zoned', 'snake', 'loop', 'walker', 'arm', 'custom'].includes(config.category)) errors.push('未知构型类别');
  if (config.baseMode !== undefined && (typeof config.baseMode !== 'string' || !['fixed', 'free'].includes(config.baseMode))) errors.push('基座模式必须为 fixed 或 free');
  if (config.geometryMode !== undefined && (typeof config.geometryMode !== 'string' || !['cad', 'envelope', 'mechanical'].includes(config.geometryMode))) errors.push('未知几何模型');
  if (!vector(config.rootPos, 3, -100, 100)) errors.push('基座位置必须是 ±100 米内的三个有限数');
  if (!vector(config.rootEuler, 3, -360, 360)) errors.push('基座角度必须是 ±360° 内的三个有限数');
  const connector = config.rootConnector !== undefined;
  if (connector && (!object(config.rootConnector) || !['cube', 'tetrahedron'].includes(String(config.rootConnector.type)) || config.rootConnector.size !== (config.rootConnector.type === 'tetrahedron' ? TETRAHEDRON_EDGE : 0.1))) errors.push('连接件须为边长 100 mm 的立方体或边长 180 mm 的正四面体');
  if (!Array.isArray(config.modules) || config.modules.length < (connector ? 0 : 1) || config.modules.length > 24) return [...errors, '最多支持 24 个模块，单链至少需要 1 个模块'];
  const faces = new Set<string>();
  const parents = new Set<string>();
  const ids = new Set<string>();
  config.modules.forEach((m: unknown, i: number) => {
    if (!object(m)) { errors.push(`模块 ${i + 1} 必须为对象`); return; }
    if (!label(m.id) || !/^[A-Za-z0-9_-]+$/.test(String(m.id)) || ids.has(String(m.id))) errors.push(`模块 ${i + 1} ID 非法或重复`);

    if (!label(m.name)) errors.push(`模块 ${i + 1} 名称无效`);
    const preceding = (config.modules as unknown[])[i - 1];
    if (connector) {
      if (m.parentId === null) {
        if (typeof m.mountFace !== 'string' || !connectorFaces(object(config.rootConnector) ? config.rootConnector.type as RootConnectorType : undefined).includes(m.mountFace as ConnectorFace) || faces.has(m.mountFace)) errors.push(`模块 ${i + 1} 必须连接未占用的连接件表面`);
        else faces.add(m.mountFace);
      } else if (typeof m.parentId !== 'string' || !ids.has(m.parentId) || parents.has(m.parentId)) errors.push(`模块 ${i + 1} 必须连接前面的模块空闲输出端（不支持循环）`);
      if (typeof m.parentId === 'string') parents.add(m.parentId);
      if (m.parentId !== null && m.mountFace !== undefined) errors.push(`模块 ${i + 1} 仅在连接根结构件时指定表面`);
    } else {
      if (m.parentId !== (i === 0 ? null : object(preceding) ? preceding.id : undefined)) errors.push(`模块 ${i + 1} 必须连接前一模块（不支持分支、循环或乱序）`);
      if (m.mountFace !== undefined) errors.push('只有连接件根构型可指定连接面');
    }
    if (typeof m.id === 'string') ids.add(m.id);
    if (!finite(m.dockAngle, -360, 360)) errors.push(`模块 ${i + 1} σ 必须在 ±360° 内`);
    if (!vector(m.jointAxis, 3, -100, 100) || Math.hypot(...m.jointAxis as number[]) < 1e-8) errors.push(`模块 ${i + 1} 关节轴必须为非零有限向量`);
    if (!vector(m.jointRange, 2, -180, 180) || (m.jointRange as number[])[0] >= (m.jointRange as number[])[1]) errors.push(`模块 ${i + 1} 关节范围必须递增且在 ±180° 内`);
    if (m.initialAngle !== undefined && (!finite(m.initialAngle, -180, 180) || Array.isArray(m.jointRange) && !finite(m.initialAngle, m.jointRange[0], m.jointRange[1]))) errors.push(`模块 ${i + 1} 初始 q 超出关节范围`);
    if (m.customEuler !== undefined && !vector(m.customEuler, 3, -360, 360)) errors.push(`模块 ${i + 1} 自定义安装角无效`);
    for (const key of ['colorA', 'colorB']) if (m[key] !== undefined && (typeof m[key] !== 'string' || !/^#[0-9a-fA-F]{6}$/.test(m[key] as string))) errors.push(`模块 ${i + 1} 颜色必须为 #RRGGBB`);
  });
  const g = config.defaultGait;
  if (!object(g)) return [...errors, '缺少控制参数'];
  if (typeof g.type !== 'string' || !['manual', 'serpentine', 'inchworm', 'rolling', 'sidewind', 'trot'].includes(g.type)) errors.push('未知控制模式');
  for (const [key, min, max] of [['frequency', 0, 10], ['amplitude', 0, 180], ['phaseLag', -360, 360], ['steering', -180, 180], ['speed', 0, 10]] as const) if (!finite(g[key], min, max)) errors.push(`控制参数 ${key} 超出 ${min}–${max}`);
  if (!object(g.manualAngles)) errors.push('manualAngles 必须为对象');
  else for (const [key, value] of Object.entries(g.manualAngles)) {
    const index = /^joint_(\d+)$/.exec(key);
    const m = index ? config.modules[Number(index[1])] : undefined;
    if (!index || !object(m) || !Array.isArray(m.jointRange) || !finite(value, m.jointRange[0], m.jointRange[1])) errors.push(`动作角 ${key} 无效或超出关节范围`);
  }
  return errors;
}

export function parseConfiguration(text: string): ZbotConfiguration {
  if (text.length > 200_000) throw new Error('构型文件不得超过 200 KB');
  const value: unknown = JSON.parse(text);
  if (object(value) && object(value.rootConnector) && value.rootConnector.type === 'tetrahedron' && value.rootConnector.size === .1) value.rootConnector = { ...value.rootConnector, size: TETRAHEDRON_EDGE };
  const errors = validateConfiguration(value);
  if (errors.length) throw new Error(errors.join('；'));
  const config = value as ZbotConfiguration;
  return { ...config, baseMode: config.baseMode ?? (config.category === 'arm' ? 'fixed' : 'free'),
    modules: config.modules.map((m, i) => ({ ...m, initialAngle: m.initialAngle ?? config.defaultGait.manualAngles[`joint_${i}`] ?? 0 })) };
}

/** Rebuild serial links and preserve each module's q and controller target by stable module identity. */
export function withSerialModules(config: ZbotConfiguration, modules: ZbotModule[]): ZbotConfiguration {
  const targets = new Map(config.modules.map((m, i) => [m.id, config.defaultGait.manualAngles[`joint_${i}`] ?? m.initialAngle ?? 0]));
  return { ...config, id: 'custom', category: 'custom',
    modules: modules.map((m, i) => ({ ...m, parentId: i ? modules[i - 1].id : null, dockAngle: i ? m.dockAngle : 0 })),
    defaultGait: { ...config.defaultGait, manualAngles: Object.fromEntries(modules.map((m, i) => [`joint_${i}`, targets.get(m.id) ?? m.initialAngle ?? 0])) } };
}

/** Start a fixed-base ZBot design with one module and zero joint angle. */
export function createConfiguration(): ZbotConfiguration {
  return {
    id: 'custom', name: '新建 ZBot 构型', category: 'custom', description: '',
    baseMode: 'fixed', geometryMode: 'cad', rootPos: [0, 0, 0.35], rootEuler: [0, 0, 0],
    modules: [{ id: 'mod_0', name: '模块 1', parentId: null, dockAngle: 0, initialAngle: 0,
      jointAxis: [0, -1, 1], jointRange: [-180, 180], colorA: '#3687b4', colorB: '#57ab98' }],
    defaultGait: { type: 'manual', frequency: 0.7, amplitude: 35, phaseLag: 60,
      steering: 0, speed: 1, manualAngles: { joint_0: 0 } },
  };
}

/** Insert into the serial chain using IDs that also work on LAN HTTP pages. */
export function insertModule(config: ZbotConfiguration, after: number, copy = false): ZbotConfiguration {
  if (config.rootConnector) return appendConnectorModule(config, config.modules[after]?.id, undefined, copy);
  if (config.modules.length >= 24) throw new Error('最多支持 24 个模块');
  if (!Number.isInteger(after) || after < 0 || after >= config.modules.length) throw new Error('请选择要连接的模块');
  let suffix = config.modules.length;
  while (config.modules.some(module => module.id === `mod_${suffix}`)) suffix++;
  const source = config.modules[after];
  const module: ZbotModule = {
    ...(copy ? source : createConfiguration().modules[0]),
    id: `mod_${suffix}`, name: copy ? `${source.name} 副本` : `模块 ${suffix + 1}`,
    dockAngle: copy ? source.dockAngle : 180,
  };
  const modules = [...config.modules];
  modules.splice(after + 1, 0, module);
  return withSerialModules(config, modules);
}

/** Reindex manual targets by stable module identity without changing branch connections. */
export function withTreeModules(config: ZbotConfiguration, modules: ZbotModule[]): ZbotConfiguration {
  const targets = new Map(config.modules.map((m, i) => [m.id, config.defaultGait.manualAngles[`joint_${i}`] ?? m.initialAngle ?? 0]));
  return { ...config, modules, defaultGait: { ...config.defaultGait, manualAngles: Object.fromEntries(modules.map((m, i) => [`joint_${i}`, targets.get(m.id) ?? m.initialAngle ?? 0])) } };
}

export function createCubeConfiguration(): ZbotConfiguration {
  return { ...createConfiguration(), name: '立方体根构型', rootConnector: { type: 'cube', size: 0.1 }, modules: [], rootPos: [0, 0, .35], defaultGait: { ...createConfiguration().defaultGait, manualAngles: {} } };
}

export function appendConnectorModule(config: ZbotConfiguration, parentId?: string, face?: ConnectorFace, copy = false): ZbotConfiguration {
  if (!config.rootConnector) throw new Error('需要连接件根构型');
  if (config.modules.length >= 24) throw new Error('最多支持 24 个模块');
  const parent = config.modules.find(m => m.id === parentId);
  if (parentId && !parent) throw new Error('请选择有效父模块');
  if (parent && config.modules.some(m => m.parentId === parent.id)) throw new Error('此模块输出端已连接，请选择分支末端');
  if (!parent && (!face || !connectorFaces(config.rootConnector.type).includes(face) || config.modules.some(m => m.parentId === null && m.mountFace === face))) throw new Error('请选择未占用的连接件表面');
  let suffix = config.modules.length;
  while (config.modules.some(m => m.id === `mod_${suffix}`)) suffix++;
  const module: ZbotModule = { ...(copy && parent ? parent : createConfiguration().modules[0]), id: `mod_${suffix}`, name: `模块 ${suffix + 1}`, parentId: parent?.id ?? null, mountFace: parent ? undefined : face, dockAngle: 0 };
  if (parent) delete module.mountFace;
  return withTreeModules(config, [...config.modules, module]);
}

export function removeConnectorBranch(config: ZbotConfiguration, id: string): ZbotConfiguration {
  const removed = new Set([id]);
  for (const m of config.modules) if (m.parentId && removed.has(m.parentId)) removed.add(m.id);
  return withTreeModules(config, config.modules.filter(m => !removed.has(m.id)));
}

export function createCubeQuadruped(): ZbotConfiguration {
  let config = createCubeConfiguration();
  config.name = '立方体四足 · 8 模块'; config.category = 'walker'; config.baseMode = 'free';
  config.description = '100 mm立方体中心连接件，四个侧面各连接两模块腿。候选静态四足结构，运动能力需实验验证。';
  config.rootPos = [0, 0, .12];
  for (const face of ['+x', '-x', '+y', '-y'] as const) {
    config = appendCubeModule(config, undefined, face);
    const parent = config.modules[config.modules.length - 1];
    parent.colorA = '#646b73'; parent.colorB = '#757d86';
    // The OBJ mating-plane normal is [0,-1,1]. Preserve that physical hinge;
    // docking twists align the bent leg radially, and the knee output points down.
    parent.name = `${face} 髋模块`; parent.dockAngle = 16.324949936895234; parent.initialAngle = -45;
    config = appendCubeModule(config, parent.id);
    const child = config.modules[config.modules.length - 1];
    child.colorA = '#646b73'; child.colorB = '#757d86';
    child.name = `${face} 足模块`; child.dockAngle = 50.46219115757686; child.initialAngle = -87.5922517865766;
  }
  config.defaultGait.manualAngles = Object.fromEntries(config.modules.map((m, i) => [`joint_${i}`, m.initialAngle ?? 0]));
  return config;
}

/** Preserve the cube helper API for existing callers. */
export const appendCubeModule = appendConnectorModule;
export const removeCubeBranch = removeConnectorBranch;

export function createTetrahedronConfiguration(): ZbotConfiguration {
  return { ...createCubeConfiguration(), name: '正四面体根构型', rootConnector: { type: 'tetrahedron', size: TETRAHEDRON_EDGE } };
}
