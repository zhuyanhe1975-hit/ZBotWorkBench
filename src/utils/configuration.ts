import { ZbotConfiguration, ZbotModule } from '../types/zbot';

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
  if (!Array.isArray(config.modules) || config.modules.length < 1 || config.modules.length > 24) return [...errors, '仅支持 1–24 个模块的串联链'];
  const ids = new Set<string>();
  config.modules.forEach((m: unknown, i: number) => {
    if (!object(m)) { errors.push(`模块 ${i + 1} 必须为对象`); return; }
    if (!label(m.id) || !/^[A-Za-z0-9_-]+$/.test(String(m.id)) || ids.has(String(m.id))) errors.push(`模块 ${i + 1} ID 非法或重复`);
    if (typeof m.id === 'string') ids.add(m.id);
    if (!label(m.name)) errors.push(`模块 ${i + 1} 名称无效`);
    const preceding = (config.modules as unknown[])[i - 1];
    if (m.parentId !== (i === 0 ? null : object(preceding) ? preceding.id : undefined)) errors.push(`模块 ${i + 1} 必须连接前一模块（不支持分支、循环或乱序）`);
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
