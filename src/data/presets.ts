import { ZbotConfiguration } from '../types/zbot';

function six(id: string, name: string, category: ZbotConfiguration['category'], sigma: number[], q: number[], hypothesis: string): ZbotConfiguration {
  const zones = [['#58748b', '#8fa5b3'], ['#3687b4', '#71b6d2'], ['#268f78', '#72bda2']];
  return {
    id: `zbot_${id}`, name, category, hypothesis,
    description: `${hypothesis}。六模块单链候选设计，尚未验证接口兼容、自碰撞、承载及任务性能。`,
    rootPos: [0, 0, 0.35], rootEuler: [90, 0, 0], baseMode: 'fixed',
    defaultGait: { type: 'manual', frequency: 0.7, amplitude: 35, phaseLag: 60, steering: 0, speed: 1,
      manualAngles: Object.fromEntries(q.map((angle, i) => [`joint_${i}`, angle])) },
    modules: q.map((angle, i) => ({ id: `mod_${i}`, name: `模块 ${i + 1}`, parentId: i ? `mod_${i - 1}` : null,
      dockAngle: i ? sigma[i - 1] : 0, initialAngle: angle, jointAxis: [0, -1, 1], jointRange: [-180, 180],
      colorA: category === 'zoned' ? zones[Math.floor(i / 2)][0] : '#3687b4',
      colorB: category === 'zoned' ? zones[Math.floor(i / 2)][1] : '#57ab98' })),
  };
}

/** Shared-conversation candidates, not validated robot capabilities. σ belongs to the incoming connection. */
export const PRESET_CONFIGURATIONS: ZbotConfiguration[] = [
  six('01_planar', '01 平面型（平行轴）', 'planar', [0, 0, 0, 0, 0], [0, 35, 35, 35, 35, 0], '研究平行轴的协同卷曲、钩挂'),
  six('02_orthogonal', '02 正交型', 'orthogonal', [180, 180, 180, 180, 180], [0, 55, -55, -55, 55, 0], '研究正交轴的空间弯曲与蛇形运动'),
  six('03_helical', '03 同手性螺旋型', 'helical', [90, 90, 90, 90, 90], [55, 55, 55, 55, 55, 55], '研究同向连接扭转的螺旋姿态与包覆'),
  six('04_alternating', '04 异手性均衡型', 'alternating', [90, 270, 90, 270, 90], [35, 55, -35, -55, 35, 55], '研究交替连接方位的空间可达性；均衡不代表已验证力学稳定'),
  six('05_leg', '05 腿型（单腿链）', 'leg', [180, 180, 0, 180, 0], [0, 80, 0, -120, 0, 50], '研究单腿链的落足、抬升与支撑，尚无多足机构'),
  six('06_zoned', '06 功能分区型', 'zoned', [0, 180, 180, 90, 0], [0, 0, 90, 90, 60, 60], '灰色 1–2 支撑、蓝色 3–4 定位、绿色 5–6 操作，颜色仅表示候选功能'),
];
