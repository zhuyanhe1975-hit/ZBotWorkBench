import type { ZbotConfiguration } from '../types/zbot';
import { benchmarkRobots } from '../utils/architectureBenchmark';

const names = ['7DOF · ZBot 正交', '7DOF · 标准蛇形', '7DOF · YuMi 风格'];
const descriptions = ['标准斜轴，六处 σ=180°；七模块正交串联链。', 'X/Y 横向转轴交替；七模块俯仰／偏航链。', 'Z/Y 扭转／弯曲转轴交替；等包络轴系抽象，不是 ABB 原机。'];
// Same collision-checked target from the comparison's +/-90 degree common task set.
const poses = [
  [-10.783855855211254,64.2990637384368,-90,-79.70412638129653,41.798348967625806,-49.88229930871586,37.80652232952186],
  [-46.30448559231838,90,-6.162756118729102,-63.936629033558205,-39.90106168216104,-72.60949983896015,-13.938822661037676],
  [20.26374202801271,90,-72.20689038981106,-44.11979293178073,-7.44425426888471,-83.15568718427437,18.643785919454768],
];
const colors = [['#0284c7','#7dd3fc'],['#059669','#6ee7b7'],['#9333ea','#d8b4fe']];
export const SEVEN_DOF_PRESETS: ZbotConfiguration[] = benchmarkRobots(90).map((robot,k) => ({
  id:`seven_${robot.id}`,name:names[k],category:'arm',baseMode:'fixed',geometryMode:'envelope',
  rootPos:[0,0,.35],rootEuler:[0,0,0],
  hypothesis:descriptions[k],description:`${descriptions[k]} 每模块零位包络直径100 mm、长106 mm，标称质量1 kg用于对照，不是实物质量或额定承载。`,
  modules:robot.cells.map((cell,i)=>({id:`mod_${i}`,name:`模块 ${i+1}`,parentId:i?`mod_${i-1}`:null,
    dockAngle:i&&k===0?180:0,jointAxis:[...cell.axis],jointRange:[-90,90],initialAngle:poses[k][i],colorA:colors[k][0],colorB:colors[k][1]})),
  defaultGait:{type:'manual',frequency:.7,amplitude:20,phaseLag:60,steering:0,speed:1,manualAngles:Object.fromEntries(poses[k].map((q,i)=>[`joint_${i}`,q]))},
}));
