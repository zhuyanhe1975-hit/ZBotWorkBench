import fs from 'node:fs';
import loadMujoco from '@mujoco/mujoco';
import { designConfiguration, jointSample, percentile, poseQuality } from '../src/utils/designAnalysis';
import { forwardKinematics } from '../src/utils/kinematics';
import { getEndEffectorPose, solveInverseKinematics, type EndEffectorTarget } from '../src/utils/inverseKinematics';
import { generateMujocoXML } from '../src/utils/xmlGenerator';
import { MujocoEngine } from '../src/mujoco/MujocoEngine';

const out='results/arm-design';
const ids=['design_6_22221','design_6_22222','design_7_232222','design_7_221222','design_7_222222'];
const targets:EndEffectorTarget[]=Array.from({length:64},(_,i)=>{
  const u=jointSample(i+30000,6).map(v=>(v+180)/360);
  const radius=Math.sqrt(.18**2+u[0]*(.5**2-.18**2)),azimuth=(-75+150*u[1])*Math.PI/180;
  const a=2*Math.PI*u[4],b=2*Math.PI*u[5];
  return {position:[radius*Math.cos(azimuth),radius*Math.sin(azimuth),.2+.35*u[2]],
    quaternion:[Math.sqrt(1-u[3])*Math.sin(a),Math.sqrt(1-u[3])*Math.cos(a),Math.sqrt(u[3])*Math.sin(b),Math.sqrt(u[3])*Math.cos(b)]};
});
const mujoco=await loadMujoco();
for(const part of ['ma','mb']) mujoco.FS.writeFile(`${part}.obj`,fs.readFileSync(`public/assets/${part}.obj`,'utf8'));
const distance=(a:EndEffectorTarget,b:EndEffectorTarget)=>a.position.reduce((s,v,i)=>s+(v-b.position[i])**2,0)+(.6*Math.acos(Math.min(1,Math.abs(a.quaternion.reduce((s,v,i)=>s+v*b.quaternion[i],0)))))**2;
const rows:any[]=[];
for(const id of ids) {
  const saved=JSON.parse(fs.readFileSync(`${out}/${id}.json`,'utf8'));
  const config=designConfiguration(saved.modules.slice(1).map((m:any)=>m.dockAngle));
  const engine=new MujocoEngine(); (engine as any).mujoco=mujoco;
  engine.loadModelFromXml(generateMujocoXML(config,{selfCollision:true}));
  const clear=(q:number[])=>{
    engine.applyPose(Object.fromEntries(q.map((v,i)=>[`joint_${i}`,v])));
    return engine.getData().ncon===0;
  };
  // Verify that the delivered startup posture is clear and aligns with true MuJoCo geometry.
  const start=saved.modules.map((m:any)=>m.initialAngle);
  if(!clear(start)) throw Error(`${id}: exported posture contacts geometry`);
  const actual=engine.getMetrics().endEffectorPos!, expected=getEndEffectorPose(saved,start).position;
  if(Math.hypot(...actual.map((v,i)=>v-expected[i]))>1e-8) throw Error(`${id}: FK disagrees with MuJoCo`);
  const pool:{q:number[];pose:EndEffectorTarget}[]=[];
  for(let i=0;i<8192;i++) {
    const q=jointSample(i+16000,config.modules.length);
    if(clear(q)) pool.push({q,pose:getEndEffectorPose(config,q)});
  }
  const tasks=targets.map(target=>{
    const nearest=[...pool].sort((a,b)=>distance(a.pose,target)-distance(b.pose,target)).slice(0,8);
    let solution:any=null;
    for(const seed of nearest) {
      const result=solveInverseKinematics(config,target,seed.q);
      if(!result.converged||!clear(result.angles)) continue;
      const quality=poseQuality(forwardKinematics(config,result.angles));
      if(!solution||quality.minimum>solution.quality.minimum) solution={q:result.angles,quality};
    }
    return {target,solution};
  });
  const solved=tasks.filter(t=>t.solution);
  const row={id,success:solved.length,total:targets.length,taskDexterityP25:percentile(solved.map(t=>t.solution.quality.minimum),.25),tasks};
  rows.push(row);engine.destroy();
  console.log(`${id}: independent orientations ${row.success}/64, dex ${row.taskDexterityP25.toFixed(4)}`);
}
const commonIndices=targets.map((_,i)=>i).filter(i=>rows.every(r=>r.tasks[i].solution));
const commonTasks=rows.map(r=>({id:r.id,count:commonIndices.length,
  dexterityP25:percentile(commonIndices.map(i=>r.tasks[i].solution.quality.minimum),.25),
  payloadTorqueP95NmPerKg:commonIndices.length?percentile(commonIndices.map(i=>r.tasks[i].solution.quality.maxPayloadTorquePerKg),.95):null}));
fs.writeFileSync(`${out}/validation.json`,JSON.stringify({
  assumptions:'Independent Halton task positions (cylindrical sector, radius .18-.5m, azimuth +/-75deg, z .2-.55m) and SO(3) orientations; 8192 new seed samples, nearest 8 starts, highest dexterity clear solution retained. Same CAD hull checks. Not globally exhaustive. Static export pose verified in MuJoCo.',
  rows,commonIndices,commonTasks},null,2));
console.log('Shared successful targets:',commonTasks);
