import fs from 'node:fs';
import loadMujoco from '@mujoco/mujoco';
import { benchmarkRobots, benchmarkFK, benchmarkPose, benchmarkXML, stiffnessProxy, PACKAGE_LENGTH, PACKAGE_RADIUS } from '../src/utils/architectureBenchmark';
import { jointSample, percentile, poseQuality } from '../src/utils/designAnalysis';
import { solvePoseIK, type EndEffectorTarget } from '../src/utils/inverseKinematics';
import { MujocoEngine } from '../src/mujoco/MujocoEngine';

const out='results/architecture-comparison';fs.mkdirSync(out,{recursive:true});
// Optional first-pass checkpoint: preserve validated solutions when refining the seed pool.
const initialFile=`${out}/details-initial.json`;
const initialRows:any[]=fs.existsSync(initialFile)?JSON.parse(fs.readFileSync(initialFile,'utf8')):[];
const write=(name:string,data:unknown)=>fs.writeFileSync(`${out}/${name}`,JSON.stringify(data,null,2));
const targets:EndEffectorTarget[]=Array.from({length:128},(_,i)=>{
  const u=jointSample(i+41000,6).map(v=>(v+180)/360),r=Math.sqrt(.12**2+u[0]*(.5**2-.12**2)),az=(-90+180*u[1])*Math.PI/180;
  return {position:[r*Math.cos(az),r*Math.sin(az),.15+.5*u[2]],quaternion:[
    Math.sqrt(1-u[3])*Math.sin(2*Math.PI*u[4]),Math.sqrt(1-u[3])*Math.cos(2*Math.PI*u[4]),
    Math.sqrt(u[3])*Math.sin(2*Math.PI*u[5]),Math.sqrt(u[3])*Math.cos(2*Math.PI*u[5])
  ]};
});
const distance=(a:EndEffectorTarget,b:EndEffectorTarget,positionOnly:boolean)=>a.position.reduce((s,v,i)=>s+(v-b.position[i])**2,0)+(positionOnly?0:(.6*Math.acos(Math.min(1,Math.abs(a.quaternion.reduce((s,v,i)=>s+v*b.quaternion[i],0)))))**2);
const mujoco=await loadMujoco(),results:any[]=[];
for(const limit of [90,180]) for(const robot of benchmarkRobots(limit)) {
  const count=robot.cells.length,engine=new MujocoEngine();(engine as any).mujoco=mujoco;
  engine.loadModelFromXml(benchmarkXML(robot));
  const clear=(q:number[])=>{engine.applyPose(Object.fromEntries(q.map((v,i)=>[`joint_${i}`,v])));return engine.getData().ncon===0;};
  const pool:{q:number[];pose:EndEffectorTarget}[]=[],cells= new Set<string>();
  const dex:number[]=[],reach:number[]=[],checkpoints:any[]=[];
  let valid=0;
  for(let i=0;i<16384;i++) {
    const q=jointSample(i+60000,count).map(v=>v*limit/180);
    if(clear(q)) {
      valid++;
      const fk=benchmarkFK(robot,q),pose=benchmarkPose(fk);
      cells.add(pose.position.map(v=>Math.floor(v/.05)).join(','));
      if(i%2===0)pool.push({q,pose});
      dex.push(poseQuality(fk).minimum);reach.push(Math.hypot(pose.position[0],pose.position[1],pose.position[2]-.35));
    }
    if([4095,8191,16383].includes(i))checkpoints.push({samples:i+1,clear:valid,cells:cells.size});
  }
  // Equal raw sampling can leave a very sparse seed set for a collision-heavy architecture.
  // Keep raw-sample statistics fixed, but refine IK with at least 4096 clear starting poses.
  let extraSeedSamples=0;
  for(let i=16384;pool.length<4096 && i<147456;i++) {
    extraSeedSamples++;
    const q=jointSample(i+60000,count).map(v=>v*limit/180);
    if(clear(q))pool.push({q,pose:benchmarkPose(benchmarkFK(robot,q))});
  }
  const ranges=Array.from({length:count},()=>[-limit,limit] as [number,number]);
  const tasks:any[]=[];
  for(const positionOnly of [true,false]) for(const target of targets) {
    const seeds=[...pool].sort((a,b)=>distance(a.pose,target,positionOnly)-distance(b.pose,target,positionOnly)).slice(0,32);
    // Feasible sets are nested: wider limits must retain every narrow-limit solution.
    const inherited=limit===180 ? results.find(r=>r.id===robot.id&&r.limit===90)?.tasks[tasks.length] : null;
    let solution:any=inherited?.solution ? structuredClone(inherited.solution) : null;
    let kinematicSolved=!!inherited?.kinematicSolved;
    const previous=initialRows.find(r=>r.id===robot.id&&r.limit===limit)?.tasks[tasks.length];
    if(previous?.solution && (!solution || previous.solution.quality.minimum>solution.quality.minimum)) solution=structuredClone(previous.solution);
    kinematicSolved ||= !!previous?.kinematicSolved;
    for(const seed of seeds) {
      const result=solvePoseIK(q=>benchmarkFK(robot,q),ranges,target,seed.q,positionOnly);
      if(!result.converged)continue;
      kinematicSolved=true;
      if(!clear(result.angles))continue;
      const fk=benchmarkFK(robot,result.angles),quality=poseQuality(fk);
      if(!solution||quality.minimum>solution.quality.minimum)solution={q:result.angles,quality,
        stiffnessCoefficient:stiffnessProxy(fk),selfGravityTorquePerKgModule:Array.from(engine.getData().qfrc_bias) as number[]};
    }
    tasks.push({positionOnly,target,kinematicSolved,solution});
  }
  const row={id:robot.id,label:robot.label,limit,count,robot,clearSamples:valid,totalSamples:16384,ikSeedPoolSize:pool.length,extraSeedSamples,
    sampleDexterityP25:percentile(dex,.25),sampleMaxReachM:Math.max(0,...reach),checkpoints,
    positionSolved:tasks.filter(t=>t.positionOnly&&t.solution).length,
    poseSolved:tasks.filter(t=>!t.positionOnly&&t.solution).length,tasks};
  results.push(row);engine.destroy();write('details.json',results);
  fs.writeFileSync(`${out}/${robot.id}-${limit}.xml`,benchmarkXML(robot));
  console.log(`${robot.id} +/-${limit}: valid ${valid}/16384; position ${row.positionSolved}/128; pose ${row.poseSolved}/128`);
}
const common=[];
for(const limit of [90,180]) {
  const rows=results.filter(r=>r.limit===limit),indices=targets.map((_,i)=>i).filter(i=>rows.every(r=>r.tasks[128+i].solution));
  common.push({limit,indices,rows:rows.map(r=>{
    const solutions=indices.map(i=>r.tasks[128+i].solution);
    return {id:r.id,count:indices.length,dexterityP25:percentile(solutions.map(s=>s.quality.minimum),.25),
      payloadTorqueP95:solutions.length?percentile(solutions.map(s=>s.quality.maxPayloadTorquePerKg),.95):null,
      selfGravityTorqueP95:solutions.length?percentile(solutions.map(s=>Math.max(...s.selfGravityTorquePerKgModule.map(Math.abs))),.95):null,
      stiffnessCoefficientP95:solutions.length?percentile(solutions.map(s=>s.stiffnessCoefficient),.95):null};
  })});
}
write('summary.json',{assumptions:{packageLength:PACKAGE_LENGTH,packageDiameter:2*PACKAGE_RADIUS,
  collision:'Common two half-cylinders per module, central hinge, zero-pose envelope only. Adjacent rigid-body contacts excluded, floor and nonadjacent bodies tested. Not manufacturability validation or commercial product geometry.',
  mass:'Each half-cylinder has a nominal .5 kg. Self-gravity coefficients correspond to 1 kg/module, not measured mass. No gripper or payload offset.',
  stiffness:'Identical uncoupled joint torsional stiffness, rigid links: vertical compliance=sum(Jz^2)/K. Does not measure actual accuracy.',
  targets:'128 deterministic cylinder-sector positions r=.12-.5m, azimuth +/-90deg, world z=.15-.65m, uniform SO(3) orientation; identical for all models and limit settings.',
  solver:'32 nearest clear Halton seeds; at least 4096 clear seed poses sought with at most 131072 extra samples beyond initial 16384. Highest minimum singular value solution retained. Initial checkpoint and narrow-limit solutions inherited. Task failure is not proof of unreachable geometry.',
  dexterity:'Linear Jacobian scaled by fixed .3m characteristic length; use only common successful full-pose tasks for comparative torque/dexterity.',
},rows:results.map(({tasks,robot,...r})=>r),common});
write('targets.json',targets);console.log(JSON.stringify(common,null,2));
