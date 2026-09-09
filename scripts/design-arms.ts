import fs from 'node:fs';
import path from 'node:path';
import loadMujoco from '@mujoco/mujoco';
import { Euler, Quaternion } from 'three';
import { PRESET_CONFIGURATIONS } from '../src/data/presets';
import { designConfiguration, jointSample, percentile, poseQuality } from '../src/utils/designAnalysis';
import { forwardKinematics } from '../src/utils/kinematics';
import { getEndEffectorPose, solveInverseKinematics, type EndEffectorTarget } from '../src/utils/inverseKinematics';
import { generateMujocoXML } from '../src/utils/xmlGenerator';
import { MujocoEngine } from '../src/mujoco/MujocoEngine';

const out = path.resolve('results/arm-design');
fs.mkdirSync(out, {recursive:true});
const write = (name: string, value: unknown) => fs.writeFileSync(path.join(out,name), JSON.stringify(value,null,2));
const key = (v: number[], width: number) => v.map(x=>Math.floor(x/width)).join(',');
type Screen = {sigma:number[]; n:number; cells:number; dexterityP25:number; fullRankFraction:number};
const screenFile = path.join(out,'screening.json');
let screening: Screen[] = [];
if (fs.existsSync(screenFile) && !process.argv.includes('--fresh')) {
  screening = JSON.parse(fs.readFileSync(screenFile,'utf8'));
} else {
  for (const n of [6,7]) {
    for (let code=0;code<4**(n-1);code++) {
      const sigma = Array.from({length:n-1},(_,i)=>90*(Math.floor(code/4**i)%4));
      const config = designConfiguration(sigma), cells = new Set<string>(), dex:number[]=[];
      for (let i=0;i<256;i++) {
        const fk = forwardKinematics(config,jointSample(i,n));
        cells.add(key(fk.tip,.06));
        if (i%8===0) dex.push(poseQuality(fk).minimum);
      }
      screening.push({sigma,n,cells:cells.size,dexterityP25:percentile(dex,.25),fullRankFraction:dex.filter(v=>v>1e-5).length/dex.length});
      if(code%512===0) console.log(`screen ${n}: ${code}/${4**(n-1)}`);
    }
  }
  write('screening.json',screening);
}

// Retain different screening objectives within each length; always include existing baselines.
const shortlist = new Map<string,Screen>();
const retain=(r:Screen)=>shortlist.set(r.sigma.join(','),r);
for(const n of [6,7]) {
  const rows=screening.filter(r=>r.n===n);
  for(const metric of ['cells','dexterityP25'] as const)
    [...rows].sort((a,b)=>b[metric]-a[metric]).slice(0,4).forEach(retain);
}
for(const preset of PRESET_CONFIGURATIONS) {
  const sigma=preset.modules.slice(1).map(m=>m.dockAngle);
  retain(screening.find(r=>r.sigma.join(',')===sigma.join(','))!);
}
for (const sigma of [Array(6).fill(0),Array(6).fill(90),Array(6).fill(180),[90,270,90,270,90,270]]) {
  retain(screening.find(r=>r.sigma.join(',')===sigma.join(','))!);
}
write('shortlist.json',[...shortlist.values()]);
console.log(`Detailed evaluation: ${shortlist.size} candidates`);

const mujoco=await loadMujoco();
for(const part of ['ma','mb']) mujoco.FS.writeFile(`${part}.obj`,fs.readFileSync(`public/assets/${part}.obj`,'utf8'));
const positions: [number,number,number][]=[];
for(const radius of [.2,.35,.5]) for(const azimuth of [-60,0,60]) for(const height of [.2,.35,.5])
  positions.push([radius*Math.cos(azimuth*Math.PI/180),radius*Math.sin(azimuth*Math.PI/180),height]);
const orientations = [[0,0,0],[0,90,0],[180,0,0]].map(e=>new Quaternion().setFromEuler(new Euler(...e.map(v=>v*Math.PI/180) as [number,number,number],'XYZ')).toArray() as [number,number,number,number]);
const assumptions={
  version:1, sampleSequence:'Halton primes 2,3,5,7,11,13,17; q=360*h-180 degrees',
  screeningSamples:256, screeningDexteritySamples:32, detailedSamples:4096,
  rootPos:[0,0,.35],rootEuler:[0,0,0],jointLimits:[-180,180],modulePitchM:.106,
  characteristicLengthM:.3,voxelWidthsM:[.03,.05,.08],positions,orientations,
  screening:'All 5120 connection sequences; no collision filtering in coarse screening. Shortlist top 4 voxel coverage and top 4 dexterity per length plus six presets and four regular seven-module baselines. Shortlist is heuristic, not guaranteed to retain global optimum.',
  collision:'Real CAD OBJ convex hulls, existing adjacent-body exclusions, ncon===0, includes ground and 0.5mm margin. No gripper, cable, mount or environment geometry; static pose only.',
  payload:'Additional signed joint torque per kg at last docking face under world gravity [0,0,-9.81]. No self-weight, tool COM offset, actuator or stiffness rating. Not a payload capacity.',
  limitations:'Provisional generic targets, 3 orientations only, local IK with 3 nearest sampled seeds. Failure is not proof of unreachability. No continuous path verification. Uncalibrated dynamics not used for load ranking.',
};
write('assumptions.json',assumptions);
const results:any[]=[];
for(const candidate of shortlist.values()) {
  const config=designConfiguration(candidate.sigma), engine=new MujocoEngine();
  (engine as any).mujoco=mujoco;
  engine.loadModelFromXml(generateMujocoXML(config,{selfCollision:true}));
  const collisionFree=(q:number[])=>{
    engine.applyPose(Object.fromEntries(q.map((v,i)=>[`joint_${i}`,v])));
    return engine.getData().ncon===0;
  };
  const pool:{q:number[];pose:EndEffectorTarget}[]=[], dex:number[]=[], torque:number[]=[];
  const widths=[.03,.05,.08], cells=widths.map(()=>new Set<string>());
  let valid=0, bestQ:number[]|null=null,bestQuality=-1;
  for(let i=0;i<4096;i++) {
    const q=jointSample(i+4096,candidate.n);
    if(!collisionFree(q)) continue;
    valid++;
    const fk=forwardKinematics(config,q), quality=poseQuality(fk);
    dex.push(quality.minimum); torque.push(quality.maxPayloadTorquePerKg);
    widths.forEach((w,j)=>cells[j].add(key(fk.tip,w)));
    if(i%4===0) pool.push({q,pose:getEndEffectorPose(config,q)});
    if(quality.minimum>bestQuality) {bestQuality=quality.minimum;bestQ=q;}
  }
  const taskResults:any[]=[];
  for(const positionOnly of [true,false]) for(const position of positions) {
    for(const quaternion of positionOnly ? [orientations[0]] : orientations) {
      const target={position,quaternion};
      const nearest=[...pool].sort((a,b)=>distance(a.pose,target,positionOnly)-distance(b.pose,target,positionOnly)).slice(0,3);
      let solution:any=null;
      for(const seed of nearest) {
        const solved=solveInverseKinematics(config,target,seed.q,positionOnly);
        if(!solved.converged || !collisionFree(solved.angles)) continue;
        const quality=poseQuality(forwardKinematics(config,solved.angles));
        if(!solution || quality.minimum>solution.quality.minimum) solution={q:solved.angles,quality,positionError:solved.positionError,orientationError:solved.orientationError};
      }
      taskResults.push({positionOnly,target,solution});
    }
  }
  const poseTasks=taskResults.filter(t=>!t.positionOnly), solvedTasks=poseTasks.filter(t=>t.solution);
  const row={...candidate,id:config.id,validSamples:valid,collisionFreeFraction:valid/4096,
    sampledOccupiedVolumeM3:widths.map((w,i)=>({width:w,cells:cells[i].size,volume:cells[i].size*w**3})),
    collisionFreeDexterityP25:percentile(dex,.25),payloadTorqueP95NmPerKg:percentile(torque,.95),
    positionSuccess:taskResults.filter(t=>t.positionOnly&&t.solution).length,positionTotal:positions.length,
    poseSuccess:solvedTasks.length,poseTotal:poseTasks.length,
    taskDexterityP25:percentile(solvedTasks.map(t=>t.solution.quality.minimum),.25),
    taskPayloadTorqueP95NmPerKg:solvedTasks.length ? percentile(solvedTasks.map(t=>t.solution.quality.maxPayloadTorquePerKg),.95) : null,
    taskResults};
  // Export the successful task posture closest to [0.35,0,0.35], or a collision-free sample.
  const ordered=solvedTasks.sort((a,b)=>distance(a.target,{position:[.35,0,.35],quaternion:orientations[0]},false)-distance(b.target,{position:[.35,0,.35],quaternion:orientations[0]},false));
  const q=ordered[0]?.solution.q ?? bestQ;
  if(q) {
    config.modules.forEach((m,i)=>{m.initialAngle=q[i];});
    config.defaultGait.manualAngles=Object.fromEntries(q.map((v:number,i:number)=>[`joint_${i}`,v]));
    write(`${config.id}.json`,config);
  }
  results.push(row);engine.destroy();write('detailed.json',results);
  console.log(`${row.id}: valid ${valid}/4096; positions ${row.positionSuccess}/27; poses ${row.poseSuccess}/81; dex ${row.taskDexterityP25.toFixed(4)}`);
}

function distance(a:EndEffectorTarget,b:EndEffectorTarget,positionOnly:boolean) {
  const position=a.position.reduce((sum,v,i)=>sum+(v-b.position[i])**2,0);
  const dot=Math.min(1,Math.abs(a.quaternion.reduce((sum,v,i)=>sum+v*b.quaternion[i],0)));
  return position+(positionOnly?0:(.3*2*Math.acos(dot))**2);
}

// Pareto objectives: position/pose coverage and lower-tail dexterity up; module count down.
// Torque at different successful task subsets is not a comparable load objective.
const dominates=(a:any,b:any)=>a.positionSuccess>=b.positionSuccess && a.poseSuccess>=b.poseSuccess && a.taskDexterityP25>=b.taskDexterityP25 && a.n<=b.n &&
  (a.positionSuccess>b.positionSuccess||a.poseSuccess>b.poseSuccess||a.taskDexterityP25>b.taskDexterityP25||a.n<b.n);
const pareto=results.filter(r=>!results.some(s=>dominates(s,r)));
write('pareto.json',pareto.map(({taskResults,...r})=>r));
const headers=['id','n','sigma','positionSuccess','poseSuccess','collisionFreeFraction','taskDexterityP25','taskPayloadTorqueP95NmPerKg'];
fs.writeFileSync(path.join(out,'comparison.csv'),'\uFEFF'+[headers,...results.map(r=>headers.map(h=>h==='sigma'?r.sigma.join(' '):r[h]))].map(row=>row.map(v=>'"'+String(v??'').replaceAll('"','""')+'"').join(',')).join('\r\n'));
console.log('Pareto:',pareto.map(r=>r.id));
