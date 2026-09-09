import test from 'node:test';
import assert from 'node:assert/strict';
import loadMujoco from '@mujoco/mujoco';
import { benchmarkRobots, benchmarkFK, benchmarkPose, benchmarkXML, stiffnessProxy } from '../src/utils/architectureBenchmark';
import { designConfiguration, jointSample, poseQuality } from '../src/utils/designAnalysis';
import { forwardKinematics } from '../src/utils/kinematics';
import { solvePoseIK } from '../src/utils/inverseKinematics';
import { MujocoEngine } from '../src/mujoco/MujocoEngine';

test('benchmark ZBot reproduces original FK at nonzero angles',()=>{
  const robot=benchmarkRobots()[0],config=designConfiguration([180,180,180,180,180,180]);
  for(let i=0;i<10;i++) {
    const q=jointSample(i+123,7),a=benchmarkFK(robot,q),b=forwardKinematics(config,q);
    assert.ok(Math.hypot(...a.tip.map((v,i)=>v-b.tip[i]))<1e-10);
    a.joints.forEach((j,i)=>assert.ok(Math.hypot(...j.axis.map((v,k)=>v-b.joints[i].axis[k]))<1e-10));
  }
});

test('generic Jacobian agrees with finite differences for all architecture models',()=>{
  for(const robot of benchmarkRobots()) {
    const q=jointSample(83,7),quality=poseQuality(benchmarkFK(robot,q));
    for(let i=0;i<7;i++) {
      const a=[...q],b=[...q],delta=1e-5;
      a[i]+=delta*180/Math.PI;b[i]-=delta*180/Math.PI;
      const numerical=-9.81*(benchmarkFK(robot,a).tip[2]-benchmarkFK(robot,b).tip[2])/(2*delta);
      assert.ok(Math.abs(numerical-quality.payloadTorquePerKg[i])<1e-7);
    }
    assert.ok(stiffnessProxy(benchmarkFK(robot,q))>=0);
    const target=benchmarkPose(benchmarkFK(robot,q.map(v=>v*.99)));
    assert.ok(solvePoseIK(angles=>benchmarkFK(robot,angles),Array.from({length:7},()=>[-180,180]),target,q).converged);
  }
});

test('matched envelope models compile in MuJoCo and align with FK',async()=>{
  const mujoco=await loadMujoco();
  for(const robot of benchmarkRobots()) {
    const e=new MujocoEngine();(e as any).mujoco=mujoco;
    e.loadModelFromXml(benchmarkXML(robot));
    assert.ok(Math.abs(Array.from(e.getModel().body_mass as Float64Array).reduce((a,b)=>a+b,0)-7)<1e-9);
    for(let i=0;i<5;i++) {
      const q=jointSample(i+83,7);e.applyPose(Object.fromEntries(q.map((v,i)=>[`joint_${i}`,v])));
      const expected=benchmarkFK(robot,q).tip,actual=e.getMetrics().endEffectorPos!;
      assert.ok(Math.hypot(...actual.map((v,i)=>v-expected[i]))<1e-8);
    }
    const q=jointSample(333,7);
    e.applyPose(Object.fromEntries(q.map((v,i)=>[`joint_${i}`,v])));
    const restoring=Array.from(e.getData().qfrc_bias) as number[];
    const potential=(angles:number[])=>{
      e.applyPose(Object.fromEntries(angles.map((v,i)=>[`joint_${i}`,v])));
      const m=e.getModel(),d=e.getData();let energy=0;
      for(let i=0;i<m.nbody;i++) energy+=m.body_mass[i]*9.81*d.xipos[3*i+2];
      return energy;
    };
    for(let i=0;i<7;i++) {
      const a=[...q],b=[...q],delta=1e-5;
      a[i]+=delta*180/Math.PI;b[i]-=delta*180/Math.PI;
      assert.ok(Math.abs((potential(a)-potential(b))/(2*delta)-restoring[i])<1e-6);
    }
    e.destroy();
  }
});
