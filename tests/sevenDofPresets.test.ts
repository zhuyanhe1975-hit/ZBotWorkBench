import test from 'node:test';
import assert from 'node:assert/strict';
import loadMujoco from '@mujoco/mujoco';
import { SEVEN_DOF_PRESETS } from '../src/data/sevenDofPresets';
import { parseConfiguration } from '../src/utils/configuration';
import { generateMujocoXML } from '../src/utils/xmlGenerator';
import { benchmarkRobots, benchmarkFK, benchmarkXML } from '../src/utils/architectureBenchmark';
import { forwardKinematics } from '../src/utils/kinematics';
import { getEndEffectorPose, solveInverseKinematics } from '../src/utils/inverseKinematics';
import { MujocoEngine } from '../src/mujoco/MujocoEngine';

test('seven DOF presets retain geometry through save/import and IK works',()=>{
  for(const preset of SEVEN_DOF_PRESETS) {
    const c=parseConfiguration(JSON.stringify(preset));
    assert.equal(c.geometryMode,'envelope');assert.equal(c.modules.length,7);
    const seed=c.modules.map(m=>m.initialAngle!);
    const q=seed.map(v=>v*.99);
    assert.ok(solveInverseKinematics(c,getEndEffectorPose(c,q),seed).converged);
  }
  assert.throws(()=>parseConfiguration(JSON.stringify({...SEVEN_DOF_PRESETS[0],geometryMode:'unknown'})));
  assert.throws(()=>parseConfiguration(JSON.stringify({...SEVEN_DOF_PRESETS[0],geometryMode:['envelope']})));
});

test('workbench cylinder models match comparison geometry, mass, gravity and contact checks',async()=>{
  const mj=await loadMujoco();
  for(const [k,c] of SEVEN_DOF_PRESETS.entries()) {
    const e=new MujocoEngine(),reference=new MujocoEngine();(e as any).mujoco=mj;(reference as any).mujoco=mj;
    const xml=generateMujocoXML(c,{selfCollision:true});
    assert.ok(!xml.includes('file="ma.obj"'));
    e.loadModelFromXml(xml);e.resetSimulation(c);
    reference.loadModelFromXml(benchmarkXML(benchmarkRobots(90)[k]));reference.applyPose(c.defaultGait.manualAngles);
    const q=c.modules.map(m=>m.initialAngle!),fk=forwardKinematics(c,q),bench=benchmarkFK(benchmarkRobots(90)[k],q);
    assert.ok(Math.hypot(...fk.tip.map((v,i)=>v-bench.tip[i]))<1e-10);
    assert.ok(Math.hypot(...fk.tip.map((v,i)=>v-e.getMetrics().endEffectorPos![i]))<1e-8);
    assert.equal(e.getData().ncon,0);
    assert.ok(Math.abs(Array.from(e.getModel().body_mass as Float64Array).reduce((a,b)=>a+b,0)-7)<1e-8);
    for(let i=0;i<7;i++) assert.ok(Math.abs(e.getData().qfrc_bias[i]-reference.getData().qfrc_bias[i])<1e-8);
    const m=e.getModel();let visuals=0;
    for(let i=0;i<m.ngeom;i++) if(m.geom_group[i]===1) {
      visuals++;assert.equal(m.geom_type[i],5);assert.equal(m.geom_dataid[i],-1);
      assert.ok(Math.abs(m.geom_size[3*i]-.05)<1e-8);assert.ok(Math.abs(m.geom_size[3*i+1]-.0265)<1e-8);
    }
    assert.equal(visuals,14);
    e.step(100,c.defaultGait);assert.ok(Object.values(e.getMetrics().jointAngles).every(Number.isFinite));
    e.resetSimulation(c);assert.equal(e.getData().ncon,0);
    e.destroy();reference.destroy();
  }
});
