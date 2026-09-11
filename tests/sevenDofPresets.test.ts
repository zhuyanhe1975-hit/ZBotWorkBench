import test from 'node:test';
import assert from 'node:assert/strict';
import loadMujoco from '@mujoco/mujoco';
import { SEVEN_DOF_PRESETS } from '../src/data/sevenDofPresets';
import { parseConfiguration } from '../src/utils/configuration';
import { generateMujocoXML } from '../src/utils/xmlGenerator';
import { benchmarkRobots, benchmarkFK, benchmarkXML } from '../src/utils/architectureBenchmark';
import { forwardKinematics } from '../src/utils/kinematics';
import { getEndEffectorPose, solveInverseKinematics } from '../src/utils/inverseKinematics';
import { Vector3 } from 'three';
import { mechanicalGeoms } from '../src/utils/mechanicalGeometry';
import { MujocoEngine } from '../src/mujoco/MujocoEngine';

test('seven DOF presets retain geometry through save/import and IK works',()=>{
  for(const preset of SEVEN_DOF_PRESETS) {
    const c=parseConfiguration(JSON.stringify(preset));
    assert.equal(c.geometryMode,preset.id==='seven_zbot'?'cad':'mechanical');assert.equal(c.modules.length,7);
    const seed=c.modules.map(m=>m.initialAngle!);
    const q=seed.map(v=>v*.99);
    assert.ok(solveInverseKinematics(c,getEndEffectorPose(c,q),seed).converged);
  }
  assert.throws(()=>parseConfiguration(JSON.stringify({...SEVEN_DOF_PRESETS[0],geometryMode:'unknown'})));
  assert.throws(()=>parseConfiguration(JSON.stringify({...SEVEN_DOF_PRESETS[0],geometryMode:['envelope']})));
});

test('workbench cylinder models match comparison geometry, mass, gravity and contact checks',async()=>{
  const mj=await loadMujoco();
  for(const [k,preset] of SEVEN_DOF_PRESETS.entries()) {
    const c = {...preset, geometryMode:'envelope' as const};
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


test('physical presets use CAD or shaft-aligned connected geometry, with consistent FK and stable simulation',async()=>{
  const mj=await loadMujoco();
  const {readFile}=await import('node:fs/promises');
  mj.FS.writeFile('ma.obj',await readFile('public/assets/ma.obj','utf8'));
  mj.FS.writeFile('mb.obj',await readFile('public/assets/mb.obj','utf8'));
  for(const c of SEVEN_DOF_PRESETS) {
    const e=new MujocoEngine();(e as any).mujoco=mj;
    const xml=generateMujocoXML(c,{selfCollision:true});
    assert.equal(xml.includes('file="ma.obj"'),c.geometryMode==='cad');
    e.loadModelFromXml(xml);e.resetSimulation(c);
    const m=e.getModel(),d=e.getData();
    assert.equal(m.njnt,7);
    const tip=forwardKinematics(c,c.modules.map(v=>v.initialAngle!)).tip;
    assert.ok(Math.hypot(...tip.map((v,i)=>v-e.getMetrics().endEffectorPos![i]))<1e-8);
    assert.equal(d.ncon,0,c.id+' initial contact');
    if(c.geometryMode==='mechanical') {
      assert.ok(Math.abs(Array.from(m.body_mass as Float64Array).reduce((a,b)=>a+b,0)-7)<1e-8);
      for(let j=0;j<7;j++) {
        const barrel=mj.mj_name2id(m,5,`visual_a_${j}_2`);
        const quat=Array.from(m.geom_quat.slice(barrel*4,barrel*4+4)) as number[];
        const {Quaternion,Vector3}=await import('three');
        const axis=new Vector3(0,0,1).applyQuaternion(new Quaternion(quat[1],quat[2],quat[3],quat[0]));
        assert.ok(Math.abs(axis.dot(new Vector3(...c.modules[j].jointAxis).normalize()))>1-1e-8);
      }
    }
    e.step(100,c.defaultGait);
    assert.ok(Math.abs(e.getTime()-.2)<1e-8);
    for(let i=0;i<d.warning.size();i++) assert.equal(d.warning.get(i).number,0,c.id);
    e.destroy();
  }
});


test('snake and YuMi motors share one barrel and fit the neutral module envelope', () => {
  for (const axis of [[1,0,0], [0,1,0], [0,0,1]]) {
    const parts = ['a', 'b'].map(part => mechanicalGeoms(axis, part as 'a' | 'b', 0)
      .filter(xml => xml.includes('name="visual_'))
      .map(xml => {
        const points = xml.match(/fromto="([^"]+)"/)![1].split(' ').map(Number);
        return {p: new Vector3(...points.slice(0,3) as [number,number,number]),
          q: new Vector3(...points.slice(3) as [number,number,number]),
          r: Number(xml.match(/size="([^"]+)"/)![1])};
      }));
    const barrels = parts.flat().filter(g => g.r === .024);
    assert.equal(barrels.length, 1);
    assert.ok(Math.abs(barrels[0].p.distanceTo(barrels[0].q) - .064) < 1e-12);
    assert.ok(barrels[0].p.clone().add(barrels[0].q).multiplyScalar(.5).distanceTo(new Vector3(0,0,.053)) < 1e-12);
    for (const {p,q,r} of parts.flat()) {
      const direction = q.clone().sub(p).normalize();
      const u = new Vector3(Math.abs(direction.x) < .9 ? 1 : 0, Math.abs(direction.x) < .9 ? 0 : 1, 0)
        .cross(direction).normalize();
      const v = direction.clone().cross(u);
      // Extremal axial extent is analytic; sample both rim circles for radial bounds.
      const dz = r * Math.sqrt(1 - direction.z ** 2);
      assert.ok(Math.min(p.z,q.z) - dz >= -1e-12);
      assert.ok(Math.max(p.z,q.z) + dz <= .106 + 1e-12);
      for (const center of [p,q]) for (let i=0;i<360;i++) {
        const t=i*Math.PI/180;
        const point=center.clone().addScaledVector(u,r*Math.cos(t)).addScaledVector(v,r*Math.sin(t));
        assert.ok(Math.hypot(point.x,point.y) <= .05 + 1e-12);
      }
    }
  }
});
