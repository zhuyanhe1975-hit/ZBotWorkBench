import { Matrix4, Quaternion, Vector3 } from 'three';
import type { KinematicResult, Vec3 } from './kinematics';
import type { EndEffectorTarget } from './inverseKinematics';

export interface BenchmarkCell { direction: Vec3; axis: Vec3; endRotation: Vec3 }
export interface BenchmarkRobot { id: string; label: string; cells: BenchmarkCell[]; limit: number }
export const PACKAGE_LENGTH = .106;
export const PACKAGE_RADIUS = .05;
export const BENCHMARK_BASE: Vec3 = [0,0,.35];
const rotation=(v:Vec3)=>new Matrix4().makeRotationFromQuaternion(new Quaternion().setFromAxisAngle(new Vector3(...v).normalize(), Math.hypot(...v)));

/** Rotation vectors are radians. Each cell has a 106 mm neutral backbone and a central hinge. */
export function benchmarkRobots(limit=180): BenchmarkRobot[] {
  return [
    {id:'zbot',label:'ZBot orthogonal',limit,cells:Array.from({length:7},()=>({direction:[0,0,1],axis:[0,-1,1],endRotation:[0,0,Math.PI]}))},
    {id:'snake',label:'Alternating snake',limit,cells:Array.from({length:7},(_,i)=>({direction:[0,0,1],axis:i%2?[0,1,0]:[1,0,0],endRotation:[0,0,0]}))},
    // YuMi-inspired roll/bend hierarchy in a straight reference posture.
    // Equal central-hinge cells omit ABB's nonintersecting-axis offsets and link proportions.
    {id:'yumi_style',label:'YuMi-inspired 7R',limit,cells:Array.from({length:7},(_,i)=>({direction:[0,0,1],axis:i%2?[0,1,0]:[0,0,1],endRotation:[0,0,0]}))},
  ];
}

export function benchmarkFK(robot:BenchmarkRobot,q:number[]):KinematicResult {
  if(q.length!==robot.cells.length||q.some(v=>!Number.isFinite(v))) throw Error('One finite angle per joint required');
  let input=new Matrix4().makeTranslation(...BENCHMARK_BASE);
  const result:KinematicResult={joints:[],endpoints:[[...BENCHMARK_BASE]],tip:[0,0,0],parts:[]};
  robot.cells.forEach((cell,i)=>{
    const half=new Vector3(...cell.direction).multiplyScalar(PACKAGE_LENGTH/2);
    const axis=new Vector3(...cell.axis).normalize();
    result.joints.push({id:`mod_${i}`,position:half.clone().applyMatrix4(input).toArray(),axis:axis.clone().transformDirection(input).toArray()});
    const b=input.clone().multiply(new Matrix4().makeTranslation(...half.toArray()))
      .multiply(new Matrix4().makeRotationAxis(axis,q[i]*Math.PI/180))
      .multiply(new Matrix4().makeTranslation(...half.clone().negate().toArray()));
    result.parts.push({moduleId:`mod_${i}`,a:input.clone(),b});
    input=b.clone().multiply(new Matrix4().makeTranslation(...half.clone().multiplyScalar(2).toArray()));
    result.endpoints.push(new Vector3().setFromMatrixPosition(input).toArray());
    // End rotation is the installation frame of the next module, not the tool frame.
    if(i<robot.cells.length-1) input.multiply(rotation(cell.endRotation));
  });
  result.tip=result.endpoints[robot.cells.length];return result;
}

export function benchmarkPose(fk:KinematicResult):EndEffectorTarget {
  return {position:[...fk.tip],quaternion:new Quaternion().setFromRotationMatrix(fk.parts[fk.parts.length-1].b).normalize().toArray()};
}

export function benchmarkXML(robot:BenchmarkRobot):string {
  const vec=(v:number[])=>v.join(' ');
  const quat=(v:Vec3)=>{
    const q=new Quaternion().setFromRotationMatrix(rotation(v));return vec([q.w,q.x,q.y,q.z]);
  };
  const geom=(name:string,cell:BenchmarkCell)=>{
    const p=cell.direction.map(v=>v*PACKAGE_LENGTH/4);
    const q=new Quaternion().setFromUnitVectors(new Vector3(0,0,1),new Vector3(...cell.direction));
    return `<geom name="${name}" type="cylinder" size="${PACKAGE_RADIUS} ${PACKAGE_LENGTH/4}" pos="${vec(p)}" quat="${vec([q.w,q.x,q.y,q.z])}" mass="0.5"/>`;
  };
  const chain=(i:number):string=>{
    if(i===robot.cells.length)return '<site name="tip" size="0.005"/>';
    const c=robot.cells[i], half=c.direction.map(v=>v*PACKAGE_LENGTH/2);
    return `${geom(`a_${i}`,c)}<body name="body_${i}" pos="${vec(half)}"><joint name="joint_${i}" axis="${vec(c.axis)}" range="${-robot.limit} ${robot.limit}"/>${geom(`b_${i}`,c)}<body pos="${vec(half)}" quat="${i===robot.cells.length-1?'1 0 0 0':quat(c.endRotation)}">${chain(i+1)}</body></body>`;
  };
  const bodies=['base',...Array.from({length:7},(_,i)=>`body_${i}`)];
  // Same adjacency policy as the workbench: shared and directly neighboring rigid bodies excluded.
  const exclude=bodies.slice(1).map((b,i)=>`<exclude body1="${bodies[i]}" body2="${b}"/>`).join('');
  return `<mujoco model="${robot.id}"><compiler angle="degree"/><option gravity="0 0 -9.81"/><default><joint limited="true" damping="1"/><geom contype="1" conaffinity="1" margin="0.0005"/></default><worldbody><geom name="floor" type="plane" size="2 2 .1"/><body name="base" pos="${vec(BENCHMARK_BASE)}">${chain(0)}</body></worldbody><contact>${exclude}</contact><actuator>${robot.cells.map((_,i)=>`<position joint="joint_${i}" kp="80"/>`).join('')}</actuator></mujoco>`;
}

export function stiffnessProxy(fk:KinematicResult):number {
  // Vertical TCP deflection under unit vertical force, multiplied by common joint torsional stiffness.
  return fk.joints.reduce((sum,j)=>{
    const lever=new Vector3(...j.axis).cross(new Vector3(...fk.tip).sub(new Vector3(...j.position))).z;
    return sum+lever*lever;
  },0);
}
