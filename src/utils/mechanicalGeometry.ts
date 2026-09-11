import { Vector3 } from 'three';

/** Concept joint: fixed motor barrel + rotating output fork, all in the module frame.
 * Neutral docking planes z=0/.106; hinge passes through z=.053.
 * Every motor is one Ø48 × 64 mm cylinder; neutral assembly fits Ø100 × 106 mm.
 * Cylinders use fromto so their longitudinal axis is the physical shaft axis.
 */
export function mechanicalGeoms(axis: number[], part: 'a' | 'b', index: number): string[] {
  const a = new Vector3(...axis as [number,number,number]).normalize();
  const h = new Vector3(0,0,.053);
  const axial = Math.abs(a.z) > .999;
  const shapes: string[] = [];
  const cylinder = (p: Vector3, q: Vector3, r: number) => shapes.push(`type="cylinder" fromto="${p.toArray().join(' ')} ${q.toArray().join(' ')}" size="${r}"`);
  const z = (v: number) => new Vector3(0,0,v);
  if (axial) {
    if (part === 'a') {
      cylinder(z(0),z(.008),.04);
      cylinder(z(.008),z(.021),.015);
      cylinder(h.clone().addScaledVector(a,-.032),h.clone().addScaledVector(a,.032),.024);
    } else {
      cylinder(z(.085),z(.098),.015);
      cylinder(z(.098),z(.106),.04);
    }
  } else if (part === 'a') {
    cylinder(z(0),z(.008),.04);
    cylinder(z(.008),z(.035),.015);
    cylinder(h.clone().addScaledVector(a,-.032),h.clone().addScaledVector(a,.032),.024);
  } else {
    for (const s of [-1,1]) {
      cylinder(h.clone().addScaledVector(a,s*.033),h.clone().addScaledVector(a,s*.045),.019);
      const elbow = z(.091).addScaledVector(a,s*.04);
      cylinder(h.clone().addScaledVector(a,s*.04),elbow,.006);
      cylinder(elbow,z(.100),.006);
    }
    cylinder(z(.098),z(.106),.04);
  }
  return shapes.flatMap((shape,j)=>[
    `<geom name="visual_${part}_${index}_${j}" class="visual_${part}" ${shape}/>`,
    `<geom class="coliision_${part}" ${shape} mass="${.5/shapes.length}"/>`,
  ]);
}
