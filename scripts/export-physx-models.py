#!/usr/bin/env python3
"""Export static PhysX robot data; development-only, requires Pixar USD (pxr).

Reproducible offline pipeline (no Python is used by the deployed frontend):
1. Use scripts/capture-isaac-reference.py with the original training project,
   --device cpu --headless --steps 0 and a matching --task (no policy required).
   For motion evidence, use --steps 600 and the matching --checkpoint. Save
   --output <reference-dir>/zbot-isaac-<alias>.json and
   --colliders-output <reference-dir>/zbot-<alias>-hulls.json. Aliases are listed
   in MODELS below; --only selects new model keys without rewriting old assets.
   The capture helper requests the runtime's actual cooked convex hulls.
2. Run this script using Isaac Lab's Python (no SimulationApp is started here):
   isaaclab.sh -p scripts/export-physx-models.py --project <training-project> \
       --reference-dir <reference-dir>

Native reset body poses intentionally come from the reference captures, retaining
PhysX float32 initialization. Source USD supplies joint frames and principal
inertias. Each output embeds actual cooked hulls in rigid-body-local coordinates,
not a different engine's approximation. All quaternions are wxyz; units are SI.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from pxr import Usd, UsdPhysics

MODELS = (
    ('human', 'zbot_8s_human', 'Zbot-Direct-8dof-bipedal-v0', 'zbot_8s_human.usd'),
    ('quat', 'zbot_6s_new', 'Zbot-Direct-6dof-bipedal-quat-v0', 'zbot_6s_new.usd'),
    ('snake', 'zbot_8s_snake_v0', 'Zbot-Direct-8dof-snake-v0', 'zbot_8s_snake_v0.usd'),
    ('human_v1', 'zbot_8s_human_v1', 'Zbot-Direct-8dof-bipedal-v1', 'zbot_8s_human.usd'),
    ('human_v2', 'zbot_8s_human_v2', 'Zbot-Direct-8dof-bipedal-v2', 'zbot_8s_human_2.usd'),
    ('human_v3', 'zbot_8s_human_v3', 'Zbot-Direct-8dof-bipedal-v3', 'zbot_8s_human_2.usd'),
    ('bird', 'zbot_8s_bird', 'Zbot-Direct-8dof-bird-v0', 'zbot_8s_new.usd'),
    ('wheel', 'zbot_8s_wheel', 'Zbot-Direct-8dof-wheel-v0', 'zbot_8s_new.usd'),
    ('run', 'zbot_8s_run', 'Zbot-Direct-8dof-bipedal-run-v0', 'zbot_8s_human.usd'),
)


def quaternion(value):
    return [float(value.GetReal()), *map(float, value.GetImaginary())]


def sha256(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def export(project, reference_dir, output_dir, alias, key, task, source_usd):
    relative_usd = Path('assets/zbot_usd/zbot') / source_usd
    source = project / relative_usd
    reference_path = reference_dir / f'zbot-isaac-{alias}.json'
    hull_path = reference_dir / f'zbot-{alias}-hulls.json'
    reference = json.loads(reference_path.read_text())
    cooked = json.loads(hull_path.read_text())
    if reference['task'] != task or not reference['rows'] or reference['rows'][0]['step'] != 0:
        raise ValueError(f'{key}: expected the matching task reset reference')
    if not cooked.get('available') or cooked.get('coordinateFrame') != 'rigid-body-local':
        raise ValueError(f'{key}: missing or unsuccessful cooked collider capture')
    expected_joints = [f'joint{i}' for i in (range(1, 7) if alias == 'quat' else range(8))]
    if reference['jointNames'] != expected_joints:
        raise ValueError(f'{key}: unexpected policy joint order')
    stage = Usd.Stage.Open(str(source))
    rigid = {p.GetName(): p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    roots = [p.GetName() for p in rigid.values() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if roots != ['foot_0'] or set(rigid) != set(reference['bodyNames']):
        raise ValueError(f'{key}: USD and reference rigid bodies disagree')
    children = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        if joint.GetJointEnabledAttr().Get() is False:
            raise ValueError(f'{key}: disabled joints require a separate export contract')
        a = joint.GetBody0Rel().GetTargets()
        b = joint.GetBody1Rel().GetTargets()
        if len(a) != 1 or len(b) != 1:
            raise ValueError(f'{key}: unsupported world or multi-body joint')
        hinge = prim.IsA(UsdPhysics.RevoluteJoint)
        if not hinge and not prim.IsA(UsdPhysics.FixedJoint):
            raise ValueError(f'{key}: unsupported joint {prim.GetTypeName()}')
        axis = str(UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get()) if hinge else None
        record = dict(name=prim.GetName(), a=a[0].name, b=b[0].name,
                      p0=list(joint.GetLocalPos0Attr().Get()), q0=quaternion(joint.GetLocalRot0Attr().Get()),
                      p1=list(joint.GetLocalPos1Attr().Get()), q1=quaternion(joint.GetLocalRot1Attr().Get()),
                      axis=axis, hinge=hinge, collisionEnabled=bool(joint.GetCollisionEnabledAttr().Get()))
        if hinge:
            revolute = UsdPhysics.RevoluteJoint(prim)
            lo, hi = revolute.GetLowerLimitAttr().Get(), revolute.GetUpperLimitAttr().Get()
            record['limits'] = [math.radians(lo), math.radians(hi)] if math.isfinite(lo) and math.isfinite(hi) else None
        children.setdefault(record['a'], []).append(record)
    order, joints = [], []

    def visit(name):
        if name in order or name not in rigid:
            raise ValueError(f'{key}: invalid articulation tree')
        order.append(name)
        for joint in sorted(children.get(name, []), key=lambda value: value['name']):
            joints.append(joint)
            visit(joint['b'])

    visit('foot_0')
    if set(order) != set(rigid) or len(joints) != len(order) - 1:
        raise ValueError(f'{key}: disconnected articulation')
    bodies, colliders = [], []
    for name in order:
        mass = UsdPhysics.MassAPI(rigid[name])
        index = reference['bodyNames'].index(name)
        bodies.append(dict(name=name, mass=float(mass.GetMassAttr().Get()),
                           inertia=list(mass.GetDiagonalInertiaAttr().Get()),
                           com=list(mass.GetCenterOfMassAttr().Get()), axes=quaternion(mass.GetPrincipalAxesAttr().Get()),
                           pos=reference['rows'][0]['bodyPos'][index], quat=reference['rows'][0]['bodyQuat'][index]))
        hulls = [hull for collider in cooked['colliders'] if collider['bodyName'] == name
                 and collider['status'] == 'completed' for hull in collider['hulls']]
        if not hulls:
            raise ValueError(f'{key}: missing cooked collider for {name}')
        for hull in hulls:
            if not hull['vertices'] or not hull['faces'] or any(len(v) != 3 for v in hull['vertices']):
                raise ValueError(f'{key}: invalid cooked hull for {name}')
            if any(i < 0 or i >= len(hull['vertices']) for face in hull['faces'] for i in face):
                raise ValueError(f'{key}: invalid triangle indices for {name}')
        colliders.append(dict(bodyName=name, hulls=hulls))
    result = dict(version=1, key=key, rootBody='foot_0', baseBody='base',
                  bodies=bodies, joints=joints, hulls=dict(colliders=colliders),
                  jointNames=reference['jointNames'], defaultQ=reference['defaultQ'],
                  metadata=dict(units='metres-kilograms-seconds-radians', quaternionOrder='wxyz',
                                hullCoordinateFrame='rigid-body-local', sourceUsd=relative_usd.as_posix(),
                                sourceSha256=sha256(source), referenceSha256=sha256(reference_path),
                                cookedHullsSha256=sha256(hull_path), sourceTask=task,
                                physicsDt=reference['physicsDt'], controlDt=reference['controlDt'],
                                provenance='Original USD mass/joint frames, native PhysX reset poses and cooked convex hulls'))
    if alias == 'run':
        # ContactSensor follows USD body traversal, whereas articulation joints
        # follow the kinematic tree. Preserve the sensor's actual foot order.
        contact_reference_path = reference_path
        contact_reference = reference
        if 'contactBodyNames' not in contact_reference:
            contact_reference_path = reference_dir / f'zbot-isaac-{alias}-sensor.json'
            contact_reference = json.loads(contact_reference_path.read_text())
        if contact_reference['task'] != task:
            raise ValueError(f'{key}: contact order reference belongs to another task')
        foot_names = [contact_reference['contactBodyNames'][i] for i in contact_reference['feetSensorIds']]
        if len(foot_names) != 2 or set(foot_names) != {'foot_0', 'foot_1'}:
            raise ValueError(f'{key}: unexpected native contact sensor feet')
        result['simulation'] = dict(staticFriction=1.2, dynamicFriction=1.1, restitution=0,
                                    stiffness=50, damping=5, maxForce=2000, maxVelocity=1000,
                                    contactObservations=True, footNames=foot_names)
        result['metadata']['contactOrderReferenceSha256'] = sha256(contact_reference_path)
    target = output_dir / (key + '.json')
    target.write_text(json.dumps(result, separators=(',', ':'), allow_nan=False) + '\n')
    print(f'{target.name}: {len(bodies)} bodies, {len(joints)} joints, {target.stat().st_size} bytes')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project', type=Path, required=True, help='Original zbot_rl_student project')
    parser.add_argument('--reference-dir', type=Path, required=True, help='Directory of paired native reference/collider JSON captures')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'public/rl/physx')
    parser.add_argument('--only', nargs='+', choices=tuple(model[1] for model in MODELS), help='Only export selected model keys')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for alias, key, task, source_usd in MODELS:
        if args.only is None or key in args.only:
            export(args.project, args.reference_dir, args.output, alias, key, task, source_usd)


if __name__ == '__main__':
    main()
