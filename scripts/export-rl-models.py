#!/usr/bin/env python3
"""Offline USD → self-contained browser MJCF assets (requires Pixar USD's pxr).

Run with Isaac Lab's CPU Python, without launching SimulationApp:
  TERM=xterm /home/yhzhu/isaaclab/isaaclab.sh -p scripts/export-rl-models.py

USD meshes, authored mass/inertia, joint frames and limits are retained. Contacts
use MuJoCo convex hulls matching the USD convexHull approximation, but PhysX and
MuJoCo solvers/implicit drives differ: this is a cross-engine policy preview.
Physics uses 600 Hz subdivision for numerical stability; policy remains 30 Hz
(training used 60 Hz physics).
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from pxr import Gf, Usd, UsdGeom, UsdPhysics

PROFILES = {
    'zbot_6s_new': {
        'task': 'Zbot-Direct-6dof-bipedal-quat-v0',
        'rootPosition': [0, -0.06, 0], 'rootQuaternion': [1, 0, 0, 0],
        'initialQ': [0.312, 0.837, -2.02, 2.02, -0.837, -0.312],
    },
    'zbot_8s_human': {
        'task': 'Zbot-Direct-8dof-bipedal-v0',
        'rootPosition': [0, -0.06, 0], 'rootQuaternion': [1, 0, 0, 0],
        'initialQ': [0, 0.312, 0.837, -2.02, 2.02, -0.837, -0.312, 0],
    },
    'zbot_8s_snake_v0': {
        'task': 'Zbot-Direct-8dof-snake-v0',
        'rootPosition': [0, 0, 0.053], 'rootQuaternion': [2 ** -0.5, 0, -(2 ** -0.5), 0],
        'initialQ': [0] * 8,
    },
}
PROFILES.update({
    'zbot_8s_human_v1': {
        'task': 'Zbot-Direct-8dof-bipedal-v1', 'sourceUsd': 'zbot_8s_human.usd',
        'rootPosition': [0, -0.06, 0], 'rootQuaternion': [1, 0, 0, 0],
        'initialQ': [-.08, 0, math.radians(25), math.radians(-125), math.radians(125), math.radians(-25), 0, .08],
    },
    'zbot_8s_human_v2': {
        'task': 'Zbot-Direct-8dof-bipedal-v2', 'sourceUsd': 'zbot_8s_human_2.usd',
        'rootPosition': [0, -0.06, 0], 'rootQuaternion': [1, 0, 0, 0],
        'initialQ': list(map(math.radians, [0, 10, -10, 140, -140, 10, -10, 0])),
    },
    'zbot_8s_human_v3': {
        'task': 'Zbot-Direct-8dof-bipedal-v3', 'sourceUsd': 'zbot_8s_human_2.usd',
        'rootPosition': [0, -0.06, 0], 'rootQuaternion': [1, 0, 0, 0],
        'initialQ': list(map(math.radians, [15, 10, 0, 120, -120, 0, -10, -15])),
    },
    'zbot_8s_bird': {
        **PROFILES['zbot_8s_human'], 'task': 'Zbot-Direct-8dof-bird-v0', 'sourceUsd': 'zbot_8s_new.usd',
    },
    'zbot_8s_wheel': {
        'task': 'Zbot-Direct-8dof-wheel-v0', 'sourceUsd': 'zbot_8s_new.usd',
        'rootPosition': [0, 0, .053], 'rootQuaternion': [2 ** -.5, 2 ** -.5, 0, 0],
        'initialQ': [1.2, -1.2, 1.8, -1.2, 1.8, -1.2, 1.2, -1.2],
    },
    'zbot_8s_run': {
        **PROFILES['zbot_8s_human'], 'task': 'Zbot-Direct-8dof-bipedal-run-v0', 'sourceUsd': 'zbot_8s_human.usd',
        'staticFriction': 1.2, 'dynamicFriction': 1.1,
    },
})


def numbers(values):
    return ' '.join(format(float(value), '.10g') for value in values)


def quat(value):
    return Gf.Quatd(float(value.GetReal()), Gf.Vec3d(value.GetImaginary())).GetNormalized()


def quat_values(value):
    return [value.GetReal(), *value.GetImaginary()]


def export(source, destination, key, profile):
    stage = Usd.Stage.Open(str(source))
    if UsdGeom.GetStageMetersPerUnit(stage) != 1:
        raise ValueError('Only metre-based USD stages are supported')
    bodies = {str(p.GetPath()): p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    roots = [p for p in bodies.values() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if len(roots) != 1:
        raise ValueError('Expected exactly one rigid articulation root')
    root = roots[0]
    children = {}
    joints = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        if joint.GetJointEnabledAttr().Get() is False:
            continue
        a, b = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if len(a) != 1 or len(b) != 1 or str(a[0]) not in bodies or str(b[0]) not in bodies:
            raise ValueError(f'Unsupported joint {prim.GetPath()}')
        children.setdefault(str(a[0]), []).append((str(b[0]), joint))
        if prim.IsA(UsdPhysics.RevoluteJoint):
            joints.append(prim.GetName())
        elif not prim.IsA(UsdPhysics.FixedJoint):
            raise ValueError(f'Unsupported joint type {prim.GetTypeName()}')
    joint_names = sorted(joints, key=lambda name: int(name.removeprefix('joint')))
    expected = range(1, 7) if len(profile['initialQ']) == 6 else range(8)
    if joint_names != [f'joint{i}' for i in expected]:
        raise ValueError(f'Unexpected joint names {joint_names}')
    xml = ET.Element('mujoco', model=key)
    ET.SubElement(xml, 'compiler', angle='radian', autolimits='true', fusestatic='false')
    ET.SubElement(xml, 'option', timestep=numbers([1 / 600]), gravity='0 0 -9.81', integrator='implicitfast', iterations='100')
    defaults = ET.SubElement(xml, 'default')
    ET.SubElement(defaults, 'geom', friction=f"{profile.get('dynamicFriction', 1)} 0.005 0.0001", condim='3')
    ET.SubElement(defaults, 'joint', damping='5', armature='0')
    assets = ET.SubElement(xml, 'asset')
    world = ET.SubElement(xml, 'worldbody')
    ET.SubElement(world, 'light', pos='0 -2 3', dir='0 0 -1')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='10 10 0.1', rgba='0.16 0.19 0.24 1', group='0')
    contacts = ET.SubElement(xml, 'contact')
    mesh_cache = {}
    visited = set()
    traversal_joints = []
    total_mass = 0

    def add_mesh(prim, rigid, body_element):
        mesh = UsdGeom.Mesh(prim)
        # Bake only mesh-to-rigid-body transforms, retaining each body frame.
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0) * UsdGeom.Xformable(rigid).ComputeLocalToWorldTransform(0).GetInverse()
        points = [matrix.Transform(Gf.Vec3d(point)) for point in mesh.GetPointsAttr().Get()]
        vertices = numbers(component for point in points for component in point)
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        triangles, offset = [], 0
        left_handed = mesh.GetOrientationAttr().Get() == 'leftHanded'
        for count in counts:
            face = indices[offset:offset + count]
            for i in range(1, count - 1):
                triangle = [face[0], face[i], face[i + 1]]
                triangles.extend(reversed(triangle) if left_handed else triangle)
            offset += count
        faces = ' '.join(str(index) for index in triangles)
        digest = hashlib.sha256((vertices + '/' + faces).encode()).hexdigest()
        if digest not in mesh_cache:
            name = 'mesh_' + digest[:12]
            ET.SubElement(assets, 'mesh', name=name, vertex=vertices, face=faces)
            mesh_cache[digest] = name
        collision = prim.HasAPI(UsdPhysics.CollisionAPI)
        if collision and prim.GetAttribute('physics:approximation').Get() not in (None, 'convexHull'):
            raise ValueError(f'Unsupported collision approximation: {prim.GetPath()}')
        color = '0.26 0.52 0.80 1' if rigid.GetName().startswith('b') else '0.76 0.80 0.85 1'
        ET.SubElement(body_element, 'geom', name=rigid.GetName() + '_' + prim.GetName(), type='mesh', mesh=mesh_cache[digest],
                      group='4' if collision else '1', contype='1' if collision else '0',
                      conaffinity='1' if collision else '0', density='0', rgba='0.7 0.3 0.2 0.2' if collision else color)

    def add_body(body_path, parent, incoming=None):
        nonlocal total_mass
        if body_path in visited:
            raise ValueError('USD articulation is not a tree')
        visited.add(body_path)
        prim = bodies[body_path]
        if incoming is None:
            position = profile['rootPosition']
            rotation = profile['rootQuaternion']
        else:
            q0, q1 = quat(incoming.GetLocalRot0Attr().Get()), quat(incoming.GetLocalRot1Attr().Get())
            q = q0 * q1.GetInverse()
            position = Gf.Vec3d(incoming.GetLocalPos0Attr().Get()) - q.Transform(Gf.Vec3d(incoming.GetLocalPos1Attr().Get()))
            rotation = quat_values(q)
        element = ET.SubElement(parent, 'body', name=prim.GetName(), pos=numbers(position), quat=numbers(rotation))
        mass = UsdPhysics.MassAPI(prim)
        mass_value = mass.GetMassAttr().Get()
        inertia = mass.GetDiagonalInertiaAttr().Get()
        if not mass_value or not inertia or min(inertia) <= 0:
            raise ValueError(f'Missing authored inertia for {body_path}')
        total_mass += mass_value
        ET.SubElement(element, 'inertial', mass=numbers([mass_value]), pos=numbers(mass.GetCenterOfMassAttr().Get()),
                      diaginertia=numbers(inertia), quat=numbers(quat_values(quat(mass.GetPrincipalAxesAttr().Get()))))
        if incoming is None:
            ET.SubElement(element, 'freejoint', name='root')
        elif incoming.GetPrim().IsA(UsdPhysics.RevoluteJoint):
            revolute = UsdPhysics.RevoluteJoint(incoming.GetPrim())
            axis = {'X': Gf.Vec3d(1, 0, 0), 'Y': Gf.Vec3d(0, 1, 0), 'Z': Gf.Vec3d(0, 0, 1)}[revolute.GetAxisAttr().Get()]
            axis = quat(incoming.GetLocalRot1Attr().Get()).Transform(axis)
            attrs = dict(name=incoming.GetPrim().GetName(), type='hinge', pos=numbers(incoming.GetLocalPos1Attr().Get()), axis=numbers(axis))
            lo, hi = revolute.GetLowerLimitAttr().Get(), revolute.GetUpperLimitAttr().Get()
            if math.isfinite(lo) and math.isfinite(hi):
                attrs['range'] = numbers([math.radians(lo), math.radians(hi)])
            else:
                attrs['limited'] = 'false'
            ET.SubElement(element, 'joint', **attrs)
            traversal_joints.append(attrs['name'])
        for mesh in Usd.PrimRange(prim):
            if mesh.IsA(UsdGeom.Mesh):
                add_mesh(mesh, prim, element)
        for child_path, joint in sorted(children.get(body_path, []), key=lambda item: item[0]):
            # USD joints explicitly disable adjacent-body contact.
            if not joint.GetCollisionEnabledAttr().Get():
                ET.SubElement(contacts, 'exclude', body1=prim.GetName(), body2=bodies[child_path].GetName())
            add_body(child_path, element, joint)

    add_body(str(root.GetPath()), world)
    if visited != set(bodies):
        raise ValueError('Disconnected rigid bodies in USD articulation')
    actuators = ET.SubElement(xml, 'actuator')
    for joint in joint_names:
        ET.SubElement(actuators, 'position', name=joint, joint=joint, kp='50', forcelimited='true', forcerange='-2000 2000')
    sensors = ET.SubElement(xml, 'sensor')
    for kind, name in [('framequat', 'quat'), ('frameangvel', 'angvel'), ('framepos', 'pos'), ('framelinvel', 'linvel')]:
        ET.SubElement(sensors, kind, name='rl_base_' + name, objtype='xbody', objname='base')
    initial = dict(zip(joint_names, profile['initialQ']))
    keyframes = ET.SubElement(xml, 'keyframe')
    ET.SubElement(keyframes, 'key', name='initial', qpos=numbers(profile['rootPosition'] + profile['rootQuaternion'] + [initial[name] for name in traversal_joints]), ctrl=numbers(profile['initialQ']))
    ET.indent(xml)
    output = destination / (key + '.xml')
    ET.ElementTree(xml).write(output, encoding='utf-8', xml_declaration=True)
    return {**profile, 'key': key, 'url': '/rl/models/' + output.name, 'jointNames': joint_names,
            'qposJointNames': traversal_joints, 'rootBody': root.GetName(), 'baseBody': 'base',
            'sourceUsd': source.name, 'sourceSha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'physicsDt': 1 / 600, 'trainingPhysicsDt': 1 / 60, 'controlDt': 1 / 30,
            'rigidBodyCount': len(bodies), 'meshCount': len(mesh_cache), 'massKg': total_mass,
            'fidelity': 'USD geometry, joint frames, mass and principal inertia; MuJoCo convex contacts and implicit PD approximate PhysX dynamics.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path('/home/yhzhu/myWorks_vips/zbot_rl_student'))
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'public/rl/models')
    parser.add_argument('--only', nargs='+', choices=tuple(PROFILES), help='Export selected model keys, preserving other assets')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    models = [export(args.project / 'assets/zbot_usd/zbot' / profile.get('sourceUsd', key + '.usd'), args.output, key, profile)
              for key, profile in PROFILES.items() if args.only is None or key in args.only]
    manifest = args.output / 'manifest.json'
    previous = json.loads(manifest.read_text())['models'] if args.only and manifest.exists() else []
    replaced = {model['key'] for model in models}
    combined = [model for model in previous if model['key'] not in replaced] + models
    manifest.write_text(json.dumps({'version': 1, 'models': combined}, indent=2) + '\n')
    for model in models:
        print(model['key'], model['rigidBodyCount'], 'bodies;', model['meshCount'], 'unique meshes;', model['massKg'], 'kg')


if __name__ == '__main__':
    main()
