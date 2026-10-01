#!/usr/bin/env python3
"""Export the archived 6DOF bipedal-walking model for browser replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vertices(path: Path) -> list[list[float]]:
    result = [[float(value) for value in line.split()[1:4]] for line in path.read_text().splitlines() if line.startswith("v ")]
    if len(result) < 4:
        raise ValueError(f"{path}: OBJ has too few vertices")
    return result


def principal_inertia(matrix: list[list[float]]) -> tuple[list[float], list[float]]:
    values, axes = np.linalg.eigh(np.asarray(matrix, dtype=np.float64))
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    xyzw = Rotation.from_matrix(axes).as_quat()
    return values.tolist(), [float(xyzw[3]), *map(float, xyzw[:3])]


def multiply(a: list[float], b: list[float]) -> list[float]:
    return [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]]


def rotate(q: list[float], value: list[float]) -> list[float]:
    return multiply(multiply(q, [0.0, *value]), [q[0], -q[1], -q[2], -q[3]])[1:]


def export_display(source: Path, target: Path, root_state: list[float], default_q: list[float]) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    compiler = root.find("compiler")
    compiler.set("autolimits", "true")
    compiler.set("fusestatic", "false")
    for parent in root.iter():
        for geom in list(parent):
            if geom.tag == "geom" and (geom.get("class") or "").startswith("coliision"):
                parent.remove(geom)
    ET.SubElement(root, "option", timestep="0.0166", gravity="0 0 -9.81", integrator="implicitfast")
    actuator = ET.SubElement(root, "actuator")
    for index in range(6):
        ET.SubElement(actuator, "position", name=f"joint_{index}", joint=f"joint_{index}", kp="10",
                      forcelimited="false", ctrllimited="false")
    sensor = ET.SubElement(root, "sensor")
    for kind, name in (("framequat", "quat"), ("frameangvel", "angvel"),
                       ("framepos", "pos"), ("framelinvel", "linvel")):
        ET.SubElement(sensor, kind, name=f"rl_base_{name}", objtype="xbody", objname="body_3")
    xyzw = root_state[3:7]
    qpos = [*root_state[:3], xyzw[3], *xyzw[:3], *default_q]
    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(keyframe, "key", name="initial", qpos=" ".join(map(str, qpos)), ctrl=" ".join(map(str, default_q)))
    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")
    tree.write(target, encoding="unicode", xml_declaration=True)


def export(project: Path, capture_path: Path, output_root: Path) -> None:
    capture = json.loads(capture_path.read_text())
    body_names = capture["body_names"]
    if body_names != [f"body_{i}" for i in range(7)]:
        raise ValueError("unexpected bipedal body order")
    properties = {value["name"]: value for value in capture["asset"]["rigid_body_properties"]}
    poses = capture["clean_reset"]["body_poses"]
    bodies = []
    for name in body_names:
        prop, pose = properties[name], poses[name]
        inertia, axes = principal_inertia(prop["inertia"])
        xyzw = pose["q_xyzw"]
        bodies.append({"name": name, "mass": prop["mass"], "inertia": inertia, "com": prop["com"], "axes": axes,
                       "pos": pose["p"], "quat": [xyzw[3], *xyzw[:3]]})

    axis_frame = [math.cos(math.pi / 8), math.sin(math.pi / 8), 0.0, 0.0]
    default_q = [-.312, -.837, 2.02, -2.02, .837, .312]
    joints = []
    for index in range(6):
        parent_name, child_name = f"body_{index}", f"body_{index + 1}"
        parent_pose, child_pose = poses[parent_name], poses[child_name]
        parent_q = [parent_pose["q_xyzw"][3], *parent_pose["q_xyzw"][:3]]
        child_q = [child_pose["q_xyzw"][3], *child_pose["q_xyzw"][:3]]
        inverse_parent = [parent_q[0], -parent_q[1], -parent_q[2], -parent_q[3]]
        p1 = [0.0, 0.0, -.053]
        world_anchor = [value + offset for value, offset in zip(child_pose["p"], rotate(child_q, p1))]
        p0 = rotate(inverse_parent, [value - origin for value, origin in zip(world_anchor, parent_pose["p"])])
        q0_zero = multiply(multiply(inverse_parent, child_q), axis_frame)
        # Captured link poses already contain the reset joint coordinate. PhysX
        # joint frames describe the zero-coordinate relation, so remove that
        # rotation before assigning defaultQ to the articulation.
        angle = -default_q[index]
        q0 = multiply(q0_zero, [math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])
        joints.append({"name": f"joint_{index}", "a": parent_name, "b": child_name, "p0": p0, "p1": p1,
                       "q0": q0, "q1": axis_frame, "axis": "Z", "hinge": True,
                       "collisionEnabled": False, "limits": [-math.pi, math.pi]})

    asset_dir = project / "assets/mjcf/zbot"
    ma, mb = vertices(asset_dir / "ma.obj"), vertices(asset_dir / "mb.obj")
    middle_ma = [[-x, -y, z] for x, y, z in ma]
    shifted_mb = [[x, y, z - .106] for x, y, z in mb]
    colliders = []
    for index, name in enumerate(body_names):
        hulls = [{"vertices": ma}] if index == 0 else [{"vertices": shifted_mb}] if index == 6 else [
            {"vertices": shifted_mb}, {"vertices": middle_ma}]
        colliders.append({"bodyName": name, "hulls": hulls})

    result = {"version": 1, "key": "legacy_zbot_6dof_bipedal_walking", "rootBody": "body_0", "baseBody": "body_3",
              "bodies": bodies, "joints": joints, "hulls": {"colliders": colliders},
              "jointNames": [f"joint_{index}" for index in range(6)], "defaultQ": default_q,
              "legacyBodyGroups": [[name] for name in body_names],
              "simulation": {"staticFriction": 1.0, "dynamicFriction": 1.0, "restitution": 0.0,
                             "stiffness": 10.0, "damping": 1.0, "maxForce": 3.4028234e38, "maxVelocity": 100.0,
                             "contactOffset": .02, "restOffset": 0.0, "bounceThresholdVelocity": .2,
                             "maxDepenetrationVelocity": 10.0, "positionIterations": 4, "velocityIterations": 0,
                             "disableSelfCollision": True, "legacyObservations": True,
                             "contactObservations": True, "footNames": ["body_0", "body_6"]},
              "metadata": {"units": "metres-kilograms-seconds-radians", "quaternionOrder": "wxyz",
                           "hullCoordinateFrame": "rigid-body-local", "sourceMjcf": "assets/mjcf/zbot/ZBot_R1.xml",
                           "sourceMjcfSha256": sha256(asset_dir / "ZBot_R1.xml"), "nativeCaptureSha256": sha256(capture_path),
                           "physicsDt": .0166, "controlDt": .0166,
                           "provenance": "Archived Isaac Gym 6DOF bipedal-walking rigid properties, reset poses and MJCF frames"}}
    physx_dir, model_dir = output_root / "public/rl/physx", output_root / "public/rl/models"
    physx_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    (physx_dir / "legacy_zbot_6dof_bipedal_walking.json").write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
    export_display(asset_dir / "ZBot_R1.xml", model_dir / "legacy_zbot_6dof_bipedal_walking.xml",
                   capture["clean_reset"]["root_state"], default_q)
    print(f"exported 6DOF bipedal walking: {len(bodies)} bodies, {len(joints)} joints, {sum(len(c['hulls']) for c in colliders)} hulls")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    export(args.project, args.capture, args.output_root)


if __name__ == "__main__":
    main()
