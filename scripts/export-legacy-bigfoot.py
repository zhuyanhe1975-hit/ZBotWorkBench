#!/usr/bin/env python3
"""Export archived BigFoot 6/8DOF Isaac Gym models for browser PhysX."""

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
    return [[float(v) for v in line.split()[1:4]] for line in path.read_text().splitlines() if line.startswith("v ")]


def cylinder(radius: float, half_height: float, z: float, segments: int = 32) -> list[list[float]]:
    return [[radius * math.cos(2 * math.pi * i / segments), radius * math.sin(2 * math.pi * i / segments), z + side * half_height]
            for side in (-1, 1) for i in range(segments)]


def principal_inertia(matrix: list[list[float]]) -> tuple[list[float], list[float]]:
    values, axes = np.linalg.eigh(np.asarray(matrix, dtype=np.float64))
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    xyzw = Rotation.from_matrix(axes).as_quat()
    return values.tolist(), [float(xyzw[3]), *map(float, xyzw[:3])]


def multiply(a, b):
    return [a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3], a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
            a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1], a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]]


def rotate(q, value):
    return multiply(multiply(q, [0.0, *value]), [q[0], -q[1], -q[2], -q[3]])[1:]


def display_xml(source: Path, target: Path, root_state, default_q, base_name: str) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    root.find("compiler").set("autolimits", "true")
    root.find("compiler").set("fusestatic", "false")
    for parent in root.iter():
        for geom in list(parent):
            if geom.tag != "geom" or not (geom.get("class") or "").startswith("coliision"):
                continue
            if geom.get("type") == "cylinder":
                geom.attrib.pop("class", None)
                geom.set("contype", "0"); geom.set("conaffinity", "0"); geom.set("group", "1")
            else:
                parent.remove(geom)
    ET.SubElement(root, "option", timestep="0.0166", gravity="0 0 -9.81", integrator="implicitfast")
    actuator = ET.SubElement(root, "actuator")
    for i in range(len(default_q)):
        ET.SubElement(actuator, "position", name=f"joint_{i}", joint=f"joint_{i}", kp="10", forcelimited="false", ctrllimited="false")
    sensor = ET.SubElement(root, "sensor")
    for kind, name in (("framequat", "quat"), ("frameangvel", "angvel"), ("framepos", "pos"), ("framelinvel", "linvel")):
        ET.SubElement(sensor, kind, name=f"rl_base_{name}", objtype="xbody", objname=base_name)
    xyzw = root_state[3:7]
    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(keyframe, "key", name="initial", qpos=" ".join(map(str, [*root_state[:3], xyzw[3], *xyzw[:3], *default_q])),
                  ctrl=" ".join(map(str, default_q)))
    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")
    tree.write(target, encoding="unicode", xml_declaration=True)


def export(project: Path, capture_path: Path, key: str, output_root: Path) -> None:
    capture = json.loads(capture_path.read_text())
    body_names, joint_names = capture["body_names"], capture["dof_names"]
    count = len(joint_names)
    if len(body_names) != count + 1 or body_names != [f"body_{i}" for i in range(count + 1)]:
        raise ValueError("unexpected BigFoot serial topology")
    default_q = capture["effective_steps"][0]["pos_d"]
    poses = capture["clean_reset"]["body_poses"]
    props = {value["name"]: value for value in capture["asset"]["rigid_body_properties"]}
    bodies = []
    for name in body_names:
        inertia, axes = principal_inertia(props[name]["inertia"])
        xyzw = poses[name]["q_xyzw"]
        bodies.append({"name": name, "mass": props[name]["mass"], "inertia": inertia, "com": props[name]["com"], "axes": axes,
                       "pos": poses[name]["p"], "quat": [xyzw[3], *xyzw[:3]]})
    axis_frame = [math.cos(math.pi/8), math.sin(math.pi/8), 0.0, 0.0]
    joints = []
    for i in range(count):
        parent, child = f"body_{i}", f"body_{i+1}"
        pq, cq = poses[parent], poses[child]
        parent_q, child_q = [pq["q_xyzw"][3], *pq["q_xyzw"][:3]], [cq["q_xyzw"][3], *cq["q_xyzw"][:3]]
        inverse = [parent_q[0], -parent_q[1], -parent_q[2], -parent_q[3]]
        p1 = [0.0, 0.0, -.053]
        anchor = [v + d for v, d in zip(cq["p"], rotate(child_q, p1))]
        p0 = rotate(inverse, [v-o for v, o in zip(anchor, pq["p"])])
        q0_zero = multiply(multiply(inverse, child_q), axis_frame)
        angle = -default_q[i]
        q0 = multiply(q0_zero, [math.cos(angle/2), 0.0, 0.0, math.sin(angle/2)])
        joints.append({"name": joint_names[i], "a": parent, "b": child, "p0": p0, "p1": p1, "q0": q0, "q1": axis_frame,
                       "axis": "Z", "hinge": True, "collisionEnabled": False, "limits": [-math.pi, math.pi]})
    asset_dir = project / "assets/mjcf/zbot"
    ma, mb = vertices(asset_dir / "ma.obj"), vertices(asset_dir / "mb.obj")
    middle_ma, shifted_mb = [[-x, -y, z] for x, y, z in ma], [[x, y, z-.106] for x, y, z in mb]
    colliders = []
    for i, name in enumerate(body_names):
        if i == 0:
            hulls = [{"vertices": ma}, {"vertices": cylinder(.06, .02, -.02)}]
        elif i == count:
            hulls = [{"vertices": shifted_mb}, {"vertices": cylinder(.06, .02, .02)}]
        else:
            hulls = [{"vertices": shifted_mb}, {"vertices": middle_ma}]
        colliders.append({"bodyName": name, "hulls": hulls})
    base_name = body_names[len(body_names)//2]
    result = {"version": 1, "key": key, "rootBody": body_names[0], "baseBody": base_name, "bodies": bodies, "joints": joints,
              "hulls": {"colliders": colliders}, "jointNames": joint_names, "defaultQ": default_q,
              "legacyBodyGroups": [[name] for name in body_names],
              "simulation": {"staticFriction": 1.0, "dynamicFriction": 1.0, "restitution": 0.0, "stiffness": 10.0, "damping": 1.0,
                             "maxForce": 3.4028234e38, "maxVelocity": 100.0, "contactOffset": .02, "restOffset": 0.0,
                             "bounceThresholdVelocity": .2, "maxDepenetrationVelocity": 10.0, "positionIterations": 4,
                             "velocityIterations": 0, "disableSelfCollision": False, "legacyObservations": True,
                             "contactObservations": True, "footNames": [body_names[0], body_names[-1]]},
              "metadata": {"physicsDt": .0166, "controlDt": .0166, "sourceMjcf": f"assets/mjcf/zbot/{capture['asset']['file']}",
                           "sourceMjcfSha256": sha256(asset_dir / capture["asset"]["file"]), "nativeCaptureSha256": sha256(capture_path),
                           "provenance": "Archived run.sh BigFoot task capture"}}
    physx_dir, model_dir = output_root / "public/rl/physx", output_root / "public/rl/models"
    physx_dir.mkdir(parents=True, exist_ok=True); model_dir.mkdir(parents=True, exist_ok=True)
    (physx_dir / f"{key}.json").write_text(json.dumps(result, separators=(",", ":"), allow_nan=False)+"\n")
    display_xml(asset_dir / capture["asset"]["file"], model_dir / f"{key}.xml", capture["clean_reset"]["root_state"], default_q, base_name)
    print(f"exported {key}: {len(body_names)} bodies, {count} joints")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    export(args.project, args.capture, args.key, args.output_root)


if __name__ == "__main__":
    main()
