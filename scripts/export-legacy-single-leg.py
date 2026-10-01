#!/usr/bin/env python3
"""Export the run.sh SingleLeg ZBot.xml capture as a dedicated model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def multiply(a, b):
    return [a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3], a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
            a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1], a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]]


def rotate(q, value):
    return multiply(multiply(q, [0.0, *value]), [q[0], -q[1], -q[2], -q[3]])[1:]


def vertices(path: Path):
    return [[float(v) for v in line.split()[1:4]] for line in path.read_text().splitlines() if line.startswith("v ")]


def principal(matrix):
    values, axes = np.linalg.eigh(np.asarray(matrix, dtype=np.float64))
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    xyzw = Rotation.from_matrix(axes).as_quat()
    return values.tolist(), [float(xyzw[3]), *map(float, xyzw[:3])]


def export(project: Path, capture_path: Path, output_root: Path) -> None:
    capture = json.loads(capture_path.read_text())
    names = ["root", "body_0", "body_1", "body_2", "body_3", "body_4", "body_5"]
    if capture["body_names"] != names:
        raise ValueError("unexpected SingleLeg topology")
    poses = capture["clean_reset"]["body_poses"]
    props = {value["name"]: value for value in capture["asset"]["rigid_body_properties"]}
    bodies = []
    for name in names:
        inertia, axes = principal(props[name]["inertia"])
        xyzw = poses[name]["q_xyzw"]
        bodies.append({"name": name, "mass": props[name]["mass"], "inertia": inertia, "com": props[name]["com"], "axes": axes,
                       "pos": poses[name]["p"], "quat": [xyzw[3], *xyzw[:3]]})
    axis_frame = [math.cos(math.pi/8), math.sin(math.pi/8), 0.0, 0.0]
    joints = []
    for i in range(6):
        parent, child = names[i], names[i+1]
        pq, cq = poses[parent], poses[child]
        parent_q, child_q = [pq["q_xyzw"][3], *pq["q_xyzw"][:3]], [cq["q_xyzw"][3], *cq["q_xyzw"][:3]]
        inverse = [parent_q[0], -parent_q[1], -parent_q[2], -parent_q[3]]
        p1 = [0.0, 0.0, .053]
        anchor = [v+d for v, d in zip(cq["p"], rotate(child_q, p1))]
        p0 = rotate(inverse, [v-o for v, o in zip(anchor, pq["p"])])
        q0 = multiply(multiply(inverse, child_q), axis_frame)
        joints.append({"name": f"joint_{i}", "a": parent, "b": child, "p0": p0, "p1": p1, "q0": q0, "q1": axis_frame,
                       "axis": "Z", "hinge": True, "collisionEnabled": False, "limits": [-math.pi, math.pi]})
    asset_dir = project / "assets/mjcf/zbot"
    ma, mb = vertices(asset_dir / "ma.obj"), vertices(asset_dir / "mb.obj")
    upper_ma = [[-x, -y, z+.106] for x, y, z in ma]
    colliders = [{"bodyName": name, "hulls": [{"vertices": ma}] if i == 0 else [{"vertices": mb}] if i == 6 else
                  [{"vertices": mb}, {"vertices": upper_ma}]} for i, name in enumerate(names)]
    key = "legacy_zbot_single_leg"
    result = {"version": 1, "key": key, "rootBody": "root", "baseBody": "body_2", "bodies": bodies, "joints": joints,
              "hulls": {"colliders": colliders}, "jointNames": [f"joint_{i}" for i in range(6)], "defaultQ": [0.0]*6,
              "legacyBodyGroups": [[name] for name in names],
              "simulation": {"staticFriction": 1.0, "dynamicFriction": 1.0, "restitution": 0.0, "stiffness": 10.0, "damping": 1.0,
                             "maxForce": 3.4028234e38, "maxVelocity": 100.0, "contactOffset": .02, "restOffset": 0.0,
                             "bounceThresholdVelocity": .2, "maxDepenetrationVelocity": 10.0, "positionIterations": 4,
                             "velocityIterations": 0, "disableSelfCollision": False, "legacyObservations": True},
              "metadata": {"physicsDt": .0166, "controlDt": .0166, "sourceMjcf": "assets/mjcf/zbot/ZBot.xml",
                           "sourceMjcfSha256": hashlib.sha256((asset_dir/"ZBot.xml").read_bytes()).hexdigest(),
                           "nativeCaptureSha256": hashlib.sha256(capture_path.read_bytes()).hexdigest(),
                           "provenance": "Archived run.sh SingleLeg native capture"}}
    physx_dir, model_dir = output_root/"public/rl/physx", output_root/"public/rl/models"
    physx_dir.mkdir(parents=True, exist_ok=True); model_dir.mkdir(parents=True, exist_ok=True)
    (physx_dir/f"{key}.json").write_text(json.dumps(result, separators=(",", ":"), allow_nan=False)+"\n")

    tree = ET.parse(asset_dir/"ZBot.xml")
    root = tree.getroot(); root.find("compiler").set("autolimits", "true"); root.find("compiler").set("fusestatic", "false")
    for parent in root.iter():
        for geom in list(parent):
            if geom.tag == "geom" and (geom.get("class") or "").startswith("coliision"):
                parent.remove(geom)
    ET.SubElement(root, "option", timestep="0.0166", gravity="0 0 -9.81", integrator="implicitfast")
    actuator = ET.SubElement(root, "actuator")
    for i in range(6): ET.SubElement(actuator, "position", name=f"joint_{i}", joint=f"joint_{i}", kp="10", forcelimited="false", ctrllimited="false")
    sensor = ET.SubElement(root, "sensor")
    for kind, name in (("framequat", "quat"), ("frameangvel", "angvel"), ("framepos", "pos"), ("framelinvel", "linvel")):
        ET.SubElement(sensor, kind, name=f"rl_base_{name}", objtype="xbody", objname="body_2")
    state = capture["clean_reset"]["root_state"]; xyzw = state[3:7]
    frame = ET.SubElement(root, "keyframe")
    ET.SubElement(frame, "key", name="initial", qpos=" ".join(map(str, [*state[:3], xyzw[3], *xyzw[:3], *([0.0]*6)])), ctrl="0 0 0 0 0 0")
    if hasattr(ET, "indent"): ET.indent(tree, space="  ")
    tree.write(model_dir/f"{key}.xml", encoding="unicode", xml_declaration=True)
    print(f"exported {key}: 7 bodies, 6 joints")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(); export(args.project, args.capture, args.output_root)


if __name__ == "__main__": main()
