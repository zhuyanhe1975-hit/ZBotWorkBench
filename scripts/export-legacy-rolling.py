#!/usr/bin/env python3
"""Export the old Isaac Gym rolling MJCF as dedicated browser assets.

Run with the Isaac Gym environment after producing the native asset capture:
  python scripts/export-legacy-rolling.py --project /path/to/zbot_evo_isaacgym \
    --capture /tmp/rolling-asset-capture.json
"""

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
    result = []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            result.append([float(value) for value in line.split()[1:4]])
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
    return [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]


def rotate(q: list[float], value: list[float]) -> list[float]:
    rotated = multiply(multiply(q, [0.0, *value]), [q[0], -q[1], -q[2], -q[3]])
    return rotated[1:]


def export_display(source: Path, target: Path, root_state: list[float]) -> None:
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
    world = root.find("worldbody")
    world.insert(0, ET.Element("geom", name="floor", type="plane", size="5 5 .01", rgba=".55 .58 .62 1"))
    actuator = ET.SubElement(root, "actuator")
    for index in range(6):
        ET.SubElement(actuator, "position", name=f"joint_{index}", joint=f"joint_{index}", kp="10",
                      forcelimited="false", ctrllimited="false")
    sensor = ET.SubElement(root, "sensor")
    ET.SubElement(sensor, "framequat", name="rl_base_quat", objtype="xbody", objname="body_3")
    ET.SubElement(sensor, "frameangvel", name="rl_base_angvel", objtype="xbody", objname="body_3")
    ET.SubElement(sensor, "framepos", name="rl_base_pos", objtype="xbody", objname="body_3")
    ET.SubElement(sensor, "framelinvel", name="rl_base_linvel", objtype="xbody", objname="body_3")
    keyframe = ET.SubElement(root, "keyframe")
    position = root_state[:3]
    xyzw = root_state[3:7]
    qpos = [*position, xyzw[3], xyzw[0], xyzw[1], xyzw[2], *([0.0] * 6)]
    ET.SubElement(keyframe, "key", name="initial", qpos=" ".join(map(str, qpos)), ctrl="0 0 0 0 0 0")
    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")
    tree.write(target, encoding="unicode", xml_declaration=True)


def export(project: Path, capture_path: Path, golden_path: Path, output_root: Path) -> None:
    capture = json.loads(capture_path.read_text())
    golden = json.loads(golden_path.read_text())
    state = golden["clean_reset_reference"]
    body_names = capture["asset_rigid_body_names"]
    if body_names != ["root", "body_0", "body_1", "body_2", "body_3", "body_4", "body_5"]:
        raise ValueError("unexpected rolling body order")
    properties = {value["name"]: value for value in capture["rigid_body_properties"]}
    poses = state["body_pose_pq_by_body"]
    bodies = []
    for name in body_names:
        prop, pose = properties[name], poses[name]
        inertia, axes = principal_inertia(prop["inertia"])
        xyzw = pose["q_xyzw"]
        bodies.append({"name": name, "mass": prop["mass"], "inertia": inertia, "com": prop["com"], "axes": axes,
                       "pos": pose["p"], "quat": [xyzw[3], xyzw[0], xyzw[1], xyzw[2]]})

    axis_frame = [math.cos(math.pi / 8), math.sin(math.pi / 8), 0.0, 0.0]
    joints = []
    for index in range(6):
        parent_name = "root" if index == 0 else f"body_{index - 1}"
        child_name = f"body_{index}"
        parent_pose, child_pose = poses[parent_name], poses[child_name]
        parent_xyzw, child_xyzw = parent_pose["q_xyzw"], child_pose["q_xyzw"]
        parent_q = [parent_xyzw[3], *parent_xyzw[:3]]
        child_q = [child_xyzw[3], *child_xyzw[:3]]
        inverse_parent = [parent_q[0], -parent_q[1], -parent_q[2], -parent_q[3]]
        p1 = [0.0, 0.0, .053]
        world_anchor = [value + offset for value, offset in zip(child_pose["p"], rotate(child_q, p1))]
        p0 = rotate(inverse_parent, [value - origin for value, origin in zip(world_anchor, parent_pose["p"])])
        q0 = multiply(multiply(inverse_parent, child_q), axis_frame)
        joints.append({"name": f"joint_{index}", "a": "root" if index == 0 else f"body_{index - 1}", "b": f"body_{index}",
                       "p0": p0, "p1": p1, "q0": q0, "q1": axis_frame, "axis": "Z", "hinge": True,
                       "collisionEnabled": False, "limits": [-math.pi, math.pi]})

    asset_dir = project / "assets/mjcf/zbot"
    ma, mb = vertices(asset_dir / "ma.obj"), vertices(asset_dir / "mb.obj")
    ma_output = [[-x, -y, z + .106] for x, y, z in ma]
    colliders = []
    for index, name in enumerate(body_names):
        if index == 0:
            hulls = [{"vertices": ma}]
        elif index == len(body_names) - 1:
            hulls = [{"vertices": mb}]
        else:
            hulls = [{"vertices": mb}, {"vertices": ma_output}]
        colliders.append({"bodyName": name, "hulls": hulls})

    result = {
        "version": 1, "key": "legacy_zbot_rolling", "rootBody": "root", "baseBody": "body_3",
        "bodies": bodies, "joints": joints, "hulls": {"colliders": colliders},
        "jointNames": [f"joint_{index}" for index in range(6)], "defaultQ": [0.0] * 6,
        "legacyBodyGroups": [[name] for name in body_names],
        "simulation": {"staticFriction": 1.0, "dynamicFriction": 1.0, "restitution": 0.0,
                       "stiffness": 10.0, "damping": 1.0, "maxForce": 3.4028234e38, "maxVelocity": 100.0,
                       "contactOffset": .02, "restOffset": 0.0, "bounceThresholdVelocity": .2,
                       "maxDepenetrationVelocity": 10.0, "positionIterations": 4, "velocityIterations": 0,
                       "disableSelfCollision": False, "legacyObservations": True},
        "metadata": {"units": "metres-kilograms-seconds-radians", "quaternionOrder": "wxyz",
                     "hullCoordinateFrame": "rigid-body-local", "sourceMjcf": "assets/mjcf/zbot/ZBot.xml",
                     "sourceMjcfSha256": sha256(asset_dir / "ZBot.xml"), "nativeCaptureSha256": sha256(capture_path),
                     "nativeGoldenSha256": sha256(golden_path),
                     "physicsDt": .0166, "controlDt": .0166,
                     "provenance": "Legacy Isaac Gym ZBotRolling rigid properties and reset poses; MJCF joint/collider frames"},
    }
    physx_dir = output_root / "public/rl/physx"
    model_dir = output_root / "public/rl/models"
    physx_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    (physx_dir / "legacy_zbot_rolling.json").write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
    export_display(asset_dir / "ZBot.xml", model_dir / "legacy_zbot_rolling.xml", state["root_state_13"])
    print(f"exported rolling: {len(bodies)} bodies, {len(joints)} joints, {sum(len(c['hulls']) for c in colliders)} hulls")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    export(args.project, args.capture, args.golden, args.output_root)


if __name__ == "__main__":
    main()
