"""Export one actual mjlab world and its ordinary MLP for standalone replay."""
import base64
import hashlib
import io
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


def export_bundle(env, checkpoint_path, job_dir, card, versions):
    import mujoco
    import numpy as np
    import torch
    try:
        from zbot_rl_mjlab.task_card import JOINT_NAMES, TASK_ID
    except ModuleNotFoundError:
        # The vendored upstream task keeps its contract in robot.py rather
        # than the legacy WorkBench task_card module.
        from zbot_rl_mjlab.robot import JOINT_NAMES
        TASK_ID = "Mjlab-Zbot-6dof-Walking"
    from zbot_rl_mjlab.robot import ASSET_PATH

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint["actor_state_dict"]
    model_class = ((checkpoint.get("infos") or {}).get("walking_config", {}).get("ppo", {}).get("actor", {}) or {}).get("class_name", "MLPModel")
    input_size = int(state["mlp.0.weight"].shape[1])
    feature_policy = model_class == "zbot_rl_mjlab.models:QuatFeatureMLP" or input_size == 30
    if model_class not in ("MLPModel", "zbot_rl_mjlab.models:QuatFeatureMLP") or input_size not in (26, 30, 31):
        raise ValueError("Replay export requires an allowlisted MLP observation contract")
    # Do not ship optimizer state or arbitrary training metadata to the browser.
    portable = io.BytesIO()
    torch.save({"actor_state_dict": {k: v.detach().cpu() for k, v in state.items()}}, portable)

    directory = Path(job_dir) / "model"
    env.scene.write(directory)
    # File-backed MjSpecs need not populate spec.assets; mjlab's generic writer
    # then writes the XML only. Supply the fixed task's packaged mesh files.
    for mesh in ET.parse(directory / "scene.xml").getroot().findall("./asset/mesh[@file]"):
        relative = Path(mesh.get("file"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe mesh path in exported task")
        target = directory / "assets" / relative
        if not target.exists():
            source = ASSET_PATH.parent / "meshes" / relative.name
            if not source.is_file():
                raise ValueError(f"Missing packaged task mesh: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    native = env.sim.mj_model
    compiled = directory / "compiled.xml"
    # MjSpec compilation does not initialize MuJoCo's legacy XML writer. Load
    # this exact exported specification first, then copy actual runtime options
    # and actuator parameters back from the trained model.
    mujoco.MjModel.from_xml_path(str(directory / "scene.xml"))
    mujoco.mj_saveLastXML(str(compiled), native)
    xml = ET.parse(compiled).getroot()
    # MuJoCo's XML writer rounds scalars to six significant figures. Preserve
    # the control clock exactly rather than changing 1/60 to 0.0166667.
    option = xml.find("option")
    if option is None:
        option = ET.SubElement(xml, "option")
    option.set("timestep", repr(float(native.opt.timestep)))
    if xml.findall(".//include"):
        raise ValueError("Replay export must be a single MJCF file")
    compiler = xml.find("compiler")
    if compiler is not None:
        for attribute in ("meshdir", "texturedir", "assetdir"):
            compiler.attrib.pop(attribute, None)
    assets = {}
    files = [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() == ".obj"]
    for element in xml.findall("./asset/*[@file]"):
        source = element.get("file")
        matches = [p for p in files if p.name == Path(source).name]
        if len(matches) != 1 or element.tag != "mesh":
            raise ValueError(f"Unsupported or ambiguous replay asset: {source}")
        text = matches[0].read_text(encoding="utf8")
        filename = "training_" + hashlib.sha256(text.encode()).hexdigest()[:20] + ".obj"
        assets[filename] = text
        element.set("file", filename)

    def joint_name(name):
        candidates = [mujoco.mj_id2name(native, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(native.njnt)]
        matches = [candidate for candidate in candidates if candidate == name or candidate.endswith("/" + name)]
        if len(matches) != 1:
            raise ValueError(f"Cannot map policy joint {name}")
        return matches[0]

    joints = [joint_name(name) for name in JOINT_NAMES]
    body_names = [mujoco.mj_id2name(native, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, native.nbody)]
    bases = [name for name in body_names if name == "base" or name.endswith("/base")]
    if len(bases) != 1:
        raise ValueError("Replay model must have one base body")
    sensor = xml.find("sensor")
    if sensor is None:
        sensor = ET.SubElement(xml, "sensor")
    for kind, name in (("framequat", "quat"), ("frameangvel", "angvel"), ("framepos", "pos"), ("framelinvel", "linvel")):
        ET.SubElement(sensor, kind, name="rl_base_" + name, objtype="xbody", objname=bases[0])
    for geom in xml.findall("./worldbody//geom"):
        if geom.get("type") == "plane":
            geom.set("group", "0")
        elif geom.get("contype", "1") == "0" and geom.get("conaffinity", "1") == "0":
            geom.set("group", "1")
        else:
            geom.set("group", "4")

    env.reset(seed=card["seed"])
    action = env.action_manager.get_term("joint_position") if "joint_position" in env.action_manager._terms else env.action_manager.get_term("joint_pos")
    lower, upper = card["task_card"]["joint_speed_range"]
    speed = max(lower, min(1.0, upper))
    if hasattr(action, "speed"):
        action.speed.fill_(speed)
    defaults = env.scene["robot"].data.default_joint_pos[0].detach().cpu().tolist()
    qpos = env.sim.data.qpos[0].detach().cpu().numpy()
    ctrl = np.zeros(native.nu)
    for name, value in zip(joints, defaults):
        index = mujoco.mj_name2id(native, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuators = np.flatnonzero(native.actuator_trnid[:, 0] == index)
        if len(actuators) != 1:
            raise ValueError(f"Cannot map position actuator {name}")
        ctrl[actuators[0]] = value
    for keyframes in xml.findall("keyframe"):
        xml.remove(keyframes)
    keyframe = ET.SubElement(xml, "keyframe")
    numbers = lambda values: " ".join(format(float(v), ".17g") for v in values)
    ET.SubElement(keyframe, "key", name="initial", qpos=numbers(qpos), ctrl=numbers(ctrl))
    text = ET.tostring(xml, encoding="unicode")
    checked = mujoco.MjModel.from_xml_string(text, {name: data.encode() for name, data in assets.items()})
    if checked.nu != 6 or abs(checked.opt.timestep - env.physics_dt) > 1e-10:
        raise ValueError("Exported scene differs from the training timestep or actuator count")
    bundle = {
        "schemaVersion": 1, "source": "mjlab", "taskId": card.get("taskId", TASK_ID),
        "name": "ZBot mjlab " + Path(checkpoint_path).stem,
        "xml": text, "assets": assets,
        "checkpointBase64": base64.b64encode(portable.getvalue()).decode("ascii"),
        "profile": {"jointNames": joints, "defaultAngles": defaults, "observation": "mjlab-zbot" if input_size == 31 else "quaternion",
                    "inputSize": input_size, "physicsDt": env.physics_dt, "controlDt": env.step_dt,
                    "jointSpeedLimit": speed, **({"policyFeatures": "quat-gravity-heading-v1"} if feature_policy else {})},
        "versions": versions,
    }
    destination = Path(job_dir) / "bundle.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(bundle, separators=(",", ":"), allow_nan=False))
    temporary.replace(destination)
    return destination
