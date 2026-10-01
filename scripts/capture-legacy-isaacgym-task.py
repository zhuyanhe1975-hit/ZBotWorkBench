#!/usr/bin/env python3
"""Capture one archived Isaac Gym task/checkpoint for browser replay alignment.

Run this script with the archived project's Isaac Gym Python environment. The
source project is read-only; only --output is written.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import types


def plain(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def vec(value):
    return [float(value.x), float(value.y), float(value.z)]


def quat(value):
    return [float(value.x), float(value.y), float(value.z), float(value.w)]


def matrix(value):
    # Isaac Gym Mat33 stores three column vectors.
    columns = [vec(value.x), vec(value.y), vec(value.z)]
    return [[columns[column][row] for column in range(3)] for row in range(3)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--task", default="ZBotBipedalWalking_20250527")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--joint-speed", type=float)
    parser.add_argument("--camera-output", type=Path)
    args = parser.parse_args()
    if args.steps < 1 or args.num_envs < 1 or args.joint_speed is not None and (not math.isfinite(args.joint_speed) or args.joint_speed <= 0):
        raise ValueError("invalid capture parameters")

    project = args.project.resolve()
    output_path = args.output.resolve()
    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else project / args.checkpoint
    os.chdir(project)
    sys.path.insert(0, str(project))
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_ext_zbot_capture")

    import isaacgym  # noqa: F401 - must precede torch
    import torch
    import torch.nn.functional as functional
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from tasks import isaacgym_task_map
    from utils.reformat import omegaconf_to_dict
    from utils.utils import set_seed

    # Match run.sh: it fixes the seed but does not enable PyTorch's deterministic
    # CUDA indexing path (which is broken for the archived scalar assignments).
    set_seed(42, torch_deterministic=False)
    for name, resolver in {
        "eq": lambda x, y: x.lower() == y.lower(),
        "contains": lambda x, y: x.lower() in y.lower(),
        "if": lambda pred, a, b: a if pred else b,
        "resolve_default": lambda default, value: default if value == "" else value,
    }.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver)
    with initialize_config_dir(version_base="1.1", config_dir=str(project / "cfg")):
        cfg = compose(config_name="config", overrides=[
            "task=ZBot", f"num_envs={args.num_envs}", "sim_device=cuda:0", "rl_device=cuda:0",
            "pipeline=gpu", "headless=True", "graphics_device_id=0",
        ])
    task_config = omegaconf_to_dict(cfg.task)
    task_config["name"] = args.task
    if args.camera_output:
        task_config["env"]["enableCameraSensors"] = True
    task = isaacgym_task_map[args.task](task_config, "cuda:0", "cuda:0", 0, True, False, False)
    if args.joint_speed is not None and hasattr(task, "joint_speed_limit"):
        task.joint_speed_limit.fill_(args.joint_speed)
    if hasattr(task, "feet_step_length"):
        # Torch 1.13's CUDA scalar advanced-index assignment fails for a
        # one-environment capture. This is operation-for-operation equivalent
        # to the archived task reset and does not modify the source project.
        def reset_idx_compatible(self, env_ids):
            self.root_states[env_ids] = self.initial_root_states[env_ids]
            self.root_states[env_ids, 3:7] = torch.tensor([0, 0, 0, 1], device=self.device, dtype=self.root_states.dtype)
            self.root_states[env_ids, 0] = .06
            self.root_states[env_ids, 2] = 0
            self.contact_force_z_sum.index_fill_(0, env_ids, 0)
            self.pos_d[env_ids] = self.initial_dof_pos[env_ids]
            self.feet_step_length.index_fill_(0, env_ids, 0)
            self.heading_sum.index_fill_(0, env_ids, 0)
            self.feet_step_frequency.index_fill_(0, env_ids, 5)
            self.dof_pos[env_ids] = self.initial_dof_pos[env_ids]
            self.dof_vel.index_fill_(0, env_ids, 0)
            actor_indices = self.all_actor_indices[env_ids, 0].flatten().to(dtype=torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, self.root_tensor, isaacgym.gymtorch.unwrap_tensor(actor_indices), len(actor_indices))
            self.gym.set_dof_state_tensor_indexed(
                self.sim, self.dof_tensor, isaacgym.gymtorch.unwrap_tensor(actor_indices), len(actor_indices))
            self.dead_count.index_fill_(0, env_ids, 0)
            self.progress_buf.index_fill_(0, env_ids, 0)
            self.reset_buf.index_fill_(0, env_ids, 0)
            self.sim_count.index_fill_(0, env_ids, 0)

        task.reset_idx = types.MethodType(reset_idx_compatible, task)

    checkpoint = torch.load(checkpoint_path, map_location="cuda:0")
    state = checkpoint["model"]
    mean = state["running_mean_std.running_mean"]
    variance = state["running_mean_std.running_var"]
    weights = [state[f"a2c_network.actor_mlp.{index}.weight"] for index in (0, 2, 4)]
    biases = [state[f"a2c_network.actor_mlp.{index}.bias"] for index in (0, 2, 4)]
    mu_weight, mu_bias = state["a2c_network.mu.weight"], state["a2c_network.mu.bias"]

    def infer(observation):
        dtype, device = observation.dtype, observation.device
        value = torch.clamp((observation - mean.to(device=device, dtype=dtype))
                            / torch.sqrt(variance.to(device=device, dtype=dtype) + 1e-5), -5, 5)
        for weight, bias in zip(weights, biases):
            value = functional.elu(functional.linear(value, weight.to(device=device, dtype=dtype), bias.to(device=device, dtype=dtype)))
        return functional.linear(value, mu_weight.to(device=device, dtype=dtype), mu_bias.to(device=device, dtype=dtype))

    gym = task.gym
    env, actor = task.envs[0], task.zbot_handles[0]
    camera = None
    camera_output = args.camera_output.resolve() if args.camera_output else None
    if camera_output:
        from isaacgym import gymapi
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 1280
        camera_properties.height = 960
        camera = gym.create_camera_sensor(env, camera_properties)
        gym.set_camera_location(camera, env, gymapi.Vec3(1.0, -1.0, .75), gymapi.Vec3(0.0, 0.0, .28))
    body_names = list(gym.get_actor_rigid_body_names(env, actor))
    dof_names = list(gym.get_actor_dof_names(env, actor))

    rigid_properties = []
    for index, prop in enumerate(gym.get_actor_rigid_body_properties(env, actor)):
        com = prop.com.p if hasattr(prop.com, "p") else prop.com
        rigid_properties.append({"index": index, "name": body_names[index], "mass": float(prop.mass),
                                 "com": vec(com), "inertia": matrix(prop.inertia)})
    dof_array = gym.get_actor_dof_properties(env, actor)
    dof_properties = [{"index": i, "name": dof_names[i], **{key: plain(dof_array[key][i]) for key in
        ("driveMode", "hasLimits", "lower", "upper", "stiffness", "damping", "velocity", "effort", "friction", "armature")}}
        for i in range(task.num_dof)]
    shape_properties = gym.get_actor_rigid_shape_properties(env, actor)
    shape_properties = [{"index": i, **{key: float(getattr(prop, key)) for key in
        ("friction", "rolling_friction", "torsion_friction", "restitution", "compliance", "contact_offset", "rest_offset")},
        "filter": int(prop.filter)} for i, prop in enumerate(shape_properties)]

    initial = task.reset()["obs"].clone()
    rows = []
    observation = initial
    for step in range(args.steps):
        before = observation.clone()
        mu = infer(before)
        actions = torch.clamp(mu, -1, 1)
        observation, reward, reset, _ = task.step(actions)
        observation = observation["obs"]
        rows.append({"step": step, "obs_before": plain(before[0]), "mu": plain(mu[0]), "actions": plain(actions[0]),
                     "obs_after": plain(observation[0]), "pos_d": plain(task.pos_d[0]),
                     "dof_pos": plain(task.dof_pos[0]), "dof_vel": plain(task.dof_vel[0]),
                     "body_state_3": plain(task.body_states[0, 3]), "body_states": plain(task.body_states[0]),
                     "root_state": plain(task.root_states[0]),
                     "reward": float(reward[0]), "reset": float(reset[0]), "progress": float(task.progress_buf[0]),
                     "sim_count": float(task.sim_count[0])})

    ids = torch.tensor([0], device=task.device, dtype=torch.long)
    task.reset_idx(ids)
    if args.joint_speed is not None and hasattr(task, "joint_speed_limit"):
        task.joint_speed_limit.fill_(args.joint_speed)
    gym.set_dof_position_target_tensor(task.sim, isaacgym.gymtorch.unwrap_tensor(task.pos_d))
    gym.simulate(task.sim)
    gym.fetch_results(task.sim, True)
    gym.refresh_actor_root_state_tensor(task.sim)
    gym.refresh_dof_state_tensor(task.sim)
    gym.refresh_rigid_body_state_tensor(task.sim)
    gym.refresh_net_contact_force_tensor(task.sim)
    task.compute_observations()
    if camera_output:
        camera_output.parent.mkdir(parents=True, exist_ok=True)
        gym.step_graphics(task.sim)
        gym.render_all_camera_sensors(task.sim)
        gym.write_camera_image_to_file(task.sim, env, camera, gymapi.IMAGE_COLOR, str(camera_output))
    clean_bodies = {name: {"p": plain(task.body_states[0, i, :3]), "q_xyzw": plain(task.body_states[0, i, 3:7])}
                    for i, name in enumerate(body_names)}

    result = {"task": args.task, "checkpoint": str(checkpoint_path.relative_to(project)),
              "checkpoint_epoch": checkpoint.get("epoch"), "checkpoint_frame": checkpoint.get("frame"),
              "dt": float(task.dt), "joint_speed_limit": plain(getattr(task, "joint_speed_limit", getattr(task, "speed_limit", None))),
              "body_names": body_names, "dof_names": dof_names,
              "initial_env_reset_obs": plain(initial[0]), "effective_steps": rows,
              "clean_reset": {"root_state": plain(task.root_states[0]), "dof_pos": plain(task.dof_pos[0]),
                              "dof_vel": plain(task.dof_vel[0]), "body_poses": clean_bodies,
                              "observation": plain(task.obs_buf[0])},
              "asset": {"file": task.asset_file, "rigid_body_properties": rigid_properties,
                        "dof_properties": dof_properties, "shape_properties": shape_properties}}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"captured {args.task}: {len(body_names)} bodies, {len(dof_names)} dofs, {len(rows)} steps -> {output_path}")


if __name__ == "__main__":
    main()
