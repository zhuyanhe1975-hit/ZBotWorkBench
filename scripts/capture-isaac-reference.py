"""Development-only Isaac Lab reference capture; never used by the browser app."""
import argparse
import json
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--task', default='Zbot-Direct-8dof-bipedal-v0')
parser.add_argument('--checkpoint', help='Required for motion; --steps 0 only captures the model and does not load a policy')
parser.add_argument('--output', required=True)
parser.add_argument('--steps', type=int, default=600)
parser.add_argument('--project', default='/home/yhzhu/myWorks_vips/zbot_rl_student')
parser.add_argument('--colliders-output')
parser.add_argument('--free-space', action='store_true')
parser.add_argument('--joint-speed', type=float, default=2.0)
parser.add_argument('--root-yaw-deg', type=float, help='Explicit historical root-yaw comparison; recorded in capture metadata')
parser.add_argument('--commands', type=float, nargs=3, metavar=('VX', 'VY', 'WZ'), help='Fixed velocity task commands')
parser.add_argument('--phase-frequency', type=float, help='Fixed velocity-task stepping frequency in Hz')
parser.add_argument('--phase-offset', type=float, help='Fixed velocity-task initial phase in radians')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 0 or (args.steps > 0 and not args.checkpoint):
    parser.error('Use non-negative --steps and supply --checkpoint for motion capture')
sys.path.insert(0, str(Path(args.project) / 'zbot_direct/source/zbot_direct'))
launcher = AppLauncher(args)

import gymnasium as gym
import torch
import zbot_direct
from isaaclab_tasks.utils import parse_env_cfg

if not Path(zbot_direct.__file__).resolve().is_relative_to(Path(args.project).resolve()):
    raise RuntimeError(f'Wrong task package loaded: {zbot_direct.__file__}')

if args.device == 'cpu':
    # The source reward helper hardcodes CUDA masses even for CPU environments.
    # Align only that diagnostic tensor; no physics/observation values change.
    from zbot_direct.tasks.direct.zbot_direct import bipedal_env, ground_env, transition_env
    from zbot_direct.tasks.direct.zbot_direct.env_utils import compute_com
    def cpu_com(positions, masses):
        return compute_com(positions, masses.to(positions.device))
    bipedal_env.compute_com = cpu_com
    ground_env.compute_com = cpu_com
    transition_env.compute_com = cpu_com


def values(tensor):
    return tensor.detach().cpu().tolist()


cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
cfg.seed = 42
# These archived networks retain their original compact observations. The source
# repository expanded its generic bipedal default without migrating checkpoints.
# Evidence: original zbot_rl/zbot_direct/.../zbot_direct_6dof_bipedal_env.py:326;
# zbot_direct_8dof_bipedal_{1,2,3}_env.py:265 and zbot_direct_8dof_bird_env.py:265.
legacy_compact_tasks = {
    'Zbot-Direct-6dof-bipedal-v0': 24,
    'Zbot-Direct-8dof-bipedal-v1': 30,
    'Zbot-Direct-8dof-bipedal-v2': 30,
    'Zbot-Direct-8dof-bipedal-v3': 30,
    'Zbot-Direct-8dof-bird-v0': 30,
}
if args.task in legacy_compact_tasks:
    cfg.policy_observation_terms = ('base_ang_vel_z_b', 'base_projected_gravity_b', 'base_heading_x_err',
                                    'joint_pos_error', 'joint_vel', 'actions', 'joint_speed_limit')
    cfg.observation_space = legacy_compact_tasks[args.task]
if getattr(cfg, 'running_curriculum', None):
    cfg.running_curriculum = {**cfg.running_curriculum, 'warmup_steps': 0, 'transition_steps': 1}
if hasattr(cfg, 'curriculum_warmup_steps'):
    cfg.curriculum_warmup_steps = 0
    cfg.curriculum_transition_steps = 1
if args.root_yaw_deg is not None:
    angle = math.radians(args.root_yaw_deg) / 2
    cfg.robot.init_state.rot = (math.cos(angle), 0., 0., math.sin(angle))
if args.free_space:
    cfg.robot.spawn.rigid_props.disable_gravity = True
    cfg.robot.init_state.pos = (0., -.06, 2.)
env = gym.make(args.task, cfg=cfg).unwrapped
try:
    obs, _ = env.reset()
    if args.colliders_output:
        from isaac_cooked_colliders import capture_cooked_colliders
        cooked = capture_cooked_colliders(env.sim.stage)
        Path(args.colliders_output).write_text(json.dumps(cooked))
    env.episode_length_buf.zero_()
    # One uninterrupted trial; capture original done flags but do not hide falls by resetting.
    env._get_dones = lambda: (torch.zeros(1, dtype=torch.bool, device=env.device), torch.zeros(1, dtype=torch.bool, device=env.device))
    env.common_step_counter = 1
    if hasattr(env, '_joint_speed_sample'):
        lower, upper = env._current_joint_speed_range()
        if not lower <= args.joint_speed <= upper:
            raise ValueError(f'Joint speed {args.joint_speed} outside deployment range {lower, upper}')
        env._joint_speed_sample.fill_((args.joint_speed - lower) / (upper - lower) if upper > lower else 0)
    env.joint_speed_limit.fill_(args.joint_speed)
    if args.commands is not None:
        if not hasattr(env, 'enable_manual_command_override'):
            raise ValueError('--commands is only available for velocity tasks')
        env.enable_manual_command_override(True)
        env.set_manual_commands(args.commands)
    if args.phase_frequency is not None:
        env._step_frequency_cmd.fill_(args.phase_frequency)
    if args.phase_offset is not None:
        env._step_phase_offset.fill_(args.phase_offset)
    if hasattr(env, '_step_phase_cache_step'):
        env._step_phase_cache_step = -1
        env._state_buffer_step = -1
        env._state_filters_initialized.zero_()
    obs = env._get_observations()
    group, state, network = None, {}, None
    if args.steps > 0:
        checkpoint = torch.load(args.checkpoint, weights_only=True, map_location=env.device)
        group = next((name for name in ('student_state_dict', 'actor_state_dict', 'model_state_dict') if name in checkpoint), None)
        if group is None:
            raise ValueError('Unsupported checkpoint: no actor or student state')
        state = checkpoint[group]
        prefix = 'actor.' if group == 'model_state_dict' else 'mlp.'
        keys = sorted([k for k in state if k.startswith(prefix) and k.endswith('.weight')], key=lambda k: int(k.split('.')[1]))
        if not keys:
            raise ValueError('Checkpoint has no MLP actor weights')
        layers = []
        for i, key in enumerate(keys):
            weight = state[key]
            layer = torch.nn.Linear(weight.shape[1], weight.shape[0], device=env.device)
            layer.weight.data.copy_(weight)
            layer.bias.data.copy_(state[key.replace('weight', 'bias')])
            layers.append(layer)
            if i < len(keys)-1:
                layers.append(torch.nn.ELU())
        network = torch.nn.Sequential(*layers).eval()

    def actor(observation):
        if network is None:
            return torch.zeros_like(env._actions)
        if 'obs_normalizer._mean' in state:
            observation = (observation - state['obs_normalizer._mean']) / (state['obs_normalizer._std'] + 1e-2)
        return network(observation)

    robot = env._robot
    view = robot.root_physx_view
    sensor = env._contact_sensor
    feet_sensor_ids = values(env._feet_ids) if isinstance(env._feet_ids, torch.Tensor) else list(env._feet_ids)
    feet_body_ids = values(env.feet_body_idx) if isinstance(env.feet_body_idx, torch.Tensor) else list(env.feet_body_idx)
    metadata = {'task': args.task, 'source': zbot_direct.__file__, 'physicsDt': cfg.sim.dt, 'controlDt': env.step_dt,
                'jointNames': robot.joint_names, 'bodyNames': robot.body_names,
                'defaultQ': values(getattr(env, '_joint_reference_pos', robot.data.default_joint_pos)[0]),
                'robotDefaultQ': values(robot.data.default_joint_pos[0]),
                'observationTerms': getattr(cfg, 'policy_observation_terms', None),
                'checkpointState': group, 'jointSpeed': args.joint_speed, 'modelOnly': args.steps == 0,
                'rootYawOverrideDegrees': args.root_yaw_deg,
                'commands': args.commands, 'phaseFrequency': args.phase_frequency, 'phaseOffset': args.phase_offset,
                'contactBodyNames': sensor.body_names, 'feetSensorIds': feet_sensor_ids, 'feetBodyIds': feet_body_ids,
                'masses': values(view.get_masses()[0]), 'inertias': values(view.get_inertias()[0]),
                'coms': values(view.get_coms()[0]),
                'stiffness': values(view.get_dof_stiffnesses()[0]), 'damping': values(view.get_dof_dampings()[0]),
                'armature': values(view.get_dof_armatures()[0]), 'rows': []}

    def snapshot(observation, raw):
        return {'obs': values(observation[0]), 'raw': values(raw[0]),
                'q': values(robot.data.joint_pos[0]), 'qd': values(robot.data.joint_vel[0]),
                'root': values(robot.data.root_state_w[0]),
                'bodyPos': values(robot.data.body_pos_w[0]), 'bodyQuat': values(robot.data.body_quat_w[0]),
                'bodyVel': values(robot.data.body_vel_w[0]),
                'feetForceHistory': values(sensor.data.net_forces_w_history[0][:, feet_sensor_ids, :]),
                'feetCurrentAirTime': values(sensor.data.current_air_time[0, feet_sensor_ids]),
                'feetCurrentContactTime': values(sensor.data.current_contact_time[0, feet_sensor_ids]),
                'delta': values((env.p_delta if hasattr(env, 'p_delta') else env._joint_pos_target_delta)[0]),
                'actions': values(env._actions[0])}

    with torch.inference_mode():
        for step in range(args.steps+1):
            observation = obs['policy']
            raw = actor(observation)
            row = snapshot(observation, raw)
            row['step'] = step
            metadata['rows'].append(row)
            if step == args.steps:
                break
            obs, _, _, _, _ = env.step(raw)
            if step % 100 == 0:
                print('REFERENCE_STEP', step, flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(metadata))
    print('REFERENCE_WRITTEN', args.output, flush=True)
finally:
    env.close()
    launcher.app.close()
