"""MDP terms matching the original six-DOF quaternion task.

Source: zbot_rl_student curriculum_env.Zbot6DofQuatEnv and shared_rewards.
Isaac Lab body_lin_vel_w is COM velocity; body pose is the link frame.
"""

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse


def walking_frame(quat):
    """Preserve the source's gravity-cross-local-Z forward vector, unnormalized."""
    axis_z = torch.zeros_like(quat[..., :3])
    axis_z[..., 2] = 1
    gravity = quat_apply_inverse(quat, -axis_z)
    forward_b = torch.cross(gravity, axis_z, dim=-1)
    heading_error = -quat_apply(quat, forward_b)[..., 1]
    return gravity, forward_b, heading_error


def _base(env):
    robot = env.scene["robot"]
    return robot, robot.find_bodies("base")[0][0]


def contact_point_slip(data, sensor, feet_ids, card):
    """Force-weighted tangential contact-point speed with a Huber penalty."""
    slots = 1
    found = sensor.found.reshape(data.body_com_pos_w.shape[0], 2, slots) > 0
    force = sensor.force.reshape(data.body_com_pos_w.shape[0], 2, slots, 3).norm(dim=-1)
    position = sensor.pos.reshape(data.body_com_pos_w.shape[0], 2, slots, 3)
    normal = sensor.normal.reshape(data.body_com_pos_w.shape[0], 2, slots, 3)
    com_position = data.body_com_pos_w[:, feet_ids, None, :]
    com_velocity = data.body_com_lin_vel_w[:, feet_ids, None, :]
    angular_velocity = data.body_com_ang_vel_w[:, feet_ids, None, :]
    point_velocity = com_velocity + torch.cross(angular_velocity.expand_as(position), position - com_position, dim=-1)
    normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    tangent_velocity = point_velocity - (point_velocity * normal).sum(dim=-1, keepdim=True) * normal
    speed = tangent_velocity.norm(dim=-1)
    delta = card.slip_huber_delta
    huber = torch.where(speed <= delta, 0.5 * speed.square() / delta, speed - 0.5 * delta)
    weight = torch.where(found & (force > card.slip_force_threshold), force, 0.0)
    return (huber * weight).sum(dim=(1, 2)) / weight.sum(dim=(1, 2)).clamp_min(1e-8)


def mechanical_power(data):
    """Positive joint mechanical power; integration over env.step_dt yields joules."""
    return (data.qfrc_actuator * data.joint_vel).abs().sum(dim=-1)


def small_step_penalty(lengths, air_time, touchdown, card):
    """Penalize touchdown events with too little advance or swing time."""
    advance = card.moving_direction * lengths
    distance_shortfall = (card.minimum_step_advancement - advance).clamp_min(0) / card.minimum_step_advancement
    duration_shortfall = (card.minimum_swing_duration - air_time).clamp_min(0) / card.minimum_swing_duration
    return ((distance_shortfall + duration_shortfall) * touchdown).sum(dim=-1)


def body_shake_penalty(angular, previous_angular, linear, previous_linear, initializing):
    """Control-step torso velocity variation, excluding steady forward speed."""
    value = (angular - previous_angular).square().sum(dim=-1)
    value += 10 * (linear[:, 1:] - previous_linear[:, 1:]).square().sum(dim=-1)
    return torch.where(initializing, 0.0, value)


def policy_observation(env):
    robot, base_id = _base(env)
    data = robot.data
    action = env.action_manager.get_term("joint_position")
    ids = action.joint_ids
    return torch.cat(
        (
            data.body_link_quat_w[:, base_id],
            data.body_link_ang_vel_w[:, base_id],
            data.joint_pos[:, ids] - data.default_joint_pos[:, ids],
            data.joint_vel[:, ids],
            action.bounded_action,
            action.speed,
        ),
        dim=-1,
    )


def fallen(env, card):
    robot, base_id = _base(env)
    pos = robot.data.body_link_pos_w[:, base_id] - env.scene.env_origins
    forces = env.scene["body_contact"].data.force_history
    contact = forces.norm(dim=-1).amax(dim=(-1, -2)) > card.non_foot_contact_force
    return (
        contact
        | (pos[:, 2] < card.termination_height)
        | (pos[:, 1].abs() > card.maximum_lateral_deviation)
    )


def source_time_out(env):
    # Source DirectRLEnv checks length >= max_episode_length - 1.
    return env.episode_length_buf >= env.max_episode_length - 1


def forward_velocity(env, card):
    robot, base_id = _base(env)
    quat = robot.data.body_link_quat_w[:, base_id]
    _, forward_b, _ = walking_frame(quat)
    vel_b = quat_apply_inverse(quat, robot.data.body_com_lin_vel_w[:, base_id])
    return card.moving_direction * (vel_b * forward_b).sum(dim=-1)


def base_vel_forward(env, card):
    velocity = forward_velocity(env, card)
    env.extras.setdefault("log", {})["Metrics/forward_velocity"] = float(velocity.mean().item())
    return torch.exp(-((velocity - card.target_forward_speed) / card.velocity_sigma).square())


def feet_downward(env, card):
    del card
    robot = env.scene["robot"]
    feet = robot.find_bodies(("foot_0", "foot_1"), preserve_order=True)[0]
    quat = robot.data.body_link_quat_w[:, feet]
    axis_z = torch.zeros_like(quat[..., :3])
    axis_z[..., 2] = 1
    return quat_apply(quat, axis_z)[..., :2].norm(dim=-1).sum(dim=-1)


def feet_forward(env, card):
    del card
    robot, base_id = _base(env)
    feet = robot.find_bodies(("foot_0", "foot_1"), preserve_order=True)[0]
    quat = robot.data.body_link_quat_w[:, base_id]
    feet_quat = robot.data.body_link_quat_w[:, feet]
    _, forward_b, _ = walking_frame(quat)
    axis_x = torch.zeros_like(feet_quat[..., :3])
    axis_x[..., 0] = 1
    feet_x_b = quat_apply_inverse(quat[:, None].expand(-1, 2, -1), quat_apply(feet_quat, axis_x))
    return (feet_x_b - forward_b[:, None]).norm(dim=-1).sum(dim=-1)


def terminal_penalty(env, card):
    # RewardManager integrates all terms with dt. This keeps the source's discrete cost.
    return -env.reset_terminated.float() * card.terminated_reward_penalty / env.step_dt


class WalkingReward:
    """Two-stage reward with global, consecutive-step promotion.

    Stage is global and intentionally survives episode resets. Per-episode
    force/heading integrals and touchdown history reset only for selected envs.
    """

    def __init__(self, cfg, env):
        self.env = env
        self.card = cfg.params["card"]
        self.stage = self.card.initial_stage
        self.promotion_counter = 0
        self.last_metric = 0.0
        self.robot, self.base_id = _base(env)
        self.feet_ids = self.robot.find_bodies(("foot_0", "foot_1"), preserve_order=True)[0]
        self.force_integral = torch.zeros(env.num_envs, device=env.device)
        self.heading_integral = torch.zeros_like(self.force_integral)
        self.last_force = torch.zeros((env.num_envs, 2), device=env.device)
        self.step_length = torch.zeros_like(self.last_force)
        self.touchdown_pos = torch.zeros((env.num_envs, 2, 3), device=env.device)
        self.initialize_touchdown = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.energy_integral = torch.zeros(env.num_envs, device=env.device)
        self.previous_base_angular = torch.zeros((env.num_envs, 3), device=env.device)
        self.previous_base_linear = torch.zeros_like(self.previous_base_angular)
        self.initialize_motion = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.previous_joint_velocity = torch.zeros_like(self.robot.data.joint_vel)
        self.step_timer = torch.zeros((env.num_envs, 2), device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for tensor in (
            self.force_integral,
            self.heading_integral,
            self.last_force,
            self.step_length,
            self.touchdown_pos,
            self.energy_integral,
            self.previous_base_angular,
            self.previous_base_linear,
            self.previous_joint_velocity,
            self.step_timer,
        ):
            tensor[ids] = 0
        self.initialize_touchdown[ids] = True
        self.initialize_motion[ids] = True

    def _promote(self, metric):
        if self.stage != 1:
            return
        self.last_metric = float(metric.mean().item())
        self.promotion_counter = (
            self.promotion_counter + 1 if self.last_metric > self.card.promotion_threshold else 0
        )
        if self.promotion_counter >= self.card.promotion_window_steps:
            self.stage = 2

    def __call__(self, env, card):
        data = self.robot.data
        action = env.action_manager.get_term("joint_position")
        q = data.body_link_quat_w[:, self.base_id]
        gravity, forward, heading = walking_frame(q)
        velocity = quat_apply_inverse(q, data.body_com_lin_vel_w[:, self.base_id])
        base_linear = data.body_com_lin_vel_w[:, self.base_id]
        base_angular = data.body_com_ang_vel_w[:, self.base_id]
        initializing_motion = self.initialize_motion.clone()
        body_shake = body_shake_penalty(base_angular, self.previous_base_angular, base_linear, self.previous_base_linear, initializing_motion)
        joint_acceleration = torch.where(initializing_motion, 0.0, (data.joint_vel - self.previous_joint_velocity).square().sum(dim=-1))
        self.previous_base_angular.copy_(base_angular)
        self.previous_base_linear.copy_(base_linear)
        self.previous_joint_velocity.copy_(data.joint_vel)
        self.initialize_motion[:] = False
        forward_velocity = card.moving_direction * (velocity * forward).sum(dim=-1)
        feet_q = data.body_link_quat_w[:, self.feet_ids]
        feet_pos = data.body_link_pos_w[:, self.feet_ids]
        axis_z = torch.zeros_like(feet_pos)
        axis_z[..., 2] = 1
        axis_x = torch.zeros_like(feet_pos)
        axis_x[..., 0] = 1
        feet_z = quat_apply(feet_q, axis_z)
        feet_x = quat_apply_inverse(q[:, None].expand(-1, 2, -1), quat_apply(feet_q, axis_x))
        sensor = env.scene["feet_contact"].data
        # netforce is world-frame force; use upward magnitude for load balance.
        force = sensor.force_history[..., 2].mean(dim=-1).abs()
        force_diff = self.force_integral.sign() * (force[:, 1] - force[:, 0])
        raw = {
            "feet_downward": feet_z[..., :2].norm(dim=-1).sum(dim=-1),
            "feet_forward": (feet_x - forward[:, None]).norm(dim=-1).sum(dim=-1),
            "base_heading_x": heading.abs(),
            "feet_force_diff": force_diff,
        }
        # Keep original ordering: force-difference uses the PREVIOUS integral.
        if self.stage == 1:
            self.force_integral += 0.001 * (force[:, 0] - force[:, 1])
            raw["feet_force_sum"] = self.force_integral.abs()
            scales = card.stage_1_rewards
        else:
            self.heading_integral.add_(0.01 * heading).clamp_(-1, 1)
            init = self.initialize_touchdown
            self.touchdown_pos[init] = feet_pos[init]
            self.initialize_touchdown[:] = False
            touchdown = (force > 10) & (self.last_force < 10)
            step_vec = quat_apply_inverse(
                q[:, None].expand(-1, 2, -1), feet_pos - self.touchdown_pos
            )
            lengths = (step_vec * forward[:, None]).sum(dim=-1)
            self.step_length[touchdown] = lengths[touchdown]
            self.touchdown_pos[touchdown] = feet_pos[touchdown]
            self.last_force.copy_(force)
            air = sensor.last_air_time
            pos = data.body_link_pos_w[:, self.base_id]
            origins = env.scene.env_origins
            contact = force > card.slip_force_threshold
            contact_count = contact.sum(dim=-1)
            support_center = (feet_pos[..., :2] * contact[..., None]).sum(dim=1) / contact_count.clamp_min(1)[..., None]
            support_distance = (data.body_com_pos_w[:, self.base_id, :2] - support_center).norm(dim=-1)
            support_stability = torch.exp(-(support_distance / card.support_margin_sigma).square()) * (contact_count > 0)
            self.step_timer += env.step_dt
            step_cadence = (((self.step_timer - card.target_step_period).abs() / card.step_period_sigma).clamp_max(3) * touchdown).sum(dim=-1)
            raw.update(
                {
                    "base_vel_forward": torch.tanh(10 * forward_velocity / action.speed[:, 0]),
                    "slow_speed_tracking": torch.exp(-((forward_velocity - card.target_forward_speed) / card.velocity_sigma).square()),
                    "similar_to_default": (data.joint_pos - data.default_joint_pos)
                    .abs()
                    .sum(dim=-1),
                    "base_heading_x_sum": self.heading_integral.abs(),
                    "support_stability": support_stability,
                    "base_tilt": gravity[..., :2].square().sum(dim=-1),
                    "step_length": torch.tanh(
                        15 * card.moving_direction * self.step_length.amin(dim=-1)
                    ),
                    "small_step": small_step_penalty(lengths, air, touchdown, card),
                    "step_cadence": step_cadence,
                    "airtime_balance": (air[:, 0] - air[:, 1]).abs(),
                    "airtime_sum": torch.tanh(air.sum(dim=-1)),
                    "double_flight": (contact_count == 0).float(),
                    "action_rate": (action.bounded_action - action.previous_action)
                    .square()
                    .sum(dim=-1),
                    "body_shake": body_shake,
                    "joint_velocity": data.joint_vel.square().sum(dim=-1),
                    "joint_acceleration": joint_acceleration,
                    "torques": 0.002 * data.qfrc_actuator.square().sum(dim=-1),
                    "energy_consumption": mechanical_power(data),
                    "feet_slide": contact_point_slip(data, env.scene["feet_slip_contact"].data, self.feet_ids, card),
                    "base_pos_y_err": 10
                    * (
                        (feet_pos[:, 0, 1] + feet_pos[:, 1, 1] - 2 * origins[:, 1]).abs()
                        + (pos[:, 1] - origins[:, 1]).abs()
                    ),
                }
            )
            scales = card.stage_2_rewards
            self.step_timer[touchdown] = 0
            self.energy_integral += raw["energy_consumption"] * env.step_dt
        result = sum(raw[name] * weight for name, weight in scales.items())
        self._promote(force_diff)
        logs = env.extras.setdefault("log", {})
        logs["Metrics/forward_velocity"] = float(forward_velocity.mean().item())
        for name in dict.fromkeys((*card.stage_1_rewards, *card.stage_2_rewards)):
            logs[f"Reward/{name}"] = (
                float((raw[name] * scales[name]).mean().item()) if name in scales else 0.0
            )
        logs["Curriculum/stage"] = float(self.stage)
        logs["Curriculum/feet_force_diff_mean"] = self.last_metric
        logs["Metrics/energy_j"] = float(self.energy_integral.mean().item())
        # RewardManager multiplies this rate by dt; the -20 terminal cost is discrete.
        return result - env.reset_terminated.float() * card.terminated_reward_penalty / env.step_dt
