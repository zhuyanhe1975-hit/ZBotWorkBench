"""Manager-based task and PPO factories for ZBot six-DOF walking."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig
from .actions import IntegratedJointPositionActionCfg
from . import mdp
from .robot import robot_cfg
from .task_card import WalkingTaskCard


def walking_env_cfg(play=False, card=None):
    card = card or WalkingTaskCard()
    observations = {
        name: ObservationGroupCfg(
            {"walking": ObservationTermCfg(func=mdp.policy_observation)},
            enable_corruption=False,
        )
        for name in ("actor", "critic")
    }
    if card.reward_mode == "bipedal_curriculum":
        rewards = {
            "walking": RewardTermCfg(func=mdp.WalkingReward, weight=1.0, params={"card": card})
        }
    else:
        rewards = {
            **{
                name: RewardTermCfg(func=getattr(mdp, name), weight=weight, params={"card": card})
                for name, weight in card.reward_scales.items()
            },
            "termination": RewardTermCfg(
                func=mdp.terminal_penalty, weight=1.0, params={"card": card}
            ),
        }
    return ManagerBasedRlEnvCfg(
        seed=42,
        scene=SceneCfg(
            num_envs=1 if play else card.num_envs,
            env_spacing=4.0,
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": robot_cfg(card)},
            sensors=(
                ContactSensorCfg(
                    name="feet_contact",
                    primary=ContactMatch(mode="body", pattern=("foot_0", "foot_1"), entity="robot"),
                    fields=("found", "force"),
                    reduce="netforce",
                    track_air_time=True,
                    history_length=card.contact_history_length,
                ),
                ContactSensorCfg(
                    name="feet_slip_contact",
                    primary=ContactMatch(mode="body", pattern=("foot_0", "foot_1"), entity="robot"),
                    secondary=ContactMatch(mode="geom", pattern="terrain"),
                    fields=("found", "force", "pos", "normal"),
                    reduce="maxforce",
                    num_slots=1,
                ),
                ContactSensorCfg(
                    name="body_contact",
                    primary=ContactMatch(mode="body", pattern="base|a.*|b.*", entity="robot"),
                    fields=("found", "force"),
                    reduce="netforce",
                    history_length=card.contact_history_length,
                ),
            ),
        ),
        observations=observations,
        actions={
            "joint_position": IntegratedJointPositionActionCfg(
                entity_name="robot",
                action_scale=card.action_scale,
                joint_speed_range=card.joint_speed_range,
            )
        },
        rewards=rewards,
        terminations={
            "time_out": TerminationTermCfg(func=mdp.source_time_out, time_out=True),
            "fallen": TerminationTermCfg(func=mdp.fallen, params={"card": card}),
        },
        sim=SimulationCfg(
            mujoco=MujocoCfg(
                timestep=card.physics_dt,
                gravity=card.gravity,
                cone=card.friction_cone,
                impratio=card.friction_impedance_ratio,
            ),
            njmax=card.constraint_capacity,
        ),
        decimation=card.decimation,
        episode_length_s=card.episode_length_s,
        viewer=ViewerConfig(
            entity_name="robot",
            body_name="base",
            distance=1.2,
            elevation=-20.0,
            azimuth=135.0,
            width=640,
            height=480,
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
        ),
    )


def walking_ppo_cfg():
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(128, 128, 128),
            activation="elu",
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(hidden_dims=(128, 128, 128), activation="elu"),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=0.001,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="6dof_bipedal_walking",
        num_steps_per_env=24,
        max_iterations=500,
        save_interval=50,
        logger="tensorboard",
    )
