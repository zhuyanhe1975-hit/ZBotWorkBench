#!/usr/bin/env python3
"""Finite local mjlab PPO job. Protocol events go to stdout; library logs to stderr."""
import argparse
from dataclasses import asdict, replace
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import sys
import time
import traceback

TASK_ID = "Mjlab-Zbot-6dof-Bipedal-Walking"
WALKING_TASK_ID = "Mjlab-Zbot-6dof-Walking-Finetune"
QUASISTATIC_TASK_ID = "Mjlab-Zbot-6dof-Quasistatic-Walking"
IN_PLACE_TASK_ID = "Mjlab-Zbot-6dof-InPlace-Stepping"
QUAT_MODEL = "zbot_rl_mjlab.models:QuatFeatureMLP"
SEED_SHA256 = "ddd61fd509a2d9bca3dd027c8e62b93d019c34be8a3da6e0f2262ee1f15181c6"
STOPPED = False


def request_stop(*_):
    global STOPPED
    STOPPED = True


def validate_request(request):
    if request.get("taskId") not in (TASK_ID, WALKING_TASK_ID, QUASISTATIC_TASK_ID, IN_PLACE_TASK_ID):
        raise ValueError("Unsupported training task")
    if not re.fullmatch(r"cpu|cuda:\d+", request.get("device", "")):
        raise ValueError("Invalid training device")
    limits = {"cpuThreads": (1, 256), "numEnvs": (1, 4096), "iterations": (1, 100000),
              "saveInterval": (1, 10000), "seed": (0, 2147483647), "maxSeconds": (1, 86400)}
    for name, (lower, upper) in limits.items():
        value = request.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
            raise ValueError(f"Invalid {name}: expected integer in [{lower}, {upper}]")
    if request["device"] == "cpu" and request["numEnvs"] > 64:
        raise ValueError("CPU training supports at most 64 environments")


def preset_checkpoint():
    import hashlib
    checkpoint = Path(__file__).resolve().parent / "presets/walking/model_801.pt"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != SEED_SHA256:
        raise ValueError("Packaged walking seed checkpoint checksum mismatch")
    return checkpoint


def validate_policy_config(config, state, task_id):
    expected = QUAT_MODEL if task_id in (WALKING_TASK_ID, QUASISTATIC_TASK_ID) else "MLPModel"
    width = 30 if expected == QUAT_MODEL else (34 if task_id == IN_PLACE_TASK_ID else 26)
    ppo = config["ppo"]
    for name in ("actor", "critic"):
        model = ppo[name]
        if (model.get("class_name", "MLPModel") != expected or model.get("activation") != "elu"
                or model.get("obs_normalization") or model.get("rnn_type") or model.get("cnn_cfg")):
            raise ValueError("Unsupported policy architecture for this task")
    if (ppo["algorithm"].get("class_name", "PPO") != "PPO"
            or state["actor_state_dict"]["mlp.0.weight"].shape[1] != width):
        raise ValueError("Unsupported checkpoint algorithm or observation contract")
    if task_id == TASK_ID:
        card = config["task_card"]
        if (card.get("physics_dt") != 1 / 240 or card.get("decimation") != 8
                or card.get("contact_history_length") != 8):
            raise ValueError("Legacy 60 Hz checkpoint is incompatible with the verified 240 Hz training task")


def from_scratch_overrides():
    """Configuration that produced the verified model_299 baseline."""
    return {"physics_dt": 1 / 240, "decimation": 8, "contact_history_length": 8}


def quasistatic_overrides(card):
    weights = {"base_vel_forward": 0.0, "slow_speed_tracking": 2.0, "feet_downward": -2.0,
               "feet_forward": -1.0, "similar_to_default": -1.5, "base_heading_x": -2.0,
               "base_heading_x_sum": -1.0, "support_stability": 4.0, "base_tilt": -2.0,
               "step_length": 0.5, "small_step": -2.0, "step_cadence": -1.0,
               "airtime_balance": -3.0, "airtime_sum": -0.5, "double_flight": -10.0,
               "action_rate": -2.0, "body_shake": -5.0, "joint_velocity": -0.5,
               "joint_acceleration": -0.2, "torques": -0.2, "energy_consumption": -0.05,
               "feet_slide": -20.0, "base_pos_y_err": -2.0}
    return replace(card, joint_speed_range=(0.2, 0.4), target_forward_speed=0.04,
                   velocity_sigma=0.02, stage_2_rewards=weights)


def apply_task_card_settings(card, value):
    if value is None:
        return card
    if not isinstance(value, dict):
        raise ValueError("taskCard must be an object")
    result = {}
    for field, current in (("stage1Rewards", card.stage_1_rewards), ("stage2Rewards", card.stage_2_rewards)):
        rewards = value.get(field)
        if card.reward_mode == "in_place" and field == "stage1Rewards" and rewards == {}:
            result["stage_1_rewards"] = {}
            continue
        if not isinstance(rewards, dict) or set(rewards) != set(current):
            raise ValueError(f"{field} reward terms do not match the task")
        if any(isinstance(weight, bool) or not isinstance(weight, (int, float))
               or not math.isfinite(weight) or not -100 <= weight <= 100 for weight in rewards.values()):
            raise ValueError(f"{field} weights must be finite and within [-100, 100]")
        result["stage_1_rewards" if field == "stage1Rewards" else "stage_2_rewards"] = dict(rewards)
    penalty = value.get("terminatedRewardPenalty")
    if isinstance(penalty, bool) or not isinstance(penalty, (int, float)) or not math.isfinite(penalty) or not 0 <= penalty <= 1000:
        raise ValueError("terminatedRewardPenalty must be within [0, 1000]")
    return replace(card, terminated_reward_penalty=float(penalty), **result)


def prepare_device(requested, environment):
    """Map a physical nvidia-smi index to the worker's sole visible CUDA device."""
    if requested == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu"
    if not re.fullmatch(r"cuda:\d+", requested):
        raise ValueError("Invalid training device")
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = requested.split(":")[1]
    return "cuda:0"


def apply_cpu_affinity(threads):
    """Restrict only this worker to a subset of its inherited allowed CPUs."""
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        return None
    try:
        selected = sorted(os.sched_getaffinity(0))[:threads]
        if not selected:
            return None
        os.sched_setaffinity(0, set(selected))
        return selected
    except OSError:
        # Non-Linux/restricted runtimes still retain all library thread limits.
        return None


class Stopped(Exception):
    pass


def live_frame(robot, environment_origins, iteration, limit=9):
    """Return a bounded real training-batch pose snapshot for the web viewer."""
    names = list(robot.body_names)
    count = min(limit, int(robot.data.body_link_pos_w.shape[0]))
    positions = robot.data.body_link_pos_w[:count].detach().cpu().tolist()
    quaternions = robot.data.body_link_quat_w[:count].detach().cpu().tolist()
    origins = environment_origins[:count].detach().cpu().tolist()
    if len(names) > 32 or any(len(env) != len(names) for env in positions + quaternions):
        raise ValueError("Unexpected robot body layout for live preview")
    return {"iteration": int(iteration), "bodyNames": names,
            "environments": [{"environmentOrigin": origins[index], "bodyPositions": positions[index], "bodyQuaternions": quaternions[index]}
                             for index in range(count)]}


def live_preview_enabled(job):
    return (job / "LIVE_PREVIEW").is_file()


def aggregate_reward_terms(extras):
    values = {}
    for info in extras:
        for key, value in info.items():
            if not isinstance(key, str) or not key.startswith("Reward/"):
                continue
            if hasattr(value, "detach"):
                value = value.detach().float().mean().cpu().item()
            number = float(value)
            if math.isfinite(number):
                values.setdefault(key[7:], []).append(number)
    return {key: statistics.mean(samples) for key, samples in values.items() if samples}


def run(request, job, emit):
    started = time.monotonic()
    validate_request(request)
    if STOPPED or (job / "STOP").exists():
        emit("finished", stopped=True, iterations=0)
        return
    threads = request["cpuThreads"]
    affinity = apply_cpu_affinity(threads)
    if affinity:
        threads = min(threads, len(affinity))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)
    device = prepare_device(request["device"], os.environ)
    os.environ["MUJOCO_GL"] = "wgl" if sys.platform == "win32" else "egl"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ.setdefault("MPLCONFIGDIR", str(job / "cache" / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(job.parent / ".zbot-training-cache"))
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(__file__).resolve().parent / "tasks"))

    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends
    from zbot_rl_mjlab.env_cfg import walking_env_cfg, walking_ppo_cfg
    from zbot_rl_mjlab.runner import WalkingRunner, checkpoint_config
    from zbot_rl_mjlab.task_card import WalkingTaskCard
    from export_bundle import export_bundle

    versions = {name: importlib.metadata.version(name) for name in ("mjlab", "mujoco", "mujoco-warp", "warp-lang", "rsl-rl-lib", "torch")}
    if versions["mjlab"] != "1.3.0" or versions["rsl-rl-lib"] != "5.2.0":
        raise ValueError("Unsupported training runtime; use pinned training/requirements.txt")
    # Match mjlab.scripts.train exactly; the verified run used TF32 on CUDA.
    configure_torch_backends()
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(max(1, min(threads, 4)))
    torch.manual_seed(request["seed"])
    if device.startswith("cuda") and (not torch.cuda.is_available() or int(device.split(":")[1]) >= torch.cuda.device_count()):
        raise ValueError(f"Requested physical CUDA device {request['device']} is unavailable")
    resume = Path(request["resumePath"]).resolve(strict=True) if request.get("resumePath") else None
    seed_checkpoint = preset_checkpoint() if not resume and request["taskId"] in (WALKING_TASK_ID, QUASISTATIC_TASK_ID) else None
    load_path = resume or seed_checkpoint
    if load_path:
        saved = torch.load(load_path, map_location="cpu", weights_only=True)
        if resume:
            marker = (saved.get("infos") or {}).get("workbench_training", {})
            if marker.get("format_version") != 1 or marker.get("task_id", TASK_ID) != request["taskId"]:
                raise ValueError("Resume requires a checkpoint produced by this workbench training task")
        config = checkpoint_config(load_path)
        validate_policy_config(config, saved, request["taskId"])
        ppo = config["ppo"]
        card = replace(WalkingTaskCard(**config["task_card"]), num_envs=request["numEnvs"])
        if seed_checkpoint and request["taskId"] == QUASISTATIC_TASK_ID:
            card = quasistatic_overrides(card)
        for name in ("actor", "critic"):
            ppo[name] = RslRlModelCfg(**ppo[name])
        ppo["algorithm"] = RslRlPpoAlgorithmCfg(**ppo["algorithm"])
        agent = RslRlOnPolicyRunnerCfg(**ppo)
    else:
        card = replace(WalkingTaskCard(num_envs=request["numEnvs"]), **from_scratch_overrides())
        if request["taskId"] == IN_PLACE_TASK_ID:
            card = replace(card, reward_mode="in_place", target_frequency=1.0, joint_speed_range=(2.0, 2.0), foot_sliding_friction=2.0, stage_1_rewards={},
                           stage_2_rewards={"support_phase": 4.0, "com_phase": 2.0, "heading": -2.0, "joint_pose": -0.2,
                                            "body_velocity": -0.5, "upright": -1.0,
                                            "action_rate": -0.05, "single_support": 1.0})
        agent = walking_ppo_cfg()
        if request["taskId"] == IN_PLACE_TASK_ID:
            # Start the standing curriculum conservatively: large Gaussian
            # exploration impulses make the freshly initialized robot bounce.
            agent.actor.distribution_cfg["init_std"] = 0.3
            agent.algorithm.entropy_coef = 0.001
    card = apply_task_card_settings(card, request.get("taskCard"))
    agent.seed = request["seed"]
    agent.save_interval = request["saveInterval"]
    agent.max_iterations = request["iterations"]
    agent.logger = "tensorboard"
    agent.upload_model = False
    cfg = walking_env_cfg(card=card)
    cfg.seed = request["seed"]
    checkpoints = job / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    (job / "versions.json").write_text(json.dumps(versions, indent=2))
    env = None
    runner = None
    completed = 0
    latest = None

    def stopping():
        return STOPPED or (job / "STOP").exists() or time.monotonic() - started >= request["maxSeconds"]

    class JobRunner(WalkingRunner):
        def save(self, path, infos=None):
            nonlocal latest
            name = f"model_{self.current_learning_iteration}.pt"
            destination = checkpoints / name
            if latest == destination:
                return
            info = {**(infos or {}), "workbench_training": {"format_version": 1,
                    "completed_updates": self.current_learning_iteration + 1, "versions": versions,
                    "task_id": request["taskId"],
                    "seed_checkpoint_sha256": SEED_SHA256 if request["taskId"] in (WALKING_TASK_ID, QUASISTATIC_TASK_ID) else None,
                    "requested_device": request["device"], "runtime_device": device,
                    "cpu_affinity": affinity, "effective_cpu_threads": threads}}
            temporary = destination.with_suffix(".pt.tmp")
            super().save(str(temporary), info)
            temporary.replace(destination)
            latest = destination
            emit("checkpoint", name=name)

    if stopping():
        emit("finished", stopped=True, iterations=0)
        return
    try:
        env = ManagerBasedRlEnv(cfg, device=device)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        robot = env.scene["robot"]
        preview = {"enabled": live_preview_enabled(job), "checked": time.monotonic(), "emitted": 0.0}
        original_step = wrapped.step

        def step_with_preview(actions):
            result = original_step(actions)
            now = time.monotonic()
            if now - preview["checked"] >= 0.25:
                preview["enabled"] = live_preview_enabled(job)
                preview["checked"] = now
            # Supply headroom for the browser's 30 Hz interpolated renderer.
            if preview["enabled"] and now - preview["emitted"] >= 1 / 45:
                relative_iteration = max(0, runner.current_learning_iteration - start_iteration + 1) if runner else 0
                emit("liveFrame", liveFrame=live_frame(robot, env.scene.env_origins, relative_iteration))
                preview["emitted"] = now
            return result

        wrapped.step = step_with_preview
        runner = JobRunner(wrapped, asdict(agent), str(job / "logs"), device=device)
        if load_path:
            runner.load(str(load_path), map_location=device,
                        # Reward weights are intentionally hot-editable.  A resumed
                        # checkpoint may therefore start with the current task-card
                        # weights instead of the values stored when it was saved.
                        # The network/PPO contract is still checked by runner.load.
                        allow_task_card_change=True)
            runner.current_learning_iteration = int(saved["iter"]) + 1
        start_iteration = runner.current_learning_iteration
        original_log = runner.logger.log
        card_revision = 0

        def refresh_task_card():
            nonlocal card, card_revision
            file = job / "TASK_CARD.json"
            if not file.is_file():
                return
            update = json.loads(file.read_text())
            revision = update.get("revision")
            if not isinstance(revision, int) or revision <= card_revision:
                return
            card = apply_task_card_settings(card, update.get("taskCard"))
            term_name = "in_place" if card.reward_mode == "in_place" else "walking"
            term = env.reward_manager.get_term_cfg(term_name)
            term.params["card"] = card
            term.func.card = card
            runner.walking_config["task_card"] = json.loads(json.dumps(asdict(card)))
            card_revision = revision
            emit("taskCard", revision=revision, taskCard=update["taskCard"])

        def report(**info):
            nonlocal completed
            reward_terms = aggregate_reward_terms(runner.logger.ep_extras)
            original_log(**info)
            refresh_task_card()
            completed = info["it"] - start_iteration + 1
            losses = [float(v) for v in info["loss_dict"].values()]
            metrics = {"iteration": completed, "totalIterations": request["iterations"],
                       "fps": request["numEnvs"] * agent.num_steps_per_env / max(1e-9, info["collect_time"] + info["learn_time"])}
            if runner.logger.rewbuffer:
                metrics["reward"] = statistics.mean(runner.logger.rewbuffer)
                metrics["episodeLength"] = statistics.mean(runner.logger.lenbuffer)
            if losses:
                metrics["loss"] = sum(losses) / len(losses)
            if not all(math.isfinite(value) for value in metrics.values()):
                raise RuntimeError("Training metrics contain non-finite values")
            if reward_terms:
                metrics["rewardTerms"] = reward_terms
            emit("progress", **metrics)
            if stopping():
                raise Stopped()

        runner.logger.log = report
        emit("ready", device=request["device"], runtimeDevice=device, versions=versions,
             cpuAffinity=affinity, effectiveCpuThreads=threads)
        try:
            if not stopping():
                runner.learn(request["iterations"], init_at_random_ep_len=True)
        except Stopped:
            pass
        if completed:
            runner.save(str(checkpoints / f"model_{runner.current_learning_iteration}.pt"))
        if latest and card.reward_mode != "in_place":
            # Sensor history buffers are created during inference-mode rollouts;
            # their reset must remain in inference mode as well.
            with torch.inference_mode():
                export_bundle(env, latest, job, {"seed": request["seed"], "taskId": request["taskId"], "task_card": asdict(card)}, versions)
            emit("bundle", name="bundle.json")
        emit("finished", stopped=stopping(), iterations=completed)
    finally:
        writer = getattr(runner.logger, "writer", None) if runner else None
        if writer is not None:
            writer.close()
        if env:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--job-dir", required=True, type=Path)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    # Capture even native library banners on stderr, keeping stdout valid NDJSON.
    events = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr

    def emit(event, **fields):
        events.write(json.dumps({"event": event, **fields}, allow_nan=False) + "\n")

    try:
        request = json.loads(args.request.read_text())
        job = args.job_dir.resolve()
        job.mkdir(parents=True, exist_ok=True)
        run(request, job, emit)
    except Exception as error:
        traceback.print_exc()
        emit("error", message=str(error))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
