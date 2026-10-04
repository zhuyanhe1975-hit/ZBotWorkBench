#!/usr/bin/env python3
import argparse, atexit, base64, json, os, select, sys, tempfile, time
from pathlib import Path

def find_checkpoint(repo: Path, mjlab_root: Path, task_root: Path) -> Path:
    configured = os.environ.get("ZBOT_MJLAB_CHECKPOINT")
    candidates = [Path(configured).expanduser()] if configured else []
    if not candidates:
        candidates.extend(sorted(
            (task_root / "logs/rsl_rl/zbot_walking").glob("*/model_*.pt"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        ))
        candidates.extend(sorted(
            (mjlab_root / "logs/rsl_rl/zbot_walking").glob("*/model_*.pt"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        ))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    reference = repo / "public/rl/training/walking-reference.json"
    if reference.is_file():
        payload = json.loads(reference.read_text())
        encoded = payload.get("checkpointBase64")
        if isinstance(encoded, str) and encoded:
            handle = tempfile.NamedTemporaryFile(prefix="zbot-mjlab-", suffix=".pt", delete=False)
            handle.write(base64.b64decode(encoded))
            handle.close()
            checkpoint = Path(handle.name)
            atexit.register(lambda: checkpoint.unlink(missing_ok=True))
            return checkpoint
    raise RuntimeError(
        "未找到 MJLab 检查点；请先完成一次 mjlab 训练，或设置 ZBOT_MJLAB_CHECKPOINT 指向 model_*.pt"
    )

def main():
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[1]
    default_mjlab = Path.home() / "AI/mjlab"
    if not default_mjlab.is_dir(): default_mjlab = Path.home() / "mjlab"
    default_task = Path.home() / "myWorks_vips/zbot_rl_mjlab"
    if not default_task.is_dir(): default_task = repo / "training/vendor/zbot_rl_mjlab"
    ap.add_argument("--root", default=os.environ.get("ZBOT_MJLAB_ROOT", str(default_mjlab)))
    ap.add_argument("--task-root", default=os.environ.get("ZBOT_MJLAB_TASK_ROOT", str(default_task)))
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--frequency", type=float, default=0.4)
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    task_root = Path(args.task_root).expanduser().resolve()
    if not (root / "src/mjlab").is_dir():
        raise RuntimeError(f"MJLab source not found: {root}")
    if not (task_root / "src/zbot_rl_mjlab").is_dir():
        raise RuntimeError(f"ZBot MJLab task source not found: {task_root}")
    import torch
    from dataclasses import asdict
    # Import the standalone MJLab checkout before exposing the task source.
    # The task repository also carries a partial `mjlab/` tree; putting its
    # src directory first would shadow installed modules such as asset_zoo.
    import mjlab
    sys.path.insert(0, str(task_root / "src"))
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    import zbot_rl_mjlab
    checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else find_checkpoint(repo, root, task_root)
    if not checkpoint.is_file():
        raise RuntimeError(f"MJLab 检查点不存在: {checkpoint}")
    cfg = load_env_cfg("Mjlab-Zbot-6dof-Walking", play=True)
    cfg.scene.env_spacing = 1.0
    cfg.device = args.device
    cfg.step_frequency_min = 0.25
    cfg.step_frequency_max = 1.0
    cfg.test_frequency = args.frequency
    agent = load_rl_cfg("Mjlab-Zbot-6dof-Walking")
    env = ManagerBasedRlEnv(cfg, device=args.device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent), device=args.device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)
    obs = wrapped.reset()[0]
    robot = env.unwrapped.scene["robot"]
    names = list(robot.body_names)
    origins = env.unwrapped.scene.env_origins[0].detach().cpu().tolist()
    print(json.dumps({"kind":"ready","bodyNames":names,"frequency":args.frequency,"device":args.device}), flush=True)
    paused = True
    step_once = False
    frequency = args.frequency
    step = 0
    def emit_frame(frame_step: int, actions=None):
        pos = robot.data.body_link_pos_w[0].detach().cpu().tolist()
        quat = robot.data.body_link_quat_w[0].detach().cpu().tolist()
        root = robot.data.root_link_pos_w[0].detach().cpu().tolist()
        vel = robot.data.root_link_lin_vel_w[0].detach().cpu().tolist()
        print(json.dumps({"kind":"frame","iteration":frame_step,"capturedAt":time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),"step":frame_step,"time":frame_step * 0.02,
                          "bodyNames":names,"environments":[{"environmentOrigin":origins,
                          "bodyPositions":pos,"bodyQuaternions":quat}],
                          "rootPos":root,"speed":float((vel[0]**2+vel[1]**2+vel[2]**2)**0.5),
                          "actionPeak":float(actions.abs().max().item()) if actions is not None else 0.0}), flush=True)
    emit_frame(0)
    with torch.inference_mode():
        while step < args.steps:
            if select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline()
                if line:
                    try:
                        command = json.loads(line)
                        kind = command.get('command')
                        if kind == 'pause': paused = True
                        elif kind == 'resume': paused = False
                        elif kind == 'step': paused = True; step_once = True
                        elif kind == 'reset': obs = wrapped.reset()[0]; step = 0; paused = True; step_once = False; emit_frame(0)
                        elif kind == 'frequency':
                            frequency = float(command['value']); cfg.test_frequency = frequency
                        print(json.dumps({'kind':'status','paused':paused,'frequency':frequency,'step':step}), flush=True)
                    except Exception as exc:
                        print(json.dumps({'kind':'error','message':f'控制命令无效: {exc}'}), flush=True)
            if paused and not step_once:
                time.sleep(.01)
                continue
            step += 1
            loop_started = time.perf_counter()
            actions = policy(obs)
            result = wrapped.step(actions)
            obs = result[0]
            emit_frame(step, actions)
            step_once = False
            time.sleep(max(0.0, 0.02 - (time.perf_counter() - loop_started)))
    env.close()
    print(json.dumps({"kind":"done","steps":args.steps}), flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"kind":"error","message":str(exc)}), flush=True)
        raise
