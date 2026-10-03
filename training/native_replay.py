#!/usr/bin/env python3
import argparse, json, os, sys, time
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("ZBOT_MJLAB_ROOT", str(Path.home() / "zbot_rl_mjlab")))
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--frequency", type=float, default=0.4)
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    if not (root / "src").is_dir():
        raise RuntimeError(f"MJLab source not found: {root}")
    sys.path.insert(0, str(root / "src"))
    import torch
    from dataclasses import asdict
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    import zbot_rl_mjlab
    checkpoint = Path(args.checkpoint or (root / "logs/rsl_rl/zbot_walking/2026-10-03_18-40-37/model_599.pt")).resolve()
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
    with torch.inference_mode():
        for step in range(1, args.steps + 1):
            if step == 1:
                time.sleep(5)
            actions = policy(obs)
            result = wrapped.step(actions)
            obs = result[0]
            pos = robot.data.body_link_pos_w[0].detach().cpu().tolist()
            quat = robot.data.body_link_quat_w[0].detach().cpu().tolist()
            root = robot.data.root_link_pos_w[0].detach().cpu().tolist()
            vel = robot.data.root_link_lin_vel_w[0].detach().cpu().tolist()
            print(json.dumps({"kind":"frame","step":step,"time":step * 0.02,
                              "bodyNames":names,"environments":[{"environmentOrigin":origins,
                              "bodyPositions":pos,"bodyQuaternions":quat}],
                              "rootPos":root,"speed":float((vel[0]**2+vel[1]**2+vel[2]**2)**0.5),
                              "actionPeak":float(actions.abs().max().item())}), flush=True)
            time.sleep(0.04)
    env.close()
    print(json.dumps({"kind":"done","steps":args.steps}), flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"kind":"error","message":str(exc)}), flush=True)
        raise
