#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

run_cmd=("$PROJECT_DIR/scripts/python.sh" play.py Mjlab-Zbot-6dof-Walking \
    --env.step-frequency-min 0.25 \
    --env.step-frequency-max 1.0 \
    --env.scene.num-envs 16)

# In WSL, give MuJoCo a short graceful-shutdown window, then force exit if
# the renderer/GPU cleanup hangs after Ctrl+C.
if command -v timeout >/dev/null 2>&1; then
  exec timeout --foreground --signal=INT --kill-after="${ZBOT_STOP_TIMEOUT:-5}s" "${ZBOT_RUN_TIMEOUT:-365d}" "${run_cmd[@]}"
else
  exec "${run_cmd[@]}"
fi
