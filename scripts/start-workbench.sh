#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="$ROOT_DIR/.training/runtime-logs"
PID_DIR="$ROOT_DIR/.training"
NPM_BIN="${NPM_BIN:-/usr/bin/npm}"
mkdir -p "$RUNTIME_DIR"
stop_one() {
  local name="$1"
  local pid_file="$PID_DIR/$name.pid"
  if [[ -f "$pid_file" ]]; then
    local pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      pkill -TERM -P "$pid" 2>/dev/null || true
    fi
    rm -f "$pid_file"
  fi
}
start_one() {
  local name="$1" log="$2"; shift 2
  local pid_file="$PID_DIR/$name.pid"
  cd "$ROOT_DIR"
  setsid env NODE_ENV=production ZBOT_TRAINING_PYTHON="${ZBOT_TRAINING_PYTHON:-$HOME/mjlab/.venv/bin/python}" "$NPM_BIN" "$@" >"$RUNTIME_DIR/$log" 2>&1 &
  echo $! >"$pid_file"
  echo "$name started (PID $(cat "$pid_file"))"
}
stop_one training
stop_one frontend
# Clean up descendants left by an earlier launcher whose PID file was lost.
pkill -TERM -f '[t]rainingServer.ts' 2>/dev/null || true
pkill -TERM -f '[v]ite --port=3000' 2>/dev/null || true
start_one training training.log run training
start_one frontend frontend.log run dev
echo "frontend: http://127.0.0.1:3000"
echo "training: http://127.0.0.1:${ZBOT_TRAINING_PORT:-8767}"
