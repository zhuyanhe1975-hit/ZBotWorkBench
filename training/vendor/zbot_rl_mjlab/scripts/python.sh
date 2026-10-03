#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MJLAB_ROOT="${MJLAB_ROOT:-/home/yhzhu/AI/mjlab}"
PYTHON_BIN="${ZBOT_PYTHON:-}"
if [[ -z "$PYTHON_BIN" && -x "$MJLAB_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$MJLAB_ROOT/.venv/bin/python"
fi
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "Python was not found. Set ZBOT_PYTHON to your Python executable." >&2
    exit 127
  fi
fi
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
if [[ -z "${MUJOCO_GL:-}" && "${OSTYPE:-}" != msys* && "${OSTYPE:-}" != cygwin* && "${OSTYPE:-}" != win32* ]]; then
  export MUJOCO_GL=egl
fi
if [[ -z "${XDG_CACHE_HOME:-}" ]]; then
  if [[ "${OSTYPE:-}" == msys* || "${OSTYPE:-}" == cygwin* || "${OSTYPE:-}" == win32* ]]; then
    export XDG_CACHE_HOME="$PROJECT_DIR/.cache"
  else
    export XDG_CACHE_HOME="/tmp/zbot-cache"
  fi
fi
cd "$PROJECT_DIR"
exec "$PYTHON_BIN" "$@"
