#!/usr/bin/env bash
set -euo pipefail

# Entry point for Baidu AIHC custom training jobs.
#
# Set MJLAB_TASK to one of:
#   Mjlab-Velocity-Rough-Ultra-GameYaw-AMP-HIM-V12
#   Mjlab-Velocity-Rough-Ultra-GameYaw-AMP-HIM-V13

cd "${MJLAB_WORKDIR:-/app}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

MJLAB_TASK="${MJLAB_TASK:?Set MJLAB_TASK to the V12 or V13 task id}"
LOG_ROOT="${LOG_ROOT:-/workspace/logs/rsl_rl}"
GPU_IDS="${GPU_IDS:-all}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-20000}"

exec uv run train "${MJLAB_TASK}" \
  --log-root "${LOG_ROOT}" \
  --env.scene.num-envs "${NUM_ENVS}" \
  --agent.max-iterations "${MAX_ITERATIONS}" \
  --gpu-ids "${GPU_IDS}"
