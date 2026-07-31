#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

STARVLA_PYTHON_ENV="${STARVLA_PYTHON_ENV:-/hdd4/MaTianran/rtc_starvla/envs/mtr_star}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${STARVLA_PYTHON_ENV}/bin/accelerate}"
TRAIN_CONFIG="${TRAIN_CONFIG:-starVLA/config/training/starvla_go2_qwen3vl_waypoint_router.yaml}"
VISIBLE_GPUS="${VISIBLE_GPUS:-1,2,3,4}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29541}"
WANDB_MODE="${WANDB_MODE:-offline}"

export CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS}"
export WANDB_MODE
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${ACCELERATE_BIN}" launch \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  --mixed_precision bf16 \
  starVLA/training/train_starvla_cotrain_router.py \
  --config_yaml "${TRAIN_CONFIG}" \
  "$@"
