#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

STARVLA_PYTHON_ENV="${STARVLA_PYTHON_ENV:-/hdd4/MaTianran/rtc_starvla/envs/mtr_star}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${STARVLA_PYTHON_ENV}/bin/accelerate}"
TRAIN_CONFIG="${TRAIN_CONFIG:-starVLA/config/training/starvla_go2_qwen3vl_waypoint_router.yaml}"
: "${VISIBLE_GPUS:?Set VISIBLE_GPUS explicitly after confirming the training server and idle GPU ownership}"
IFS=',' read -r -a GPU_IDS <<< "${VISIBLE_GPUS}"
EXPECTED_PROCESSES="${#GPU_IDS[@]}"
NUM_PROCESSES="${NUM_PROCESSES:-${EXPECTED_PROCESSES}}"
if [[ "${NUM_PROCESSES}" -ne "${EXPECTED_PROCESSES}" ]]; then
  echo "NUM_PROCESSES=${NUM_PROCESSES} must match VISIBLE_GPUS count=${EXPECTED_PROCESSES}" >&2
  exit 2
fi
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29541}"
WANDB_MODE="${WANDB_MODE:-offline}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
OFFLOAD_OPTIMIZER_DEVICE="${OFFLOAD_OPTIMIZER_DEVICE:-cpu}"
CHECK_GPU_IDLE="${CHECK_GPU_IDLE:-1}"

if [[ ! -x "${ACCELERATE_BIN}" ]]; then
  echo "accelerate executable not found: ${ACCELERATE_BIN}; set STARVLA_PYTHON_ENV or ACCELERATE_BIN" >&2
  exit 2
fi

if [[ "${CHECK_GPU_IDLE}" == "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  for gpu_id in "${GPU_IDS[@]}"; do
    gpu_id="${gpu_id//[[:space:]]/}"
    [[ "${gpu_id}" =~ ^[0-9]+$ ]] || {
      echo "Invalid GPU index in VISIBLE_GPUS: ${gpu_id}" >&2
      exit 2
    }
    gpu_pids="$(nvidia-smi --id="${gpu_id}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ -n "${gpu_pids//[[:space:]]/}" ]]; then
      echo "GPU ${gpu_id} already has compute process(es): ${gpu_pids//$'\n'/, }" >&2
      exit 2
    fi
  done
fi

export CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS}"
export WANDB_MODE
export PYTHONDONTWRITEBYTECODE=1
export PATH="${STARVLA_PYTHON_ENV}/bin:${PATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

exec "${ACCELERATE_BIN}" launch \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  --mixed_precision bf16 \
  starVLA/training/train_starvla_cotrain_router.py \
  --config_yaml "${TRAIN_CONFIG}" \
  --datasets.router_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --trainer.deepspeed.offload_optimizer_device "${OFFLOAD_OPTIMIZER_DEVICE}" \
  "$@"
