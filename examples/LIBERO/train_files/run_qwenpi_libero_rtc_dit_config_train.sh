#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=${WANDB_MODE:-offline}
export NCCL_BLOCKING_WAIT=${NCCL_BLOCKING_WAIT:-1}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-10000}
export NCCL_SOCKET_TIMEOUT_MS=${NCCL_SOCKET_TIMEOUT_MS:-360000}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export PYTHONPATH=${PYTHONPATH:-/diff/wallx_workspace/khl_179/envs/mtr_star/lib/python3.10/site-packages:.}

config_yaml=${CONFIG_YAML:-./examples/LIBERO/train_files/starvla_qwenpi_libero_rtc_dit_cuda1.yaml}
python_bin=${PYTHON_BIN:-python3}

mkdir -p ./RTC_log/Checkpoints

exec "${python_bin}" -m accelerate.commands.launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2_1gpu.yaml \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}"
