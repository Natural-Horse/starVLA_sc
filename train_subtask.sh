#!/bin/bash

cd /diff/wallx_workspace/starVLA

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0

# 临时创建 libcuda.so 符号链接（不需要 sudo）
LIBCUDA_LINK=/tmp/starvla_libcuda/libcuda.so
if [ ! -L "$LIBCUDA_LINK" ]; then
    mkdir -p /tmp/starvla_libcuda
    ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$LIBCUDA_LINK"
fi
export LIBRARY_PATH=/tmp/starvla_libcuda:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

/diff/wallx_workspace/miniconda3/envs/starvla/bin/accelerate launch \
  --config_file starVLA/config/deepseeds/zero2.yaml \
  --num_processes 1 \
  --main_process_port 29523 \
  starVLA/training/train_starvla_subtask.py \
  --config_yaml starVLA/config/training/starvla_subtask_wallx.yaml
