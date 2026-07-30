#!/usr/bin/env bash
set -euo pipefail

cd /diff/wallx_workspace/starVLA

DELAY_SECONDS="${DELAY_SECONDS:-10800}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-600}"
VISIBLE_GPUS="${VISIBLE_GPUS:-0,1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29534}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
RUN_ID="${RUN_ID:-qwenpi_wallx_stage2_flow_from_fast_custom_tokenizer_h24_bbox_detach_action_grad}"

ACCELERATE=/diff/wallx_workspace/miniconda3/envs/starvla/bin/accelerate
DEEPSPEED_CONFIG=starVLA/config/deepseeds/deepspeed_zero3.yaml
TRAIN_SCRIPT=starVLA/training/train_starvla_cotrain_router.py
TRAIN_CONFIG=starVLA/config/training/starvla_cotrain_wallx_qwenpi_router.yaml
BASE_VLM=/diff/wallx_workspace/wallx_data_ckp/checkpoints/Qwen/Qwen3-VL-4B-Instruct-ActionRouter
FAST_TOKENIZER=/diff/wallx_workspace/wallx_data_ckp/checkpoints/fast_tokenizers/wallx_ego_h24_v2048
STAGE1_CKPT=/diff/wallx_workspace/starVLA/results/Checkpoints/qwenpi_wallx_fast_custom_tokenizer_h24_bbox/final_model/pytorch_model.pt

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

log "Sleeping ${DELAY_SECONDS}s before checking Stage1 custom-tokenizer FAST checkpoint."
sleep "${DELAY_SECONDS}"

until [[ -f "${STAGE1_CKPT}" ]]; do
  log "Stage1 checkpoint not ready: ${STAGE1_CKPT}"
  log "Waiting another ${CHECK_INTERVAL_SECONDS}s."
  sleep "${CHECK_INTERVAL_SECONDS}"
done

log "Stage1 checkpoint is ready. Starting detached-gradient Stage2 flow training."
log "Checkpoint: ${STAGE1_CKPT}"
log "Run ID: ${RUN_ID}"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS}" \
"${ACCELERATE}" launch \
  --config_file "${DEEPSPEED_CONFIG}" \
  --num_processes 2 \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  "${TRAIN_SCRIPT}" \
  --config_yaml "${TRAIN_CONFIG}" \
  --run_id "${RUN_ID}" \
  --framework.qwenvl.base_vlm "${BASE_VLM}" \
  --framework.qwenvl.special_tokens.policy strict \
  --framework.qwenvl.special_tokens.require_fast_action_tokens true \
  --framework.router.action_supervision flow_matching \
  --framework.router.action_loss_grad_to_vlm false \
  --framework.action_tokenizer.path "${FAST_TOKENIZER}" \
  --framework.action_tokenizer.token_count 2048 \
  --trainer.pretrained_checkpoint "${STAGE1_CKPT}" \
  --trainer.reload_modules qwen_vl_interface \
  --framework.qwenvl.freeze false \
  --framework.action_model.freeze false \
  --datasets.router_data.action_horizon 24 \
  --framework.action_model.action_horizon 24 \
  --framework.action_model.future_action_window_size 23 \
  --datasets.router_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}"
