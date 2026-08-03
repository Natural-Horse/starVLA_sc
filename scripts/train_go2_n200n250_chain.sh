#!/usr/bin/env bash
set -euo pipefail

# 阶段1 VLM(n200+n250, 5 epoch) -> 阶段2 Action(n200+n250, 10 epoch) 串行编排。
# 由 train_go2_n200n250_vlm5_action10_tmux.sh 在 tmux 内调用。

REPO_ROOT="${REPO_ROOT:-/hdd4/MaTianran/pct_workspace/starVLA_sc}"
cd "${REPO_ROOT}"

STARVLA_PYTHON_ENV="${STARVLA_PYTHON_ENV:-/hdd4/MaTianran/rtc_starvla/envs/mtr_star}"
VISIBLE_GPUS="${VISIBLE_GPUS:-3,4}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/starVLA/config/training/starvla_go2_qwen3vl_waypoint_router.yaml}"
N200_ROOT="${N200_ROOT:-${REPO_ROOT}/datasets/liangzhu_0729_n200/lerobot_dataset}"
N250_ROOT="${N250_ROOT:-${REPO_ROOT}/datasets/liangzhu_0729_n250/lerobot_dataset}"
STAGE1_ROOT="${STAGE1_ROOT:-${N250_ROOT}}"
STAGE2_ROOT="${STAGE2_ROOT:-${N200_ROOT},${N250_ROOT}}"
STAGE2_ROOT_LIST="[$(printf '"%s",' "${STAGE2_ROOT}" | sed 's/,$//' | sed "s|,|\",\"|g")]"
VLM_PRETRAINED="${VLM_PRETRAINED:-${REPO_ROOT}/results/Checkpoints/go2_n200_vlm_instruction_a40x2_b1_0802/final_model/pytorch_model.pt}"
STAGE1_STEPS="${STAGE1_STEPS:?set by orchestrator}"
STAGE2_STEPS="${STAGE2_STEPS:?set by orchestrator}"
STAGE1_WARMUP="${STAGE1_WARMUP:-100}"
STAGE2_WARMUP="${STAGE2_WARMUP:-150}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/results/launcher_logs}"
MARKER="${MARKER:?set by orchestrator}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
CHECK_GPU_IDLE="${CHECK_GPU_IDLE:-1}"
DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK:-1}"
OFFLOAD_OPTIMIZER_DEVICE="${OFFLOAD_OPTIMIZER_DEVICE:-none}"

export PATH="${STARVLA_PYTHON_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1

STAGE1_RUN_ID="go2_n250_vlm_1ep_$(date +%m%d_%H%M%S)"
STAGE1_OUT="${REPO_ROOT}/results/Checkpoints/${STAGE1_RUN_ID}"
echo ">> stage1 vlm run_id=${STAGE1_RUN_ID} steps=${STAGE1_STEPS}" | tee "${LOG_DIR}/${STAGE1_RUN_ID}.log"
mkdir -p "${STAGE1_OUT}/tensorboard"
echo "${STAGE1_OUT}/tensorboard" > "${MARKER}"

VISIBLE_GPUS="${VISIBLE_GPUS}" \
NUM_PROCESSES="${NUM_PROCESSES}" \
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE}" \
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS}" \
RUN_ID="${STAGE1_RUN_ID}" \
PRETRAINED_CHECKPOINT="${VLM_PRETRAINED}" \
MAX_TRAIN_STEPS="${STAGE1_STEPS}" \
WARMUP_STEPS="${STAGE1_WARMUP}" \
SAVE_INTERVAL="${SAVE_INTERVAL}" \
EVAL_INTERVAL="${EVAL_INTERVAL}" \
CHECK_GPU_IDLE="${CHECK_GPU_IDLE}" \
DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK}" \
OFFLOAD_OPTIMIZER_DEVICE="${OFFLOAD_OPTIMIZER_DEVICE}" \
STARVLA_PYTHON_ENV="${STARVLA_PYTHON_ENV}" \
TRAIN_CONFIG="${TRAIN_CONFIG}" \
  "${REPO_ROOT}/scripts/run_go2_staged_training.sh" vlm_instruction \
    --datasets.router_data.root "${STAGE1_ROOT}" \
    2>&1 | tee -a "${LOG_DIR}/${STAGE1_RUN_ID}.log"

[[ -f "${STAGE1_OUT}/final_model/pytorch_model.pt" ]] || {
  echo "stage1 final model missing: ${STAGE1_OUT}/final_model/pytorch_model.pt" >&2
  exit 1
}

STAGE2_RUN_ID="go2_n200n250_action_5ep_$(date +%m%d_%H%M%S)"
STAGE2_OUT="${REPO_ROOT}/results/Checkpoints/${STAGE2_RUN_ID}"
echo ">> stage2 action run_id=${STAGE2_RUN_ID} steps=${STAGE2_STEPS}" | tee "${LOG_DIR}/${STAGE2_RUN_ID}.log"
mkdir -p "${STAGE2_OUT}/tensorboard"
echo "${STAGE2_OUT}/tensorboard" > "${MARKER}"

VISIBLE_GPUS="${VISIBLE_GPUS}" \
NUM_PROCESSES="${NUM_PROCESSES}" \
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE}" \
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS}" \
RUN_ID="${STAGE2_RUN_ID}" \
PRETRAINED_CHECKPOINT="${STAGE1_OUT}/final_model/pytorch_model.pt" \
MAX_TRAIN_STEPS="${STAGE2_STEPS}" \
WARMUP_STEPS="${STAGE2_WARMUP}" \
SAVE_INTERVAL="${SAVE_INTERVAL}" \
EVAL_INTERVAL="${EVAL_INTERVAL}" \
CHECK_GPU_IDLE="${CHECK_GPU_IDLE}" \
DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK}" \
OFFLOAD_OPTIMIZER_DEVICE="${OFFLOAD_OPTIMIZER_DEVICE}" \
STARVLA_PYTHON_ENV="${STARVLA_PYTHON_ENV}" \
TRAIN_CONFIG="${TRAIN_CONFIG}" \
  "${REPO_ROOT}/scripts/run_go2_staged_training.sh" action \
    --datasets.router_data.root "${STAGE2_ROOT_LIST}" \
    --datasets.router_data.include_routes "[nav,grasp,place]" \
    2>&1 | tee -a "${LOG_DIR}/${STAGE2_RUN_ID}.log"

rm -f "${MARKER}"
echo ">> ALL STAGES DONE: ${STAGE1_RUN_ID} -> ${STAGE2_RUN_ID}"
