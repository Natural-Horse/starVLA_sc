#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:?Usage: launch_go2_stage_tmux.sh <vlm|action|manip>}"
shift

case "${STAGE}" in
  vlm)
    ROUTES=(nav grasp place done recover)
    DEFAULT_WARMUP=100
    ;;
  action)
    ROUTES=(nav)
    DEFAULT_WARMUP=150
    : "${PRETRAINED_CHECKPOINT:?Set PRETRAINED_CHECKPOINT to the VLM final checkpoint}"
    ;;
  manip)
    ROUTES=(grasp place)
    DEFAULT_WARMUP=100
    : "${PRETRAINED_CHECKPOINT:?Set PRETRAINED_CHECKPOINT to the NAV action final checkpoint}"
    ;;
  *)
    echo "Unknown stage: ${STAGE}" >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/starVLA/config/training/starvla_go2_qwen3vl_waypoint_router.yaml}"
STARVLA_PYTHON_ENV="${STARVLA_PYTHON_ENV:-/hdd4/MaTianran/rtc_starvla/envs/mtr_star}"
PYTHON_BIN="${PYTHON_BIN:-${STARVLA_PYTHON_ENV}/bin/python}"
TENSORBOARD_BIN="${TENSORBOARD_BIN:-${STARVLA_PYTHON_ENV}/bin/tensorboard}"
: "${VISIBLE_GPUS:?Set VISIBLE_GPUS after checking GPU ownership, for example 5,6}"

IFS=',' read -r -a GPU_IDS <<< "${VISIBLE_GPUS}"
NUM_PROCESSES="${NUM_PROCESSES:-${#GPU_IDS[@]}}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
count_root_args=()
if [[ -n "${DATASET_ROOT:-}" ]]; then
  count_root_args=(--dataset-root "${DATASET_ROOT}")
fi
read -r SAMPLE_COUNT MAX_TRAIN_STEPS < <(
  PYTHONPATH="${REPO_ROOT}" "${PYTHON_BIN}" "${REPO_ROOT}/scripts/compute_go2_epoch_steps.py" \
    --config "${TRAIN_CONFIG}" \
    "${count_root_args[@]}" \
    --routes "${ROUTES[@]}" \
    --world-size "${NUM_PROCESSES}" \
    --batch-size "${PER_DEVICE_BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
)

WARMUP_STEPS="${WARMUP_STEPS:-${DEFAULT_WARMUP}}"
if (( WARMUP_STEPS >= MAX_TRAIN_STEPS )); then
  WARMUP_STEPS=$((MAX_TRAIN_STEPS / 10))
fi

RUN_ID="${RUN_ID:-go2_n200_${STAGE}_epoch1_$(date +%m%d_%H%M%S)}"
TMUX_SESSION="${TMUX_SESSION:-${RUN_ID}}"
TENSORBOARD_PORT="${TENSORBOARD_PORT:-6006}"
OUTPUT_DIR="${REPO_ROOT}/results/Checkpoints/${RUN_ID}"
LOG_DIR="${REPO_ROOT}/results/launcher_logs"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"
mkdir -p "${LOG_DIR}"

if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${TMUX_SESSION}" >&2
  exit 2
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Output directory already exists: ${OUTPUT_DIR}" >&2
  exit 2
fi

RUN_ENV=(
  env
  VISIBLE_GPUS="${VISIBLE_GPUS}"
  NUM_PROCESSES="${NUM_PROCESSES}"
  PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE}"
  RUN_ID="${RUN_ID}"
  MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS}"
  WARMUP_STEPS="${WARMUP_STEPS}"
  SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
  CHECK_GPU_IDLE="${CHECK_GPU_IDLE:-1}"
  DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK:-1}"
  OFFLOAD_OPTIMIZER_DEVICE="${OFFLOAD_OPTIMIZER_DEVICE:-cpu}"
  STARVLA_PYTHON_ENV="${STARVLA_PYTHON_ENV}"
  TRAIN_CONFIG="${TRAIN_CONFIG}"
)
if [[ "${STAGE}" != "vlm" ]]; then
  RUN_ENV+=(PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT}")
fi
EXTRA_ARGS=("$@")
if [[ -n "${DATASET_ROOT:-}" ]]; then
  EXTRA_ARGS+=(--datasets.router_data.root "${DATASET_ROOT}")
fi
printf -v RUN_COMMAND '%q ' "${RUN_ENV[@]}" "${REPO_ROOT}/scripts/run_go2_staged_training.sh" "${STAGE}" \
  --trainer.gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" "${EXTRA_ARGS[@]}"
printf -v TRAIN_SHELL 'set -o pipefail; cd %q; echo %q; %s 2>&1 | tee %q' \
  "${REPO_ROOT}" \
  "stage=${STAGE} samples=${SAMPLE_COUNT} optimizer_steps=${MAX_TRAIN_STEPS}" \
  "${RUN_COMMAND}" \
  "${LOG_FILE}"
printf -v TB_SHELL 'while [[ ! -d %q ]]; do sleep 5; done; exec %q --logdir %q --port %q --bind_all' \
  "${OUTPUT_DIR}/tensorboard" "${TENSORBOARD_BIN}" "${OUTPUT_DIR}/tensorboard" "${TENSORBOARD_PORT}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "train_command=${TRAIN_SHELL}"
  echo "tensorboard_command=${TB_SHELL}"
  exit 0
fi

tmux new-session -d -s "${TMUX_SESSION}" -n train "bash -lc $(printf '%q' "${TRAIN_SHELL}")"
tmux set-option -w -t "${TMUX_SESSION}:train" remain-on-exit on
tmux new-window -t "${TMUX_SESSION}" -n tensorboard "bash -lc $(printf '%q' "${TB_SHELL}")"

echo "tmux_session=${TMUX_SESSION}"
echo "stage=${STAGE} samples=${SAMPLE_COUNT} optimizer_steps=${MAX_TRAIN_STEPS}"
echo "log=${LOG_FILE}"
echo "tensorboard_port=${TENSORBOARD_PORT}"
