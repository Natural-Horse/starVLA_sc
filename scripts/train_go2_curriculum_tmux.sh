#!/usr/bin/env bash
set -euo pipefail

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

count_stage() {
  PYTHONPATH="${REPO_ROOT}" "${PYTHON_BIN}" "${REPO_ROOT}/scripts/compute_go2_epoch_steps.py" \
    --config "${TRAIN_CONFIG}" \
    "${count_root_args[@]}" \
    --routes "$@" \
    --world-size "${NUM_PROCESSES}" \
    --batch-size "${PER_DEVICE_BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
}

read -r VLM_SAMPLES VLM_STEPS < <(count_stage nav grasp place done recover)
read -r NAV_SAMPLES NAV_STEPS < <(count_stage nav)
read -r MANIP_SAMPLES MANIP_STEPS < <(count_stage grasp place)

BASE_RUN_ID="${BASE_RUN_ID:-go2_n200_curriculum_$(date +%m%d_%H%M%S)}"
VLM_RUN_ID="${BASE_RUN_ID}_vlm"
NAV_RUN_ID="${BASE_RUN_ID}_nav"
MANIP_RUN_ID="${BASE_RUN_ID}_pick_place"
TMUX_SESSION="${TMUX_SESSION:-${BASE_RUN_ID}}"
TENSORBOARD_PORT="${TENSORBOARD_PORT:-6006}"
RUN_ROOT="${REPO_ROOT}/results/Checkpoints"
LOG_DIR="${REPO_ROOT}/results/launcher_logs/${BASE_RUN_ID}"
mkdir -p "${LOG_DIR}" "${RUN_ROOT}"

if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${TMUX_SESSION}" >&2
  exit 2
fi
for run_id in "${VLM_RUN_ID}" "${NAV_RUN_ID}" "${MANIP_RUN_ID}"; do
  if [[ -e "${RUN_ROOT}/${run_id}" ]]; then
    echo "Output directory already exists: ${RUN_ROOT}/${run_id}" >&2
    exit 2
  fi
done

common_env=(
  env
  VISIBLE_GPUS="${VISIBLE_GPUS}"
  NUM_PROCESSES="${NUM_PROCESSES}"
  PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE}"
  SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
  CHECK_GPU_IDLE="${CHECK_GPU_IDLE:-1}"
  DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK:-1}"
  STARVLA_PYTHON_ENV="${STARVLA_PYTHON_ENV}"
  TRAIN_CONFIG="${TRAIN_CONFIG}"
)
dataset_args=()
if [[ -n "${DATASET_ROOT:-}" ]]; then
  dataset_args=(--datasets.router_data.root "${DATASET_ROOT}")
fi

printf -v VLM_COMMAND '%q ' "${common_env[@]}" RUN_ID="${VLM_RUN_ID}" \
  MAX_TRAIN_STEPS="${VLM_STEPS}" WARMUP_STEPS="${VLM_WARMUP_STEPS:-100}" \
  "${REPO_ROOT}/scripts/run_go2_staged_training.sh" vlm \
  --trainer.gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" "${dataset_args[@]}"
VLM_CHECKPOINT="${RUN_ROOT}/${VLM_RUN_ID}/final_model/pytorch_model.pt"

printf -v NAV_COMMAND '%q ' "${common_env[@]}" RUN_ID="${NAV_RUN_ID}" \
  MAX_TRAIN_STEPS="${NAV_STEPS}" WARMUP_STEPS="${NAV_WARMUP_STEPS:-150}" \
  PRETRAINED_CHECKPOINT="${VLM_CHECKPOINT}" \
  "${REPO_ROOT}/scripts/run_go2_staged_training.sh" action \
  --trainer.gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" "${dataset_args[@]}"
NAV_CHECKPOINT="${RUN_ROOT}/${NAV_RUN_ID}/final_model/pytorch_model.pt"

printf -v MANIP_COMMAND '%q ' "${common_env[@]}" RUN_ID="${MANIP_RUN_ID}" \
  MAX_TRAIN_STEPS="${MANIP_STEPS}" WARMUP_STEPS="${MANIP_WARMUP_STEPS:-100}" \
  PRETRAINED_CHECKPOINT="${NAV_CHECKPOINT}" \
  "${REPO_ROOT}/scripts/run_go2_staged_training.sh" manip \
  --trainer.gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" "${dataset_args[@]}"

printf -v TRAIN_SHELL \
  'set -euo pipefail; cd %q; echo %q; %s 2>&1 | tee %q; test -f %q; echo %q; %s 2>&1 | tee %q; test -f %q; echo %q; %s 2>&1 | tee %q' \
  "${REPO_ROOT}" \
  "stage=vlm samples=${VLM_SAMPLES} optimizer_steps=${VLM_STEPS}" \
  "${VLM_COMMAND}" "${LOG_DIR}/vlm.log" "${VLM_CHECKPOINT}" \
  "stage=nav samples=${NAV_SAMPLES} optimizer_steps=${NAV_STEPS}" \
  "${NAV_COMMAND}" "${LOG_DIR}/nav.log" "${NAV_CHECKPOINT}" \
  "stage=pick_place samples=${MANIP_SAMPLES} optimizer_steps=${MANIP_STEPS}" \
  "${MANIP_COMMAND}" "${LOG_DIR}/pick_place.log"
printf -v TB_SHELL 'exec %q --logdir %q --port %q --bind_all' \
  "${TENSORBOARD_BIN}" "${RUN_ROOT}" "${TENSORBOARD_PORT}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "train_command=${TRAIN_SHELL}"
  echo "tensorboard_command=${TB_SHELL}"
  exit 0
fi

tmux new-session -d -s "${TMUX_SESSION}" -n train "bash -lc $(printf '%q' "${TRAIN_SHELL}")"
tmux set-option -w -t "${TMUX_SESSION}:train" remain-on-exit on
tmux new-window -t "${TMUX_SESSION}" -n tensorboard "bash -lc $(printf '%q' "${TB_SHELL}")"

echo "tmux_session=${TMUX_SESSION}"
echo "vlm=${VLM_SAMPLES} samples/${VLM_STEPS} steps"
echo "nav=${NAV_SAMPLES} samples/${NAV_STEPS} steps"
echo "pick_place=${MANIP_SAMPLES} samples/${MANIP_STEPS} steps"
echo "logs=${LOG_DIR}"
echo "tensorboard_port=${TENSORBOARD_PORT}"
