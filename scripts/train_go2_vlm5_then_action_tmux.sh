#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/starVLA/config/training/starvla_go2_qwen3vl_waypoint_router.yaml}"
STARVLA_PYTHON_ENV="${STARVLA_PYTHON_ENV:-/hdd4/MaTianran/rtc_starvla/envs/mtr_star}"
PYTHON_BIN="${PYTHON_BIN:-${STARVLA_PYTHON_ENV}/bin/python}"
TENSORBOARD_BIN="${TENSORBOARD_BIN:-${STARVLA_PYTHON_ENV}/bin/tensorboard}"
DATASET_ROOT="${DATASET_ROOT:-/hdd4/MaTianran/pct_workspace/starVLA_sc/datasets/liangzhu_0729_n250/lerobot_dataset}"

: "${VISIBLE_GPUS:?Set VISIBLE_GPUS after checking GPU ownership, for example 3,4}"
: "${PRETRAINED_VLM_CHECKPOINT:?Set PRETRAINED_VLM_CHECKPOINT to the previous final VLM checkpoint}"

VLM_EPOCHS="${VLM_EPOCHS:-5}"
NAV_EPOCHS="${NAV_EPOCHS:-1}"
MANIP_EPOCHS="${MANIP_EPOCHS:-1}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"

for value_name in VLM_EPOCHS NAV_EPOCHS MANIP_EPOCHS PER_DEVICE_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS; do
  value="${!value_name}"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${value_name} must be a positive integer, got: ${value}" >&2
    exit 2
  fi
done

[[ -x "${PYTHON_BIN}" ]] || {
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 2
}
[[ -x "${TENSORBOARD_BIN}" ]] || {
  echo "TensorBoard executable not found: ${TENSORBOARD_BIN}" >&2
  exit 2
}
[[ -f "${PRETRAINED_VLM_CHECKPOINT}" ]] || {
  echo "Previous VLM checkpoint not found: ${PRETRAINED_VLM_CHECKPOINT}" >&2
  exit 2
}
[[ -f "${DATASET_ROOT}/meta/info.json" ]] || {
  echo "LeRobot dataset not found: ${DATASET_ROOT}" >&2
  exit 2
}

IFS=',' read -r -a GPU_IDS <<< "${VISIBLE_GPUS}"
NUM_PROCESSES="${NUM_PROCESSES:-${#GPU_IDS[@]}}"
if (( NUM_PROCESSES != ${#GPU_IDS[@]} )); then
  echo "NUM_PROCESSES=${NUM_PROCESSES} must match VISIBLE_GPUS count=${#GPU_IDS[@]}" >&2
  exit 2
fi

count_stage() {
  local epochs="$1"
  shift
  PYTHONPATH="${REPO_ROOT}" "${PYTHON_BIN}" "${REPO_ROOT}/scripts/compute_go2_epoch_steps.py" \
    --config "${TRAIN_CONFIG}" \
    --dataset-root "${DATASET_ROOT}" \
    --routes "$@" \
    --world-size "${NUM_PROCESSES}" \
    --batch-size "${PER_DEVICE_BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --epochs "${epochs}"
}

read -r VLM_SAMPLES VLM_STEPS < <(count_stage "${VLM_EPOCHS}" nav grasp place done recover)
read -r NAV_SAMPLES NAV_STEPS < <(count_stage "${NAV_EPOCHS}" nav)
read -r MANIP_SAMPLES MANIP_STEPS < <(count_stage "${MANIP_EPOCHS}" grasp place)

normalize_warmup_steps() {
  local requested="$1"
  local total_steps="$2"
  if (( requested < total_steps )); then
    echo "${requested}"
    return
  fi
  local fallback=$(( total_steps / 10 ))
  (( fallback >= total_steps )) && fallback=$(( total_steps - 1 ))
  (( fallback < 0 )) && fallback=0
  echo "warmup ${requested} must be below ${total_steps}; using ${fallback}" >&2
  echo "${fallback}"
}

VLM_WARMUP_STEPS="$(normalize_warmup_steps "${VLM_WARMUP_STEPS:-100}" "${VLM_STEPS}")"
NAV_WARMUP_STEPS="$(normalize_warmup_steps "${NAV_WARMUP_STEPS:-150}" "${NAV_STEPS}")"
MANIP_WARMUP_STEPS="$(normalize_warmup_steps "${MANIP_WARMUP_STEPS:-100}" "${MANIP_STEPS}")"

BASE_RUN_ID="${BASE_RUN_ID:-go2_n250_vlm5_then_action_$(date +%m%d_%H%M%S)}"
VLM_RUN_ID="${BASE_RUN_ID}_vlm"
NAV_RUN_ID="${BASE_RUN_ID}_nav"
MANIP_RUN_ID="${BASE_RUN_ID}_pick_place"
TMUX_SESSION="${TMUX_SESSION:-${BASE_RUN_ID}}"
TENSORBOARD_PORT="${TENSORBOARD_PORT:-6016}"
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
  OFFLOAD_OPTIMIZER_DEVICE="${OFFLOAD_OPTIMIZER_DEVICE:-none}"
  STARVLA_PYTHON_ENV="${STARVLA_PYTHON_ENV}"
  TRAIN_CONFIG="${TRAIN_CONFIG}"
)
dataset_args=(--datasets.router_data.root "${DATASET_ROOT}")
optimizer_args=(--trainer.gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}")

printf -v VLM_COMMAND '%q ' "${common_env[@]}" RUN_ID="${VLM_RUN_ID}" \
  MAX_TRAIN_STEPS="${VLM_STEPS}" WARMUP_STEPS="${VLM_WARMUP_STEPS}" \
  PRETRAINED_CHECKPOINT="${PRETRAINED_VLM_CHECKPOINT}" \
  "${REPO_ROOT}/scripts/run_go2_staged_training.sh" vlm_instruction \
  "${optimizer_args[@]}" "${dataset_args[@]}"
VLM_CHECKPOINT="${RUN_ROOT}/${VLM_RUN_ID}/final_model/pytorch_model.pt"

printf -v NAV_COMMAND '%q ' "${common_env[@]}" RUN_ID="${NAV_RUN_ID}" \
  MAX_TRAIN_STEPS="${NAV_STEPS}" WARMUP_STEPS="${NAV_WARMUP_STEPS}" \
  PRETRAINED_CHECKPOINT="${VLM_CHECKPOINT}" \
  "${REPO_ROOT}/scripts/run_go2_staged_training.sh" action \
  "${optimizer_args[@]}" "${dataset_args[@]}"
NAV_CHECKPOINT="${RUN_ROOT}/${NAV_RUN_ID}/final_model/pytorch_model.pt"

printf -v MANIP_COMMAND '%q ' "${common_env[@]}" RUN_ID="${MANIP_RUN_ID}" \
  MAX_TRAIN_STEPS="${MANIP_STEPS}" WARMUP_STEPS="${MANIP_WARMUP_STEPS}" \
  PRETRAINED_CHECKPOINT="${NAV_CHECKPOINT}" \
  "${REPO_ROOT}/scripts/run_go2_staged_training.sh" manip \
  "${optimizer_args[@]}" "${dataset_args[@]}"

printf -v TRAIN_SHELL \
  'set -euo pipefail; cd %q; echo %q; %s 2>&1 | tee %q; test -f %q; echo %q; %s 2>&1 | tee %q; test -f %q; echo %q; %s 2>&1 | tee %q' \
  "${REPO_ROOT}" \
  "stage=vlm_instruction epochs=${VLM_EPOCHS} samples_per_epoch=${VLM_SAMPLES} optimizer_steps=${VLM_STEPS}" \
  "${VLM_COMMAND}" "${LOG_DIR}/vlm.log" "${VLM_CHECKPOINT}" \
  "stage=nav epochs=${NAV_EPOCHS} samples_per_epoch=${NAV_SAMPLES} optimizer_steps=${NAV_STEPS}" \
  "${NAV_COMMAND}" "${LOG_DIR}/nav.log" "${NAV_CHECKPOINT}" \
  "stage=pick_place epochs=${MANIP_EPOCHS} samples_per_epoch=${MANIP_SAMPLES} optimizer_steps=${MANIP_STEPS}" \
  "${MANIP_COMMAND}" "${LOG_DIR}/pick_place.log"

TB_LOGDIR_SPEC="vlm:${RUN_ROOT}/${VLM_RUN_ID}/tensorboard,nav:${RUN_ROOT}/${NAV_RUN_ID}/tensorboard,pick_place:${RUN_ROOT}/${MANIP_RUN_ID}/tensorboard"
printf -v TB_SHELL 'exec %q --logdir_spec %q --port %q --bind_all' \
  "${TENSORBOARD_BIN}" "${TB_LOGDIR_SPEC}" "${TENSORBOARD_PORT}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "vlm=${VLM_SAMPLES} samples/epoch x ${VLM_EPOCHS} epochs = ${VLM_STEPS} optimizer steps"
  echo "nav=${NAV_SAMPLES} samples/epoch x ${NAV_EPOCHS} epochs = ${NAV_STEPS} optimizer steps"
  echo "pick_place=${MANIP_SAMPLES} samples/epoch x ${MANIP_EPOCHS} epochs = ${MANIP_STEPS} optimizer steps"
  echo "train_command=${TRAIN_SHELL}"
  echo "tensorboard_command=${TB_SHELL}"
  exit 0
fi

tmux new-session -d -s "${TMUX_SESSION}" -n train "bash -lc $(printf '%q' "${TRAIN_SHELL}")"
tmux set-option -w -t "${TMUX_SESSION}:train" remain-on-exit on
tmux new-window -t "${TMUX_SESSION}" -n tensorboard "bash -lc $(printf '%q' "${TB_SHELL}")"

echo "tmux_session=${TMUX_SESSION}"
echo "vlm=${VLM_SAMPLES} samples/epoch x ${VLM_EPOCHS} epochs = ${VLM_STEPS} optimizer steps"
echo "nav=${NAV_SAMPLES} samples/epoch x ${NAV_EPOCHS} epochs = ${NAV_STEPS} optimizer steps"
echo "pick_place=${MANIP_SAMPLES} samples/epoch x ${MANIP_EPOCHS} epochs = ${MANIP_STEPS} optimizer steps"
echo "logs=${LOG_DIR}"
echo "tensorboard_port=${TENSORBOARD_PORT}"
