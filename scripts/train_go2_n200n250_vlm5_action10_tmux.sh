#!/usr/bin/env bash
set -euo pipefail

# 一键编排：阶段1 VLM(n200+n250, 5 epoch) -> 阶段2 Action(n200+n250, 10 epoch)
# 串行自动衔接；tmux 内启动；TensorBoard 只显示当前阶段数据。

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

STARVLA_PYTHON_ENV="${STARVLA_PYTHON_ENV:-/hdd4/MaTianran/rtc_starvla/envs/mtr_star}"
TENSORBOARD_BIN="${TENSORBOARD_BIN:-${STARVLA_PYTHON_ENV}/bin/tensorboard}"
TMUX_SESSION="${TMUX_SESSION:-go2_n200n250_vlm5_action10}"
TB_PORT="${TB_PORT:-6016}"
VISIBLE_GPUS="${VISIBLE_GPUS:-3,4}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/starVLA/config/training/starvla_go2_qwen3vl_waypoint_router.yaml}"
N200_ROOT="${N200_ROOT:-${REPO_ROOT}/datasets/liangzhu_0729_n200/lerobot_dataset}"
N250_ROOT="${N250_ROOT:-${REPO_ROOT}/datasets/liangzhu_0729_n250/lerobot_dataset}"
VLM_PRETRAINED="${VLM_PRETRAINED:-${REPO_ROOT}/results/Checkpoints/go2_n200_vlm_instruction_a40x2_b1_0802/final_model/pytorch_model.pt}"
STAGE1_EPOCHS=5
STAGE2_EPOCHS=10
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/results/launcher_logs}"
MARKER="/tmp/go2_tb_target_${TMUX_SESSION}.txt"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
CHECK_GPU_IDLE="${CHECK_GPU_IDLE:-1}"
DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK:-1}"
OFFLOAD_OPTIMIZER_DEVICE="${OFFLOAD_OPTIMIZER_DEVICE:-none}"
STAGE1_WARMUP="${STAGE1_WARMUP:-100}"
STAGE2_WARMUP="${STAGE2_WARMUP:-150}"

# 子 shell（tmux 窗口）需要的变量全部导出。
export REPO_ROOT STARVLA_PYTHON_ENV TENSORBOARD_BIN TMUX_SESSION TB_PORT
export VISIBLE_GPUS NUM_PROCESSES PER_DEVICE_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS
export TRAIN_CONFIG N200_ROOT N250_ROOT VLM_PRETRAINED
export STAGE1_EPOCHS STAGE2_EPOCHS LOG_DIR MARKER
export SAVE_INTERVAL EVAL_INTERVAL CHECK_GPU_IDLE DS_SKIP_CUDA_CHECK OFFLOAD_OPTIMIZER_DEVICE
export STAGE1_WARMUP STAGE2_WARMUP

for path in "${N200_ROOT}" "${N250_ROOT}" "${VLM_PRETRAINED}"; do
  [[ -e "${path}" ]] || {
    echo "missing path: ${path}" >&2
    exit 2
  }
done
if ! command -v "${TENSORBOARD_BIN}" >/dev/null 2>&1 && [[ ! -x "${TENSORBOARD_BIN}" ]]; then
  echo "tensorboard not found: ${TENSORBOARD_BIN}" >&2
  exit 2
fi
if ss -lnt 2>/dev/null | grep -q "[:.]${TB_PORT}[[:space:]]"; then
  echo "port ${TB_PORT} is already in use" >&2
  exit 2
fi
if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${TMUX_SESSION}" >&2
  exit 2
fi

PYTHON_BIN="${STARVLA_PYTHON_ENV}/bin/python"
read -r STAGE1_SAMPLES STAGE1_STEPS < <(
  PYTHONPATH="${REPO_ROOT}" "${PYTHON_BIN}" "${REPO_ROOT}/scripts/compute_go2_epoch_steps.py" \
    --config "${TRAIN_CONFIG}" \
    --dataset-root "${N200_ROOT},${N250_ROOT}" \
    --routes nav grasp place done recover \
    --world-size "${NUM_PROCESSES}" \
    --batch-size "${PER_DEVICE_BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --epochs "${STAGE1_EPOCHS}"
)
read -r STAGE2_SAMPLES STAGE2_STEPS < <(
  PYTHONPATH="${REPO_ROOT}" "${PYTHON_BIN}" "${REPO_ROOT}/scripts/compute_go2_epoch_steps.py" \
    --config "${TRAIN_CONFIG}" \
    --dataset-root "${N200_ROOT},${N250_ROOT}" \
    --routes nav grasp place \
    --world-size "${NUM_PROCESSES}" \
    --batch-size "${PER_DEVICE_BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --epochs "${STAGE2_EPOCHS}"
)
export STAGE1_SAMPLES STAGE1_STEPS STAGE2_SAMPLES STAGE2_STEPS

mkdir -p "${LOG_DIR}"
rm -f "${MARKER}"

# TensorBoard 窗口：跟随 marker 指向的当前阶段 logdir，阶段切换时重启。
TB_WATCHER='
last=""
while true; do
  target="$(cat "${MARKER}" 2>/dev/null || true)"
  if [[ -n "${target}" && "${target}" != "${last}" && -d "${target}" ]]; then
    pkill -f "tensorboard.*--port ${TB_PORT}" 2>/dev/null || true
    sleep 1
    nohup "${TENSORBOARD_BIN}" --logdir "${target}" --port "${TB_PORT}" --bind_all \
      > "${LOG_DIR}/tb_${TMUX_SESSION}.log" 2>&1 &
    last="${target}"
  elif [[ -z "${target}" ]]; then
    last=""
  fi
  sleep 3
done
'

CHAIN='
set -euo pipefail
cd "${REPO_ROOT}"
export PATH="${STARVLA_PYTHON_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1

STAGE1_RUN_ID="go2_n200n250_vlm_5ep_$(date +%m%d_%H%M%S)"
STAGE1_OUT="${REPO_ROOT}/results/Checkpoints/${STAGE1_RUN_ID}"
echo ">> stage1 vlm run_id=${STAGE1_RUN_ID} samples=${STAGE1_SAMPLES} steps=${STAGE1_STEPS}" | tee "${LOG_DIR}/${STAGE1_RUN_ID}.log"
mkdir -p "${STAGE1_OUT}/tensorboard"
echo "${STAGE1_OUT}/tensorboard" > "${MARKER}"

VISIBLE_GPUS="${VISIBLE_GPUS}" \
NUM_PROCESSES="${NUM_PROCESSES}" \
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE}" \
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS}" \
RUN_ID="${STAGE1_RUN_ID}" \
PRETRAINED_CHECKPOINT="${VLM_PRETRAINED}" \
MAX_TRAIN_STEPS="${STAGE1_STEPS}" \
WARMUP_STEPS="${STAGE1_WARMUP:-100}" \
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}" \
EVAL_INTERVAL="${EVAL_INTERVAL:-100}" \
CHECK_GPU_IDLE="${CHECK_GPU_IDLE:-1}" \
DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK:-1}" \
OFFLOAD_OPTIMIZER_DEVICE="${OFFLOAD_OPTIMIZER_DEVICE:-none}" \
STARVLA_PYTHON_ENV="${STARVLA_PYTHON_ENV}" \
TRAIN_CONFIG="${TRAIN_CONFIG}" \
  "${REPO_ROOT}/scripts/run_go2_staged_training.sh" vlm_instruction \
    --datasets.router_data.root "[\"${N200_ROOT}\",\"${N250_ROOT}\"]" \
    2>&1 | tee -a "${LOG_DIR}/${STAGE1_RUN_ID}.log"

[[ -f "${STAGE1_OUT}/final_model/pytorch_model.pt" ]] || {
  echo "stage1 final model missing: ${STAGE1_OUT}/final_model/pytorch_model.pt" >&2
  exit 1
}

STAGE2_RUN_ID="go2_n200n250_action_10ep_$(date +%m%d_%H%M%S)"
STAGE2_OUT="${REPO_ROOT}/results/Checkpoints/${STAGE2_RUN_ID}"
echo ">> stage2 action run_id=${STAGE2_RUN_ID} samples=${STAGE2_SAMPLES} steps=${STAGE2_STEPS}" | tee "${LOG_DIR}/${STAGE2_RUN_ID}.log"
mkdir -p "${STAGE2_OUT}/tensorboard"
echo "${STAGE2_OUT}/tensorboard" > "${MARKER}"

VISIBLE_GPUS="${VISIBLE_GPUS}" \
NUM_PROCESSES="${NUM_PROCESSES}" \
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE}" \
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS}" \
RUN_ID="${STAGE2_RUN_ID}" \
PRETRAINED_CHECKPOINT="${STAGE1_OUT}/final_model/pytorch_model.pt" \
MAX_TRAIN_STEPS="${STAGE2_STEPS}" \
WARMUP_STEPS="${STAGE2_WARMUP:-150}" \
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}" \
EVAL_INTERVAL="${EVAL_INTERVAL:-100}" \
CHECK_GPU_IDLE="${CHECK_GPU_IDLE:-1}" \
DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK:-1}" \
OFFLOAD_OPTIMIZER_DEVICE="${OFFLOAD_OPTIMIZER_DEVICE:-none}" \
STARVLA_PYTHON_ENV="${STARVLA_PYTHON_ENV}" \
TRAIN_CONFIG="${TRAIN_CONFIG}" \
  "${REPO_ROOT}/scripts/run_go2_staged_training.sh" action \
    --datasets.router_data.root "[\"${N200_ROOT}\",\"${N250_ROOT}\"]" \
    --datasets.router_data.include_routes "[nav,grasp,place]" \
    2>&1 | tee -a "${LOG_DIR}/${STAGE2_RUN_ID}.log"

rm -f "${MARKER}"
echo ">> ALL STAGES DONE: ${STAGE1_RUN_ID} -> ${STAGE2_RUN_ID}"
'

# 先建空会话并设置 remain-on-exit，再用 send-keys 启动命令，
# 避免窗口命令先退出导致窗口在设置选项前被销毁。
tmux new-session -d -s "${TMUX_SESSION}"
tmux set-option -w -t "${TMUX_SESSION}:0" remain-on-exit on
tmux rename-window -t "${TMUX_SESSION}:0" train
tmux send-keys -t "${TMUX_SESSION}:train" "bash -lc $(printf '%q' "${CHAIN}")" Enter
tmux new-window -t "${TMUX_SESSION}" -n tensorboard
tmux set-option -w -t "${TMUX_SESSION}:tensorboard" remain-on-exit on
tmux send-keys -t "${TMUX_SESSION}:tensorboard" "bash -lc $(printf '%q' "${TB_WATCHER}")" Enter

echo "tmux_session=${TMUX_SESSION}"
echo "gpu=${VISIBLE_GPUS}"
echo "stage1_samples=${STAGE1_SAMPLES} stage1_steps=${STAGE1_STEPS}"
echo "stage2_samples=${STAGE2_SAMPLES} stage2_steps=${STAGE2_STEPS}"
echo "stage1_run_id=go2_n200n250_vlm_5ep_*"
echo "stage2_run_id=go2_n200n250_action_10ep_*"
echo "tensorboard=http://127.0.0.1:${TB_PORT}"
echo "logs=${LOG_DIR}"
