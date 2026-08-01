#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  VISIBLE_GPUS=5,6 scripts/run_go2_staged_training.sh vlm
  VISIBLE_GPUS=5,6 PRETRAINED_CHECKPOINT=/path/to/stage1.pt scripts/run_go2_staged_training.sh action
  VISIBLE_GPUS=5,6 PRETRAINED_CHECKPOINT=/path/to/stage2.pt scripts/run_go2_staged_training.sh joint

Optional environment variables:
  RUN_ID, MAX_TRAIN_STEPS, WARMUP_STEPS, SAVE_INTERVAL, EVAL_INTERVAL
  PER_DEVICE_BATCH_SIZE, OFFLOAD_OPTIMIZER_DEVICE, TRAIN_CONFIG
EOF
}

STAGE="${1:-}"
if [[ -z "${STAGE}" || "${STAGE}" == "-h" || "${STAGE}" == "--help" ]]; then
  usage
  [[ -n "${STAGE}" ]] && exit 0 || exit 2
fi
shift

case "${STAGE}" in
  vlm)
    DEFAULT_STEPS=1000
    DEFAULT_WARMUP=100
    STAGE_ARGS=(
      --framework.qwenvl.freeze false
      --framework.action_model.freeze true
      --framework.router.action_loss_grad_to_vlm false
      --trainer.loss_scale.vlm 1.0
      --trainer.loss_scale.action 0.0
      --trainer.learning_rate.qwen_vl_interface 5.0e-6
      --trainer.skip_no_grad_batches false
    )
    ;;
  action)
    DEFAULT_STEPS=3000
    DEFAULT_WARMUP=150
    : "${PRETRAINED_CHECKPOINT:?Set PRETRAINED_CHECKPOINT to a VLM-stage checkpoint}"
    [[ -f "${PRETRAINED_CHECKPOINT}" ]] || {
      echo "Checkpoint does not exist: ${PRETRAINED_CHECKPOINT}" >&2
      exit 2
    }
    STAGE_ARGS=(
      --framework.qwenvl.freeze true
      --framework.action_model.freeze false
      --framework.router.action_loss_grad_to_vlm false
      --trainer.loss_scale.vlm 0.0
      --trainer.loss_scale.action 1.0
      --trainer.learning_rate.action_model 5.0e-6
      --trainer.pretrained_checkpoint "${PRETRAINED_CHECKPOINT}"
      --trainer.reload_modules null
      --trainer.skip_no_grad_batches true
    )
    ;;
  joint)
    DEFAULT_STEPS=1000
    DEFAULT_WARMUP=50
    : "${PRETRAINED_CHECKPOINT:?Set PRETRAINED_CHECKPOINT to an action-stage checkpoint}"
    [[ -f "${PRETRAINED_CHECKPOINT}" ]] || {
      echo "Checkpoint does not exist: ${PRETRAINED_CHECKPOINT}" >&2
      exit 2
    }
    STAGE_ARGS=(
      --framework.qwenvl.freeze false
      --framework.action_model.freeze false
      --framework.router.action_loss_grad_to_vlm true
      --trainer.loss_scale.vlm 0.25
      --trainer.loss_scale.action 1.0
      --trainer.learning_rate.qwen_vl_interface 5.0e-7
      --trainer.learning_rate.action_model 2.0e-6
      --trainer.pretrained_checkpoint "${PRETRAINED_CHECKPOINT}"
      --trainer.reload_modules null
      --trainer.skip_no_grad_batches false
    )
    ;;
  *)
    echo "Unknown stage: ${STAGE}" >&2
    usage >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-${DEFAULT_STEPS}}"
WARMUP_STEPS="${WARMUP_STEPS:-${DEFAULT_WARMUP}}"
SAVE_INTERVAL="${SAVE_INTERVAL:-300}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
RUN_ID="${RUN_ID:-go2_n200_${STAGE}_$(date +%m%d_%H%M)}"

if (( MAX_TRAIN_STEPS <= 0 )); then
  echo "MAX_TRAIN_STEPS must be positive" >&2
  exit 2
fi
if (( WARMUP_STEPS < 0 || WARMUP_STEPS >= MAX_TRAIN_STEPS )); then
  echo "WARMUP_STEPS must satisfy 0 <= WARMUP_STEPS < MAX_TRAIN_STEPS" >&2
  exit 2
fi

exec "${REPO_ROOT}/scripts/run_go2_waypoint_training.sh" \
  --run_id "${RUN_ID}" \
  --trainer.stage "${STAGE}" \
  --trainer.max_train_steps "${MAX_TRAIN_STEPS}" \
  --trainer.num_warmup_steps "${WARMUP_STEPS}" \
  --trainer.save_interval "${SAVE_INTERVAL}" \
  --trainer.eval_interval "${EVAL_INTERVAL}" \
  --trainer.enable_tensorboard true \
  "${STAGE_ARGS[@]}" \
  "$@"
