#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/hdd4/MaTianran/rtc_starvla/envs/mtr_star/bin/python}"
CHECKPOINT="${CHECKPOINT:-}"
GPU="${GPU:-0}"
PORT="${PORT:-10093}"
TMUX_SESSION="${TMUX_SESSION:-go2_vla_eval_server}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/results/evaluation_logs}"

if [[ -z "${CHECKPOINT}" ]] || [[ ! -f "${CHECKPOINT}" ]]; then
  echo "请用 CHECKPOINT=/path/to/pytorch_model.pt 指定完整训练权重。" >&2
  exit 2
fi
if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
  echo "tmux session 已存在: ${TMUX_SESSION}" >&2
  exit 2
fi
if nvidia-smi -i "${GPU}" --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  echo "GPU ${GPU} 已有计算进程，拒绝启动推理服务。" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/${TMUX_SESSION}.log"
printf -v SERVER_COMMAND \
  'cd %q; exec env CUDA_VISIBLE_DEVICES=%q PYTHONDONTWRITEBYTECODE=1 %q -B -m deployment.go2_remote.server --checkpoint %q --host 127.0.0.1 --port %q 2>&1 | tee %q' \
  "${REPO_ROOT}" "${GPU}" "${PYTHON_BIN}" "${CHECKPOINT}" "${PORT}" "${LOG_PATH}"

tmux new-session -d -s "${TMUX_SESSION}" -n server "bash -lc $(printf '%q' "${SERVER_COMMAND}")"
tmux set-option -w -t "${TMUX_SESSION}:server" remain-on-exit on
echo "tmux_session=${TMUX_SESSION}"
echo "listen=127.0.0.1:${PORT}"
echo "log=${LOG_PATH}"
