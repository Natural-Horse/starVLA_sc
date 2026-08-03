#!/usr/bin/env bash
set -euo pipefail

# TensorBoard watcher：跟随 marker 文件指向的当前阶段 logdir。
# marker 内容变化时杀掉旧 TB，并只对新 logdir 启动 tensorboard。

MARKER="${MARKER:?set by orchestrator}"
TB_PORT="${TB_PORT:-6016}"
TENSORBOARD_BIN="${TENSORBOARD_BIN:-tensorboard}"
LOG_DIR="${LOG_DIR:-/tmp}"
TMUX_SESSION="${TMUX_SESSION:-go2}"
TB_LOG="${LOG_DIR}/tb_${TMUX_SESSION}.log"

last=""
while true; do
  target="$(cat "${MARKER}" 2>/dev/null || true)"
  if [[ -n "${target}" && "${target}" != "${last}" && -d "${target}" ]]; then
    pkill -f "tensorboard.*--port ${TB_PORT}" 2>/dev/null || true
    sleep 1
    nohup "${TENSORBOARD_BIN}" --logdir "${target}" --port "${TB_PORT}" --bind_all \
      > "${TB_LOG}" 2>&1 &
    last="${target}"
  elif [[ -z "${target}" ]]; then
    last=""
  fi
  sleep 3
done
