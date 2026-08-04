#!/usr/bin/env bash
set -euo pipefail

# 从 YAML 启动 Go2 StarVLA 远程推理服务（tmux 内，只监听回环）。
#
# 用法：
#   bash scripts/evaluation/start_vla_inference.sh \
#     --config configs/vla_eval/server.yaml

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

usage() {
  echo "用法: $0 --config <server.yaml> [--dry-run]" >&2
  exit 2
}

CONFIG=""
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      usage
      ;;
  esac
done
[[ -n "${CONFIG}" ]] || usage
CONFIG_PATH="$(cd "$(dirname "${CONFIG}")" && pwd)/$(basename "${CONFIG}")"
[[ -f "${CONFIG_PATH}" ]] || {
  echo "config yaml 不存在: ${CONFIG_PATH}" >&2
  exit 2
}

# 读取 yaml 字段（依赖 python + pyyaml；若缺 pyyaml 则用简单 grep 兜底）。
get_cfg() {
  local key="$1"
  python3 - "${CONFIG_PATH}" "${key}" <<'PYEOF'
import sys
try:
    import yaml
except ImportError:
    yaml = None
path, key = sys.argv[1], sys.argv[2]
if yaml is not None:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    node = data
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            sys.exit(0)
        node = node[part]
    value = node
    if value is None:
        sys.exit(0)
    if isinstance(value, bool):
        print("true" if value else "false")
    elif isinstance(value, (int, float)):
        print(value)
    else:
        print(str(value))
else:
    # 无 pyyaml 时只支持 server.checkpoint / server.gpu / server.port / server.tmux_session
    import re
    text = open(path, encoding="utf-8").read()
    for part in key.split("."):
        m = re.search(r"^\s*" + re.escape(part) + r"\s*:\s*(.+?)\s*$", text, re.M)
        if not m:
            sys.exit(0)
        text = m.group(1)
    print(text.strip().strip('"').strip("'"))
PYEOF
}

CHECKPOINT="$(get_cfg server.checkpoint)"
GPU="$(get_cfg server.gpu)"
PORT="$(get_cfg server.port)"
TMUX_SESSION="$(get_cfg server.tmux_session)"
LOG_DIR="$(get_cfg server.log_dir)"
PYTHON_BIN="$(get_cfg server.python_bin)"
DEVICE="$(get_cfg server.device)"
USE_BF16="$(get_cfg server.use_bf16)"

[[ -n "${CHECKPOINT}" ]] || {
  echo "config 缺少 server.checkpoint" >&2
  exit 2
}
[[ -f "${CHECKPOINT}" ]] || {
  echo "checkpoint 不存在: ${CHECKPOINT}" >&2
  exit 2
}
GPU="${GPU:-0}"
PORT="${PORT:-10093}"
TMUX_SESSION="${TMUX_SESSION:-go2_vla_eval_server}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/results/evaluation_logs}"
PYTHON_BIN="${PYTHON_BIN:-/hdd4/MaTianran/rtc_starvla/envs/mtr_star/bin/python}"
DEVICE="${DEVICE:-cuda}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "checkpoint=${CHECKPOINT}"
  echo "gpu=${GPU} port=${PORT} tmux=${TMUX_SESSION}"
  echo "python=${PYTHON_BIN} device=${DEVICE} use_bf16=${USE_BF16}"
  exit 0
fi

if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
  echo "tmux session 已存在: ${TMUX_SESSION}" >&2
  exit 2
fi
if nvidia-smi -i "${GPU}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; then
  echo "GPU ${GPU} 已有计算进程，拒绝启动推理服务。" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/${TMUX_SESSION}.log"
BF16_ARGS=()
if [[ "${USE_BF16}" == "false" ]]; then
  BF16_ARGS+=(--no-bf16)
fi

printf -v SERVER_COMMAND \
  'cd %q; exec env CUDA_VISIBLE_DEVICES=%q PYTHONDONTWRITEBYTECODE=1 %q -B -m deployment.go2_remote.server --checkpoint %q --host 127.0.0.1 --port %q --device %q %s 2>&1 | tee %q' \
  "${REPO_ROOT}" "${GPU}" "${PYTHON_BIN}" "${CHECKPOINT}" "${PORT}" "${DEVICE}" \
  "${BF16_ARGS[*]:-}" "${LOG_PATH}"

tmux new-session -d -s "${TMUX_SESSION}" -n server "bash -lc $(printf '%q' "${SERVER_COMMAND}")"
tmux set-option -w -t "${TMUX_SESSION}:server" remain-on-exit on

echo "tmux_session=${TMUX_SESSION}"
echo "listen=127.0.0.1:${PORT}"
echo "log=${LOG_PATH}"
echo "checkpoint=${CHECKPOINT}"
