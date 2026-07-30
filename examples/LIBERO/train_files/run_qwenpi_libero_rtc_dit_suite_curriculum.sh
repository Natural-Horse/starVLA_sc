#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=${WANDB_MODE:-offline}
export NCCL_BLOCKING_WAIT=${NCCL_BLOCKING_WAIT:-1}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-10000}
export NCCL_SOCKET_TIMEOUT_MS=${NCCL_SOCKET_TIMEOUT_MS:-360000}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export PYTHONPATH=${PYTHONPATH:-/diff/wallx_workspace/khl_179/envs/mtr_star/lib/python3.10/site-packages:.}

base_vlm=${BASE_VLM:-/diff/wallx_workspace/khl_179/wallx_data_ckp/checkpoints/Qwen/Qwen3-VL-4B-Instruct}
config_yaml=${CONFIG_YAML:-./examples/LIBERO/train_files/starvla_qwenpi_libero.yaml}
libero_data_root=${LIBERO_DATA_ROOT:-playground/Datasets/LEROBOT_LIBERO_DATA}
run_root_dir=${RUN_ROOT_DIR:-./RTC_log/Checkpoints}
run_root_dir="$(realpath -m "${run_root_dir}")"
start_checkpoint=${START_CHECKPOINT:?START_CHECKPOINT must point to a checkpoint}
suites=${SUITES:-"libero_object"}
action_mode=${ACTION_MODE:-abs}

num_processes=${NUM_PROCESSES:-2}
main_process_port_base=${MAIN_PROCESS_PORT_BASE:-29630}
per_device_batch_size=${PER_DEVICE_BATCH_SIZE:-2}
gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-8}
export GRADIENT_ACCUMULATION_STEPS="${gradient_accumulation_steps}"
epochs=${EPOCHS:-50}
max_train_steps=${MAX_TRAIN_STEPS:-0}
save_interval=${SAVE_INTERVAL:-500}
logging_frequency=${LOGGING_FREQUENCY:-10}
rtc_simulated_delay=${RTC_SIMULATED_DELAY:-5}
current_tb_link=${CURRENT_TB_LINK:-${run_root_dir}/current_suite_tensorboard}

prev_checkpoint="${start_checkpoint}"
suite_index=0
for suite in ${suites}; do
  run_id="qwenpi_${suite}_qwen3vl4b_rtc_dit_delay${rtc_simulated_delay}_gradacc${gradient_accumulation_steps}_effective_50ep"
  output_dir="${run_root_dir}/${run_id}"
  mkdir -p "${output_dir}/tensorboard"
  ln -sfn "${output_dir}/tensorboard" "${current_tb_link}"
  cp "$0" "${output_dir}/"

  main_process_port=$((main_process_port_base + suite_index))
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${suite} from ${prev_checkpoint}"

  python3 -m accelerate.commands.launch \
    --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
    --num_processes "${num_processes}" \
    --main_process_port "${main_process_port}" \
    starVLA/training/train_starvla.py \
    --config_yaml "${config_yaml}" \
    --framework.name QwenPI \
    --framework.rtc.enable true \
    --framework.rtc.simulated_delay "${rtc_simulated_delay}" \
    --framework.rtc.conditioning_mode dit_token \
    --framework.qwenvl.base_vlm "${base_vlm}" \
    --datasets.vla_data.data_root_dir "${libero_data_root}" \
    --datasets.vla_data.data_mix "${suite}" \
    --datasets.vla_data.action_mode "${action_mode}" \
    --datasets.vla_data.per_device_batch_size "${per_device_batch_size}" \
    --trainer.freeze_modules qwen_vl_interface \
    --trainer.epochs "${epochs}" \
    --trainer.max_train_steps "${max_train_steps}" \
    --trainer.gradient_accumulation_steps "${gradient_accumulation_steps}" \
    --trainer.pretrained_checkpoint "${prev_checkpoint}" \
    --trainer.is_resume false \
    --trainer.enable_tensorboard true \
    --trainer.save_interval "${save_interval}" \
    --trainer.logging_frequency "${logging_frequency}" \
    --run_root_dir "${run_root_dir}" \
    --run_id "${run_id}" \
    --wandb_project starVLA_Libero_QwenPI \
    --wandb_entity local

  prev_checkpoint="$(python3 - "$output_dir" <<'PY'
import glob
import os
import re
import sys

output_dir = sys.argv[1]
final_pt = os.path.join(output_dir, "final_model", "pytorch_model.pt")
if os.path.exists(final_pt):
    print(final_pt)
    raise SystemExit

items = []
for path in glob.glob(os.path.join(output_dir, "checkpoints", "steps_*_pytorch_model.pt")):
    match = re.search(r"steps_(\d+)_pytorch_model\.pt$", path)
    if match:
        items.append((int(match.group(1)), path))
if not items:
    raise SystemExit(f"No checkpoint found under {output_dir}")
print(max(items)[1])
PY
)"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished ${suite}; next checkpoint ${prev_checkpoint}"
  suite_index=$((suite_index + 1))
done
