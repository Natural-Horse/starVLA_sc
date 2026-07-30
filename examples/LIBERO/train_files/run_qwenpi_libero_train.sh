#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=${WANDB_MODE:-offline}

export NCCL_BLOCKING_WAIT=${NCCL_BLOCKING_WAIT:-1}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-10000}
export NCCL_SOCKET_TIMEOUT_MS=${NCCL_SOCKET_TIMEOUT_MS:-360000}

framework_name=${FRAMEWORK_NAME:-QwenPI}
base_vlm=${BASE_VLM:-/diff/wallx_workspace/khl_179/wallx_data_ckp/checkpoints/Qwen/Qwen3-VL-4B-Instruct}
config_yaml=${CONFIG_YAML:-./examples/LIBERO/train_files/starvla_qwenpi_libero.yaml}
libero_data_root=${LIBERO_DATA_ROOT:-playground/Datasets/LEROBOT_LIBERO_DATA}
data_mix=${DATA_MIX:-libero_object}
action_mode=${ACTION_MODE:-abs}
run_root_dir=${RUN_ROOT_DIR:-./RTC_log/Checkpoints}
run_id=${RUN_ID:-qwenpi_libero_qwen3vl4b_smoke}
rtc_enable=${RTC_ENABLE:-false}
rtc_simulated_delay=${RTC_SIMULATED_DELAY:-0}
rtc_conditioning_mode=${RTC_CONDITIONING_MODE:-action_encoder}
num_processes=${NUM_PROCESSES:-1}
main_process_port=${MAIN_PROCESS_PORT:-0}
per_device_batch_size=${PER_DEVICE_BATCH_SIZE:-2}
gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-1}
max_train_steps=${MAX_TRAIN_STEPS:-1000}
epochs=${EPOCHS:-100}
save_interval=${SAVE_INTERVAL:-500}
logging_frequency=${LOGGING_FREQUENCY:-10}
freeze_modules=${FREEZE_MODULES:-qwen_vl_interface}
enable_tensorboard=${ENABLE_TENSORBOARD:-false}
python_bin=${PYTHON_BIN:-python3}
export PYTHONPATH=${PYTHONPATH:-/diff/wallx_workspace/khl_179/envs/mtr_star/lib/python3.10/site-packages:.}
export GRADIENT_ACCUMULATION_STEPS="${gradient_accumulation_steps}"

output_dir="${run_root_dir}/${run_id}"
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

"${python_bin}" -m accelerate.commands.launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${num_processes}" \
  --main_process_port "${main_process_port}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name "${framework_name}" \
  --framework.rtc.enable "${rtc_enable}" \
  --framework.rtc.simulated_delay "${rtc_simulated_delay}" \
  --framework.rtc.conditioning_mode "${rtc_conditioning_mode}" \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --datasets.vla_data.data_root_dir "${libero_data_root}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.action_mode "${action_mode}" \
  --datasets.vla_data.per_device_batch_size "${per_device_batch_size}" \
  --trainer.freeze_modules "${freeze_modules}" \
  --trainer.epochs "${epochs}" \
  --trainer.max_train_steps "${max_train_steps}" \
  --trainer.gradient_accumulation_steps "${gradient_accumulation_steps}" \
  --trainer.enable_tensorboard "${enable_tensorboard}" \
  --trainer.save_interval "${save_interval}" \
  --trainer.logging_frequency "${logging_frequency}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project starVLA_Libero_QwenPI \
  --wandb_entity local
