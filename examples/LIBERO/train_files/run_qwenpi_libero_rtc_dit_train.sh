#!/usr/bin/env bash
set -euo pipefail

export RTC_ENABLE=${RTC_ENABLE:-true}
export RTC_SIMULATED_DELAY=${RTC_SIMULATED_DELAY:-5}
export RTC_CONDITIONING_MODE=${RTC_CONDITIONING_MODE:-dit_token}
export DATA_MIX=${DATA_MIX:-libero_object}
export ACTION_MODE=${ACTION_MODE:-abs}
export EPOCHS=${EPOCHS:-50}
export MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-0}
export ENABLE_TENSORBOARD=${ENABLE_TENSORBOARD:-true}
export RUN_ID=${RUN_ID:-qwenpi_libero_object_qwen3vl4b_rtc_dit_delay${RTC_SIMULATED_DELAY}_50ep}

exec "$(dirname "$0")/run_qwenpi_libero_train.sh"
