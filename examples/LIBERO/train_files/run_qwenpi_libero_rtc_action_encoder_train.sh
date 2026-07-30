#!/usr/bin/env bash
set -euo pipefail

export RTC_ENABLE=${RTC_ENABLE:-true}
export RTC_SIMULATED_DELAY=${RTC_SIMULATED_DELAY:-5}
export RTC_CONDITIONING_MODE=${RTC_CONDITIONING_MODE:-action_encoder}
export DATA_MIX=${DATA_MIX:-libero_object}
export ACTION_MODE=${ACTION_MODE:-abs}
export RUN_ID=${RUN_ID:-qwenpi_libero_object_qwen3vl4b_rtc_action_encoder_delay${RTC_SIMULATED_DELAY}}

exec "$(dirname "$0")/run_qwenpi_libero_train.sh"
