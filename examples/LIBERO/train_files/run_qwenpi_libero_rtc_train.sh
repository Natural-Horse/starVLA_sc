#!/usr/bin/env bash
set -euo pipefail

export RTC_ENABLE=${RTC_ENABLE:-true}
export RTC_SIMULATED_DELAY=${RTC_SIMULATED_DELAY:-5}
export RUN_ID=${RUN_ID:-qwenpi_libero_qwen3vl4b_rtc_delay${RTC_SIMULATED_DELAY}}

exec "$(dirname "$0")/run_qwenpi_libero_train.sh"
