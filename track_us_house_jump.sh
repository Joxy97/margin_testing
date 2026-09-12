#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH=src:tools OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/house-jump-plots
exec .venv/bin/python tools/manage_group8589_backtest.py "${1:-status}" --experiment group1_us_house200_pca10_jump_garch_20260912 "${@:2}"
