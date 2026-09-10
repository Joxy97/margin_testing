#!/usr/bin/env bash
set -euo pipefail
cd /workspace/group1-risky-portfolios-20260910
export PYTHONPATH=src:tools
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/risky-factor-matplotlib
python_bin=/workspace/margin-sweep-venv/bin/python
RUN_FACTOR_GPU_TESTS=1 "$python_bin" -m unittest \
  tests.test_risky_factor_portfolios tests.test_factor_stress \
  tests.test_factor_stress_repair tests.test_factor_sweep_worker -v
time -p "$python_bin" tools/run_risky_factor_backtests.py \
  --group /workspace/margin-penalty-sweep/yahoo_equities_grouped/groups/US/group_1_8590.csv \
  --output experiments/group1_risky_portfolios_20260910/results \
  --runs 32 --steps 10000 --devices 0 1 2 3 4 5 6 7 "$@"
