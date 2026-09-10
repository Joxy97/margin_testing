#!/usr/bin/env bash
set -euo pipefail
cd /workspace/group1-factor-sweep-20260910
export PYTHONPATH=src:tools
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/group1-factor-full-matplotlib
python_bin=/workspace/margin-sweep-venv/bin/python
experiment=experiments/group1_factor_full_20260910

"$python_bin" -m unittest tests.test_factor_stress tests.test_factor_stress_repair -v
time -p "$python_bin" tools/sweep_factor_stress.py \
  --group /workspace/margin-penalty-sweep/yahoo_equities_grouped/groups/US/group_1_8590.csv \
  --output "$experiment/results" --days 294 --window 125 \
  --bits 8 --multipliers 1e-11 --repeats 3 --runs 32 --steps 10000 \
  --devices 0 1 2 3 4 5 6 7 --batch-size 32
time -p "$python_bin" tools/repair_factor_sweep.py \
  "$experiment/results" "$experiment/repair_results" --max-steps 1024
time -p "$python_bin" tools/report_factor_sweep.py "$experiment/results"
time -p "$python_bin" tools/report_factor_repair.py "$experiment/repair_results"
