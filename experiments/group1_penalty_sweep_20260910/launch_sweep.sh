#!/usr/bin/env bash
set -euo pipefail
cd /workspace/margin-penalty-sweep
while [[ ! -f experiments/group1_penalty_sweep_20260910/qubos_manifest.json ]]; do
  sleep 5
done
exec /workspace/margin-sweep-venv/bin/python -u \
  experiments/group1_penalty_sweep_20260910/run_sweep.py \
  --steps 512 --runs 1 --budget 300 --policy selected --output results
