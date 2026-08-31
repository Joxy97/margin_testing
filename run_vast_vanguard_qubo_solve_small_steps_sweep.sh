#!/usr/bin/env bash
# Run the Vanguard stored-QUBO sweep with a low Torch SBM step grid.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export STEPS_VALUES="${STEPS_VALUES:-15 25 50 75 100 150 250 400 650 1000}"
export RUNS_VALUES="${RUNS_VALUES:-16 24 32 40 48 64 80 96 128}"
export BATCH_ID="${BATCH_ID:-small_steps_$(date +%Y-%m-%d_%H%M%S)}"

exec "${SCRIPT_DIR}/run_vast_vanguard_qubo_solve_sweep.sh"
