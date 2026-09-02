#!/usr/bin/env bash
# Sweep Torch SBM steps and runs for one Vanguard margin date on Vast.ai.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXECUTION_MODE="${EXECUTION_MODE:-auto}"
VAST_HOST="${VAST_HOST:-98.142.241.120}"
VAST_PORT="${VAST_PORT:-23740}"
VAST_USER="${VAST_USER:-root}"
if [[ "${EXECUTION_MODE}" == "auto" ]]; then
  if [[ "${SCRIPT_DIR}" == /workspace/* && -x /venv/main/bin/python ]]; then
    EXECUTION_MODE="server"
  else
    EXECUTION_MODE="local"
  fi
fi
if [[ "${EXECUTION_MODE}" == "server" ]]; then
  VAST_PROJECT="${VAST_PROJECT:-${SCRIPT_DIR}}"
  LOCAL_PYTHON="${LOCAL_PYTHON:-/venv/main/bin/python}"
  LOCAL_RESULTS_ROOT="${LOCAL_RESULTS_ROOT:-${VAST_PROJECT}/sweep_results}"
else
  VAST_PROJECT="${VAST_PROJECT:-/workspace/margin_testing}"
  LOCAL_PYTHON="${LOCAL_PYTHON:-${SCRIPT_DIR}/.venv/bin/python}"
  LOCAL_RESULTS_ROOT="${LOCAL_RESULTS_ROOT:-${SCRIPT_DIR}/fetched_results}"
fi
SSH_KEY="${SSH_KEY:-${SCRIPT_DIR}/margin_testing}"
BASE_CONFIG="${BASE_CONFIG:-${SCRIPT_DIR}/config/backtests/vanguard_202426_torch_sbm.yaml}"
MARGIN_DATE="${MARGIN_DATE:-2025-01-01}"
BATCH_ID="${BATCH_ID:-$(date +%Y-%m-%d_%H%M%S)}"
STATUS_INTERVAL_SECONDS="${STATUS_INTERVAL_SECONDS:-60}"
GENERATE_ONLY="${GENERATE_ONLY:-0}"
STEPS_VALUES_TEXT="${STEPS_VALUES:-1000 1500 2000 3000 4000 5000 6000 7500 9000 10000}"
RUNS_VALUES_TEXT="${RUNS_VALUES:-16 24 32 40 48 64 80 96 128}"

read -r -a STEP_VALUES <<< "${STEPS_VALUES_TEXT}"
read -r -a RUN_VALUES <<< "${RUNS_VALUES_TEXT}"

REMOTE="${VAST_USER}@${VAST_HOST}"
REMOTE_SWEEP_ROOT="${VAST_PROJECT}/sweeps/vanguard_torch_sbm/${BATCH_ID}"
LOCAL_SWEEP_ROOT="${LOCAL_RESULTS_ROOT}/vast_vanguard_torch_sbm_sweep_${BATCH_ID}"
LOCAL_CONFIG_ROOT="${LOCAL_SWEEP_ROOT}/config"
LOCAL_RESULT_ROOT="${LOCAL_SWEEP_ROOT}/results"
CONTROL_PATH="${TMPDIR:-/tmp}/margin-testing-vast-sweep-${BASHPID}-%C"

SSH_OPTIONS=(
  -i "${SSH_KEY}"
  -o ControlMaster=auto
  -o ControlPersist=10m
  -o "ControlPath=${CONTROL_PATH}"
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=6
)

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

format_duration() {
  local total_seconds="$1"
  printf '%02dh:%02dm:%02ds' \
    "$((total_seconds / 3600))" \
    "$(((total_seconds % 3600) / 60))" \
    "$((total_seconds % 60))"
}

require_positive_integers() {
  local label="$1"
  shift
  if (($# == 0)); then
    printf '%s must contain at least one value.\n' "${label}" >&2
    exit 1
  fi
  local value
  for value in "$@"; do
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
      printf '%s values must be positive integers; received: %s\n' \
        "${label}" "${value}" >&2
      exit 1
    fi
  done
}

write_summary() {
  "${LOCAL_PYTHON}" - "${LOCAL_SWEEP_ROOT}" <<'PYTHON'
from pathlib import Path
import csv
import sys

import yaml

sweep_root = Path(sys.argv[1])
rows = []
for result_path in sorted((sweep_root / "results").glob("*.result.yaml")):
    with result_path.open("r", encoding="utf-8") as stream:
        result = yaml.safe_load(stream)
    timings = result["timings"]
    stem = result_path.name.removesuffix(".result.yaml")
    rows.append(
        {
            "steps": result["steps"],
            "runs": result["runs"],
            "margin_date": result["marginDate"],
            "margin": result["margin"],
            "data_acquisition_seconds": timings["dataAcquisitionSeconds"],
            "risk_state_generation_seconds": timings[
                "riskStateGenerationSeconds"
            ],
            "margin_calculation_seconds": timings[
                "marginCalculationSeconds"
            ],
            "total_seconds": timings["totalSeconds"],
            "config": f"config/{stem}.yaml",
            "result": f"results/{result_path.name}",
        }
    )

if not rows:
    raise RuntimeError("No sweep result files were produced")
summary_path = sweep_root / "summary.csv"
with summary_path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
PYTHON
}

current_step="initialization"
ssh_connection_open=0
cleanup() {
  local exit_status="$?"
  if ((exit_status != 0)); then
    log "FAILED during ${current_step} (exit status ${exit_status})."
    log "Any completed combinations remain in ${LOCAL_SWEEP_ROOT}."
  fi
  if ((ssh_connection_open)); then
    ssh "${SSH_OPTIONS[@]}" -p "${VAST_PORT}" -O exit "${REMOTE}" \
      >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if [[ "${EXECUTION_MODE}" != "local" && "${EXECUTION_MODE}" != "server" ]]; then
  printf 'EXECUTION_MODE must be auto, local, or server.\n' >&2
  exit 1
fi
if [[ "${EXECUTION_MODE}" == "local" && ! -f "${SSH_KEY}" ]]; then
  printf 'SSH key not found: %s\n' "${SSH_KEY}" >&2
  exit 1
fi
if [[ ! -x "${LOCAL_PYTHON}" ]]; then
  printf 'Local Python interpreter not found: %s\n' "${LOCAL_PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${BASE_CONFIG}" ]]; then
  printf 'Base Vanguard YAML not found: %s\n' "${BASE_CONFIG}" >&2
  exit 1
fi
if [[ ! "${STATUS_INTERVAL_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'STATUS_INTERVAL_SECONDS must be a positive integer.\n' >&2
  exit 1
fi
if [[ ! "${GENERATE_ONLY}" =~ ^[01]$ ]]; then
  printf 'GENERATE_ONLY must be 0 or 1.\n' >&2
  exit 1
fi
if [[ ! "${MARGIN_DATE}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  printf 'MARGIN_DATE must use YYYY-MM-DD format.\n' >&2
  exit 1
fi
require_positive_integers "STEPS_VALUES" "${STEP_VALUES[@]}"
require_positive_integers "RUNS_VALUES" "${RUN_VALUES[@]}"

if [[ -e "${LOCAL_SWEEP_ROOT}" ]]; then
  printf 'Local sweep folder already exists: %s\n' "${LOCAL_SWEEP_ROOT}" >&2
  exit 1
fi
mkdir -p "${LOCAL_CONFIG_ROOT}" "${LOCAL_RESULT_ROOT}"

total_combinations=$((${#STEP_VALUES[@]} * ${#RUN_VALUES[@]}))
log "Preparing ${total_combinations} Vanguard Torch SBM combinations."
log "Margin date: ${MARGIN_DATE}"
log "Steps: ${STEP_VALUES[*]}"
log "Runs: ${RUN_VALUES[*]}"

current_step="generating YAML configurations"
for steps in "${STEP_VALUES[@]}"; do
  for runs in "${RUN_VALUES[@]}"; do
    config_name="vanguard_steps_$(printf '%06d' "${steps}")_runs_$(printf '%04d' "${runs}").yaml"
    output_config="${LOCAL_CONFIG_ROOT}/${config_name}"
    "${LOCAL_PYTHON}" - \
      "${BASE_CONFIG}" "${output_config}" "${steps}" "${runs}" \
      "${MARGIN_DATE}" "${VAST_PROJECT}" <<'PYTHON'
from pathlib import Path
import sys

import yaml

base_path, output_path, steps, runs, margin_date, remote_project = sys.argv[1:]
with Path(base_path).open("r", encoding="utf-8") as stream:
    document = yaml.safe_load(stream)

document.pop("backtest", None)
document["marginDate"] = margin_date
document["portfolio"]["csv"] = (
    f"{remote_project}/vanguard_market/"
    "vanguard_portfolio_202426_marginlab_complete_history.csv"
)
document["engine"]["downloadManager"]["requestParameters"]["locations"] = [
    (
        f"{remote_project}/vanguard_market/"
        "vanguard_portfolio_202426_marginlab_close_history_2024.csv"
    ),
    (
        f"{remote_project}/vanguard_market/"
        "vanguard_portfolio_202426_marginlab_close_backtest_2025.csv"
    ),
]
solver = document["engine"]["marginCalculator"]["solver"]
solver["type"] = "torch_sbm"
solver["constructorParameters"] = {
    "devices": [f"cuda:{index}" for index in range(8)]
}
solver["solverParameters"]["steps"] = int(steps)
solver["solverParameters"]["runs"] = int(runs)
solver["solverParameters"]["run_batch_size"] = 16

with Path(output_path).open("w", encoding="utf-8") as stream:
    yaml.safe_dump(document, stream, sort_keys=False)
PYTHON
    log "YAML READY steps=${steps}, runs=${runs}: ${config_name}"
  done
done
log "All ${total_combinations} YAML files generated."
report_python=$'import sys\nimport yaml\nfrom margin_engine import MarginApplicationConfig\napplication = MarginApplicationConfig.fromYaml(sys.argv[1])\nreport = application.generateReport()\noutput = {\n    "steps": int(sys.argv[2]),\n    "runs": int(sys.argv[3]),\n    "marginDate": application.marginDate.isoformat(),\n    "margin": report.margin,\n    "timings": {\n        "dataAcquisitionSeconds": report.timings.dataAcquisitionSeconds,\n        "riskStateGenerationSeconds": report.timings.riskStateGenerationSeconds,\n        "marginCalculationSeconds": report.timings.marginCalculationSeconds,\n        "totalSeconds": report.timings.totalSeconds,\n    },\n}\nprint(yaml.safe_dump(output, sort_keys=False), end="")'
if ((GENERATE_ONLY)); then
  current_step="complete"
  log "Generation-only mode finished: ${LOCAL_CONFIG_ROOT}"
  exit 0
fi

if [[ "${EXECUTION_MODE}" == "server" ]]; then
  current_step="server preflight checks"
  log "Server mode selected; no SSH key or nested SSH connection is required."
  if pgrep -f '[r]un_backtest.py\|[p]ython.*-m margin_engine' >/dev/null; then
    printf 'A margin or backtest process is already running; refusing to compete for the GPUs.\n' >&2
    exit 1
  fi
  gpu_count="$("${LOCAL_PYTHON}" -c 'import torch; print(torch.cuda.device_count())')"
  if [[ "${gpu_count}" != "8" ]]; then
    printf 'Expected 8 CUDA devices, but Torch reported %s.\n' "${gpu_count}" >&2
    exit 1
  fi
  log "Server preflight finished: Torch reports ${gpu_count} CUDA devices."

  sweep_started="$(date +%s)"
  combination_number=0
  cd "${VAST_PROJECT}"
  for steps in "${STEP_VALUES[@]}"; do
    for runs in "${RUN_VALUES[@]}"; do
      combination_number=$((combination_number + 1))
      stem="vanguard_steps_$(printf '%06d' "${steps}")_runs_$(printf '%04d' "${runs}")"
      config_path="${LOCAL_CONFIG_ROOT}/${stem}.yaml"
      result_path="${LOCAL_RESULT_ROOT}/${stem}.result.yaml"

      current_step="running steps=${steps}, runs=${runs}"
      combination_started="$(date +%s)"
      log "[${combination_number}/${total_combinations}] START steps=${steps}, runs=${runs}."
      PYTHONPATH="${VAST_PROJECT}/src" "${LOCAL_PYTHON}" -c \
        "${report_python}" "${config_path}" "${steps}" "${runs}" \
        >"${result_path}" &
      solver_pid="$!"
      while kill -0 "${solver_pid}" 2>/dev/null; do
        sleep "${STATUS_INTERVAL_SECONDS}"
        if kill -0 "${solver_pid}" 2>/dev/null; then
          now="$(date +%s)"
          log "[${combination_number}/${total_combinations}] RUNNING steps=${steps}, runs=${runs}; elapsed $(format_duration "$((now - combination_started))")."
        fi
      done
      wait "${solver_pid}"
      test -s "${result_path}"
      margin="$("${LOCAL_PYTHON}" -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["margin"])' "${result_path}")"
      total_seconds="$("${LOCAL_PYTHON}" -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["timings"]["totalSeconds"])' "${result_path}")"
      combination_finished="$(date +%s)"
      log "[${combination_number}/${total_combinations}] COMPLETE margin=${margin}, engine runtime=${total_seconds}s, elapsed $(format_duration "$((combination_finished - combination_started))")."
    done
  done

  current_step="writing the server summary"
  write_summary
  sweep_finished="$(date +%s)"
  current_step="complete"
  log "ALL ${total_combinations} COMBINATIONS FINISHED in $(format_duration "$((sweep_finished - sweep_started))")."
  log "Summary: ${LOCAL_SWEEP_ROOT}/summary.csv"
  log "All configs and results: ${LOCAL_SWEEP_ROOT}"
  exit 0
fi

current_step="opening the SSH control connection"
log "Opening one reusable SSH connection to ${REMOTE}:${VAST_PORT}."
log "Enter the key passphrase once if prompted."
ssh "${SSH_OPTIONS[@]}" -p "${VAST_PORT}" -N -f "${REMOTE}"
ssh_connection_open=1
log "SSH connection ready."

current_step="remote preflight checks"
printf -v remote_preflight \
  'cd %q && test -f %q && if pgrep -f '\''[r]un_backtest.py\|[m]argin_engine'\'' >/dev/null; then echo '\''A margin or backtest process is already running; refusing to compete for the GPUs.'\'' >&2; exit 1; fi && if test -e %q; then echo '\''Remote sweep folder already exists.'\'' >&2; exit 1; fi && mkdir -p %q %q' \
  "${VAST_PROJECT}" \
  "config/backtests/vanguard_202426_torch_sbm.yaml" \
  "${REMOTE_SWEEP_ROOT}" \
  "${REMOTE_SWEEP_ROOT}/config" \
  "${REMOTE_SWEEP_ROOT}/results"
ssh "${SSH_OPTIONS[@]}" -p "${VAST_PORT}" "${REMOTE}" "${remote_preflight}"
log "Remote preflight checks finished."

current_step="uploading generated YAML configurations"
scp "${SSH_OPTIONS[@]}" -P "${VAST_PORT}" \
  "${LOCAL_CONFIG_ROOT}"/*.yaml \
  "${REMOTE}:${REMOTE_SWEEP_ROOT}/config/"
log "Uploaded all generated YAML files."

sweep_started="$(date +%s)"
combination_number=0
for steps in "${STEP_VALUES[@]}"; do
  for runs in "${RUN_VALUES[@]}"; do
    combination_number=$((combination_number + 1))
    stem="vanguard_steps_$(printf '%06d' "${steps}")_runs_$(printf '%04d' "${runs}")"
    remote_config="${REMOTE_SWEEP_ROOT}/config/${stem}.yaml"
    remote_result="${REMOTE_SWEEP_ROOT}/results/${stem}.result.yaml"
    local_result="${LOCAL_RESULT_ROOT}/${stem}.result.yaml"

    current_step="running steps=${steps}, runs=${runs}"
    combination_started="$(date +%s)"
    log "[${combination_number}/${total_combinations}] START steps=${steps}, runs=${runs}."
    printf -v remote_command \
      'cd %q && PYTHONPATH=src /venv/main/bin/python -c %q %q %q %q > %q' \
      "${VAST_PROJECT}" "${report_python}" "${remote_config}" \
      "${steps}" "${runs}" "${remote_result}"
    ssh "${SSH_OPTIONS[@]}" -p "${VAST_PORT}" "${REMOTE}" "${remote_command}" &
    remote_job_pid="$!"
    while kill -0 "${remote_job_pid}" 2>/dev/null; do
      sleep "${STATUS_INTERVAL_SECONDS}"
      if kill -0 "${remote_job_pid}" 2>/dev/null; then
        now="$(date +%s)"
        log "[${combination_number}/${total_combinations}] RUNNING steps=${steps}, runs=${runs}; elapsed $(format_duration "$((now - combination_started))")."
      fi
    done
    wait "${remote_job_pid}"
    run_finished="$(date +%s)"
    log "[${combination_number}/${total_combinations}] RUN FINISHED after $(format_duration "$((run_finished - combination_started))")."

    current_step="fetching steps=${steps}, runs=${runs}"
    scp "${SSH_OPTIONS[@]}" -P "${VAST_PORT}" \
      "${REMOTE}:${remote_result}" "${local_result}"
    test -s "${local_result}"
    margin="$("${LOCAL_PYTHON}" -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["margin"])' "${local_result}")"
    total_seconds="$("${LOCAL_PYTHON}" -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["timings"]["totalSeconds"])' "${local_result}")"
    log "[${combination_number}/${total_combinations}] FETCH FINISHED margin=${margin}, engine runtime=${total_seconds}s."
    log "[${combination_number}/${total_combinations}] COMPLETE steps=${steps}, runs=${runs}."
  done
done

current_step="writing the local summary"
write_summary

sweep_finished="$(date +%s)"
current_step="complete"
log "ALL ${total_combinations} COMBINATIONS FINISHED in $(format_duration "$((sweep_finished - sweep_started))")."
log "Summary: ${LOCAL_SWEEP_ROOT}/summary.csv"
log "All configs and results: ${LOCAL_SWEEP_ROOT}"
