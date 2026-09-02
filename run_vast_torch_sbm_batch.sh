#!/usr/bin/env bash
# Run selected Torch SBM backtests on Vast.ai and fetch each result.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VAST_HOST="${VAST_HOST:-194.228.55.129}"
VAST_PORT="${VAST_PORT:-37049}"
VAST_USER="${VAST_USER:-root}"
VAST_PROJECT="${VAST_PROJECT:-/workspace/margin_testing}"
SSH_KEY="${SSH_KEY:-${SCRIPT_DIR}/margin_testing}"
LOCAL_RESULTS_ROOT="${LOCAL_RESULTS_ROOT:-${SCRIPT_DIR}/fetched_results}"
BATCH_ID="${BATCH_ID:-$(date +%Y-%m-%d_%H%M%S)}"
STATUS_INTERVAL_SECONDS="${STATUS_INTERVAL_SECONDS:-60}"
REMOTE="${VAST_USER}@${VAST_HOST}"
REMOTE_BATCH_ROOT="${VAST_PROJECT}/backtest_results/torch_sbm_batches/${BATCH_ID}"
LOCAL_BATCH_ROOT="${LOCAL_RESULTS_ROOT}/vast_torch_sbm_batch_${BATCH_ID}"
CONTROL_PATH="${TMPDIR:-/tmp}/margin-testing-vast-${BASHPID}-%C"

CASES=(
  "assets_00050:config/backtests/assets_00050_torch_sbm.yaml"
  "assets_01000:config/backtests/assets_01000_torch_sbm.yaml"
  "assets_10000:config/backtests/assets_10000_torch_sbm.yaml"
  "vanguard_202426:config/backtests/vanguard_202426_torch_sbm.yaml"
)

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

current_step="initialization"
cleanup() {
  local exit_status="$?"
  if ((exit_status != 0)); then
    log "FAILED during ${current_step} (exit status ${exit_status})."
  fi
  ssh "${SSH_OPTIONS[@]}" -p "${VAST_PORT}" -O exit "${REMOTE}" \
    >/dev/null 2>&1 || true
}
trap cleanup EXIT

if [[ ! -f "${SSH_KEY}" ]]; then
  printf 'SSH key not found: %s\n' "${SSH_KEY}" >&2
  exit 1
fi
if [[ ! "${STATUS_INTERVAL_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'STATUS_INTERVAL_SECONDS must be a positive integer.\n' >&2
  exit 1
fi

mkdir -p "${LOCAL_BATCH_ROOT}"

current_step="opening the SSH control connection"
log "Opening one reusable SSH connection to ${REMOTE}:${VAST_PORT}."
log "Enter the key passphrase once if prompted."
ssh "${SSH_OPTIONS[@]}" -p "${VAST_PORT}" -N -f "${REMOTE}"
log "SSH connection ready."

current_step="remote preflight checks"
log "Checking remote configurations and ensuring no backtest is already running."
remote_preflight="cd $(printf '%q' "${VAST_PROJECT}")"
for case_entry in "${CASES[@]}"; do
  config_path="${case_entry#*:}"
  remote_preflight+=" && test -f $(printf '%q' "${config_path}")"
done
remote_preflight+=" && if pgrep -f '[r]un_backtest.py' >/dev/null; then echo 'A backtest is already running; refusing to compete for the GPUs.' >&2; exit 1; fi"
ssh "${SSH_OPTIONS[@]}" -p "${VAST_PORT}" "${REMOTE}" "${remote_preflight}"
log "Preflight checks finished. Four jobs are ready."

batch_started="$(date +%s)"
case_number=0
for case_entry in "${CASES[@]}"; do
  case_number=$((case_number + 1))
  case_name="${case_entry%%:*}"
  config_path="${case_entry#*:}"
  remote_output="${REMOTE_BATCH_ROOT}/${case_name}"
  local_output="${LOCAL_BATCH_ROOT}/${case_name}"
  mkdir -p "${local_output}/config"

  current_step="running ${case_name}"
  case_started="$(date +%s)"
  log "[${case_number}/${#CASES[@]}] START ${case_name} using ${config_path}."
  printf -v remote_command \
    'cd %q && PYTHONPATH=src /venv/main/bin/python run_backtest.py %q --output-directory %q' \
    "${VAST_PROJECT}" "${config_path}" "${remote_output}"
  ssh "${SSH_OPTIONS[@]}" -p "${VAST_PORT}" "${REMOTE}" "${remote_command}" &
  remote_job_pid="$!"
  while kill -0 "${remote_job_pid}" 2>/dev/null; do
    sleep "${STATUS_INTERVAL_SECONDS}"
    if kill -0 "${remote_job_pid}" 2>/dev/null; then
      now="$(date +%s)"
      log "[${case_number}/${#CASES[@]}] RUNNING ${case_name}; elapsed $(format_duration "$((now - case_started))")."
    fi
  done
  wait "${remote_job_pid}"
  run_finished="$(date +%s)"
  log "[${case_number}/${#CASES[@]}] RUN FINISHED ${case_name} after $(format_duration "$((run_finished - case_started))")."

  current_step="fetching ${case_name}"
  log "[${case_number}/${#CASES[@]}] FETCH START ${case_name}."
  scp "${SSH_OPTIONS[@]}" -P "${VAST_PORT}" \
    "${REMOTE}:${remote_output}/default/breaches.csv" \
    "${REMOTE}:${remote_output}/default/performance_metrics.csv" \
    "${local_output}/"
  scp "${SSH_OPTIONS[@]}" -P "${VAST_PORT}" \
    "${REMOTE}:${VAST_PROJECT}/${config_path}" \
    "${local_output}/config/"

  test -s "${local_output}/breaches.csv"
  test -s "${local_output}/performance_metrics.csv"
  daily_rows="$(($(wc -l < "${local_output}/breaches.csv") - 1))"
  reported_seconds="$(awk -F, '$8 == "total" {print $9}' "${local_output}/performance_metrics.csv")"
  fetch_finished="$(date +%s)"
  log "[${case_number}/${#CASES[@]}] FETCH FINISHED ${case_name}: ${daily_rows} daily rows, reported runtime ${reported_seconds}s."
  log "[${case_number}/${#CASES[@]}] COMPLETE ${case_name} after $(format_duration "$((fetch_finished - case_started))")."
done

batch_finished="$(date +%s)"
current_step="complete"
log "ALL FOUR JOBS FINISHED in $(format_duration "$((batch_finished - batch_started))")."
log "Fetched results: ${LOCAL_BATCH_ROOT}"
