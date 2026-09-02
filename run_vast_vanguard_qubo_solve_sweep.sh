#!/usr/bin/env bash
# Run the stored-QUBO Torch SBM sweep on Vast.ai and fetch compact results.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VAST_HOST="${VAST_HOST:-38.49.42.46}"
VAST_PORT="${VAST_PORT:-59681}"
VAST_USER="${VAST_USER:-root}"
SSH_KEY="${SSH_KEY:-${SCRIPT_DIR}/margin_testing}"
VAST_PROJECT="${VAST_PROJECT:-/workspace/margin_testing}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/venv/main/bin/python}"
RUNNER_NAME="run_vanguard_torch_sbm_qubo_sweep.py"
LOCAL_RUNNER="${SCRIPT_DIR}/${RUNNER_NAME}"
BASE_CONFIG="${BASE_CONFIG:-config/backtests/vanguard_202426_torch_sbm.yaml}"

BATCH_ID="${BATCH_ID:-$(date +%Y-%m-%d_%H%M%S)}"
REMOTE_RUN_NAME="${REMOTE_RUN_NAME:-vanguard_qubo_solve_${BATCH_ID}}"
REMOTE_OUTPUT="${VAST_PROJECT}/sweep_results/${REMOTE_RUN_NAME}"
LOCAL_OUTPUT="${LOCAL_OUTPUT:-${SCRIPT_DIR}/fetched_results/${REMOTE_RUN_NAME}}"

STEPS_VALUES_TEXT="${STEPS_VALUES:-25 30 50 75 100 200 300 400 500 600 700 800 900 1000}"
RUNS_VALUES_TEXT="${RUNS_VALUES:-1 2 4 8 16 32}"
QUBO_BATCH_SIZE="${QUBO_BATCH_SIZE:-8}"
RUN_BATCH_SIZE="${RUN_BATCH_SIZE:-16}"
EXPECTED_QUBOS="${EXPECTED_QUBOS:-105}"
STATUS_INTERVAL_SECONDS="${STATUS_INTERVAL_SECONDS:-60}"

read -r -a STEP_VALUES <<< "${STEPS_VALUES_TEXT}"
read -r -a RUN_VALUES <<< "${RUNS_VALUES_TEXT}"

REMOTE="${VAST_USER}@${VAST_HOST}"
CONTROL_PATH="${TMPDIR:-/tmp}/margin-testing-qubo-sweep-${BASHPID}-%C"
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

require_positive_integer() {
  local label="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s must be a positive integer; received: %s\n' \
      "${label}" "${value}" >&2
    exit 1
  fi
}

require_positive_integer_list() {
  local label="$1"
  shift
  if (($# == 0)); then
    printf '%s must contain at least one value.\n' "${label}" >&2
    exit 1
  fi
  local value
  for value in "$@"; do
    require_positive_integer "${label}" "${value}"
  done
}

ssh_connection_open=0
fetch_attempted=0
fetch_results() {
  if ((fetch_attempted)); then
    return
  fi
  fetch_attempted=1
  mkdir -p "${LOCAL_OUTPUT}"

  local available
  available="$(ssh "${SSH_OPTIONS[@]}" -p "${VAST_PORT}" "${REMOTE}" \
    "test -d '${REMOTE_OUTPUT}' && echo yes || echo no")"
  if [[ "${available}" != "yes" ]]; then
    log "No remote output directory is available to fetch yet."
    return
  fi

  log "Fetching completed configs and result YAML files."
  if ssh "${SSH_OPTIONS[@]}" -p "${VAST_PORT}" "${REMOTE}" \
    "test -d '${REMOTE_OUTPUT}/config'"; then
    scp "${SSH_OPTIONS[@]}" -P "${VAST_PORT}" -r \
      "${REMOTE}:${REMOTE_OUTPUT}/config" "${LOCAL_OUTPUT}/"
  fi
  if ssh "${SSH_OPTIONS[@]}" -p "${VAST_PORT}" "${REMOTE}" \
    "test -d '${REMOTE_OUTPUT}/results'"; then
    scp "${SSH_OPTIONS[@]}" -P "${VAST_PORT}" -r \
      "${REMOTE}:${REMOTE_OUTPUT}/results" "${LOCAL_OUTPUT}/"
  fi
  if ssh "${SSH_OPTIONS[@]}" -p "${VAST_PORT}" "${REMOTE}" \
    "test -f '${REMOTE_OUTPUT}/summary.csv'"; then
    scp "${SSH_OPTIONS[@]}" -P "${VAST_PORT}" \
      "${REMOTE}:${REMOTE_OUTPUT}/summary.csv" "${LOCAL_OUTPUT}/"
  fi
  if ssh "${SSH_OPTIONS[@]}" -p "${VAST_PORT}" "${REMOTE}" \
    "test -f '${REMOTE_OUTPUT}/qubos/manifest.json'"; then
    scp "${SSH_OPTIONS[@]}" -P "${VAST_PORT}" \
      "${REMOTE}:${REMOTE_OUTPUT}/qubos/manifest.json" \
      "${LOCAL_OUTPUT}/qubo_manifest.json"
  fi
  log "Fetch finished: ${LOCAL_OUTPUT}"
  log "Stored QUBO .npz files remain on the server at ${REMOTE_OUTPUT}/qubos."
}

current_step="initialization"
cleanup() {
  local exit_status="$?"
  trap - EXIT INT TERM
  if ((ssh_connection_open)) && ((fetch_attempted == 0)); then
    log "Attempting to fetch any completed results before exit."
    fetch_results || log "Partial-result fetch did not complete."
  fi
  if ((ssh_connection_open)); then
    ssh "${SSH_OPTIONS[@]}" -p "${VAST_PORT}" -O exit "${REMOTE}" \
      >/dev/null 2>&1 || true
  fi
  if ((exit_status != 0)); then
    log "FAILED during ${current_step} (exit status ${exit_status})."
    log "Resume with: BATCH_ID=${BATCH_ID} $0"
  fi
  exit "${exit_status}"
}
trap cleanup EXIT INT TERM

if [[ ! -f "${SSH_KEY}" ]]; then
  printf 'SSH key not found: %s\n' "${SSH_KEY}" >&2
  exit 1
fi
if [[ ! -f "${LOCAL_RUNNER}" ]]; then
  printf 'Python QUBO sweep runner not found: %s\n' "${LOCAL_RUNNER}" >&2
  exit 1
fi
require_positive_integer_list "STEPS_VALUES" "${STEP_VALUES[@]}"
require_positive_integer_list "RUNS_VALUES" "${RUN_VALUES[@]}"
require_positive_integer "QUBO_BATCH_SIZE" "${QUBO_BATCH_SIZE}"
require_positive_integer "RUN_BATCH_SIZE" "${RUN_BATCH_SIZE}"
require_positive_integer "EXPECTED_QUBOS" "${EXPECTED_QUBOS}"
require_positive_integer "STATUS_INTERVAL_SECONDS" "${STATUS_INTERVAL_SECONDS}"

log "Vanguard stored-QUBO sweep: ${#STEP_VALUES[@]} step values x ${#RUN_VALUES[@]} run values."
log "Remote output: ${REMOTE_OUTPUT}"
log "Local fetched output: ${LOCAL_OUTPUT}"

current_step="opening the SSH control connection"
log "Opening one reusable SSH connection to ${REMOTE}:${VAST_PORT}."
log "Enter the margin_testing key passphrase when prompted."
ssh "${SSH_OPTIONS[@]}" -p "${VAST_PORT}" -N -f "${REMOTE}"
ssh_connection_open=1

current_step="remote preflight"
printf -v preflight_command \
  'cd %q && test -x %q && test -f %q && if pgrep -f %q >/dev/null; then echo %q >&2; exit 1; fi' \
  "${VAST_PROJECT}" \
  "${REMOTE_PYTHON}" \
  "${BASE_CONFIG}" \
  '[r]un_vanguard_torch_sbm_qubo_sweep.py' \
  'Another stored-QUBO sweep is already running; refusing to compete for the GPUs.'
ssh "${SSH_OPTIONS[@]}" -p "${VAST_PORT}" "${REMOTE}" "${preflight_command}"
log "Remote preflight finished."

current_step="uploading the Python runner"
scp "${SSH_OPTIONS[@]}" -P "${VAST_PORT}" \
  "${LOCAL_RUNNER}" "${REMOTE}:${VAST_PROJECT}/${RUNNER_NAME}"
ssh "${SSH_OPTIONS[@]}" -p "${VAST_PORT}" "${REMOTE}" \
  "chmod +x '${VAST_PROJECT}/${RUNNER_NAME}'"
log "Latest Python runner uploaded."

remote_arguments=(
  "${REMOTE_PYTHON}"
  -u
  "${VAST_PROJECT}/${RUNNER_NAME}"
  "${VAST_PROJECT}/${BASE_CONFIG}"
  --output "${REMOTE_OUTPUT}"
  --steps "${STEP_VALUES[@]}"
  --runs "${RUN_VALUES[@]}"
  --qubo-batch-size "${QUBO_BATCH_SIZE}"
  --run-batch-size "${RUN_BATCH_SIZE}"
  --expected-qubos "${EXPECTED_QUBOS}"
  --status-interval "${STATUS_INTERVAL_SECONDS}"
)
printf -v quoted_remote_arguments '%q ' "${remote_arguments[@]}"

current_step="running the remote QUBO sweep"
log "Starting or resuming ${REMOTE_RUN_NAME}."
ssh "${SSH_OPTIONS[@]}" -p "${VAST_PORT}" "${REMOTE}" \
  "cd '${VAST_PROJECT}' && exec ${quoted_remote_arguments}"
log "Remote sweep completed successfully."

current_step="fetching results"
fetch_results

current_step="complete"
log "ALL DONE. Summary: ${LOCAL_OUTPUT}/summary.csv"
