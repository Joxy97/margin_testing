#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
HOST=${VAST_HOST:-root@13.56.204.87}
PORT=${VAST_PORT:-21963}
KEY=${SSH_KEY:-"$ROOT/margin_testing"}
REMOTE=/workspace/group1-categorical-trf-20260910/experiments/group1_categorical_trf_20260910/results
LOCAL="$ROOT/experiments/group1_categorical_trf_20260910/results"
OPTIONS=(-p "$PORT" -i "$KEY" -o IdentitiesOnly=yes -o ControlMaster=auto -o ControlPersist=600 -o ControlPath=/tmp/margin-group1-second)
case "${1:-status}" in
  status) ssh "${OPTIONS[@]}" "$HOST" "supervisorctl status group1_categorical_trf; cat $REMOTE/status.json" ;;
  logs) ssh "${OPTIONS[@]}" "$HOST" "tail -n 30 -F $REMOTE/progress.log" ;;
  fetch)
    mkdir -p "$LOCAL"
    printf -v TRANSPORT '%q ' ssh "${OPTIONS[@]}"
    rsync -az --exclude='*.tmp' -e "$TRANSPORT" "$HOST:$REMOTE/" "$LOCAL/"
    printf 'Results fetched to %s\n' "$LOCAL"
    ;;
  *) printf 'Usage: bash tools/track_group1_categorical_trf.sh [status|logs|fetch]\n' >&2; exit 2 ;;
esac
