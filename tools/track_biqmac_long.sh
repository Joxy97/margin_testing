#!/usr/bin/env bash
# Track the 10k/20k/40k-step sweep independently of the original benchmark.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
HOST=${BIQMAC_HOST:-root@45.59.100.176}
PORT=${BIQMAC_PORT:-43437}
KEY=${SSH_KEY:-"$ROOT/margin_testing"}
REMOTE=/workspace/biqmac-maxcut-20260909/results_long_20260910
LOCAL="$ROOT/experiments/biqmac_maxcut_long_20260910/results"
OPTIONS=(-p "$PORT" -i "$KEY" -o IdentitiesOnly=yes -o ControlMaster=auto -o ControlPersist=600 -o 'ControlPath=/tmp/biqmac-track-%C')
case "${1:-status}" in
  status) ssh "${OPTIONS[@]}" "$HOST" "supervisorctl status biqmac_maxcut_long; cat $REMOTE/status.json" ;;
  logs) ssh "${OPTIONS[@]}" "$HOST" "tail -n 40 -F $REMOTE/progress.log" ;;
  fetch)
    mkdir -p "$LOCAL"
    printf -v TRANSPORT '%q ' ssh "${OPTIONS[@]}"
    rsync -az --exclude='*.tmp' --exclude='.lock' -e "$TRANSPORT" "$HOST:$REMOTE/" "$LOCAL/"
    printf 'Results fetched to %s\n' "$LOCAL"
    ;;
  *) printf 'Usage: bash tools/track_biqmac_long.sh [status|logs|fetch]\n' >&2; exit 2 ;;
esac
