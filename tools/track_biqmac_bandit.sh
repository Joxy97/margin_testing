#!/usr/bin/env bash
# Track only the isolated eight-RTX-5090 bandit experiment.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
HOST=${BIQMAC_BANDIT_HOST:-root@13.56.204.87}
PORT=${BIQMAC_BANDIT_PORT:-21963}
KEY=${SSH_KEY:-"$ROOT/margin_testing"}
REMOTE=/workspace/biqmac-bandit-20260910/results
LOCAL="$ROOT/experiments/biqmac_bandit_20260910/results"
OPTIONS=(-p "$PORT" -i "$KEY" -o IdentitiesOnly=yes -o ControlMaster=auto -o ControlPersist=600 -o ControlPath=/tmp/margin-group1-second)
case "${1:-status}" in
  status) ssh "${OPTIONS[@]}" "$HOST" "supervisorctl status biqmac_bandit; cat $REMOTE/status.json" ;;
  logs) ssh "${OPTIONS[@]}" "$HOST" "tail -n 40 -F $REMOTE/progress.log" ;;
  fetch)
    mkdir -p "$LOCAL"
    printf -v TRANSPORT '%q ' ssh "${OPTIONS[@]}"
    rsync -az --exclude='*.tmp' -e "$TRANSPORT" "$HOST:$REMOTE/" "$LOCAL/"
    printf 'Results fetched to %s\n' "$LOCAL"
    ;;
  *) printf 'Usage: bash tools/track_biqmac_bandit.sh [status|logs|fetch]\n' >&2; exit 2 ;;
esac
