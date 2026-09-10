#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
HOST=${QOBLIB_HOST:-root@45.59.100.176}
PORT=${QOBLIB_PORT:-43437}
KEY=${SSH_KEY:-"$ROOT/margin_testing"}
REMOTE=/workspace/qoblib-sac-comparison-20260910/results
LOCAL="$ROOT/experiments/qoblib_sac_comparison_20260910/results"
OPTIONS=(-p "$PORT" -i "$KEY" -o IdentitiesOnly=yes -o ControlMaster=auto -o ControlPersist=600 -o 'ControlPath=/tmp/biqmac-track-%C')
case "${1:-status}" in
  status) ssh "${OPTIONS[@]}" "$HOST" "supervisorctl status qoblib_sac_comparison; cat $REMOTE/status.json" ;;
  logs) ssh "${OPTIONS[@]}" "$HOST" "tail -n 30 -F /workspace/qoblib-sac-comparison-20260910/progress.log" ;;
  fetch)
    mkdir -p "$LOCAL"
    printf -v TRANSPORT '%q ' ssh "${OPTIONS[@]}"
    rsync -az --exclude='*.tmp' -e "$TRANSPORT" "$HOST:$REMOTE/" "$LOCAL/"
    ;;
  *) printf 'Usage: bash tools/track_qoblib_sac_comparison.sh [status|logs|fetch]\n' >&2; exit 2 ;;
esac
