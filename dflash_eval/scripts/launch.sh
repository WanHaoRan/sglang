#!/usr/bin/env bash
# Launch one arm's SGLang server inside the dev container and wait for it to be ready.
#
#   scripts/launch.sh <ARM>        # e.g. A0 A1 A1-auto A1-w8 A1-card A2 A2-w8 A3 B0 B1-4 B1-8 B2 B3
#   scripts/launch.sh stop         # kill whatever server is running in the container
#   scripts/launch.sh print <ARM>  # only print the command that would run
#
# Boot log goes to $RESULTS/server_<ARM>.log (host path; same file at the container path).
# After READY it greps the log for the sanity lines listed in RUNBOOK.md §5.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=env.sh
source "$HERE/env.sh"

stop_server() {
  $DOCKER exec "$CONTAINER" bash -lc 'killall_sglang >/dev/null 2>&1 || true; pkill -f "sglang.launch_server" || true; pkill -f "sglang serve" || true; sleep 2; nvidia-smi --query-gpu=memory.used --format=csv,noheader'
}

if [[ "${1:-}" == "stop" ]]; then stop_server; exit 0; fi
if [[ "${1:-}" == "print" ]]; then PRINT_ONLY=1; shift; else PRINT_ONLY=0; fi
ARM=${1:?usage: launch.sh <ARM>|stop|print <ARM>}

EXTRA=$(arm_flags "$ARM")
case "$ARM" in
  A*) COMMON=$COMMON_A; ENVSTR="" ;;
  B*) COMMON=$COMMON_B; ENVSTR="$ENV_B" ;;
esac
CMD="cd /sgl-workspace/sglang && ${ENVSTR} python3 -m sglang.launch_server ${COMMON} ${EXTRA}"

mkdir -p "$RESULTS"
LOG_HOST="$RESULTS/server_${ARM}.log"
LOG_CTR="$RESULTS_CTR/server_${ARM}.log"

echo "# arm=$ARM"
echo "# $CMD"
echo "# boot log: $LOG_HOST  (container path: $LOG_CTR)"
if [[ $PRINT_ONLY == 1 ]]; then exit 0; fi

# refuse to start on top of another server
if curl -sf "$BASE_URL/health" >/dev/null 2>&1; then
  echo "a server already answers on $BASE_URL; run 'launch.sh stop' first" >&2; exit 1
fi

# append, never clobber: an earlier boot of the same arm in this results dir stays readable
{ echo; echo "# ===== $(date -Is) arm=$ARM"; echo "# $CMD"; } >> "$LOG_HOST"
$DOCKER exec -d "$CONTAINER" bash -lc "$CMD >> $LOG_CTR 2>&1"

# fail fast if the process never came up (bad path, bad flag) instead of waiting 20 min
sleep 5
if ! $DOCKER exec "$CONTAINER" pgrep -f "sglang.launch_server" >/dev/null; then
  echo "server process did not start in $CONTAINER; last lines of $LOG_HOST:" >&2
  tail -n 20 "$LOG_HOST" >&2
  echo "(if the log has only the header, the container could not open $LOG_CTR)" >&2
  exit 1
fi

echo "waiting for $BASE_URL/health (boot log: $LOG_HOST)"
for i in $(seq 1 240); do            # up to 20 min: first FA3 load re-downloads cubins, 27B loads 66 shards
  if curl -sf "$BASE_URL/health" >/dev/null 2>&1; then
    echo "READY after ~$((i*5)) s"
    echo "--- sanity lines from the boot log ---"
    grep -E "max_total_num_tokens|max_running_requests|speculative_num_draft_tokens|KV Cache is allocated|Disable DFLASH draft cuda graph|Disable DSpark draft cuda graph|folded sampling disabled|kept eager|Non-overlap|ragged-verify scheduler|fired up" "$LOG_HOST" || true
    exit 0
  fi
  if grep -qE "Traceback|RuntimeError|Not enough GPU memory" "$LOG_HOST"; then
    echo "server died; tail of log:" >&2; tail -n 40 "$LOG_HOST" >&2; exit 1
  fi
  sleep 5
done
echo "timed out waiting for the server; see $LOG_HOST" >&2
exit 1
