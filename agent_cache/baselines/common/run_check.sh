#!/usr/bin/env bash
# Boot a system with small pools, run its correctness checks, stop it. Host side; GPU busy for 3-10 minutes.
#   host: bash /home/wanhr/sglang/agent_cache/baselines/common/run_check.sh <folder> <name> [VAR=VALUE ...] -- <check> [args]
# <folder> is a baseline folder (its start_server.sh boots the system with the VAR=VALUE knobs); <check> is
# check_server.py or check_session.py with its arguments, run in the sglang_hicache container where the replay runs.
# Output: <run dir>/check_<name>.json and check_<name>.log. Exit status: the check's.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/lib.sh"
FOLDER=$1 NAME=$2; shift 2
KNOBS=(); while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do KNOBS+=("$1"); shift; done
[ "${1:-}" = "--" ] || { echo "STOP: usage: run_check.sh <folder> <name> [VAR=VALUE ...] -- <check> [args]"; exit 1; }
shift
CHECK=("$@")
case "$FOLDER" in
  sglang_*) STOP=(docker exec sglang_hicache bash /sgl-workspace/sglang/agent_cache/scripts/stop_server.sh) ;;
  *)        STOP=(bash "$HERE/stop_server.sh") ;;
esac
env "${KNOBS[@]}" bash "$HERE/../$FOLDER/start_server.sh" | tail -6 || { "${STOP[@]}"; exit 1; }
RUN="$BL_RESULTS/$(cat "$BL_RESULTS/.current_run")"
CRUN="/sgl-workspace/sglang/agent_cache/results/$(basename "$RUN")"
printf '%q ' "${CHECK[@]}" > "$RUN/check_$NAME.command.txt"
docker exec -e PYTHONDONTWRITEBYTECODE=1 sglang_hicache bash -c "cd /sgl-workspace/sglang/agent_cache/baselines/common && python3 $(printf '%q ' "${CHECK[@]}") --json $CRUN/check_$NAME.json" 2>&1 | tee "$RUN/check_$NAME.log" | grep -v -E "^\s*$"
rc=${PIPESTATUS[0]}
"${STOP[@]}" | tail -1
exit "$rc"
