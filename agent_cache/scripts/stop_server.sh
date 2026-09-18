#!/usr/bin/env bash
# Stop the server of the current run (agent_cache/results/.current_run) and wait until its pid is gone and port 30000 is free.
#   host:      docker exec -it sglang_hicache bash /sgl-workspace/sglang/agent_cache/scripts/stop_server.sh
#   container: bash /sgl-workspace/sglang/agent_cache/scripts/stop_server.sh
set -uo pipefail
AC=/sgl-workspace/sglang/agent_cache
[ -d "$AC" ] || { echo "STOP: $AC not found; run inside the container"; exit 1; }
RESULTS="$AC/results"
[ -s "$RESULTS/.current_run" ] || { echo "nothing to stop: no .current_run in $RESULTS"; exit 0; }
RUN="$RESULTS/$(cat "$RESULTS/.current_run")"
[ -f "$RUN/server.pid" ] || { echo "nothing to stop: no server.pid in $RUN"; exit 0; }
P=$(cat "$RUN/server.pid")
if ! kill -0 "$P" 2>/dev/null; then echo "already stopped: $(basename "$RUN") (pid $P)"; exit 0; fi
kill "$P"
for i in $(seq 1 120); do kill -0 "$P" 2>/dev/null || break; sleep 1; done
kill -0 "$P" 2>/dev/null && { echo "pid $P still alive after 120 s, SIGKILL"; kill -9 "$P"; sleep 3; }
for i in $(seq 1 30); do (exec 3<>/dev/tcp/127.0.0.1/30000) 2>/dev/null || break; sleep 1; done
echo "stopped: $(basename "$RUN") (pid $P); port 30000 $( (exec 3<>/dev/tcp/127.0.0.1/30000) 2>/dev/null && echo STILL BUSY || echo free)"
