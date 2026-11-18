#!/usr/bin/env bash
# Run the program-trace replay client (replay_programs.py) against the server on port 30000, then summarize the run.
#
#   host:      docker exec -it sglang_hicache bash /sgl-workspace/sglang/agent_cache/scripts/start_programs.sh
#   container: bash /sgl-workspace/sglang/agent_cache/scripts/start_programs.sh
#
# Knobs (environment variable, default, the replay_programs.py flag it sets; MAINSTREAM_REPLAY.md explains each):
#   TRACE      data/converted/agentx_062126_256k.programs.jsonl.gz   --trace   a converted program trace
#   RECIPE     agentx-mvp   --recipe     agentx-mvp | mori | tracelab | none: a published setup's defaults
#   LANES      -            --lanes      session trees kept live (unset: the recipe's)
#   RATE       -            --arrival-rate  trees per second, Poisson (replaces the recipe's lanes)
#   MAX_ACTIVE -            --max-active    with RATE: most trees at once
#   DURATION   -            --duration   seconds of profiling (unset: the recipe's; 0 = until done)
#   HINTS      none         --hints      none | program | release | dist | oracle (dist needs a server with the hint
#                                        reply-key source: start_server.sh's *_table / *_jit / *_owned arms)
#   MAX_CONTEXT 262144      --max-context   the served model's window: trees with a longer request are dropped
#   SEED       0            --seed
#   TAG        prog         --tag        request ids <TAG>-c<stream>-t<index>
#   POOL       -            --token-pool a token pool for the served model (make_token_pool.py)
#   Any further replay_programs.py flags go after the script name: start_programs.sh --plays 10
# Output: <server run dir>/programs_<UTC HHMMSS>_<TAG>/client.jsonl, client.log and summary.txt, where the server run dir
# is the one named by agent_cache/results/.current_run (written by start_server.sh).
set -euo pipefail

AC=/sgl-workspace/sglang/agent_cache
PORT=30000
[ -d "$AC" ] || { echo "STOP: $AC not found; run inside the container (docker exec -it sglang_hicache bash $0)"; exit 1; }
TRACE=${TRACE:-$AC/data/converted/agentx_062126_256k.programs.jsonl.gz}
RECIPE=${RECIPE:-agentx-mvp}; HINTS=${HINTS:-none}; MAX_CONTEXT=${MAX_CONTEXT:-262144}; SEED=${SEED:-0}; TAG=${TAG:-prog}
[ -s "$TRACE" ] || { echo "STOP: no trace at $TRACE (MAINSTREAM_REPLAY.md 6.2 converts one)"; exit 1; }
FLAGS=(--trace "$TRACE" --recipe "$RECIPE" --hints "$HINTS" --max-context "$MAX_CONTEXT" --seed "$SEED" --tag "$TAG")
[ -n "${LANES:-}" ] && FLAGS+=(--lanes "$LANES")
[ -n "${RATE:-}" ] && FLAGS+=(--arrival-rate "$RATE")
[ -n "${MAX_ACTIVE:-}" ] && FLAGS+=(--max-active "$MAX_ACTIVE")
[ -n "${DURATION:-}" ] && FLAGS+=(--duration "$DURATION")
[ -n "${POOL:-}" ] && FLAGS+=(--token-pool "$POOL")

RESULTS="$AC/results"
[ -s "$RESULTS/.current_run" ] || { echo "STOP: $RESULTS/.current_run missing: no server booted by start_server.sh yet"; exit 1; }
RUN="$RESULTS/$(cat "$RESULTS/.current_run")"
# A pid counts as running unless it is gone or a zombie (the container's PID 1 does not reap orphans).
alive() { local s; s=$(ps -o stat= -p "$1" 2>/dev/null | tr -d ' '); [ -n "$s" ] && [ "${s#Z}" = "$s" ]; }
[ -f "$RUN/server.pid" ] && alive "$(cat "$RUN/server.pid")" || { echo "STOP: the server of $RUN is not running (start_server.sh first)"; exit 1; }
curl -sf -o /dev/null --max-time 3 "http://127.0.0.1:$PORT/health" || { echo "STOP: no server on port $PORT (start_server.sh first)"; exit 1; }
OUT="$RUN/programs_$(date -u +%H%M%S)_$TAG"; mkdir -p "$OUT"

echo "client: ${FLAGS[*]} $* -> $OUT"
cd "$AC/scripts"
python3 replay_programs.py --url "http://127.0.0.1:$PORT" "${FLAGS[@]}" "$@" --output "$OUT/client.jsonl" 2>&1 \
  | tee "$OUT/client.log"
python3 program_summary.py "$OUT/client.jsonl" --json "$OUT/summary.json" | tee "$OUT/summary.txt"
