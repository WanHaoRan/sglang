#!/usr/bin/env bash
# Run the trace replay against a baseline server on port 30000 (replay_baseline.py), then summarize the run.
# The counterpart of scripts/start_programs.sh for servers that are not our SGLang fork: same knobs, same output
# layout, same summary. It runs in the SGLang container whatever the server, so every system meets one client.
#
#   host: docker exec sglang_hicache bash /sgl-workspace/sglang/agent_cache/baselines/common/start_client.sh
#
# Knobs (environment variable, default, the flag it sets):
#   BACKEND      (required)  --backend      vllm | continuum | tokencake | openai (backends.py)
#   BACKEND_OPTS -           --backend-opt  space-separated KEY=VALUE options of the backend
#   TRACE RECIPE RATE MAX_ACTIVE DURATION MAX_CONTEXT SEED TAG LANES POOL: as scripts/start_programs.sh
#   Any further replay_programs.py flags go after the script name: start_client.sh --gap-scale 1 --gap-cap 1200
# Output: <server run dir>/programs_<UTC HHMMSS>_<TAG>/client.jsonl, client.log, summary.txt and summary.json, where the
# server run dir is the one named by agent_cache/results/.current_run (written by the system's start_server.sh).
set -euo pipefail

AC=/sgl-workspace/sglang/agent_cache
PORT=${PORT:-30000}
[ -d "$AC" ] || { echo "STOP: $AC not found; run inside the container (docker exec sglang_hicache bash $0)"; exit 1; }
[ -n "${BACKEND:-}" ] || { echo "STOP: set BACKEND (vllm | continuum | tokencake | openai)"; exit 1; }
TRACE=${TRACE:?STOP: set TRACE to a converted program trace}
RECIPE=${RECIPE:-none}; MAX_CONTEXT=${MAX_CONTEXT:-262144}; SEED=${SEED:-0}; TAG=${TAG:-prog}
[ -s "$TRACE" ] || { echo "STOP: no trace at $TRACE"; exit 1; }
FLAGS=(--backend "$BACKEND" --trace "$TRACE" --recipe "$RECIPE" --hints none --max-context "$MAX_CONTEXT"
       --seed "$SEED" --tag "$TAG")
for opt in ${BACKEND_OPTS:-}; do FLAGS+=(--backend-opt "$opt"); done
[ -n "${LANES:-}" ] && FLAGS+=(--lanes "$LANES")
[ -n "${RATE:-}" ] && FLAGS+=(--arrival-rate "$RATE")
[ -n "${MAX_ACTIVE:-}" ] && FLAGS+=(--max-active "$MAX_ACTIVE")
[ -n "${DURATION:-}" ] && FLAGS+=(--duration "$DURATION")
[ -n "${POOL:-}" ] && FLAGS+=(--token-pool "$POOL")

RESULTS="$AC/results"
[ -s "$RESULTS/.current_run" ] || { echo "STOP: $RESULTS/.current_run missing: no server booted yet"; exit 1; }
RUN="$RESULTS/$(cat "$RESULTS/.current_run")"
[ -d "$RUN" ] || { echo "STOP: run dir $RUN is missing"; exit 1; }
# The server lives in another container, so its pid is not visible here: the HTTP check is the liveness test.
curl -sf -o /dev/null --max-time 5 "http://127.0.0.1:$PORT/v1/models" \
  || { echo "STOP: no server answers on port $PORT (start the system's server first)"; exit 1; }
OUT="$RUN/programs_$(date -u +%H%M%S)_$TAG"; mkdir -p "$OUT"

echo "client: ${FLAGS[*]} $* -> $OUT"
cd "$AC/baselines/common"
python3 replay_baseline.py --url "http://127.0.0.1:$PORT" "${FLAGS[@]}" "$@" --output "$OUT/client.jsonl" 2>&1 \
  | tee "$OUT/client.log"
python3 "$AC/scripts/program_summary.py" "$OUT/client.jsonl" --json "$OUT/summary.json" | tee "$OUT/summary.txt"
