#!/usr/bin/env bash
# Run the gap-faithful replay client (RUNBOOK 4.6) against the server on port 30000 and print a per-turn view.
#
#   host:      docker exec -it sglang_hicache bash /sgl-workspace/sglang/agent_cache/scripts/start_client.sh
#   container: bash /sgl-workspace/sglang/agent_cache/scripts/start_client.sh
#
# Knobs (environment variables, defaults = the 4-conversation observation run):
#   NCONV=4  TURNS=10  C=4  GAP=1.0  GAP_CAP=0  K=0  THINKING=off  CHECK_IDS=1  ARRIVAL=0  TAG=observe_c4  OFFSET=0  SEED=0  MODEL=<the server's>  MAX_SECONDS=0
#   MAX_SECONDS>0 is the wall cap (--max-seconds): conversations stop at their next turn boundary after that many seconds, in-flight
#   requests complete, every completed turn is kept and the summary line carries "capped": true.
#   GAP_CAP>0 caps every (scaled) sleep at that many seconds (the trace has 300 s gaps that GAP=10 turns into 50 min).
#   THINKING must match the server's template (nothink -> off, think -> on). CHECK_IDS=1 returns ~20K ids per turn: fine for a
#   short run, set 0 for a measured cell. ARRIVAL>0 = open loop (conversations/s), C becomes the cap.
# Output: <server run dir>/client_<UTC HHMMSS>_<TAG>/client.jsonl (one line per turn) + client.log, where the server run dir is the one
# named by agent_cache/results/.current_run (written by start_server.sh), so every client run sits under the boot it measured.
set -euo pipefail

AC=/sgl-workspace/sglang/agent_cache
PORT=30000
NCONV=${NCONV:-4}; TURNS=${TURNS:-10}; C=${C:-4}; GAP=${GAP:-1.0}; K=${K:-0}
THINKING=${THINKING:-off}; CHECK_IDS=${CHECK_IDS:-1}; ARRIVAL=${ARRIVAL:-0}; GAP_CAP=${GAP_CAP:-0}
TAG=${TAG:-observe_c${C}}; OFFSET=${OFFSET:-0}; SEED=${SEED:-0}; MAX_SECONDS=${MAX_SECONDS:-0}
TRACE=${TRACE:-$AC/traces/lmcache_agentic_trace.json}   # the committed 32K-cut trace; campaign 7 uses the 262K re-conversion

[ -d "$AC" ] || { echo "STOP: $AC not found; run inside the container (docker exec -it sglang_hicache bash $0)"; exit 1; }
RESULTS="$AC/results"
[ -s "$RESULTS/.current_run" ] || { echo "STOP: $RESULTS/.current_run missing: no server booted by start_server.sh yet"; exit 1; }
RUN="$RESULTS/$(cat "$RESULTS/.current_run")"
[ -f "$RUN/server.pid" ] && kill -0 "$(cat "$RUN/server.pid")" 2>/dev/null || { echo "STOP: the server of $RUN is not running (start_server.sh first)"; exit 1; }
OUT="$RUN/client_$(date -u +%H%M%S)_$TAG"; mkdir -p "$OUT"
curl -sf -o /dev/null --max-time 3 "http://127.0.0.1:$PORT/health" || { echo "STOP: no server on port $PORT (start_server.sh first)"; exit 1; }

EXTRA=()
[ "$CHECK_IDS" = 1 ] && EXTRA+=(--check-ids)
[ "$ARRIVAL" != 0 ] && EXTRA+=(--arrival-rate "$ARRIVAL")
[ "$GAP_CAP" != 0 ] && EXTRA+=(--gap-cap "$GAP_CAP")
[ "$MAX_SECONDS" != 0 ] && EXTRA+=(--max-seconds "$MAX_SECONDS")

echo "client: $NCONV conversations x <= $TURNS turns, concurrency $C, gap x$GAP cap ${GAP_CAP}s, wall cap ${MAX_SECONDS}s, controls every ${K:-0}, thinking $THINKING -> $OUT"
cd "$AC/scripts"
python3 replay_agentic.py \
  --url "http://127.0.0.1:$PORT" --model "${MODEL:-Qwen/Qwen3-32B-FP8}" \
  --trace "$TRACE" \
  --num-conversations "$NCONV" --max-turns "$TURNS" --offset "$OFFSET" --seed "$SEED" \
  --concurrency "$C" --gap-scale "$GAP" --control-every "$K" --thinking "$THINKING" \
  --tag "$TAG" --output "$OUT/client.jsonl" "${EXTRA[@]}" 2>&1 | tee "$OUT/client.log"

# per-turn view: when it was sent, the gap slept before it, TTFT, prompt/cached tokens, reply reuse (exact ids if CHECK_IDS=1) and reprefill
python3 - "$OUT/client.jsonl" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
print(f"{'conv':>4} {'turn':>4} {'t_send':>8} {'gap':>6} {'ttft':>7} {'prompt':>7} {'cached':>7} {'reply_reused':>13} {'reprefill':>9}")
for d in rows:
    if d["kind"] != "turn":
        continue
    if "t_send" not in d:  # the request raised before/while sending (e.g. a keep-alive reset): no timing fields
        print(f"{d['conv']:>4} {d['turn']:>4} {'ERROR':>8} {str(d.get('error', ''))[:70]}")
        continue
    reused = f"{d.get('reply_ids_reused')}/{d.get('reply_ids_total')}" if d.get("reply_ids_total") is not None else "-"
    print(f"{d['conv']:>4} {d['turn']:>4} {d['t_send']:>8.1f} {d['gap_slept']:>6.1f} {(d.get('ttft') or 0):>7.2f} "
          f"{d.get('prompt_tokens', 0):>7} {d.get('cached_tokens', 0):>7} {reused:>13} {str(d.get('reply_reprefilled', '-')):>9}")
s = [d for d in rows if d["kind"] == "summary"]
if s:
    print("summary:", json.dumps(s[-1]))
EOF
