#!/usr/bin/env bash
# Run the replay client (replay_agentic.py) against the server on port 30000 and print a per-turn view.
#
#   host:      docker exec -it sglang_hicache bash /sgl-workspace/sglang/agent_cache/scripts/start_client.sh
#   container: bash /sgl-workspace/sglang/agent_cache/scripts/start_client.sh
#
# Knobs (environment variable, default, the replay_agentic.py flag it sets; the defaults are a 4-user smoke run):
#   TRACE         traces/lmcache_agentic_trace.json  --trace         the 32K-cut trace; campaigns 13-17 used the 262K one
#   USERS         4      --num-users     users, each replaying a different conversation
#   TURNS         10     --turns         turns per user, at most (0 = all recorded)
#   ARRIVAL       0      --arrival-rate  users per second, Poisson (0 = all at once)
#   GAP           1.0    --gap-scale     sleep = recorded gap x GAP, at most 1,200 s
#   ORACLE_HINTS  0      --oracle-hints  1 = each turn sends the next turn's rid and gap as kv_hints
#   C             4      --concurrency   most users in session at once, idle ones included (0 = no cap)
#   TIMEOUT       0      --timeout       seconds; then no user starts and no request is sent (0 = none)
#   SEED          0      --seed          the arrival times and which user replays which conversation
#   TAG           smoke  --tag           request ids <TAG>-c<user>-t<turn>
# Output: <server run dir>/client_<UTC HHMMSS>_<TAG>/client.jsonl + client.log, where the server run dir is the one
# named by agent_cache/results/.current_run (written by start_server.sh), so every client run sits under the boot it measured.
set -euo pipefail

AC=/sgl-workspace/sglang/agent_cache
PORT=30000
[ -d "$AC" ] || { echo "STOP: $AC not found; run inside the container (docker exec -it sglang_hicache bash $0)"; exit 1; }
# knobs of the client before its trim (2026-10-17): refuse them rather than run a different workload
for v in NCONV OFFSET MAX_SECONDS GAP_CAP K THINKING CHECK_IDS TURNS_RANGE ARRIVAL_UNTIL STOP_AFTER_USERS SESSION_TASKS \
         CONTEXT_POLICY CAP_MODE ORDER START_PHASE VALIDATE_ONLY; do
  [ -z "${!v:-}" ] || { echo "STOP: $v is not a client knob any more (replay_agentic.py was trimmed; knobs: $0)"; exit 1; }
done
TRACE=${TRACE:-$AC/traces/lmcache_agentic_trace.json}
USERS=${USERS:-4}; TURNS=${TURNS:-10}; ARRIVAL=${ARRIVAL:-0}; GAP=${GAP:-1.0}; ORACLE_HINTS=${ORACLE_HINTS:-0}
C=${C:-4}; TIMEOUT=${TIMEOUT:-0}; SEED=${SEED:-0}; TAG=${TAG:-smoke}

RESULTS="$AC/results"
[ -s "$RESULTS/.current_run" ] || { echo "STOP: $RESULTS/.current_run missing: no server booted by start_server.sh yet"; exit 1; }
RUN="$RESULTS/$(cat "$RESULTS/.current_run")"
# A pid counts as running unless it is gone or a zombie: the container's PID 1 (a bare /bin/zsh, no init) does not
# reap orphans, so an exited server stays a zombie that kill -0 still reports alive.
alive() { local s; s=$(ps -o stat= -p "$1" 2>/dev/null | tr -d ' '); [ -n "$s" ] && [ "${s#Z}" = "$s" ]; }
[ -f "$RUN/server.pid" ] && alive "$(cat "$RUN/server.pid")" || { echo "STOP: the server of $RUN is not running (start_server.sh first)"; exit 1; }
OUT="$RUN/client_$(date -u +%H%M%S)_$TAG"; mkdir -p "$OUT"
curl -sf -o /dev/null --max-time 3 "http://127.0.0.1:$PORT/health" || { echo "STOP: no server on port $PORT (start_server.sh first)"; exit 1; }

FLAGS=(--trace "$TRACE" --num-users "$USERS" --turns "$TURNS" --arrival-rate "$ARRIVAL" --gap-scale "$GAP"
  --concurrency "$C" --timeout "$TIMEOUT" --seed "$SEED")
[ "$ORACLE_HINTS" = 1 ] && FLAGS+=(--oracle-hints)
echo "client: ${FLAGS[*]} -> $OUT"
cd "$AC/scripts"
python3 replay_agentic.py --url "http://127.0.0.1:$PORT" "${FLAGS[@]}" \
  --tag "$TAG" --output "$OUT/client.jsonl" 2>&1 | tee "$OUT/client.log"

# per-turn view: when it was sent, the gap slept before it, TTFT, prompt/cached tokens and reprefill
python3 - "$OUT/client.jsonl" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
print(f"{'conv':>4} {'turn':>4} {'t_send':>8} {'gap':>6} {'ttft':>7} {'prompt':>7} {'cached':>7} {'reprefill':>9}")
for d in rows:
    if d["kind"] != "turn":
        continue
    if "t_send" not in d:  # the request raised before/while sending (e.g. a keep-alive reset): no timing fields
        print(f"{d['conv']:>4} {d['turn']:>4} {'ERROR':>8} {str(d.get('error', ''))[:70]}")
        continue
    print(f"{d['conv']:>4} {d['turn']:>4} {d['t_send']:>8.1f} {d['gap_slept']:>6.1f} {(d.get('ttft') or 0):>7.2f} "
          f"{d.get('prompt_tokens', 0):>7} {d.get('cached_tokens', 0):>7} {str(d.get('reply_reprefilled', '-')):>9}")
s = [d for d in rows if d["kind"] == "summary"]
if s:
    print("summary:", json.dumps(s[-1]))
EOF
