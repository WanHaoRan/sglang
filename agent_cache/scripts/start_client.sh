#!/usr/bin/env bash
# Run the gap-faithful replay client (RUNBOOK 4.6) against the server on port 30000 and print a per-turn view.
#
#   host:      docker exec -it sglang_hicache bash /sgl-workspace/sglang/agent_cache/scripts/start_client.sh
#   container: bash /sgl-workspace/sglang/agent_cache/scripts/start_client.sh
#
# Knobs (environment variables, defaults = the 4-conversation observation run):
#   NCONV=4  TURNS=10  C=4  GAP=1.0  GAP_CAP=0  K=0  THINKING=off  CHECK_IDS=1  ARRIVAL=0  TAG=observe_c4  OFFSET=0  SEED=0  MODEL=<the server's>  MAX_SECONDS=0
#   ORACLE_HINTS=0  TURNS_RANGE=  TURNS_SEED=0
#   ARRIVAL_UNTIL=0  SESSION_TASKS=  SESSION_SEED=0  TASK_GAP=-1  CONTEXT_POLICY=none  MAX_CONTEXT=0  CONTEXT_TRIGGER=0.9
#   COMPACT_TOKENS=4096  COMPACT_KEEP=20000  COMPACT_MODE=cached  TRUNCATE_TO=0.5
#   ARRIVAL_UNTIL>0 (with ARRIVAL>0) keeps new users arriving until that many seconds, cycling through the NCONV conversations
#   (--arrival-until); C caps the live users, so ARRIVAL above the server's capacity holds C users live. SESSION_TASKS=LO:HI
#   chains that many tasks per user into one growing context (--session-tasks); TASK_GAP<0 draws the idle between tasks from
#   the trace's within-task gaps, >=0 fixes it (trace seconds, scaled by GAP). CONTEXT_POLICY (none | end | compact |
#   truncate) is what a session does when its prompt would not fit MAX_CONTEXT (0 = the server's max_model_len); see
#   replay_agentic.py. The compact defaults suit a 131072-token window; at 32K-40K use COMPACT_KEEP of about 5000.
#   STOP_AFTER_USERS=N stops the run like MAX_SECONDS once N users have finished (--stop-after-users); MAX_SECONDS stays the guard.
#   VALIDATE_ONLY=1 checks the flags and plans every session (replay_agentic.py --validate) without a server, then exits;
#   with a context policy and MAX_CONTEXT=0 it checks against CTX. run_compare.sh runs it before booting any arm.
#   TURNS_RANGE=LO:HI gives each conversation min(its recorded turns, a seeded uniform draw in [LO, HI]) turns (--turns-range).
#   ORACLE_HINTS=1 sends each non-last turn's next rid and next gap as a kv_hints envelope (--oracle-hints, agent_cache/ORACLE_PLAN.md).
#   MAX_SECONDS>0 is the wall cap (--max-seconds): conversations stop at their next turn boundary after that many seconds, in-flight
#   requests complete, every completed turn is kept and the summary line carries "capped": true.
#   GAP_CAP>0 caps every (scaled) sleep at that many seconds (the trace has 300 s gaps that GAP=10 turns into 50 min).
#   THINKING must match the server's template (nothink -> off, think -> on). CHECK_IDS=1 returns ~20K ids per turn: fine for a
#   short run, set 0 for a measured cell. ARRIVAL>0 = open loop (conversations/s), C becomes the cap.
#   CAP_MODE CAP_TARGET CAP_INTERVAL CAP_GAIN CAP_STEP CAP_MIN CAP_MAX ADMIT_SPACING ORDER ORDER_SEED START_PHASE START_USERS
#   START_SEED START_SERVICE each reach the client as their flag only when set (unset = the client's defaults; see
#   replay_agentic.py). CAP_MODE=backlog moves the live-user cap, from C, to hold the client-measured backlog of requests
#   awaiting their first token at CAP_TARGET (--cap-mode). ORDER=shuffle gives every later block of NCONV users a seeded
#   permutation of the conversations (--order). START_PHASE=stationary starts the first START_USERS (0 = C) users at a
#   time-uniform point of their session (--start-phase).
# Output: <server run dir>/client_<UTC HHMMSS>_<TAG>/client.jsonl (one line per turn) + client.log, where the server run dir is the one
# named by agent_cache/results/.current_run (written by start_server.sh), so every client run sits under the boot it measured.
set -euo pipefail

AC=/sgl-workspace/sglang/agent_cache
PORT=30000
NCONV=${NCONV:-4}; TURNS=${TURNS:-10}; C=${C:-4}; GAP=${GAP:-1.0}; K=${K:-0}
THINKING=${THINKING:-off}; CHECK_IDS=${CHECK_IDS:-1}; ARRIVAL=${ARRIVAL:-0}; GAP_CAP=${GAP_CAP:-0}
TAG=${TAG:-observe_c${C}}; OFFSET=${OFFSET:-0}; SEED=${SEED:-0}; MAX_SECONDS=${MAX_SECONDS:-0}; ORACLE_HINTS=${ORACLE_HINTS:-0}
TURNS_RANGE=${TURNS_RANGE:-}; TURNS_SEED=${TURNS_SEED:-0}
ARRIVAL_UNTIL=${ARRIVAL_UNTIL:-0}; SESSION_TASKS=${SESSION_TASKS:-}; SESSION_SEED=${SESSION_SEED:-0}; TASK_GAP=${TASK_GAP:--1}
CONTEXT_POLICY=${CONTEXT_POLICY:-none}; MAX_CONTEXT=${MAX_CONTEXT:-0}; CONTEXT_TRIGGER=${CONTEXT_TRIGGER:-0.9}
COMPACT_TOKENS=${COMPACT_TOKENS:-4096}; COMPACT_KEEP=${COMPACT_KEEP:-20000}; COMPACT_MODE=${COMPACT_MODE:-cached}
TRUNCATE_TO=${TRUNCATE_TO:-0.5}; STOP_AFTER_USERS=${STOP_AFTER_USERS:-0}
TRACE=${TRACE:-$AC/traces/lmcache_agentic_trace.json}   # the committed 32K-cut trace; campaign 7 uses the 262K re-conversion

[ -d "$AC" ] || { echo "STOP: $AC not found; run inside the container (docker exec -it sglang_hicache bash $0)"; exit 1; }

EXTRA=()
[ "$CHECK_IDS" = 1 ] && EXTRA+=(--check-ids)
[ "$ARRIVAL" != 0 ] && EXTRA+=(--arrival-rate "$ARRIVAL")
[ "$GAP_CAP" != 0 ] && EXTRA+=(--gap-cap "$GAP_CAP")
[ "$MAX_SECONDS" != 0 ] && EXTRA+=(--max-seconds "$MAX_SECONDS")
[ "$ORACLE_HINTS" = 1 ] && EXTRA+=(--oracle-hints)
[ -n "$TURNS_RANGE" ] && EXTRA+=(--turns-range "$TURNS_RANGE" --turns-seed "$TURNS_SEED")
[ "$ARRIVAL_UNTIL" != 0 ] && EXTRA+=(--arrival-until "$ARRIVAL_UNTIL")
[ "$STOP_AFTER_USERS" != 0 ] && EXTRA+=(--stop-after-users "$STOP_AFTER_USERS")
[ -n "$SESSION_TASKS" ] && EXTRA+=(--session-tasks "$SESSION_TASKS" --session-seed "$SESSION_SEED" --task-gap "$TASK_GAP")
[ "$CONTEXT_POLICY" != none ] && EXTRA+=(--context-policy "$CONTEXT_POLICY" --max-context "$MAX_CONTEXT"
  --context-trigger "$CONTEXT_TRIGGER" --compact-tokens "$COMPACT_TOKENS" --compact-keep "$COMPACT_KEEP"
  --compact-mode "$COMPACT_MODE" --truncate-to "$TRUNCATE_TO")
opt() { [ -z "$2" ] || EXTRA+=("$1" "$2"); }   # a flag only when its knob is set
opt --cap-mode "${CAP_MODE:-}"; opt --cap-target "${CAP_TARGET:-}"; opt --cap-interval "${CAP_INTERVAL:-}"
opt --cap-gain "${CAP_GAIN:-}"; opt --cap-step "${CAP_STEP:-}"; opt --cap-min "${CAP_MIN:-}"; opt --cap-max "${CAP_MAX:-}"
opt --admit-spacing "${ADMIT_SPACING:-}"; opt --order "${ORDER:-}"; opt --order-seed "${ORDER_SEED:-}"
opt --start-phase "${START_PHASE:-}"; opt --start-users "${START_USERS:-}"; opt --start-seed "${START_SEED:-}"
opt --start-service "${START_SERVICE:-}"
FLAGS=(--model "${MODEL:-Qwen/Qwen3-32B-FP8}" --trace "$TRACE"
  --num-conversations "$NCONV" --max-turns "$TURNS" --offset "$OFFSET" --seed "$SEED"
  --concurrency "$C" --gap-scale "$GAP" --control-every "$K" --thinking "$THINKING" "${EXTRA[@]}")

if [ "${VALIDATE_ONLY:-0}" = 1 ]; then
  VAL=(--validate)
  [ "$CONTEXT_POLICY" != none ] && [ "$MAX_CONTEXT" = 0 ] && VAL+=(--max-context "${CTX:?VALIDATE_ONLY with a context policy needs MAX_CONTEXT or CTX}")
  cd "$AC/scripts" && exec python3 replay_agentic.py "${FLAGS[@]}" "${VAL[@]}"
fi

RESULTS="$AC/results"
[ -s "$RESULTS/.current_run" ] || { echo "STOP: $RESULTS/.current_run missing: no server booted by start_server.sh yet"; exit 1; }
RUN="$RESULTS/$(cat "$RESULTS/.current_run")"
# A pid counts as running unless it is gone or a zombie: the container's PID 1 (a bare /bin/zsh, no init) does not
# reap orphans, so an exited server stays a zombie that kill -0 still reports alive.
alive() { local s; s=$(ps -o stat= -p "$1" 2>/dev/null | tr -d ' '); [ -n "$s" ] && [ "${s#Z}" = "$s" ]; }
[ -f "$RUN/server.pid" ] && alive "$(cat "$RUN/server.pid")" || { echo "STOP: the server of $RUN is not running (start_server.sh first)"; exit 1; }
OUT="$RUN/client_$(date -u +%H%M%S)_$TAG"; mkdir -p "$OUT"
curl -sf -o /dev/null --max-time 3 "http://127.0.0.1:$PORT/health" || { echo "STOP: no server on port $PORT (start_server.sh first)"; exit 1; }

echo "client: $NCONV conversations x <= $TURNS turns, concurrency $C, gap x$GAP cap ${GAP_CAP}s, wall cap ${MAX_SECONDS}s, controls every ${K:-0}, thinking $THINKING, oracle hints $ORACLE_HINTS, turns ${TURNS_RANGE:-fixed} (seed $TURNS_SEED), arrival ${ARRIVAL}/s until ${ARRIVAL_UNTIL}s, tasks per user ${SESSION_TASKS:-1} (seed $SESSION_SEED, gap ${TASK_GAP}s), context $CONTEXT_POLICY (window ${MAX_CONTEXT})${CAP_MODE:+, cap $CAP_MODE}${ORDER:+, order $ORDER}${START_PHASE:+, start $START_PHASE} -> $OUT"
cd "$AC/scripts"
python3 replay_agentic.py --url "http://127.0.0.1:$PORT" "${FLAGS[@]}" \
  --tag "$TAG" --output "$OUT/client.jsonl" 2>&1 | tee "$OUT/client.log"

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
