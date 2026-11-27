#!/usr/bin/env bash
# Pre-flight of the engine copy with the admission pass-over, before campaign 44's arm. Host side, GPU free:
#   bash /home/wanhr/sglang/agent_cache/results/campaign44/preflight_passover.sh engine_unified_po_20261127 \
#     >> /home/wanhr/sglang/agent_cache/results/campaign44/preflight.log 2>&1
# 1. Unit tests on the frozen copy: the two scheduler components, the scheduler tests that build a Scheduler without
#    __init__, the weight-version tests (they read the in-flight requests) and the flag checks.
# 2. One ten-minute smoke with every switch of settings.env on (the unified switches and the pass-over), the arm's
#    flags, GPU pool 200,000 tokens, host 48 GB, 64 users, cleaner at 90/80 % so the GPU pool is what fills. Passes
#    with no client error, no traceback logged before the server began to exit, and at least one pass_over line.
# The engine with the pass-over off is engine_unified_20261126 plus code that is not reached; no switches-off smoke.
# PREFLIGHT_SKIP (optional): a test name to leave out (pytest -k "not NAME"), for a test of a change this copy lacks.
set -uo pipefail
ENGINE=${1:?engine folder under agent_cache/frozen}
A=/home/wanhr/sglang/agent_cache
C=/sgl-workspace/sglang/agent_cache
D=$A/results/campaign44
log() { echo "$(date -u +%FT%TZ) $*"; }
stop_server() { docker exec sglang_hicache bash -c "cd $C/scripts && bash stop_server.sh" > /dev/null 2>&1; : > "$A/results/.current_run"; }
serving_tracebacks() { awk '/Gracefully exiting/ {exit} /Traceback/ {n++} END {print n + 0}' "$1"; }

[ -d "$A/frozen/$ENGINE/python/sglang" ] || { log "STOP: frozen/$ENGINE is missing"; exit 1; }
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
[ "$used" -lt 2000 ] || { log "STOP: the GPU holds $used MiB (an arm is running)"; exit 1; }
(exec 3<>/dev/tcp/127.0.0.1/30000) 2>/dev/null && { log "STOP: something listens on port 30000"; exit 1; }
ENVS=()
while IFS= read -r line; do
  case "$line" in ''|'#'*) ;; *) ENVS+=(-e "$line") ;; esac
done < "$D/settings.env"
FLAGS=$(sed -e 's/^#HOLD //' "$D/queue.txt" | grep '^unified_load_po ' | cut -d' ' -f6-)
[ -n "$FLAGS" ] || { log "STOP: campaign 44's queue line was not found"; exit 1; }
grep -q '^SGLANG_PREFILL_PASS_OVER_LOOKAHEAD=[1-9]' "$D/settings.env" || { log "STOP: the pass-over switch is not on in settings.env"; exit 1; }
log "pre-flight of frozen/$ENGINE"

UNIT=$D/preflight_unit_tests.log; : > "$UNIT"
for f in managers/scheduler_components/test_admission_pass_over managers/scheduler_components/test_prefill_load_hold \
         managers/test_scheduler_hicache_events managers/test_scheduler_chunked_req_gate utils/test_weight_versions \
         mem_cache/test_oracle_v2_flags mem_cache/test_next_turn_prefetch mem_cache/test_next_use_eviction; do
  docker exec -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES=9 -e PYTHONPATH="$C/frozen/$ENGINE/python" sglang_hicache \
    bash -c "cd /sgl-workspace/sglang && timeout 1800 python3 -m pytest -q -p no:cacheprovider ${PREFLIGHT_SKIP:+-k \"not $PREFLIGHT_SKIP\"} test/registered/unit/$f.py" >> "$UNIT" 2>&1
  rc=$?
  log "unit tests $f: rc=$rc: $(grep -E 'passed|failed|error' "$UNIT" | tail -1)"
  [ "$rc" -eq 0 ] || { log "STOP: unit tests $f failed ($UNIT)"; exit 1; }
done

log "smoke: three_tier_wc_oracle_v2_nu_owned, every switch on with the pass-over, GPU 200,000 tokens, host 48 GB, 64 users, 600 s"
docker exec sglang_hicache bash -c 'find /mnt/ssd/hicache_l3 -mindepth 1 -delete 2>/dev/null; true'
docker exec "${ENVS[@]}" \
  -e MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 -e CTX=262144 -e KV_DTYPE=auto -e TEMPLATE= -e HOST_GB=48 -e L3_CLEANER_PCT=90,80 \
  -e SGLANG_TIMEOUT_KEEP_ALIVE=3600 -e ENGINE_PYTHONPATH="$C/frozen/$ENGINE/python" \
  -e SERVER_EXTRA="$(sed -e 's/--max-total-tokens [0-9]*/--max-total-tokens 200000/' <<< "$FLAGS")" \
  sglang_hicache bash -c "cd $C/scripts && bash start_server.sh three_tier_wc_oracle_v2_nu_owned NAT160" > "$D/preflight_smoke_boot.log" 2>&1 \
  || { log "STOP: the smoke server did not boot ($D/preflight_smoke_boot.log): $(grep -a 'Error' "$D/preflight_smoke_boot.log" | tail -2 | cut -c1-200)"; stop_server; exit 1; }
RUN=$(cat "$A/results/.current_run"); S=$A/results/$RUN/server.log
docker exec -e TRACE=$C/data/converted/freeinference_16w.programs.jsonl.gz -e RECIPE=none -e RATE=1 -e MAX_ACTIVE=64 \
  -e DURATION=600 -e HINTS=dist -e MAX_CONTEXT=65536 -e SEED=0 -e TAG=smoke_po \
  sglang_hicache bash -c "cd $C/scripts && bash start_programs.sh --gap-scale 1 --gap-cap 1200 --hint-no-call" > "$D/preflight_smoke_client.log" 2>&1
rc=$?
OUT=$(ls -d "$A/results/$RUN"/programs_*_smoke_po 2>/dev/null | tail -1)
SUM=$(grep -a '"kind":"summary"' "$OUT/client.jsonl" 2>/dev/null | tail -1)
ERR=$(grep -o '"errors":[0-9]*' <<< "$SUM" | cut -d: -f2)
PO=$(grep -ac 'HICACHE_EVT pass_over' "$S")
log "smoke: client rc=$rc, run $RUN; requests $(grep -o '"profile":[0-9]*' <<< "$SUM" | cut -d: -f2), errors ${ERR:-?}; tracebacks while serving $(serving_tracebacks "$S"), in all $(grep -ac Traceback "$S"); pass_over lines $PO, requests admitted behind a refused one $(grep -a 'HICACHE_EVT pass_over' "$S" | grep -o 'admitted=[0-9]*' | cut -d= -f2 | paste -sd+ | bc 2>/dev/null), largest pass count $(grep -a 'HICACHE_EVT pass_over' "$S" | grep -o 'passes=[0-9]*' | cut -d= -f2 | sort -n | tail -1); load_hold $(grep -ac 'HICACHE_EVT load_hold' "$S")"
log "smoke: $(python3 - "$OUT/client.jsonl" <<'PY'
import json, statistics as st, sys
t = [d["ttft"] for d in map(json.loads, open(sys.argv[1])) if d.get("kind") == "req" and d.get("ttft") is not None]
t.sort()
print(f"requests with a first token {len(t)}, TTFT mean {st.mean(t):.2f} s, p50 {t[len(t) // 2]:.2f}, p90 {t[int(len(t) * 0.9)]:.2f}, max {t[-1]:.1f}" if t else "no request")
PY
)"
stop_server
[ "$rc" -eq 0 ] && [ "${ERR:-1}" = 0 ] && [ "$(serving_tracebacks "$S")" = 0 ] \
  || { log "STOP: the smoke failed (client rc $rc, errors ${ERR:-?}, tracebacks while serving $(serving_tracebacks "$S")): $S"; exit 1; }
[ "$PO" -gt 0 ] || { log "STOP: no request was admitted behind a refused one ($S)"; exit 1; }
log "pre-flight passed for frozen/$ENGINE"
