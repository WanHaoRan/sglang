#!/usr/bin/env bash
# Pre-flight of the load-hold engine copy before its campaign arm. Host side, GPU free:
#   bash /home/wanhr/sglang/agent_cache/results/campaign39/preflight_loadhold.sh engine_loadhold_20261126 \
#     >> /home/wanhr/sglang/agent_cache/results/campaign39/preflight.log 2>&1
# 1. unit tests on the frozen copy; 2. tier check with the hold on, quiet and then with six generations decoding
# in the background, so that the checked prompt's batch is held: a prompt served from host memory and from the
# SSD must still give the warm run's logprobs; 3. two ten-minute smokes of the current design on small pools,
# hold off and hold on (same users), for errors and a first look at TTFT. Stops at the first failure (STOP line).
set -uo pipefail
ENGINE=${1:?engine folder under agent_cache/frozen}
A=/home/wanhr/sglang/agent_cache
C=/sgl-workspace/sglang/agent_cache
D=$A/results/campaign39
log() { echo "$(date -u +%FT%TZ) $*"; }
stop_server() { docker exec sglang_hicache bash -c "cd $C/scripts && bash stop_server.sh" > /dev/null 2>&1; : > "$A/results/.current_run"; }

[ -d "$A/frozen/$ENGINE/python/sglang" ] || { log "STOP: frozen/$ENGINE is missing"; exit 1; }
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
[ "$used" -lt 2000 ] || { log "STOP: the GPU holds $used MiB (an arm is running)"; exit 1; }
(exec 3<>/dev/tcp/127.0.0.1/30000) 2>/dev/null && { log "STOP: something listens on port 30000"; exit 1; }
ENVS=(); while IFS= read -r line; do case "$line" in ''|'#'*) ;; *) ENVS+=(-e "$line") ;; esac; done < "$D/settings.env"
FLAGS=$(sed -e 's/^#HOLD //' "$D/queue.txt" | grep '^ec_full_load ' | cut -d' ' -f6-)
[ -n "$FLAGS" ] || { log "STOP: no ec_full_load line in $D/queue.txt"; exit 1; }
log "pre-flight of frozen/$ENGINE; settings: $(grep SGLANG "$D/settings.env" | tr '\n' ' ')"

# 1. Unit tests on the frozen copy.
UNIT=$D/preflight_unit_tests.log; : > "$UNIT"
for f in managers/scheduler_components/test_prefill_load_hold mem_cache/test_next_use_eviction \
         mem_cache/test_next_turn_prefetch mem_cache/test_storage_read_scheduler mem_cache/test_storage_read_dispatch; do
  docker exec -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH="$C/frozen/$ENGINE/python" sglang_hicache \
    bash -c "cd /sgl-workspace/sglang && timeout 1800 python3 -m pytest -q -p no:cacheprovider test/registered/unit/$f.py" >> "$UNIT" 2>&1
  rc=$?
  log "unit tests $f: rc=$rc: $(grep -E 'passed|failed|error' "$UNIT" | tail -1)"
  [ "$rc" -eq 0 ] || { log "STOP: unit tests $f failed ($UNIT)"; exit 1; }
done

# 2. Tier check, hold on: quiet, then with background decoding.
log "tier check: three_tier_wc on frozen/$ENGINE, GPU 100,000 tokens, host 30 GB, hold on"
docker exec sglang_hicache bash -c 'find /mnt/ssd/hicache_l3 -mindepth 1 -delete 2>/dev/null; true'
docker exec "${ENVS[@]}" \
  -e MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 -e CTX=262144 -e KV_DTYPE=auto -e TEMPLATE= -e HOST_GB=30 -e L3_CLEANER_PCT=90,80 \
  -e L1_TOKENS=100000 -e SGLANG_TIMEOUT_KEEP_ALIVE=3600 -e ENGINE_PYTHONPATH="$C/frozen/$ENGINE/python" \
  -e SERVER_EXTRA="--attention-backend fa3 --host 127.0.0.1" \
  sglang_hicache bash -c "cd $C/scripts && bash start_server.sh three_tier_wc NAT160" > "$D/preflight_tier_boot.log" 2>&1 \
  || { log "STOP: the tier-check server did not boot ($D/preflight_tier_boot.log)"; stop_server; exit 1; }
RUN=$(cat "$A/results/.current_run"); S=$A/results/$RUN/server.log
check() {  # name
  docker exec -e PYTHONDONTWRITEBYTECODE=1 sglang_hicache bash -c "cd $C/baselines/common && python3 check_server.py \
    --backend openai --backend-opt details=on --prompt-tokens 20000 --block-tokens 64 \
    --settle-s 30 --stage host:150000:cached --stage ssd:400000:cached --json $C/results/$RUN/check_$1.json" > "$A/results/$RUN/check_$1.log" 2>&1
}
check quiet; rc=$?
log "tier check quiet: rc=$rc, run $RUN: $(grep -a -E 'passed|failed|FAIL' "$A/results/$RUN/check_quiet.log" | tail -1 | cut -c1-120); holds $(grep -ac 'HICACHE_EVT load_hold' "$S"); tracebacks $(grep -ac Traceback "$S")"
[ "$rc" -eq 0 ] || { log "STOP: the quiet tier check failed ($A/results/$RUN/check_quiet.log)"; stop_server; exit 1; }
docker exec sglang_hicache rm -f /tmp/bg_decode.stop
docker exec -d sglang_hicache bash -c "python3 $C/results/campaign39/bg_decode.py /tmp/bg_decode.stop 6 1500 > $C/results/$RUN/bg_decode.log 2>&1"
sleep 5
check busy; rc=$?
docker exec sglang_hicache touch /tmp/bg_decode.stop
H=$(grep -ac 'HICACHE_EVT load_hold' "$S")
log "tier check with six generations decoding: rc=$rc: $(grep -a -E 'passed|failed|FAIL' "$A/results/$RUN/check_busy.log" | tail -1 | cut -c1-120); holds $H, landed $(grep -a 'HICACHE_EVT load_hold' "$S" | grep -c 'landed=1'), steps $(grep -a 'HICACHE_EVT load_hold' "$S" | grep -o 'steps=[0-9]*' | sort | uniq -c | tr '\n' ' '); tracebacks $(grep -ac Traceback "$S")"
sleep 20
stop_server
# With other requests decoding, the cold run's logprobs can differ from the warm run's by more than the check's
# tolerance (batching changes the kernel path; no load and no hold is involved in either). What the hold must keep
# is the copy: the host and SSD stages compare a held batch's logprobs with the warm run's at tolerance 0.02.
STAGES=$(grep -ac '^ok    host \|^ok    ssd ' "$A/results/$RUN/check_busy.log")
log "tier check with six generations decoding: host and SSD stages ok: $STAGES of 2; cold against warm: $(grep -a '^\(ok\|FAIL\)  *warm' "$A/results/$RUN/check_busy.log" | grep -o 'logprobs.*' | cut -c1-60)"
[ "$STAGES" -eq 2 ] || { log "STOP: the busy tier check failed in a tier stage ($A/results/$RUN/check_busy.log)"; exit 1; }
[ "$H" -gt 0 ] || { log "STOP: no batch was held during the busy tier check"; exit 1; }

# 3. Smokes: the current design, 64 users arriving 1 a second, GPU pool 200,000 tokens, host 48 GB, 600 s.
SMOKE_FLAGS=$(sed -e 's/--max-total-tokens [0-9]*/--max-total-tokens 200000/' <<< "$FLAGS")
for HOLD in 0 16; do
  log "smoke, hold $HOLD: three_tier_wc_oracle_v2_nu_owned (ec_full flags), GPU 200,000 tokens, host 48 GB, 64 users, 600 s"
  docker exec sglang_hicache bash -c 'find /mnt/ssd/hicache_l3 -mindepth 1 -delete 2>/dev/null; true'
  docker exec "${ENVS[@]}" -e SGLANG_HICACHE_PREFILL_LOAD_HOLD_STEPS=$HOLD \
    -e MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 -e CTX=262144 -e KV_DTYPE=auto -e TEMPLATE= -e HOST_GB=48 -e L3_CLEANER_PCT=90,80 \
    -e SGLANG_TIMEOUT_KEEP_ALIVE=3600 -e ENGINE_PYTHONPATH="$C/frozen/$ENGINE/python" -e SERVER_EXTRA="$SMOKE_FLAGS" \
    sglang_hicache bash -c "cd $C/scripts && bash start_server.sh three_tier_wc_oracle_v2_nu_owned NAT160" > "$D/preflight_smoke${HOLD}_boot.log" 2>&1 \
    || { log "STOP: the smoke server did not boot ($D/preflight_smoke${HOLD}_boot.log)"; stop_server; exit 1; }
  RUN=$(cat "$A/results/.current_run"); S=$A/results/$RUN/server.log
  docker exec -e TRACE=$C/data/converted/freeinference_16w.programs.jsonl.gz -e RECIPE=none -e RATE=1 -e MAX_ACTIVE=64 \
    -e DURATION=600 -e HINTS=dist -e MAX_CONTEXT=65536 -e SEED=0 -e TAG=smoke$HOLD \
    sglang_hicache bash -c "cd $C/scripts && bash start_programs.sh --gap-scale 1 --gap-cap 1200 --hint-no-call" > "$D/preflight_smoke${HOLD}_client.log" 2>&1
  rc=$?
  OUT=$(ls -d "$A/results/$RUN"/programs_*_smoke$HOLD 2>/dev/null | tail -1)
  SUM=$(grep -a '"kind":"summary"' "$OUT/client.jsonl" 2>/dev/null | tail -1)
  ERR=$(grep -o '"errors":[0-9]*' <<< "$SUM" | cut -d: -f2)
  log "smoke, hold $HOLD: client rc=$rc, run $RUN; requests $(grep -o '"profile":[0-9]*' <<< "$SUM" | cut -d: -f2), errors ${ERR:-?}; holds $(grep -ac 'HICACHE_EVT load_hold' "$S"), landed $(grep -a 'HICACHE_EVT load_hold' "$S" | grep -c 'landed=1'), steps $(grep -a 'HICACHE_EVT load_hold' "$S" | grep -o 'steps=[0-9]*' | sort | uniq -c | sort -k2 -t= -n | tr '\n' ' '); tracebacks $(grep -ac Traceback "$S")"
  log "smoke, hold $HOLD: $(tail -4 "$OUT/summary.txt" 2>/dev/null | tr '\n' ' ' | cut -c1-600)"
  stop_server
  [ "$rc" -eq 0 ] && [ "${ERR:-1}" = 0 ] && [ "$(grep -ac Traceback "$S")" = 0 ] \
    || { log "STOP: the smoke with hold $HOLD failed (client rc $rc, errors ${ERR:-?}): $S"; exit 1; }
done
log "pre-flight passed for frozen/$ENGINE"
