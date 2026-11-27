#!/usr/bin/env bash
# Pre-flight of the frozen unified engine before its campaign arms. Host side, GPU free:
#   bash /home/wanhr/sglang/agent_cache/results/campaign40/preflight_unified.sh engine_unified_20261126 \
#     >> /home/wanhr/sglang/agent_cache/results/campaign40/preflight.log 2>&1
# 1. Unit tests on the frozen copy (CPU files, then the write/read path cases that need the GPU).
# 2. Tier check on the unified arm with every switch of this folder's settings.env on, small pools: quiet, then
#    with six generations decoding in the background so that the checked prompt's batch is held. A prompt served
#    from host memory and from the SSD must give the warm run's logprobs.
# 3. Two ten-minute smokes on small pools with the same users: every switch off (the engine's default path, with
#    ec_full's flags) and every switch on (the unified arm). The cleaner's marks are low, so passes run.
# Stops at the first failure (a STOP line) and always stops its server.
set -uo pipefail
ENGINE=${1:?engine folder under agent_cache/frozen}
# all (default), or smoke_on: only the smoke with every switch on (after the earlier stages passed in a former run).
ONLY=${2:-all}
A=/home/wanhr/sglang/agent_cache
C=/sgl-workspace/sglang/agent_cache
D=$A/results/campaign40
log() { echo "$(date -u +%FT%TZ) $*"; }
stop_server() { docker exec sglang_hicache bash -c "cd $C/scripts && bash stop_server.sh" > /dev/null 2>&1; : > "$A/results/.current_run"; }
# Tracebacks logged before the server began to exit; one logged after ("Gracefully exiting") is a thread caught by the
# interpreter's shutdown (the SSD cleaner mid-pass: "cannot schedule new futures after shutdown"), not a serving fault.
serving_tracebacks() { awk '/Gracefully exiting/ {exit} /Traceback/ {n++} END {print n + 0}' "$1"; }

[ -d "$A/frozen/$ENGINE/python/sglang" ] || { log "STOP: frozen/$ENGINE is missing"; exit 1; }
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
[ "$used" -lt 2000 ] || { log "STOP: the GPU holds $used MiB (an arm is running)"; exit 1; }
(exec 3<>/dev/tcp/127.0.0.1/30000) 2>/dev/null && { log "STOP: something listens on port 30000"; exit 1; }
# The replay settings and the write-path fixes, common to both smokes; the new switches apart.
BASE=(); NEW=()
while IFS= read -r line; do
  case "$line" in
    ''|'#'*) ;;
    SGLANG_HICACHE_PREFILL_LOAD_HOLD_STEPS=*|SGLANG_LOG_HICACHE_PRICES_INTERVAL=*|SGLANG_HICACHE_NEXT_TURN_MEASURED_PRICES=*|SGLANG_HICACHE_NEXT_TURN_LATENESS_WEIGHT=*|SGLANG_HICACHE_NEXT_TURN_READ_LINE=*|SGLANG_HICACHE_NEXT_TURN_HOLD_QUANTILE=*|SGLANG_HICACHE_NEXT_TURN_FREE_EMPTY_LOOKUPS=*|SGLANG_HICACHE_STORAGE_TRACK_STORED=*|SGLANG_HICACHE_STORAGE_SKIP_ENDED_WRITES=*|SGLANG_HICACHE_NIXL_TOUCH_INTERVAL_S=*|SGLANG_HICACHE_NIXL_RETIRE_ENDED_PAGES=*) NEW+=(-e "$line") ;;
    *) BASE+=(-e "$line") ;;
  esac
done < "$D/settings.env"
UNIFIED_FLAGS=$(sed -e 's/^#HOLD //' "$D/queue.txt" | grep '^unified_load ' | cut -d' ' -f6-)
BASE_FLAGS=$(grep '^ec_full ' "$A/results/campaign35/queue.txt" | cut -d' ' -f6-)
[ -n "$UNIFIED_FLAGS" ] && [ -n "$BASE_FLAGS" ] || { log "STOP: the queue lines of campaigns 40 and 35 were not found"; exit 1; }
[ $(( ${#NEW[@]} / 2 )) -eq 11 ] || { log "STOP: expected 11 new switches in $D/settings.env, found $(( ${#NEW[@]} / 2 ))"; exit 1; }
log "pre-flight of frozen/$ENGINE; new switches: $(printf '%s ' "${NEW[@]}" | sed 's/-e //g')"

if [ "$ONLY" = all ]; then
# 1. Unit tests on the frozen copy.
UNIT=$D/preflight_unit_tests.log; : > "$UNIT"
for f in managers/scheduler_components/test_prefill_load_hold mem_cache/test_storage_read_scheduler \
         mem_cache/test_storage_read_dispatch mem_cache/test_next_turn_prefetch mem_cache/test_next_turn_estimators \
         mem_cache/test_hicache_nixl_cleaner mem_cache/test_next_use_eviction mem_cache/test_host_census \
         mem_cache/test_evict_policy mem_cache/test_hicache_nixl_storage mem_cache/test_oracle_v2_flags; do
  docker exec -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH="$C/frozen/$ENGINE/python" sglang_hicache \
    bash -c "cd /sgl-workspace/sglang && timeout 1800 python3 -m pytest -q -p no:cacheprovider test/registered/unit/$f.py" >> "$UNIT" 2>&1
  rc=$?
  log "unit tests $f: rc=$rc: $(grep -E 'passed|failed|error' "$UNIT" | tail -1)"
  [ "$rc" -eq 0 ] || { log "STOP: unit tests $f failed ($UNIT)"; exit 1; }
done
docker exec -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH="$C/frozen/$ENGINE/python" sglang_hicache \
  bash -c "cd /sgl-workspace/sglang && timeout 1800 python3 -m pytest -q -p no:cacheprovider test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py -k 'hicache_l3 or l3_prefetch or release_aborted_request_l3'" >> "$UNIT" 2>&1
rc=$?; log "unit tests unified_radix_cache l3 (GPU): rc=$rc: $(grep -E 'passed|failed|error' "$UNIT" | tail -1)"
[ "$rc" -eq 0 ] || { log "STOP: unit tests failed ($UNIT)"; exit 1; }

# 2. Tier check on the unified arm, every switch on, GPU 100,000 tokens, host 30 GB.
log "tier check: three_tier_wc_oracle_v2_nu_owned on frozen/$ENGINE, unified flags and switches, GPU 100,000 tokens, host 30 GB"
docker exec sglang_hicache bash -c 'find /mnt/ssd/hicache_l3 -mindepth 1 -delete 2>/dev/null; true'
TIER_FLAGS=$(sed -e 's/--max-total-tokens [0-9]*/--max-total-tokens 100000/' <<< "$UNIFIED_FLAGS")
docker exec "${BASE[@]}" "${NEW[@]}" \
  -e MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 -e CTX=262144 -e KV_DTYPE=auto -e TEMPLATE= -e HOST_GB=30 -e L3_CLEANER_PCT=90,80 \
  -e SGLANG_TIMEOUT_KEEP_ALIVE=3600 -e ENGINE_PYTHONPATH="$C/frozen/$ENGINE/python" -e SERVER_EXTRA="$TIER_FLAGS" \
  sglang_hicache bash -c "cd $C/scripts && bash start_server.sh three_tier_wc_oracle_v2_nu_owned NAT160" > "$D/preflight_tier_boot.log" 2>&1 \
  || { log "STOP: the tier-check server did not boot ($D/preflight_tier_boot.log): $(grep -a 'Error\|error' "$D/preflight_tier_boot.log" | tail -2 | cut -c1-200)"; stop_server; exit 1; }
RUN=$(cat "$A/results/.current_run"); S=$A/results/$RUN/server.log
log "tier check server: run $RUN; $(grep -a 'HiCache retention\|HiCacheNixl: touch\|stays off' "$S" | cut -c30-200 | tr '\n' ';')"
check() {  # name
  docker exec -e PYTHONDONTWRITEBYTECODE=1 sglang_hicache bash -c "cd $C/baselines/common && python3 check_server.py \
    --backend openai --backend-opt details=on --prompt-tokens 20000 --block-tokens 64 --settle-s 30 \
    --stage host:150000:cached --stage ssd:400000:cached --json $C/results/$RUN/check_$1.json" > "$A/results/$RUN/check_$1.log" 2>&1
}
check quiet; rc=$?
log "tier check quiet: rc=$rc: $(grep -a -E 'passed|failed|FAIL' "$A/results/$RUN/check_quiet.log" | tail -1 | cut -c1-120); holds $(grep -ac 'HICACHE_EVT load_hold' "$S"); tracebacks $(grep -ac Traceback "$S")"
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
[ "$(serving_tracebacks "$S")" = 0 ] || { log "STOP: tracebacks in the tier-check server log ($S)"; exit 1; }
fi

# 3. Smokes: 64 users arriving 1 a second, GPU pool 200,000 tokens, host 48 GB, cleaner at 3 and 2 % of the volume, 600 s.
smoke() {  # tag, flags, then the docker -e arguments of the switches
  local TAG=$1 FLAGS=$2; shift 2
  log "smoke $TAG: three_tier_wc_oracle_v2_nu_owned, GPU 200,000 tokens, host 48 GB, 64 users, 600 s"
  docker exec sglang_hicache bash -c 'find /mnt/ssd/hicache_l3 -mindepth 1 -delete 2>/dev/null; true'
  docker exec "${BASE[@]}" "$@" \
    -e MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 -e CTX=262144 -e KV_DTYPE=auto -e TEMPLATE= -e HOST_GB=48 -e L3_CLEANER_PCT=3,2 \
    -e SGLANG_TIMEOUT_KEEP_ALIVE=3600 -e ENGINE_PYTHONPATH="$C/frozen/$ENGINE/python" \
    -e SERVER_EXTRA="$(sed -e 's/--max-total-tokens [0-9]*/--max-total-tokens 200000/' <<< "$FLAGS")" \
    sglang_hicache bash -c "cd $C/scripts && bash start_server.sh three_tier_wc_oracle_v2_nu_owned NAT160" > "$D/preflight_smoke_${TAG}_boot.log" 2>&1 \
    || { log "STOP: the smoke server ($TAG) did not boot ($D/preflight_smoke_${TAG}_boot.log)"; stop_server; return 1; }
  local RUN S OUT SUM ERR rc
  RUN=$(cat "$A/results/.current_run"); S=$A/results/$RUN/server.log
  docker exec -e TRACE=$C/data/converted/freeinference_16w.programs.jsonl.gz -e RECIPE=none -e RATE=1 -e MAX_ACTIVE=64 \
    -e DURATION=600 -e HINTS=dist -e MAX_CONTEXT=65536 -e SEED=0 -e TAG=smoke_$TAG \
    sglang_hicache bash -c "cd $C/scripts && bash start_programs.sh --gap-scale 1 --gap-cap 1200 --hint-no-call" > "$D/preflight_smoke_${TAG}_client.log" 2>&1
  rc=$?
  OUT=$(ls -d "$A/results/$RUN"/programs_*_smoke_$TAG 2>/dev/null | tail -1)
  SUM=$(grep -a '"kind":"summary"' "$OUT/client.jsonl" 2>/dev/null | tail -1)
  ERR=$(grep -o '"errors":[0-9]*' <<< "$SUM" | cut -d: -f2)
  log "smoke $TAG: client rc=$rc, run $RUN; requests $(grep -o '"profile":[0-9]*' <<< "$SUM" | cut -d: -f2), errors ${ERR:-?}; tracebacks while serving $(serving_tracebacks "$S"), in all $(grep -ac Traceback "$S"); cleaner passes $(grep -ac 'NIXL L3 cleanup' "$S")"
  log "smoke $TAG events: $(for e in load_hold prices l3_touch l3_retire l3_retire_ack l3_kept h2s_skip h2s_short ntp_lookup ntp_starve srs_write_hold ec_start srs_pin host_census; do printf '%s=%s ' "$e" "$(grep -ac "HICACHE_EVT $e" "$S")"; done) ends explicit=$(grep -a 'HICACHE_EVT ntp_end' "$S" | grep -c 'explicit=1') implicit=$(grep -a 'HICACHE_EVT ntp_end' "$S" | grep -c 'explicit=0') failed_pages=$(grep -ac 'failed to retrieve page' "$S")"
  log "smoke $TAG: $(python3 - "$OUT/client.jsonl" <<'PY'
import json, statistics as st, sys
t = [d["ttft"] for d in map(json.loads, open(sys.argv[1])) if d.get("kind") == "req" and d.get("ttft") is not None]
t.sort()
print(f"requests with a first token {len(t)}, TTFT mean {st.mean(t):.2f} s, p50 {t[len(t) // 2]:.2f}, p90 {t[int(len(t) * 0.9)]:.2f}" if t else "no request")
PY
)"
  [ -z "$(grep -a 'HICACHE_EVT prices' "$S" | tail -1)" ] || log "smoke $TAG: last prices line: $(grep -a 'HICACHE_EVT prices' "$S" | tail -1 | cut -c25-420)"
  stop_server
  [ "$rc" -eq 0 ] && [ "${ERR:-1}" = 0 ] && [ "$(serving_tracebacks "$S")" = 0 ] \
    || { log "STOP: the smoke $TAG failed (client rc $rc, errors ${ERR:-?}, tracebacks while serving $(serving_tracebacks "$S")): $S"; return 1; }
}
[ "$ONLY" = all ] && { smoke off "$BASE_FLAGS" || exit 1; }
smoke on "$UNIFIED_FLAGS" "${NEW[@]}" || exit 1
log "pre-flight passed for frozen/$ENGINE"
