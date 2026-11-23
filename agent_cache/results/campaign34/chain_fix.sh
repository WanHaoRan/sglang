#!/usr/bin/env bash
# Campaign 34, the verification of the FA3 engine fixes (agent_cache/OPEN_ITEMS.md section 7), after campaign 33.
# One server on the GPU at a time: it starts when no campaign or baseline launcher is alive and the GPU is free. Host side.
#   setsid nohup bash /home/wanhr/sglang/agent_cache/results/campaign34/chain_fix.sh >> /home/wanhr/sglang/agent_cache/results/campaign34/chain.log 2>&1 < /dev/null &
# Steps, each stopping the chain on a failure (a STOP line here):
#   1. unit tests on the frozen fixed copy (PYTHONPATH): the CPU files of the fixes, then the CUDA write/read path tests;
#   2. tier check: the hicache arm on the fixed copy with every write-path fix on, small pools (GPU 100,000 tokens, host
#      30 GB), check_server.py's host and SSD stages: a prompt served from host and from SSD must give the warm run's
#      first-token logprobs, so pages written by the merged write path are exact copies;
#   3. the three arms of queue.txt by run_campaigns.sh, reading the model's weights into the page cache at each launch.
set -uo pipefail
A=/home/wanhr/sglang/agent_cache
C=/sgl-workspace/sglang/agent_cache
D=$A/results/campaign34
ENGINE=engine_fa3fix_20261122
SNAP=$(ls -d /home/wanhr/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/*/ | head -1)
log() { echo "$(date -u +%FT%TZ) $*"; }

log "waiting for campaigns 32 and 33 (run_campaigns.sh, chain_fa3.sh)"
while pgrep -f "[r]un_campaigns.sh" > /dev/null || pgrep -f "[c]hain_fa3.sh" > /dev/null || pgrep -f "[r]un_baselines.sh" > /dev/null; do sleep 60; done
log "campaign 33 ended: $(tail -1 "$A/results/campaign33/launcher.log" | cut -c1-140)"
sleep 30
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
[ "$used" -lt 2000 ] || { log "STOP: the GPU holds $used MiB"; exit 1; }
[ -d "$A/frozen/$ENGINE/python/sglang" ] || { log "STOP: frozen/$ENGINE is missing"; exit 1; }

# 1. Unit tests on the frozen copy.
UNIT=$D/unit_tests.log
: > "$UNIT"
run_tests() {  # label, pytest arguments
  local label=$1; shift
  docker exec -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH="$C/frozen/$ENGINE/python" sglang_hicache \
    bash -c "cd /sgl-workspace/sglang && timeout 1800 python3 -m pytest -q -p no:cacheprovider $*" >> "$UNIT" 2>&1
  local rc=$?
  log "unit tests $label: rc=$rc: $(grep -E 'passed|failed|error' "$UNIT" | tail -1)"
  return $rc
}
T=test/registered/unit/mem_cache
for f in test_storage_read_scheduler test_storage_read_dispatch test_next_turn_prefetch test_hicache_nixl_cleaner \
         test_oracle_v2_flags test_storage_prefetch_lifecycle test_next_use_eviction test_host_census; do
  [ -f "$A/../$T/$f.py" ] || continue
  run_tests "$f" "$T/$f.py" || { log "STOP: unit tests $f failed ($UNIT)"; exit 1; }
done
run_tests test_hicache_nixl_storage "$T/test_hicache_nixl_storage.py" || { log "STOP: unit tests failed ($UNIT)"; exit 1; }
run_tests test_unified_radix_cache_l3 "$T/test_unified_radix_cache_unittest.py -k 'hicache_l3 or l3_prefetch or release_aborted_request_l3'" \
  || { log "STOP: unit tests failed ($UNIT)"; exit 1; }

# 2. Tier check with the write-path fixes on.
log "tier check: hicache arm on frozen/$ENGINE, GPU 100,000 tokens, host 30 GB, fixes on"
docker exec sglang_hicache bash -c 'find /mnt/ssd/hicache_l3 -mindepth 1 -delete 2>/dev/null; true'
docker exec \
  -e MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 -e CTX=262144 -e KV_DTYPE=auto -e TEMPLATE= -e HOST_GB=30 -e L3_CLEANER_PCT=90,80 \
  -e L1_TOKENS=100000 -e SGLANG_TIMEOUT_KEEP_ALIVE=3600 -e ENGINE_PYTHONPATH="$C/frozen/$ENGINE/python" \
  -e SERVER_EXTRA="--attention-backend fa3" \
  -e SGLANG_HICACHE_NIXL_SKIP_STORED_WRITES=1 -e SGLANG_HICACHE_STORAGE_WRITE_COALESCE_PAGES=128 \
  -e SGLANG_HICACHE_STORAGE_READ_HOLD_WRITE_TOKENS=50000 -e SGLANG_LOG_HICACHE_HOST_CENSUS_INTERVAL=10 \
  sglang_hicache bash "$C/scripts/start_server.sh" three_tier_wc NAT160 > "$D/check_boot.log" 2>&1 \
  || { log "STOP: the check server did not boot ($D/check_boot.log)"; docker exec sglang_hicache bash "$C/scripts/stop_server.sh"; exit 1; }
RUN=$(cat "$A/results/.current_run")
docker exec -e PYTHONDONTWRITEBYTECODE=1 sglang_hicache bash -c "cd $C/baselines/common && python3 check_server.py \
  --backend openai --backend-opt details=on --prompt-tokens 20000 --block-tokens 64 \
  --stage host:150000:cached --stage ssd:400000:cached --json $C/results/$RUN/check_fixes.json" > "$A/results/$RUN/check_fixes.log" 2>&1
rc=$?
docker exec sglang_hicache bash "$C/scripts/stop_server.sh" > /dev/null 2>&1
log "tier check: rc=$rc, run $RUN: $(grep -a -E 'passed|failed|FAIL' "$A/results/$RUN/check_fixes.log" | tail -1 | cut -c1-120)"
[ "$rc" -eq 0 ] || { log "STOP: the tier check failed ($A/results/$RUN/check_fixes.log)"; exit 1; }
grep -a -c 'h2s_group' "$A/results/$RUN/server.log" | sed "s/^/$(date -u +%FT%TZ) merged write groups in the check: /"

# 3. The arms.
: > "$A/results/.current_run"
docker exec -d sglang_hicache bash -c "nohup bash $C/frozen/oracle_scripts_staging/c28_users/run_campaigns.sh campaign34 >> $C/results/c34_runner.log 2>&1"
for _ in $(seq 1 18); do grep -q " LAUNCH " "$D/launcher.log" 2>/dev/null && break; sleep 10; done
grep -q " LAUNCH " "$D/launcher.log" 2>/dev/null \
  || { log "STOP: no LAUNCH after 3 minutes: $(tail -2 "$A/results/c34_runner.log" | tr '\n' ' ')"; exit 1; }
log "campaign launcher started: $(grep ' LAUNCH ' "$D/launcher.log" | tail -1 | cut -c1-120)"
seen=0
while pgrep -f "[r]un_campaigns.sh campaign34" > /dev/null; do
  now=$(grep -c ' LAUNCH ' "$D/launcher.log" 2>/dev/null)
  if [ "$now" -gt "$seen" ]; then
    seen=$now; t=$(date +%s)
    cat "$SNAP"/model-*.safetensors > /dev/null
    log "launch $now: weights read in $(( $(date +%s) - t )) s"
  fi
  sleep 5
done
log "== done: $(tail -1 "$D/launcher.log" 2>/dev/null | cut -c1-140)"
