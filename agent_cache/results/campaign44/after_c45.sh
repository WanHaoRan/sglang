#!/usr/bin/env bash
# Campaign 44's pre-flight stopped at 16:12Z on the one unit test its engine copy is meant to fail (the tie fix of
# queued turns, which was kept out of the copy so that campaign 44 against 40 measures the pass-over alone).
# After campaign 45's arm the launcher idles with campaign 44's line held: run the pre-flight again with that test
# left out, and release the line on a pass. Host side:
#   setsid nohup bash /home/wanhr/sglang/agent_cache/results/campaign44/after_c45.sh engine_unified_po_20261127 \
#     >> /home/wanhr/sglang/agent_cache/results/campaign44/after_chain.log 2>&1 < /dev/null &
set -uo pipefail
ENGINE=${1:?engine folder under agent_cache/frozen}
R=/home/wanhr/sglang/agent_cache/results
log() { echo "$(date -u +%FT%TZ) $*"; }
log "waiting for campaign 45's arm"
until grep -q "^C45 unified_load_fixedprice \(DONE\|FAIL\|STOP\)" "$R/campaign45/launcher.log" 2>/dev/null; do sleep 30; done
log "$(grep '^C45 unified_load_fixedprice \(DONE\|FAIL\|STOP\)' "$R/campaign45/launcher.log" | tail -1 | cut -c1-120)"
for _ in $(seq 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$used" -lt 2000 ] && ! (exec 3<>/dev/tcp/127.0.0.1/30000) 2>/dev/null && break
  sleep 5
done
log "pre-flight of $ENGINE again, without the tie-fix test"
PREFLIGHT_DESELECT="test/registered/unit/mem_cache/test_next_turn_prefetch.py::TestDeadlineMode::test_turns_first_seen_in_one_tick_are_queued_at_their_own_arrivals" \
  bash "$R/campaign44/preflight_passover.sh" "$ENGINE" >> "$R/campaign44/preflight.log" 2>&1
rc=$?
log "pre-flight rc=$rc: $(tail -1 "$R/campaign44/preflight.log" | cut -c1-200)"
[ "$rc" -eq 0 ] || { log "the pre-flight failed again: campaign 44's line stays held"; exit 1; }
sed -i "s/^#HOLD unified_load_po \(.*\) engine_unified_po_20261127 /unified_load_po \1 $ENGINE /" "$R/campaign44/queue.txt"
grep -q "^unified_load_po " "$R/campaign44/queue.txt" && log "released unified_load_po on $ENGINE; the idle launcher takes it" \
  || log "the queue line was not released"
