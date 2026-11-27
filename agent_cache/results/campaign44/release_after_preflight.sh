#!/usr/bin/env bash
# Releases campaign 44's held line once the pre-flight started at 18:20Z on 2026-11-27 (with the tie-fix test left
# out by name) passes; the idle launcher for campaigns 44 and 45 then takes it. Host side, in the background.
set -uo pipefail
R=/home/wanhr/sglang/agent_cache/results
log() { echo "$(date -u +%FT%TZ) $*"; }
since="2026-11-27T18:20:30Z"
until awk -v t="$since" '$1 >= t && $1 ~ /^2026/' "$R/campaign44/preflight.log" | grep -q "pre-flight passed\|STOP"; do sleep 20; done
last=$(awk -v t="$since" '$1 >= t && $1 ~ /^2026/' "$R/campaign44/preflight.log" | grep "pre-flight passed\|STOP" | tail -1)
log "pre-flight: $(cut -c1-200 <<< "$last")"
grep -q "pre-flight passed" <<< "$last" || { log "campaign 44's line stays held"; exit 1; }
sed -i "s/^#HOLD unified_load_po \(.*\) engine_unified_po_20261127 /unified_load_po \1 engine_unified_po_20261127 /" "$R/campaign44/queue.txt"
grep -q "^unified_load_po " "$R/campaign44/queue.txt" && log "released unified_load_po; the idle launcher takes it" || log "the line was not released"
