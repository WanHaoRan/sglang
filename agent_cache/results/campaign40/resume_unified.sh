#!/usr/bin/env bash
# Second step of the unified engine's pre-flight after the first attempt stopped on a shutdown traceback (the SSD
# cleaner mid-pass when the switches-off smoke's server exited; nothing while serving): runs only the switches-on
# smoke (preflight_unified.sh <engine> smoke_on), then the release of after_c39.sh. Host side:
#   setsid nohup bash /home/wanhr/sglang/agent_cache/results/campaign40/resume_unified.sh engine_unified_20261126 \
#     >> /home/wanhr/sglang/agent_cache/results/campaign40/after_c39.log 2>&1 < /dev/null &
set -uo pipefail
ENGINE=${1:?engine folder under agent_cache/frozen}
A=/home/wanhr/sglang/agent_cache
R=$A/results
log() { echo "$(date -u +%FT%TZ) $*"; }
log "pre-flight of $ENGINE, the switches-on smoke only (unit tests, tier checks and the switches-off smoke passed at 04:22Z)"
bash "$R/campaign40/preflight_unified.sh" "$ENGINE" smoke_on >> "$R/campaign40/preflight.log" 2>&1
rc=$?
log "pre-flight rc=$rc: $(tail -1 "$R/campaign40/preflight.log" | cut -c1-200)"
[ "$rc" -eq 0 ] || { log "the switches-on smoke failed: nothing released"; exit 1; }
for c in campaign40 campaign41 campaign42 campaign43; do
  sed -i "s/^#HOLD unified_load \(.*\) engine_unified_20261126 /unified_load \1 $ENGINE /" "$R/$c/queue.txt"
done
for c in campaign35 campaign38; do
  grep -q MEASURED_PRICES "$R/$c/settings.env" || grep -v '^#' "$R/campaign35/unified.env" >> "$R/$c/settings.env"
  sed -i "s/^#HOLD unified \(.*\) engine_unified_20261126 /unified \1 $ENGINE /" "$R/$c/queue.txt"
done
log "released on $ENGINE: unified_load (campaigns 40 to 43), unified (campaigns 35 and 38)"
