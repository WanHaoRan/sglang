#!/usr/bin/env bash
# After campaign 39's ec_full_load: pre-flight of the unified engine while the launcher idles (every other line is
# held), then release of the new arms. On a pass: unified_load in campaigns 40 to 43 and unified in campaigns 35
# and 38 (their settings.env gains the ten cache-rule switches of campaign35/unified.env first). On a failure
# nothing is released and the script says so. Host side:
#   setsid nohup bash /home/wanhr/sglang/agent_cache/results/campaign40/after_c39.sh engine_unified_20261126 \
#     >> /home/wanhr/sglang/agent_cache/results/campaign40/after_c39.log 2>&1 < /dev/null &
set -uo pipefail
ENGINE=${1:?engine folder under agent_cache/frozen}
A=/home/wanhr/sglang/agent_cache
R=$A/results
log() { echo "$(date -u +%FT%TZ) $*"; }
log "waiting for campaign 39's ec_full_load"
until grep -q "^C39 ec_full_load \(DONE\|FAIL\|STOP\)" "$R/campaign39/launcher.log" 2>/dev/null; do
  pgrep -f "[r]un_campaigns.sh" > /dev/null || { log "the launcher is gone before the arm ended: nothing done"; exit 1; }
  sleep 15
done
log "$(grep '^C39 ec_full_load \(DONE\|FAIL\|STOP\)' "$R/campaign39/launcher.log" | tail -1 | cut -c1-160)"
for _ in $(seq 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$used" -lt 2000 ] && ! (exec 3<>/dev/tcp/127.0.0.1/30000) 2>/dev/null && break
  sleep 5
done
log "pre-flight of the unified engine $ENGINE"
bash "$R/campaign40/preflight_unified.sh" "$ENGINE" >> "$R/campaign40/preflight.log" 2>&1
rc=$?
log "pre-flight rc=$rc: $(tail -1 "$R/campaign40/preflight.log" | cut -c1-200)"
[ "$rc" -eq 0 ] || { log "the unified pre-flight failed: nothing released"; exit 1; }
for c in campaign40 campaign41 campaign42 campaign43; do
  sed -i "s/^#HOLD unified_load \(.*\) engine_unified_20261126 /unified_load \1 $ENGINE /" "$R/$c/queue.txt"
done
for c in campaign35 campaign38; do
  grep -q MEASURED_PRICES "$R/$c/settings.env" || grep -v '^#' "$R/campaign35/unified.env" >> "$R/$c/settings.env"
  sed -i "s/^#HOLD unified \(.*\) engine_unified_20261126 /unified \1 $ENGINE /" "$R/$c/queue.txt"
done
log "released on $ENGINE: unified_load (campaigns 40 to 43), unified (campaigns 35 and 38)"
