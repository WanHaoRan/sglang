#!/usr/bin/env bash
# After campaign 36's ec_full: end the running launcher (END files), then pre-flight and release the new arms.
#   - If results/campaign40/ENGINE_READY names a frozen unified engine: preflight_unified.sh on it; on a pass,
#     campaign 39's ec_full_load runs on that engine (hold only), the unified_load arms of campaigns 40 to 43 and the
#     unified arms of campaigns 35 and 38 are released (the ten cache-rule switches are appended to the settings.env
#     of 35 and 38, whose earlier arms are done).
#   - Otherwise: the load-hold pre-flight on engine_loadhold_20261126; on a pass, campaign 39's ec_full_load is
#     released on that engine.
# Then chain_multi.sh starts over every campaign of the comparison. Host side:
#   setsid nohup bash /home/wanhr/sglang/agent_cache/results/campaign40/after_c36.sh \
#     >> /home/wanhr/sglang/agent_cache/results/campaign40/after_c36.log 2>&1 < /dev/null &
set -uo pipefail
A=/home/wanhr/sglang/agent_cache
R=$A/results
ALL="campaign39 campaign40 campaign41 campaign42 campaign43 campaign35 campaign38 campaign36 campaign37"
log() { echo "$(date -u +%FT%TZ) $*"; }
log "waiting for campaign 36's ec_full to end"
until grep -q " ec_full DONE \| ec_full STOP " "$R/campaign36/launcher.log" 2>/dev/null; do sleep 15; done
log "campaign 36: $(grep ' ec_full ' "$R/campaign36/launcher.log" | tail -1 | cut -c1-140)"
for c in $ALL; do : > "$R/$c/END"; done
while pgrep -f "[r]un_campaigns.sh campaign39" > /dev/null; do sleep 5; done
log "the launcher exited"
sleep 30
if [ -s "$R/campaign40/ENGINE_READY" ]; then
  ENGINE=$(head -1 "$R/campaign40/ENGINE_READY")
  log "pre-flight of the unified engine $ENGINE"
  bash "$R/campaign40/preflight_unified.sh" "$ENGINE" >> "$R/campaign40/preflight.log" 2>&1
  rc=$?
  log "pre-flight rc=$rc: $(tail -1 "$R/campaign40/preflight.log" | cut -c1-200)"
  if [ "$rc" -eq 0 ]; then
    sed -i "s/^#HOLD ec_full_load three_tier_wc_oracle_v2_nu_owned dist --hint-no-call engine_loadhold_20261126 /ec_full_load three_tier_wc_oracle_v2_nu_owned dist --hint-no-call $ENGINE /" "$R/campaign39/queue.txt"
    for c in campaign40 campaign41 campaign42 campaign43; do
      sed -i "s/^#HOLD unified_load \(.*\) engine_unified_20261126 /unified_load \1 $ENGINE /" "$R/$c/queue.txt"
    done
    for c in campaign35 campaign38; do
      grep -q MEASURED_PRICES "$R/$c/settings.env" || grep -v '^#' "$R/campaign35/unified.env" >> "$R/$c/settings.env"
      sed -i "s/^#HOLD unified \(.*\) engine_unified_20261126 /unified \1 $ENGINE /" "$R/$c/queue.txt"
    done
    log "released: ec_full_load (campaign 39), unified_load (campaigns 40 to 43), unified (campaigns 35 and 38) on $ENGINE"
  fi
else
  log "no unified engine yet: pre-flight of engine_loadhold_20261126"
  bash "$R/campaign39/preflight_loadhold.sh" engine_loadhold_20261126 >> "$R/campaign39/preflight.log" 2>&1
  rc=$?
  log "pre-flight rc=$rc: $(tail -1 "$R/campaign39/preflight.log" | cut -c1-200)"
  if [ "$rc" -eq 0 ]; then
    sed -i 's/^#HOLD ec_full_load /ec_full_load /' "$R/campaign39/queue.txt"
    log "released: ec_full_load (campaign 39) on engine_loadhold_20261126"
  fi
fi
setsid nohup bash "$R/campaign35/chain_multi.sh" $ALL >> "$R/campaign40/chain_multi.log" 2>&1 < /dev/null &
log "chain_multi.sh started over: $ALL"
