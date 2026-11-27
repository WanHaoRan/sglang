#!/usr/bin/env bash
# After campaign 37's ec_full (campaign 36's baseline is held, so the launcher idles): end the launcher (END
# files), pre-flight the load-hold engine again (the tier check now waits 30 s after its filler prompts), release
# campaign 39's ec_full_load line if it passes and campaign 36's ec_full in any case, and start chain_multi.sh over
# every campaign of the comparison. Lines marked #HOLD stay out until their engine is frozen
# and pre-flighted. Host side:
#   setsid nohup bash /home/wanhr/sglang/agent_cache/results/campaign39/after_c37.sh \
#     >> /home/wanhr/sglang/agent_cache/results/campaign39/after_c37.log 2>&1 < /dev/null &
set -uo pipefail
A=/home/wanhr/sglang/agent_cache
R=$A/results
ALL="campaign39 campaign40 campaign41 campaign42 campaign36 campaign43 campaign35 campaign38 campaign37"
log() { echo "$(date -u +%FT%TZ) $*"; }
log "waiting for campaign 37's ec_full to end"
until grep -q " ec_full DONE \| ec_full STOP " "$R/campaign37/launcher.log" 2>/dev/null; do sleep 15; done
log "campaign 37: $(grep ' ec_full ' "$R/campaign37/launcher.log" | tail -1 | cut -c1-140)"
for c in campaign39 campaign35 campaign38 campaign37 campaign36; do : > "$R/$c/END"; done
while pgrep -f "[r]un_campaigns.sh campaign39" > /dev/null; do sleep 5; done
log "the launcher exited"
sleep 30
bash "$R/campaign39/preflight_loadhold.sh" engine_loadhold_20261126 >> "$R/campaign39/preflight.log" 2>&1
rc=$?
log "pre-flight rc=$rc: $(tail -1 "$R/campaign39/preflight.log" | cut -c1-200)"
if [ "$rc" -eq 0 ]; then
  sed -i 's/^#HOLD ec_full_load /ec_full_load /' "$R/campaign39/queue.txt"
  log "campaign 39's ec_full_load is released"
fi
# Campaign 36's baseline runs once nothing ahead of it in the list is pending.
sed -i 's/^#HOLD ec_full /ec_full /' "$R/campaign36/queue.txt"
setsid nohup bash "$R/campaign35/chain_multi.sh" $ALL >> "$R/campaign39/chain_multi.log" 2>&1 < /dev/null &
log "chain_multi.sh started over: $ALL"
