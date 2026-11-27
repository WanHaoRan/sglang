#!/usr/bin/env bash
# After campaign 38's ec_full: end the launcher over campaigns 35, 38, 37, 36 (END files), pre-flight the load-hold
# engine, restore the held baseline lines of campaigns 37 and 36, and start chain_multi.sh over campaigns 39, 35, 38,
# 37 and 36 (campaign 39's arm first). Host side:
#   setsid nohup bash /home/wanhr/sglang/agent_cache/results/campaign39/after_c38.sh \
#     >> /home/wanhr/sglang/agent_cache/results/campaign39/after_c38.log 2>&1 < /dev/null &
set -uo pipefail
A=/home/wanhr/sglang/agent_cache
R=$A/results
log() { echo "$(date -u +%FT%TZ) $*"; }
log "waiting for campaign 38's ec_full to end"
until grep -q " ec_full DONE \| ec_full STOP " "$R/campaign38/launcher.log" 2>/dev/null; do sleep 15; done
log "campaign 38: $(grep ' ec_full ' "$R/campaign38/launcher.log" | tail -1 | cut -c1-140)"
for c in campaign35 campaign38 campaign37 campaign36; do : > "$R/$c/END"; done
while pgrep -f "[r]un_campaigns.sh campaign35" > /dev/null; do sleep 5; done
log "the launcher over campaigns 35, 38, 37, 36 exited"
sleep 30
bash "$R/campaign39/preflight_loadhold.sh" engine_loadhold_20261126 >> "$R/campaign39/preflight.log" 2>&1
rc=$?
log "pre-flight rc=$rc: $(tail -1 "$R/campaign39/preflight.log" | cut -c1-200)"
for c in campaign37 campaign36; do sed -i 's/^#HOLD ec_full /ec_full /' "$R/$c/queue.txt"; done
if [ "$rc" -ne 0 ]; then
  # The hold is not run; the two baselines still are.
  sed -i 's/^ec_full_load /#FAILED-PREFLIGHT ec_full_load /' "$R/campaign39/queue.txt"
  log "campaign 39's arm is commented out; starting the chain for the baselines only"
fi
setsid nohup bash "$R/campaign35/chain_multi.sh" campaign39 campaign35 campaign38 campaign37 campaign36 \
  >> "$R/campaign39/chain_multi.log" 2>&1 < /dev/null &
log "chain_multi.sh started over campaigns 39, 35, 38, 37, 36"
