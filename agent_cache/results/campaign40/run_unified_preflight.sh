#!/usr/bin/env bash
# Runs the unified engine's pre-flight while the launcher idles (every new arm is held), then releases the arms.
# Waits for campaign40/after_c36.sh to finish the load-hold pre-flight and start the chain. On a pass of the unified
# pre-flight: campaign 39's ec_full_load runs on the unified engine (hold only; its other switches are off, which
# the engine's off-path tests cover), the unified_load arms of campaigns 40 to 43 and the unified arms of campaigns
# 35 and 38 are released. On a failure: ec_full_load is released on engine_loadhold_20261126 if that engine's own
# pre-flight passed, so the GPU does not idle. Host side:
#   setsid nohup bash /home/wanhr/sglang/agent_cache/results/campaign40/run_unified_preflight.sh engine_unified_20261126 \
#     >> /home/wanhr/sglang/agent_cache/results/campaign40/run_unified_preflight.log 2>&1 < /dev/null &
set -uo pipefail
ENGINE=${1:?engine folder under agent_cache/frozen}
A=/home/wanhr/sglang/agent_cache
R=$A/results
log() { echo "$(date -u +%FT%TZ) $*"; }
log "waiting for the load-hold pre-flight and the chain"
until grep -q "chain_multi.sh started over" "$R/campaign40/after_c36.log" 2>/dev/null; do sleep 10; done
LOADHOLD_OK=no; grep -q "pre-flight rc=0" "$R/campaign40/after_c36.log" && LOADHOLD_OK=yes
log "load-hold pre-flight passed: $LOADHOLD_OK; $(tail -1 "$R/campaign39/preflight.log" | cut -c1-120)"
sleep 45
log "pre-flight of the unified engine $ENGINE"
bash "$R/campaign40/preflight_unified.sh" "$ENGINE" >> "$R/campaign40/preflight.log" 2>&1
rc=$?
log "pre-flight rc=$rc: $(tail -1 "$R/campaign40/preflight.log" | cut -c1-200)"
if [ "$rc" -eq 0 ]; then
  sed -i "s/^#WAIT ec_full_load three_tier_wc_oracle_v2_nu_owned dist --hint-no-call engine_loadhold_20261126 /ec_full_load three_tier_wc_oracle_v2_nu_owned dist --hint-no-call $ENGINE /" "$R/campaign39/queue.txt"
  for c in campaign40 campaign41 campaign42 campaign43; do
    sed -i "s/^#HOLD unified_load \(.*\) engine_unified_20261126 /unified_load \1 $ENGINE /" "$R/$c/queue.txt"
  done
  for c in campaign35 campaign38; do
    grep -q MEASURED_PRICES "$R/$c/settings.env" || grep -v '^#' "$R/campaign35/unified.env" >> "$R/$c/settings.env"
    sed -i "s/^#HOLD unified \(.*\) engine_unified_20261126 /unified \1 $ENGINE /" "$R/$c/queue.txt"
  done
  log "released on $ENGINE: ec_full_load (campaign 39), unified_load (campaigns 40 to 43), unified (campaigns 35 and 38)"
elif [ "$LOADHOLD_OK" = yes ]; then
  sed -i 's/^#WAIT ec_full_load /ec_full_load /' "$R/campaign39/queue.txt"
  log "the unified pre-flight failed: released ec_full_load (campaign 39) on engine_loadhold_20261126 only"
else
  log "both pre-flights failed: nothing released"
fi
