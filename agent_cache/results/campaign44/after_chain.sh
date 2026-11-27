#!/usr/bin/env bash
# After the six arms released at 04:42Z (unified_load in campaigns 40 to 43, unified in campaigns 35 and 38): the
# running launcher's folders get END so it exits, campaign 44's pre-flight runs on the free GPU, and on a pass its line
# is released; then a launcher for campaigns 44 and 45 starts (campaign35/chain_multi.sh campaign44 campaign45), so
# campaign 45 runs even if campaign 44's pre-flight fails. Host side:
#   setsid nohup bash /home/wanhr/sglang/agent_cache/results/campaign44/after_chain.sh engine_unified_po_20261127 \
#     >> /home/wanhr/sglang/agent_cache/results/campaign44/after_chain.log 2>&1 < /dev/null &
set -uo pipefail
ENGINE=${1:?engine folder under agent_cache/frozen}
A=/home/wanhr/sglang/agent_cache
R=$A/results
FOLDERS="campaign39 campaign40 campaign41 campaign42 campaign43 campaign35 campaign38 campaign36 campaign37"
log() { echo "$(date -u +%FT%TZ) $*"; }
ended() {  # campaign label: the arm has a DONE, FAIL or STOP line
  grep -q "^C${1#campaign} $2 \(DONE\|FAIL\|STOP\)" "$R/$1/launcher.log" 2>/dev/null
}
all_ended() {
  for c in campaign40 campaign41 campaign42 campaign43; do ended "$c" unified_load || return 1; done
  for c in campaign35 campaign38; do ended "$c" unified || return 1; done
}
log "waiting for the six arms released at 04:42Z"
until all_ended; do
  pgrep -f "[r]un_campaigns.sh" > /dev/null || { log "the launcher is gone before the six arms ended: nothing done"; exit 1; }
  sleep 30
done
log "the six arms have ended; END in the running launcher's folders"
for c in $FOLDERS; do touch "$R/$c/END"; done
for _ in $(seq 120); do pgrep -f "[r]un_campaigns.sh" > /dev/null || break; sleep 10; done
pgrep -f "[r]un_campaigns.sh" > /dev/null && { log "STOP: the launcher did not exit within 20 minutes"; exit 1; }
sleep 30
log "launcher exited; pre-flight of $ENGINE"
bash "$R/campaign44/preflight_passover.sh" "$ENGINE" >> "$R/campaign44/preflight.log" 2>&1
rc=$?
log "pre-flight rc=$rc: $(tail -1 "$R/campaign44/preflight.log" | cut -c1-200)"
if [ "$rc" -eq 0 ]; then
  sed -i "s/^#HOLD unified_load_po \(.*\) engine_unified_po_20261127 /unified_load_po \1 $ENGINE /" "$R/campaign44/queue.txt"
  grep -q "^unified_load_po " "$R/campaign44/queue.txt" && log "released unified_load_po on $ENGINE" \
    || log "the queue line of campaign 44 was not released"
else
  log "the pre-flight failed: campaign 44's line stays held"
fi
# Campaign 45 (campaign 40's arm with the fixed price of host room) needs no new engine: it runs either way.
log "starting the launcher for campaigns 44 and 45"
setsid nohup bash "$R/campaign35/chain_multi.sh" campaign44 campaign45 >> "$R/campaign44/chain_multi.log" 2>&1 < /dev/null &
sleep 60
log "$(tail -1 "$R/campaign44/chain_multi.log" | cut -c1-160)"
