#!/usr/bin/env bash
# Campaigns 35, 36 and 37 under one launcher: run_campaigns.sh takes the folders in the order given and re-reads
# every queue after each arm, so a line appended to campaign 35's queue runs before the next arm of 36 or 37.
# Host side:
#   setsid nohup bash /home/wanhr/sglang/agent_cache/results/campaign35/chain_multi.sh campaign35 campaign36 campaign37 \
#     >> /home/wanhr/sglang/agent_cache/results/campaign35/chain_multi.log 2>&1 < /dev/null &
# It waits for a launcher that is already running to exit (campaign 35's first launcher exits after its arm once
# results/campaign35/END exists), removes the END files so the new launcher waits for more lines, and starts
# run_campaigns.sh with the folders named. If that launcher is already running it only watches it.
# Here: read the weights into the page cache at each launch, and after each arm run arm_checks.sh
# (results/<campaign>/checks/arm_checks_<label>.log). The launcher exits once every folder holds END with nothing left.
set -uo pipefail
[ $# -ge 1 ] || { echo "name the campaign folders"; exit 1; }
A=/home/wanhr/sglang/agent_cache
C=/sgl-workspace/sglang/agent_cache
SNAP=$(ls -d /home/wanhr/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/*/ | head -1)
ARGS="$*"
log() { echo "$(date -u +%FT%TZ) $*"; }
count() { cat "$A/results/$1/launcher.log" 2>/dev/null | grep -c " $2 "; }   # campaign folder, LAUNCH or DONE
total() { local n=0 c; for c in $ARGS; do n=$((n + $(count "$c" "$1"))); done; echo "$n"; }

docker ps --format '{{.Names}}' | grep -qx sglang_hicache || docker start sglang_hicache > /dev/null \
  || { log "STOP: container sglang_hicache did not start"; exit 1; }
if pgrep -f "[r]un_campaigns.sh $ARGS" > /dev/null; then
  log "run_campaigns.sh $ARGS is running: watching it"
else
  if pgrep -f "[r]un_campaigns.sh" > /dev/null; then
    log "waiting for the running launcher to exit: $(pgrep -af '[r]un_campaigns.sh' | head -1 | cut -c1-120)"
    while pgrep -f "[r]un_campaigns.sh" > /dev/null; do sleep 10; done
    # The first launcher's watcher polls every 5 s: let it see the exit before a new launcher appears.
    sleep 20
  fi
  if pgrep -f "[r]un_baselines.sh" > /dev/null; then log "STOP: a baseline launcher is running"; exit 1; fi
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$used" -lt 2000 ] || { log "STOP: the GPU holds $used MiB"; exit 1; }
  for c in $ARGS; do rm -f "$A/results/$c/END"; done
  t=$(date +%s); cat "$SNAP"/model-*.safetensors > /dev/null
  log "gpu $(nvidia-smi --query-gpu=uuid --format=csv,noheader); ssd used $(df --output=pcent /mnt/ssd | tail -1 | tr -d ' '); weights read in $(( $(date +%s) - t )) s"
  : > "$A/results/.current_run"
  docker exec -d sglang_hicache bash -c "nohup bash $C/frozen/oracle_scripts_staging/c28_users/run_campaigns.sh $ARGS >> $C/results/c35_runner.log 2>&1"
  sleep 20
  pgrep -f "[r]un_campaigns.sh $ARGS" > /dev/null || { log "STOP: the launcher exited: $(tail -2 "$A/results/c35_runner.log" | tr '\n' ' ')"; exit 1; }
  log "run_campaigns.sh $ARGS started"
fi
seen=$(total LAUNCH)
declare -A checked
for c in $ARGS; do checked[$c]=$(count "$c" DONE); done
after_arm() {  # arm_checks.sh for every DONE line not yet checked, in the background
  local c n k label
  for c in $ARGS; do
    n=$(count "$c" DONE); k=${checked[$c]}
    while [ "$k" -lt "$n" ]; do
      k=$((k + 1)); checked[$c]=$k
      label=$(grep ' DONE ' "$A/results/$c/launcher.log" | sed -n "${k}p" | awk '{print $2}')
      log "arm done: $(grep ' DONE ' "$A/results/$c/launcher.log" | sed -n "${k}p" | cut -c1-150)"
      mkdir -p "$A/results/$c/checks"
      ( ( cd "$A" && bash frozen/oracle_scripts_staging/c28_users/arm_checks.sh "$c" "$label" ) > "$A/results/$c/checks/arm_checks_$label.log" 2>&1
        log "checks for $c $label: $(grep -c '^ok' "$A/results/$c/checks/arm_checks_$label.log") ok, $(grep -c '^FAIL' "$A/results/$c/checks/arm_checks_$label.log") failed" ) &
    done
  done
}
while pgrep -f "[r]un_campaigns.sh $ARGS" > /dev/null; do
  now=$(total LAUNCH)
  if [ "$now" -gt "$seen" ]; then
    seen=$now; t=$(date +%s)
    cat "$SNAP"/model-*.safetensors > /dev/null
    log "launch $now: weights read in $(( $(date +%s) - t )) s; gpu $(nvidia-smi --query-gpu=uuid --format=csv,noheader | cut -c1-12); $(tail -qn 1 $(for c in $ARGS; do echo "$A/results/$c/launcher.log"; done) 2>/dev/null | grep ' LAUNCH ' | tail -1 | cut -c1-60)"
  fi
  after_arm
  sleep 5
done
after_arm
wait
log "== launcher exited"
