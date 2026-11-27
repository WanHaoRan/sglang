#!/usr/bin/env bash
# Campaign 35: the unified four-action design against the current one (queue.txt). Host side:
#   setsid nohup bash /home/wanhr/sglang/agent_cache/results/campaign35/chain.sh >> /home/wanhr/sglang/agent_cache/results/campaign35/chain.log 2>&1 < /dev/null &
# Starts the container if needed, empties the baselines' SSD store (run_campaigns.sh wipes only hicache_l3, and the
# cleaner's 90/80 % marks are of the whole volume), then runs run_campaigns.sh campaign35. With that launcher already
# running it only watches it. Here: read the weights into the page cache at each launch, and after each arm run
# arm_checks.sh (checks/arm_checks_<label>.log, which also redraws results/compare_c35_designs).
# The launcher waits for more queue lines until results/campaign35/END exists; this script ends when it exits.
set -uo pipefail
A=/home/wanhr/sglang/agent_cache
C=/sgl-workspace/sglang/agent_cache
D=$A/results/campaign35
SNAP=$(ls -d /home/wanhr/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/*/ | head -1)
log() { echo "$(date -u +%FT%TZ) $*"; }
count() { cat "$D/launcher.log" 2>/dev/null | grep -c " $1 "; }

docker ps --format '{{.Names}}' | grep -qx sglang_hicache || docker start sglang_hicache > /dev/null \
  || { log "STOP: container sglang_hicache did not start"; exit 1; }
if pgrep -f "[r]un_campaigns.sh campaign35" > /dev/null; then
  seen=$(count LAUNCH); checked=$(count DONE)
  log "run_campaigns.sh campaign35 is running: watching it"
else
  if pgrep -f "[r]un_campaigns.sh" > /dev/null || pgrep -f "[r]un_baselines.sh" > /dev/null; then
    log "STOP: another campaign or baseline launcher is running"; exit 1
  fi
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$used" -lt 2000 ] || { log "STOP: the GPU holds $used MiB"; exit 1; }
  docker exec sglang_hicache bash -c 'find /mnt/ssd/baselines -mindepth 1 -delete 2>/dev/null; true'
  log "gpu $(nvidia-smi --query-gpu=uuid --format=csv,noheader); ssd used $(df --output=pcent /mnt/ssd | tail -1 | tr -d ' ') after emptying /mnt/ssd/baselines"
  t=$(date +%s); cat "$SNAP"/model-*.safetensors > /dev/null
  log "weights read in $(( $(date +%s) - t )) s"
  seen=$(count LAUNCH); checked=$(count DONE)
  : > "$A/results/.current_run"
  docker exec -d sglang_hicache bash -c "nohup bash $C/frozen/oracle_scripts_staging/c28_users/run_campaigns.sh campaign35 >> $C/results/c35_runner.log 2>&1"
  sleep 20
  pgrep -f "[r]un_campaigns.sh campaign35" > /dev/null || { log "STOP: the launcher exited: $(tail -2 "$A/results/c35_runner.log" | tr '\n' ' ')"; exit 1; }
  log "run_campaigns.sh campaign35 started"
fi
after_arm() {  # arm_checks.sh for every DONE line not yet checked, in the background
  local n label
  n=$(count DONE)
  while [ "$checked" -lt "$n" ]; do
    checked=$((checked + 1))
    label=$(grep ' DONE ' "$D/launcher.log" | sed -n "${checked}p" | awk '{print $2}')
    log "arm done: $(grep ' DONE ' "$D/launcher.log" | sed -n "${checked}p" | cut -c1-150)"
    ( ( cd "$A" && bash frozen/oracle_scripts_staging/c28_users/arm_checks.sh campaign35 "$label" ) > "$D/checks/arm_checks_$label.log" 2>&1
      log "checks for $label: $(grep -c '^ok' "$D/checks/arm_checks_$label.log") ok, $(grep -c '^FAIL' "$D/checks/arm_checks_$label.log") failed ($D/checks/arm_checks_$label.log)" ) &
  done
}
while pgrep -f "[r]un_campaigns.sh campaign35" > /dev/null; do
  now=$(count LAUNCH)
  if [ "$now" -gt "$seen" ]; then
    seen=$now; t=$(date +%s)
    cat "$SNAP"/model-*.safetensors > /dev/null
    log "launch $now ($(grep ' LAUNCH ' "$D/launcher.log" | tail -1 | awk '{print $2}')): weights read in $(( $(date +%s) - t )) s; gpu $(nvidia-smi --query-gpu=uuid --format=csv,noheader | cut -c1-12)"
  fi
  after_arm
  sleep 5
done
after_arm
wait
log "== launcher exited: $(tail -1 "$D/launcher.log" 2>/dev/null | cut -c1-140)"
