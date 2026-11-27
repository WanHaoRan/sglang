#!/usr/bin/env bash
# Campaign 34, second part: campaign 32's hicache and ec_exact arms on the fixed copy (queue.txt, added 2026-11-23 at
# the author's request), after the host restart. Host side:
#   setsid nohup bash /home/wanhr/sglang/agent_cache/results/campaign34/chain_rerun.sh >> /home/wanhr/sglang/agent_cache/results/campaign34/chain_rerun.log 2>&1 < /dev/null &
# The engine copy and the pinned inputs are the ones the first three arms ran on (run_campaigns.sh checks inputs.sha256;
# unit tests and the tier check of chain_fix.sh are not repeated). The launcher skips the arms in manifest.txt, so it runs
# only the two new lines, and exits on END. Here: read the weights into the page cache at each launch, and after each
# arm run arm_checks.sh (checks/arm_checks_<label>.log, which also redraws results/compare_c34_designs).
set -uo pipefail
A=/home/wanhr/sglang/agent_cache
C=/sgl-workspace/sglang/agent_cache
D=$A/results/campaign34
SNAP=$(ls -d /home/wanhr/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/*/ | head -1)
log() { echo "$(date -u +%FT%TZ) $*"; }

if pgrep -f "[r]un_campaigns.sh" > /dev/null || pgrep -f "[r]un_baselines.sh" > /dev/null; then
  log "STOP: a campaign or baseline launcher is running"; exit 1
fi
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
[ "$used" -lt 2000 ] || { log "STOP: the GPU holds $used MiB"; exit 1; }
t=$(date +%s); cat "$SNAP"/model-*.safetensors > /dev/null
log "gpu $(nvidia-smi --query-gpu=uuid --format=csv,noheader); weights read in $(( $(date +%s) - t )) s"

launches0=$(grep -c ' LAUNCH ' "$D/launcher.log")
: > "$A/results/.current_run"
docker exec -d sglang_hicache bash -c "nohup bash $C/frozen/oracle_scripts_staging/c28_users/run_campaigns.sh campaign34 >> $C/results/c34_runner.log 2>&1"
for _ in $(seq 1 18); do [ "$(grep -c ' LAUNCH ' "$D/launcher.log")" -gt "$launches0" ] && break; sleep 10; done
[ "$(grep -c ' LAUNCH ' "$D/launcher.log")" -gt "$launches0" ] \
  || { log "STOP: no LAUNCH after 3 minutes: $(tail -2 "$A/results/c34_runner.log" | tr '\n' ' ')"; exit 1; }
log "campaign launcher started: $(grep ' LAUNCH ' "$D/launcher.log" | tail -1 | cut -c1-120)"

seen=$launches0; done_lines=$(grep -c ' DONE ' "$D/launcher.log"); checked=$done_lines
after_arm() {  # run arm_checks.sh for every DONE line not yet checked
  local n label
  n=$(grep -c ' DONE ' "$D/launcher.log")
  while [ "$checked" -lt "$n" ]; do
    checked=$((checked + 1))
    label=$(grep ' DONE ' "$D/launcher.log" | sed -n "${checked}p" | awk '{print $2}')
    log "arm done: $(grep ' DONE ' "$D/launcher.log" | sed -n "${checked}p" | cut -c1-140)"
    # In the background, so the next launch's weights are read on time.
    ( ( cd "$A" && bash frozen/oracle_scripts_staging/c28_users/arm_checks.sh campaign34 "$label" ) > "$D/checks/arm_checks_$label.log" 2>&1
      log "checks for $label: $(grep -c '^ok' "$D/checks/arm_checks_$label.log") ok, $(grep -c '^FAIL' "$D/checks/arm_checks_$label.log") failed ($D/checks/arm_checks_$label.log)" ) &
  done
}
while pgrep -f "[r]un_campaigns.sh campaign34" > /dev/null; do
  now=$(grep -c ' LAUNCH ' "$D/launcher.log")
  if [ "$now" -gt "$seen" ]; then
    seen=$now; t=$(date +%s)
    cat "$SNAP"/model-*.safetensors > /dev/null
    log "launch $((now - launches0)): weights read in $(( $(date +%s) - t )) s"
  fi
  after_arm
  sleep 5
done
after_arm
wait
log "== done: $(tail -1 "$D/launcher.log" 2>/dev/null | cut -c1-140)"
