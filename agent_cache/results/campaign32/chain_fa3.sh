#!/usr/bin/env bash
# After tonight's chain (results/qfix_check/chain.sh: ec_exact on campaigns 28 and 29, then the 30-minute fix check):
# campaigns 32 (FreeInference) and 33 (TraceLab), the six arms on FA3 for the full 2 h, by the campaigns' launcher. One
# server on the GPU at a time: it starts only when no launcher of either kind is alive and the GPU is free; it does not
# wait on how tonight's chain ended, since these arms do not depend on its results. Host side.
#   setsid nohup bash /home/wanhr/sglang/agent_cache/results/campaign32/chain_fa3.sh >> /home/wanhr/sglang/agent_cache/results/campaign32/chain.log 2>&1 < /dev/null &
# campaign32/END and campaign33/END are in place, so the launcher exits after the last line of campaign 33. While it
# runs, this script reads the model's weights back into the page cache at each launch, as c28_users/warm_weights.sh
# does for campaigns 28 and 29 (that script watches only their logs).
set -uo pipefail
A=/home/wanhr/sglang/agent_cache
SNAP=$(ls -d /home/wanhr/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/*/ | head -1)
log() { echo "$(date -u +%FT%TZ) $*"; }

log "waiting for tonight's chain (results/qfix_check/chain.sh)"
while pgrep -f "[q]fix_check/chain.sh" > /dev/null; do sleep 60; done
log "tonight's chain ended: $(tail -1 "$A/results/qfix_check/chain.log" | cut -c1-160)"
while pgrep -f "[r]un_baselines.sh" > /dev/null || pgrep -f "[r]un_campaigns.sh" > /dev/null; do sleep 60; done
sleep 30
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
[ "$used" -lt 2000 ] || { log "STOP: the GPU holds $used MiB"; exit 1; }

: > "$A/results/.current_run"
docker exec -d sglang_hicache bash -c 'nohup bash /sgl-workspace/sglang/agent_cache/frozen/oracle_scripts_staging/c28_users/run_campaigns.sh campaign32 campaign33 >> /sgl-workspace/sglang/agent_cache/results/c32_runner.log 2>&1'
for _ in $(seq 1 18); do grep -q "^C32 hicache LAUNCH" "$A/results/campaign32/launcher.log" 2>/dev/null && break; sleep 10; done
grep -q "^C32 hicache LAUNCH" "$A/results/campaign32/launcher.log" 2>/dev/null \
  || { log "STOP: no C32 hicache LAUNCH after 3 minutes: $(tail -2 "$A/results/c32_runner.log" | tr '\n' ' ')"; exit 1; }
log "campaign launcher started: $(grep '^C32 hicache LAUNCH' "$A/results/campaign32/launcher.log" | tail -1 | cut -c1-120)"

launches() { cat "$A"/results/campaign3[23]/launcher.log 2>/dev/null | grep -c ' LAUNCH '; }
seen=0
while pgrep -f "[r]un_campaigns.sh campaign32" > /dev/null; do
  now=$(launches)
  if [ "$now" -gt "$seen" ]; then
    seen=$now; t=$(date +%s)
    cat "$SNAP"/model-*.safetensors > /dev/null
    log "launch $now: weights read in $(( $(date +%s) - t )) s"
  fi
  sleep 5
done
log "== done: $(tail -1 "$A/results/campaign33/launcher.log" 2>/dev/null | cut -c1-140)"
