#!/usr/bin/env bash
# After the FA3 batch (results/fa3_smoke) and its HiCache check: the oracle arm of campaigns 28 and 29 (ec_exact: the
# four rules given the exact next-turn time as their distribution), then the 30-minute run of the next_use fix in this
# folder. One server on the GPU at a time: each step waits for the one before to end. Host side.
#   setsid nohup bash /home/wanhr/sglang/agent_cache/results/qfix_check/chain.sh >> .../results/qfix_check/chain.log 2>&1 < /dev/null &
# ec_exact runs engine copy d, as ec_full did: the pair then differs by the hint alone (the fix does not touch the
# rules' eviction). Its campaign lines are appended only when the GPU is free, since the campaigns' launcher starts a
# queued line within a minute.
set -uo pipefail
A=/home/wanhr/sglang/agent_cache
log() { echo "$(date -u +%FT%TZ) $*"; }
gpu_idle() {
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$used" -lt 2000 ] || { log "STOP: the GPU holds $used MiB"; exit 1; }
}

log "waiting for the FA3 batch and its HiCache check"
until ! pgrep -f "[r]un_baselines.sh .*fa3_smoke" > /dev/null \
      && tail -1 "$A/results/fa3_smoke/launcher.log" | grep -q -E " STOP |ALL DONE" \
      && ! pgrep -f "[c]hain_fa3_check.sh" > /dev/null; do sleep 30; done
sleep 30
gpu_idle

LINE='ec_exact three_tier_wc_oracle_v2_nu_owned oracle - engine_ec_20261111d --hicache-storage-next-turn-read-timing expected_cost --hicache-storage-next-turn-memory-price 5e-5 --hicache-storage-next-turn-price-learning harm --hicache-storage-next-turn-soft-pins on --radix-eviction-policy return_chance --radix-eviction-policy-config {"horizon_s":30}'
for c in campaign28 campaign29; do
  grep -q '^ec_exact ' "$A/results/$c/queue.txt" || echo "$LINE" >> "$A/results/$c/queue.txt"
  rm -f "$A/results/$c/END"
done
: > "$A/results/.current_run"
docker exec -d sglang_hicache bash -c 'nohup bash /sgl-workspace/sglang/agent_cache/frozen/oracle_scripts_staging/c28_users/run_campaigns.sh campaign28 campaign29 >> /sgl-workspace/sglang/agent_cache/results/c28_runner.log 2>&1'
sleep 20
pgrep -f "[w]arm_weights.sh" > /dev/null \
  || setsid nohup bash "$A/frozen/oracle_scripts_staging/c28_users/warm_weights.sh" >> "$A/results/c28_warm.log" 2>&1 < /dev/null &
for _ in $(seq 1 18); do grep -q "^C28 ec_exact LAUNCH" "$A/results/campaign28/launcher.log" && break; sleep 10; done
grep -q "^C28 ec_exact LAUNCH" "$A/results/campaign28/launcher.log" \
  || { log "STOP: no C28 ec_exact LAUNCH after 3 minutes: $(tail -2 "$A/results/c28_runner.log" | tr '\n' ' ')"; exit 1; }
log "campaign launcher started: $(grep '^C28 ec_exact LAUNCH' "$A/results/campaign28/launcher.log" | tail -1 | cut -c1-120)"

until grep -q -E "^C29 ec_exact (DONE|STOP)" "$A/results/campaign29/launcher.log" \
      || grep -q -E "^C28 ec_exact STOP" "$A/results/campaign28/launcher.log" \
      || tail -1 "$A/results/campaign28/launcher.log" | grep -q " STOP "; do sleep 60; done
log "oracle arms: $(grep -h -E '^C2[89] ec_exact (DONE|STOP)' "$A"/results/campaign2[89]/launcher.log | cut -c1-140 | tr '\n' ';')"
# The launcher idles after its queues; END files let it exit, so no queue line can start a second server.
touch "$A/results/campaign28/END" "$A/results/campaign29/END"
until ! docker exec sglang_hicache pgrep -f "[r]un_campaigns.sh" > /dev/null; do sleep 30; done
grep -q -E "^C29 ec_exact DONE" "$A/results/campaign29/launcher.log" || { log "STOP: an oracle arm did not finish; the fix check is not started"; exit 1; }
sleep 30
gpu_idle
: > "$A/results/.current_run"
log "== the fix check (results/qfix_check)"
bash "$A/baselines/common/run_baselines.sh" qfix_check >> "$A/results/qfix_check/runner.log" 2>&1
log "== done: $(tail -1 "$A/results/qfix_check/launcher.log" | cut -c1-140)"
