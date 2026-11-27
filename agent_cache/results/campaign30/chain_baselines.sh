#!/usr/bin/env bash
# Campaign 30: the baselines at campaign 34's setting, after campaign 34's hicache and ec_exact (campaign34/chain_rerun.sh).
# One server on the GPU at a time. Host side:
#   setsid nohup bash /home/wanhr/sglang/agent_cache/results/campaign30/chain_baselines.sh >> /home/wanhr/sglang/agent_cache/results/campaign30/chain.log 2>&1 < /dev/null &
# Waits for campaign 34's chain and launcher to exit, then runs baselines/common/run_baselines.sh campaign30, which reads
# the weights before every boot and wipes the SSD stores; with that launcher already running it only watches it.
# Here: the address each arm's server listens on, once it is up.
set -uo pipefail
A=/home/wanhr/sglang/agent_cache
D=$A/results/campaign30
log() { echo "$(date -u +%FT%TZ) $*"; }
launches() { cat "$D/launcher.log" 2>/dev/null | grep -c ' LAUNCH '; }

if pgrep -f "[r]un_baselines.sh campaign30" > /dev/null; then
  seen=$(launches)
  log "run_baselines.sh campaign30 is running (pid $(pgrep -f '[r]un_baselines.sh campaign30' | head -1)): watching it"
else
  log "waiting for campaign 34 (chain_rerun.sh, run_campaigns.sh)"
  while pgrep -f "[c]hain_rerun.sh" > /dev/null || pgrep -f "[r]un_campaigns.sh" > /dev/null || pgrep -f "[r]un_baselines.sh" > /dev/null; do sleep 60; done
  log "campaign 34 ended: $(tail -1 "$A/results/campaign34/launcher.log" | cut -c1-140)"
  sleep 30
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$used" -lt 2000 ] || { log "STOP: the GPU holds $used MiB"; exit 1; }
  log "gpu $(nvidia-smi --query-gpu=uuid --format=csv,noheader)"
  seen=$(launches)
  setsid nohup bash "$A/baselines/common/run_baselines.sh" campaign30 >> "$D/runner.log" 2>&1 < /dev/null &
  sleep 20
  pgrep -f "[r]un_baselines.sh campaign30" > /dev/null || { log "STOP: run_baselines.sh exited: $(tail -2 "$D/runner.log" | tr '\n' ' ')"; exit 1; }
  log "run_baselines.sh campaign30 started"
fi
while pgrep -f "[r]un_baselines.sh campaign30" > /dev/null; do
  now=$(launches)
  if [ "$now" -gt "$seen" ]; then
    seen=$now
    label=$(grep ' LAUNCH ' "$D/launcher.log" | tail -1 | awk '{print $1}')
    b=
    for _ in $(seq 1 240); do
      b=$(ss -ltnH 'sport = :30000' | awk '{print $4}' | sort -u | tr '\n' ' ')
      [ -n "$b" ] && break
      pgrep -f "[r]un_baselines.sh campaign30" > /dev/null || break
      sleep 15
    done
    case "$b" in
      '') log "$label: nothing listened on port 30000 within 60 min" ;;
      127.0.0.1:30000\ ) log "$label: port 30000 bound on 127.0.0.1 only" ;;
      *) log "$label: WARNING: port 30000 bound on $b" ;;
    esac
  fi
  sleep 15
done
log "== done: $(tail -1 "$D/launcher.log" 2>/dev/null | cut -c1-140)"
