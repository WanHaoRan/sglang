#!/usr/bin/env bash
# Each baseline's own evaluation, one after another, on an idle GPU. Host side; about 2 h; GPU busy.
#   host: setsid nohup bash /home/wanhr/sglang/agent_cache/baselines/common/run_official.sh [lmcache] [continuum] [tokencake] \
#           >> /home/wanhr/sglang/agent_cache/results/baseline_official.log 2>&1 < /dev/null &
# With no argument all three run. WAIT_FOR=<results folder> first waits until no launcher of that folder runs and its
# launcher.log ends in ALL DONE or STOP (a replay batch holding the GPU). Each step's output goes to its own run dir; this log names them.
#   lmcache    LMCache's multi-round QA benchmark against stock vLLM, then against vLLM + LMCache (full pools)
#   continuum  Continuum's SWE-bench evaluation, fcfs then continuum (continuum/official/swebench.sh)
#   tokencake  TokenCake's DAG workload, native then offload-agent (tokencake/official/dag.sh)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/lib.sh"
STEPS=("$@"); [ ${#STEPS[@]} -gt 0 ] || STEPS=(lmcache continuum tokencake)
if [ -n "${WAIT_FOR:-}" ]; then
  echo "$(date -u +%FT%TZ) waiting for $WAIT_FOR to finish"
  # a launcher still alive holds the GPU, whatever old STOP lines its log has
  until ! pgrep -f "[r]un_baselines.sh .*$WAIT_FOR" > /dev/null \
        && tail -1 "$BL_RESULTS/$WAIT_FOR/launcher.log" 2>/dev/null | grep -q -E " STOP |ALL DONE"; do sleep 30; done
  sleep 30
fi
bl_port_busy && { echo "STOP: port $BL_PORT is busy; stop that server first"; exit 1; }
# a server still booting has no port yet but holds GPU memory
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
[ "$used" -lt 2000 ] || { echo "STOP: the GPU holds $used MiB; another server is running or booting"; exit 1; }
for step in "${STEPS[@]}"; do
  echo "$(date -u +%FT%TZ) == $step"
  case "$step" in
    lmcache)
      for sys in vllm_vanilla vllm_lmcache; do
        bash "$HERE/../$sys/start_server.sh" | tail -3 || { bash "$HERE/stop_server.sh"; continue; }
        echo "$sys run dir: $(cat "$BL_RESULTS/.current_run")"
        bash "$HERE/../vllm_lmcache/official/multi_round_qa.sh" "$sys"
        bash "$HERE/../$sys/after_run.sh" "$BL_RESULTS/$(cat "$BL_RESULTS/.current_run")"
        bash "$HERE/stop_server.sh" | tail -1
      done ;;
    continuum) bash "$HERE/../continuum/official/swebench.sh" ;;
    tokencake) bash "$HERE/../tokencake/official/dag.sh" ;;
    *) echo "unknown step $step" ;;
  esac
done
echo "$(date -u +%FT%TZ) == all done"
