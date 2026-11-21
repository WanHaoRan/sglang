#!/usr/bin/env bash
# Continuum's own evaluation, scaled down: its vendored mini-swe-agent solves SWE-bench Verified tasks in Docker
# sandboxes against the fork, once with --scheduling-policy fcfs (its "Original vLLM mode") and once with continuum,
# and the authors' continuum_exp/analyze.py turns the scheduler's own record (scheduler_timestamps) into job durations.
# The authors' shipped result (continuum_exp/, Llama-3.1-70B, 500 jobs): average job 215.5 s with Continuum against
# 465.2 s without. The gain needs KV pressure, so the jobs run at once on a small GPU pool.
#   host: bash /home/wanhr/sglang/agent_cache/baselines/continuum/official/swebench.sh
# Knobs: SLICE (0:8, tasks of SWE-bench Verified test), WORKERS (8, jobs at once), GPU_KV_TOKENS (100000), CTX (65536),
#   POLICIES ("fcfs continuum"). Needs: setup.sh, the agent venv (.venv_agent) and the tasks' sandbox images.
# Output: per policy a run dir agent_cache/results/<UTC>_continuum_<policy>/ with swebench_output/, scheduler_timestamps,
# the analysis (results.json) and official_summary.txt. Takes about 10-20 min per policy; GPU busy.
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
. "$HERE/../common/lib.sh"
AGENT=$HERE/.venv_agent/bin
SRC=$HERE/src/vllm-continuum
[ -x "$AGENT/mini-extra" ] || { echo "STOP: no agent venv (uv venv $HERE/.venv_agent; uv pip install -e $SRC/mini-swe-agent datasets)"; exit 1; }
for POLICY in ${POLICIES:-fcfs continuum}; do
  echo "== policy $POLICY: boot"
  GPU_KV_TOKENS=${GPU_KV_TOKENS:-100000} CTX=${CTX:-65536} LMCACHE=off POLICY=$POLICY bash "$HERE/start_server.sh" | tail -4 || exit 1
  RUN="$BL_RESULTS/$(cat "$BL_RESULTS/.current_run")"
  echo "== policy $POLICY: ${SLICE:-0:8} tasks, ${WORKERS:-8} at once -> $RUN/swebench_output"
  t0=$(date +%s)
  # The runner's --port reaches VllmModelConfig as an unknown "port" field; its config's model.base_url is the way in.
  "$AGENT/python" -c 'import sys, yaml; c = yaml.safe_load(open(sys.argv[1])); c["model"]["base_url"] = sys.argv[2]; yaml.safe_dump(c, open(sys.argv[3], "w"))' \
    "$SRC/mini-swe-agent/src/minisweagent/config/extra/swebench.yaml" "http://127.0.0.1:$BL_PORT/v1" "$RUN/swebench_config.yaml"
  "$AGENT/mini-extra" swebench --model-class vllm --model "$BL_MODEL" -c "$RUN/swebench_config.yaml" --subset verified \
    --split test --slice "${SLICE:-0:8}" --workers "${WORKERS:-8}" --output "$RUN/swebench_output" > "$RUN/swebench.log" 2>&1
  echo "agent rc=$? after $(( $(date +%s) - t0 )) s"
  bash "$HERE/../common/stop_server.sh"            # the scheduler writes scheduler_timestamps at shutdown
  "$AGENT/python" "$SRC/continuum_exp/analyze.py" --input-dir "$RUN" --output-dir "$RUN" > "$RUN/official_summary.txt" 2>&1
  tail -12 "$RUN/official_summary.txt"
  echo "pins: $(grep -ac 'pinned_time' "$RUN/scheduler_timestamps" 2>/dev/null)  parsed tool calls: $(grep -ac 'Extracted func_call' "$RUN/server.log")"
done
