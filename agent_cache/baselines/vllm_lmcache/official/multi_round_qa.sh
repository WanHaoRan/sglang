#!/usr/bin/env bash
# LMCache's own benchmark (benchmarks/multi_round_qa at the release tag) against the server on port 30000: many users,
# each returning every NUM_USERS/QPS seconds with a growing history. Sized so the histories outgrow the GPU pool,
# stock vLLM recomputes evicted histories and vLLM + LMCache reloads them from its CPU tier; the expected result is a
# lower TTFT with LMCache (its documented CPU-offloading use case).
#   host: bash /home/wanhr/sglang/agent_cache/baselines/vllm_lmcache/official/multi_round_qa.sh <label>
# Knobs: NUM_USERS (48), ROUNDS (10), QPS (1.2), SYSTEM (1000), HISTORY (20000), ANSWER (100), TIME (600 s).
# Output: <current run dir>/official_multi_round_qa_<label>.csv and .log (the benchmark's own summary).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../../common/lib.sh"
LABEL=${1:?label for the output files}
RUN="$BL_RESULTS/$(cat "$BL_RESULTS/.current_run")"
BENCH=$HERE/../src/LMCache/benchmarks/multi_round_qa
[ -f "$BENCH/multi-round-qa.py" ] || { echo "STOP: no LMCache source at $BENCH (git clone https://github.com/LMCache/LMCache at v0.5.7.post1)"; exit 1; }
curl -sf -o /dev/null --max-time 5 "http://127.0.0.1:$BL_PORT/v1/models" || { echo "STOP: no server on port $BL_PORT"; exit 1; }
cd "$BENCH"
"$HERE/../../common/.venv_tools/bin/python" multi-round-qa.py \
  --num-users "${NUM_USERS:-48}" --num-rounds "${ROUNDS:-10}" --qps "${QPS:-1.2}" \
  --shared-system-prompt "${SYSTEM:-1000}" --user-history-prompt "${HISTORY:-20000}" --answer-len "${ANSWER:-100}" \
  --model "$BL_MODEL" --base-url "http://127.0.0.1:$BL_PORT/v1" --time "${TIME:-600}" \
  --output "$RUN/official_multi_round_qa_$LABEL.csv" 2>&1 | tee "$RUN/official_multi_round_qa_$LABEL.log" | grep -E "Average|QPS|TTFT|finished|Error" | tail -12
