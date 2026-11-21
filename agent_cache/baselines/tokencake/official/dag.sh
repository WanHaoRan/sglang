#!/usr/bin/env bash
# TokenCake's own evaluation, scaled down: its experiment driver (tools/tokencake_experiments) runs its fixed multi-agent
# DAG workload (conversation-tools, here its first 8 of 24 applications: official/conversation-tools-8apps.json)
# against its native mode and the full system, launching the server itself with its own sizing (GPU memory
# utilization 0.5, context 32,768, CPU KV 100 GiB). The README's result at QPS 1.0 (Qwen2.5-14B, A800): application
# E2E 960.44 s native against 631.77 s with TokenCake (-34.2 %), first-pass prefill tokens -71 %.
#   host: bash /home/wanhr/sglang/agent_cache/baselines/tokencake/official/dag.sh
# Knobs: QPS (1.0), MODES ("native offload-agent"). Output: agent_cache/results/<UTC>_tokencake_official/ (the driver's
# run folder, its report). Takes about 15-30 min per mode; GPU busy; nothing else may hold the GPU.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
. "$HERE/../common/lib.sh"
OUT="$BL_RESULTS/$(date -u +%Y%m%d_%H%M%S)_tokencake_official"
MODES=(); for m in ${MODES:-native offload-agent}; do MODES+=(--mode "$m"); done
cd "$HERE/src/TokenCake"
PATH="$HERE/official/nosmi:$PATH" "$HERE/.venv/bin/python" -m tools.tokencake_experiments.driver prepare "$OUT" \
  --model "$BL_MODEL" --gpus 0 --python "$HERE/.venv/bin/python" "${MODES[@]}" --qps "${QPS:-1.0}" \
  --workload-dataset "$HERE/official/conversation-tools-8apps.json" 2>&1 | grep -v " INFO " | tail -5
# official/nosmi/nvidia-smi fails, so the driver passes CUDA_VISIBLE_DEVICES=0, not the GPU UUID, which vLLM 0.22
# cannot parse (its README: GPU discovery is optional and a missing nvidia-smi does not block launching).
PATH="$HERE/official/nosmi:$HERE/.venv/bin:$PATH" "$HERE/.venv/bin/python" -m tools.tokencake_experiments.driver run-cases "$OUT" 2>&1 | tee "$OUT/run-cases.log" | grep -v " INFO " | tail -20
"$HERE/.venv/bin/python" -m tools.tokencake_experiments.driver report "$OUT" 2>&1 | tee "$OUT/report.log" | grep -v " INFO " | tail -40
