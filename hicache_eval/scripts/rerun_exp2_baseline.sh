#!/usr/bin/env bash
# Re-run one model's Exp 2 baseline sweep. Needed for qwen8b, whose sweep ran
# before the writeload SIGTERM fix and so recorded achieved_rps = 0 for every
# point -- the plan makes achieved rate the x-axis.
set -uo pipefail
S=/sgl-workspace/sglang/hicache_eval/scripts
KEY="${1:?usage: rerun_exp2_baseline.sh <model key>}"
source "$S/models.sh" "$KEY"
source "$S/env.sh"
export HICACHE_BACKEND=nixl EXP2_SUBDIR="exp2_${KEY}_rerun"
mkdir -p "$RESULTS/$EXP2_SUBDIR"
case "$KEY" in
  qwen8b)   RATES="0,0.5,1,2,4,8,16" ;;
  qwen32b)  RATES="0,0.25,0.5,1,2,4" ;;
  llama70b) RATES="0,0.1,0.25,0.5,1,2" ;;
esac
bash "$S/stop_server.sh"
find "$L3_DIR" -mindepth 1 -delete 2>/dev/null
bash -c "source $S/env.sh && MODEL='$MODEL' REASONING_PARSER='$REASONING_PARSER' \
  bash $S/start_server.sh ${EXP2_SUBDIR}_srv $MODEL_EXTRA_ARGS \
  --enable-hierarchical-cache --hicache-size 100 --hicache-io-backend kernel \
  --hicache-mem-layout page_first --hicache-storage-backend nixl \
  --radix-eviction-policy lru --hicache-write-policy write_through \
  --hicache-storage-prefetch-policy wait_complete"
python3 "$S/exp2.py" --rates "$RATES" --reps 3 --control baseline --duration 150 \
  > "$RESULTS/$EXP2_SUBDIR/run_baseline.log" 2>&1
bash "$S/stop_server.sh"
echo "rerun done: $RESULTS/$EXP2_SUBDIR"
