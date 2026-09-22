#!/usr/bin/env bash
# Measure prefill throughput P for the 70B AWQ + fp8-KV config BEFORE committing
# to the full plan. P is the denominator of the recompute bar b*P, and the
# 1,743 tok/s figure used so far is a first-order estimate, not a measurement.
export MODEL=casperhansen/llama-3.3-70b-instruct-awq
export L3_DIR=/var/hicache_nixl
source /sgl-workspace/sglang/hicache_eval/scripts/env.sh
cd $WORK/scripts
bash $WORK/scripts/stop_server.sh >/dev/null 2>&1
find $L3_DIR -mindepth 1 -delete 2>/dev/null
# No hierarchical cache: this measures pure recompute.
bash $WORK/scripts/start_server.sh p_measure \
  --kv-cache-dtype fp8_e5m2 \
  --radix-eviction-policy lru
python3 cachectl.py scrape $RESULTS/p_measure/metrics_before.txt
EXP1_OUT=p_measure python3 exp1.py --reps 3 --tiers recompute \
  --lengths 512,1024,2048,4096,8192,16384,32512 > $RESULTS/p_measure/run.log 2>&1
echo "P MEASURE DONE"
