#!/usr/bin/env bash
# Exp 0 + Exp 1 for Llama-3.3-70B AWQ-INT4 with fp8 KV, nixl L3 backend.
# b = 163,840 B/token (fp8 halves it, and pool_host inherits store_dtype so it
# halves in L2/L3 too). Measured recompute bar b*P = 0.14-0.21 GiB/s.
export MODEL=casperhansen/llama-3.3-70b-instruct-awq
export L3_DIR=/var/hicache_nixl
source /sgl-workspace/sglang/hicache_eval/scripts/env.sh
cd $WORK/scripts

boot () {
  bash $WORK/scripts/stop_server.sh >/dev/null 2>&1
  find $L3_DIR -mindepth 1 -delete 2>/dev/null
  bash $WORK/scripts/start_server.sh "$1" \
    --kv-cache-dtype fp8_e5m2 \
    --enable-hierarchical-cache --hicache-size 100 \
    --hicache-write-policy write_through --hicache-io-backend kernel \
    --hicache-mem-layout page_first \
    --hicache-storage-backend nixl --hicache-storage-prefetch-policy wait_complete \
    --radix-eviction-policy lru
  grep -aoE "existing=[0-9]+ B \([0-9]+ entries\)|KV Cache is allocated[^,]*|max_total_num_tokens=[0-9]+" \
    $RESULTS/$1/server.log | head -3 > $RESULTS/$1/cold_check.txt
  cat $RESULTS/$1/cold_check.txt
}

# ---- Exp 0 ----
boot exp0_70b
bash $WORK/scripts/telemetry.sh start $RESULTS/exp0_70b
python3 cachectl.py scrape $RESULTS/exp0_70b/metrics_before.txt
EXP0_OUT=exp0_70b python3 exp0.py > $RESULTS/exp0_70b/run.log 2>&1
python3 cachectl.py scrape $RESULTS/exp0_70b/metrics_after.txt
bash $WORK/scripts/telemetry.sh stop $RESULTS/exp0_70b
echo "EXP0 DONE"

# ---- Exp 1: recompute + L1 + L3 ----
boot exp1_70b
bash $WORK/scripts/telemetry.sh start $RESULTS/exp1_70b
EXP1_OUT=exp1_70b python3 exp1.py --reps 3 --tiers recompute,L1,L3 \
  > $RESULTS/exp1_70b/run.log 2>&1
bash $WORK/scripts/telemetry.sh stop $RESULTS/exp1_70b
echo "EXP1 MAIN DONE"

# ---- Exp 1: dedicated L2 pass (L3 loads refill the host pool otherwise) ----
boot exp1_70b_l2
EXP1_OUT=exp1_70b python3 exp1.py --reps 3 --tiers L2 \
  > $RESULTS/exp1_70b/run_l2.log 2>&1
echo "70B EXP0+EXP1 COMPLETE"
