#!/usr/bin/env bash
# Re-run Exp 0 and Exp 1 on the nixl backend, so the tier cost curves reflect a
# storage path that is not software-throttled. Same probes, same recipe.
export L3_DIR=${L3_DIR:-/var/hicache_nixl}   # hcommon + start_server both follow this
source /sgl-workspace/sglang/hicache_eval/scripts/env.sh
mkdir -p $L3_DIR
cd $WORK/scripts

boot () {
  bash $WORK/scripts/stop_server.sh >/dev/null 2>&1
  find $L3_DIR -mindepth 1 -delete 2>/dev/null
  bash $WORK/scripts/start_server.sh "$1" \
    --enable-hierarchical-cache --hicache-size 100 \
    --hicache-write-policy write_through --hicache-io-backend kernel \
    --hicache-mem-layout page_first \
    --hicache-storage-backend nixl --hicache-storage-prefetch-policy wait_complete \
    --radix-eviction-policy lru
  df -h $L3_DIR | tail -1 > $RESULTS/$1/df.txt
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv > $RESULTS/$1/gpu_procs.txt
}

# ---- Exp 0 on nixl ------------------------------------------------------
boot exp0_nixl
bash $WORK/scripts/telemetry.sh start $RESULTS/exp0_nixl
python3 cachectl.py scrape $RESULTS/exp0_nixl/metrics_before.txt
EXP0_OUT=exp0_nixl python3 exp0.py > $RESULTS/exp0_nixl/run.log 2>&1
python3 cachectl.py scrape $RESULTS/exp0_nixl/metrics_after.txt
bash $WORK/scripts/telemetry.sh stop $RESULTS/exp0_nixl

# ---- Exp 1 on nixl: recompute + L1 + L3, then L2 -------------------------
boot exp1_nixl
bash $WORK/scripts/telemetry.sh start $RESULTS/exp1_nixl
python3 cachectl.py scrape $RESULTS/exp1_nixl/metrics_before.txt
EXP1_OUT=exp1_nixl python3 exp1.py --reps 3 --tiers ${EXP1_TIERS:-recompute,L1,L3,L2} \
  > $RESULTS/exp1_nixl/run.log 2>&1
python3 cachectl.py scrape $RESULTS/exp1_nixl/metrics_after.txt
bash $WORK/scripts/telemetry.sh stop $RESULTS/exp1_nixl
echo "NIXL EXP0+EXP1 COMPLETE"
