#!/usr/bin/env bash
# Dedicated L2 pass for nixl. The combined run put L3 before L2, and every L3
# probe loads its pages back into the host pool, so by the L2 phase the pool was
# already 99.5% full and the fillers evicted the probes out of L2 entirely.
# A fresh server with only recompute+L2 keeps the pool budget predictable.
export L3_DIR=${L3_DIR:-/var/hicache_nixl}
source /sgl-workspace/sglang/hicache_eval/scripts/env.sh
cd $WORK/scripts
bash $WORK/scripts/stop_server.sh >/dev/null 2>&1
find $L3_DIR -mindepth 1 -delete 2>/dev/null
bash $WORK/scripts/start_server.sh exp1_nixl_l2 \
  --enable-hierarchical-cache --hicache-size 100 \
  --hicache-write-policy write_through --hicache-io-backend kernel \
  --hicache-mem-layout page_first \
  --hicache-storage-backend nixl --hicache-storage-prefetch-policy wait_complete \
  --radix-eviction-policy lru
EXP1_OUT=exp1_nixl python3 exp1.py --reps 3 --tiers L2 --policy wait_complete \
  > $RESULTS/exp1_nixl/run_l2.log 2>&1
echo "NIXL L2 PASS DONE"
