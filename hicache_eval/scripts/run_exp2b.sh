#!/usr/bin/env bash
source /sgl-workspace/sglang/hicache_eval/scripts/env.sh
bash $WORK/scripts/stop_server.sh
find $L3_DIR -mindepth 1 -delete 2>/dev/null
unset HICACHE_EVAL_BACKUP_SKIP
bash $WORK/scripts/start_server.sh exp2b \
  --enable-hierarchical-cache --hicache-size 100 \
  --hicache-write-policy write_through --hicache-io-backend kernel \
  --hicache-mem-layout page_first \
  --hicache-storage-backend file --hicache-storage-prefetch-policy wait_complete \
  --radix-eviction-policy lru
cd $WORK/scripts
python3 exp2.py --rates 0,8 --reps 3 --control baseline_uniquefix --duration 300 \
  > $RESULTS/exp2/run_uniquefix.log 2>&1
