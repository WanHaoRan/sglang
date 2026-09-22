#!/usr/bin/env bash
# Exp 2 control C2: keep GPU->host writes and all Python bookkeeping, remove the
# SSD write. Probes are populated WITHOUT the skip so L3 actually holds them.
source /sgl-workspace/sglang/hicache_eval/scripts/env.sh
set -x
bash $WORK/scripts/stop_server.sh
find $L3_DIR -mindepth 1 -delete 2>/dev/null
unset HICACHE_EVAL_BACKUP_SKIP
bash $WORK/scripts/start_server.sh c2_populate \
  --enable-hierarchical-cache --hicache-size 100 \
  --hicache-write-policy write_through --hicache-io-backend kernel \
  --hicache-mem-layout page_first \
  --hicache-storage-backend file --hicache-storage-prefetch-policy wait_complete \
  --radix-eviction-policy lru
cd $WORK/scripts
python3 exp2.py --rates 0 --reps 3 --control _populate_only --duration 5 \
  > $RESULTS/exp2/c2_populate.log 2>&1
bash $WORK/scripts/stop_server.sh
export HICACHE_EVAL_BACKUP_SKIP=1
bash $WORK/scripts/start_server.sh c2_run \
  --enable-hierarchical-cache --hicache-size 100 \
  --hicache-write-policy write_through --hicache-io-backend kernel \
  --hicache-mem-layout page_first \
  --hicache-storage-backend file --hicache-storage-prefetch-policy wait_complete \
  --radix-eviction-policy lru
python3 exp2.py --rates 8 --reps 3 --control C2_backup_skip --duration 300 \
  --skip-populate > $RESULTS/exp2/c2_run.log 2>&1
