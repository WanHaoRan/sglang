#!/usr/bin/env bash
# Exp 2 control C1: remove the block device entirely by putting L3 on tmpfs.
# If the R=8 behaviour is unchanged vs baseline, the SSD was never the cause.
source /sgl-workspace/sglang/hicache_eval/scripts/env.sh
set -x
bash $WORK/scripts/stop_server.sh
unset HICACHE_EVAL_BACKUP_SKIP
mountpoint -q /mnt/tmpfs_l3 || { mkdir -p /mnt/tmpfs_l3; mount -t tmpfs -o size=48G tmpfs /mnt/tmpfs_l3; }
find /mnt/tmpfs_l3 -mindepth 1 -delete 2>/dev/null
export L3_DIR=/mnt/tmpfs_l3
export L3_MAX_SIZE=30G
bash $WORK/scripts/start_server.sh c1_tmpfs \
  --enable-hierarchical-cache --hicache-size 100 \
  --hicache-write-policy write_through --hicache-io-backend kernel \
  --hicache-mem-layout page_first \
  --hicache-storage-backend file --hicache-storage-prefetch-policy wait_complete \
  --radix-eviction-policy lru
cd $WORK/scripts
# page-cache eviction is meaningless on tmpfs; exp2 still calls it, harmless.
python3 exp2.py --rates 8 --reps 3 --control C1_tmpfs --duration 300 \
  > $RESULTS/exp2/c1_run.log 2>&1
df -h /mnt/tmpfs_l3 | tail -1
