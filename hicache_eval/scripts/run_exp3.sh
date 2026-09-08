#!/usr/bin/env bash
source /sgl-workspace/sglang/hicache_eval/scripts/env.sh
cd $WORK/scripts
python3 exp3.py --conditions "$1" \
  --client-args "--request-length 1024 --output-length 1024 --num-clients 16 --num-rounds 4 --max-parallel 8 --request-rate 2" \
  > $RESULTS/exp3/run_$2.log 2>&1
