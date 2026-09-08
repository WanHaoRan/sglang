#!/usr/bin/env bash
source /sgl-workspace/sglang/hicache_eval/scripts/env.sh
cd $WORK/scripts
python3 exp1.py --reps 3 --tiers recompute,L2 > $RESULTS/exp1/run_l2b.log 2>&1
