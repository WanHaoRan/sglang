#!/usr/bin/env bash
source /sgl-workspace/sglang/hicache_eval/scripts/env.sh
cd $WORK/scripts
python3 exp2.py --rates 0,2,8 --reps 3 --control baseline --duration 480 > $RESULTS/exp2/run.log 2>&1
