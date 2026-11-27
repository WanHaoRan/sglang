#!/bin/zsh
# All arms x four volume settings, four at a time (one arm after another), niced.
cd /home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/ev_ssd
for arm in ec_full jit_exact jit_p50; do
  for sc in "1.0 0.90 0.80" "0.5 0.90 0.80" "0.25 0.90 0.80" "1.0 0.80 0.70"; do
    set -- ${=sc}
    nice -n 10 python3 -I -B -u compare.py $arm $1 $2 $3 > out/cmp_${arm}_$1_$2_$3.txt 2>&1 &
  done
  wait
done
echo done
