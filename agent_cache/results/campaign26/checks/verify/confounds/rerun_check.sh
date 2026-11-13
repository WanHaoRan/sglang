#!/bin/sh
# Rerun every check script of 2026-11-11 one at a time and compare its output with the saved .out (byte for byte).
#   sh rerun_check.sh > rerun_check.out
cd "$(dirname "$0")" && mkdir -p rerun
for s in t0_reproduce ta_load ta_offered ta_sets tb_concentration tb_blocks tb_blocks_repeats tc_episodes tc_event td_margin te_drift te_early tf_args tf_workload tf_more tf_latency tf_latency_ci tf_r4 tf_ssd tf_shares tf_placebo tf_client tf_caps peek_users peek_convend; do
  nice -n 19 python3 -I -B $s.py > rerun/$s.out 2> rerun/$s.err
  if cmp -s rerun/$s.out $s.out; then r=SAME; else r=DIFF; fi
  echo "$s $r stderr_bytes=$(stat -c %s rerun/$s.err)"
done
nice -n 19 python3 -I -B ta_load.py fix > rerun/ta_load_fix.out 2> rerun/ta_load_fix.err
if cmp -s rerun/ta_load_fix.out ta_load_fix.out; then echo "ta_load_fix SAME"; else echo "ta_load_fix DIFF"; fi
