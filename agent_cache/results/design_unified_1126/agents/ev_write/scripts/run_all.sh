#!/bin/zsh
# Rebuild every output of the ev_write checks from the extracts and models (CPU only; about 5 minutes).
#   zsh scripts/run_all.sh            # from agents/ev_write
# Before it, once: extract.py per run (extracts/*.tsv), pick_trees.py (extracts/trees_c32_c34.jsonl.gz) and
# model.py per campaign (extracts/model_*.pkl); the lines for those are at the end of this file, commented out.
set -e
cd "${0:A:h}/.."
py() { nice -n 10 python3 -I -B "$@"; }
c34=(ec_full jit_exact jit_p50 ec_exact hicache)
for a in ${c34}; do
  py scripts/q1_writes.py campaign34 ${a} > out/q1_c34_${a}.out 2>&1
  py scripts/q2_due.py campaign34 ${a} > out/q2_c34_${a}.out 2>&1
  py scripts/q3_rw.py campaign34 ${a} > out/q3_c34_${a}.out 2>&1
done
for a in ec_full jit_exact jit_p50 ec_exact; do
  py scripts/q4_ssd.py campaign34 ${a} > out/q4_c34_${a}.out 2>&1
done
for a in ec_full jit_exact; do
  EV_WIN=600 py scripts/q1_writes.py campaign34 ${a} > out/q1_c34_${a}_10min.out 2>&1
  py scripts/q1_writes.py campaign32 ${a} > out/q1_c32_${a}.out 2>&1
  py scripts/q2_due.py campaign32 ${a} > out/q2_c32_${a}.out 2>&1
done
for a in ec_full jit_exact jit_p50; do
  py scripts/q3_rw.py campaign32 ${a} > out/q3_c32_${a}.out 2>&1
done
py scripts/q2_by_gap.py campaign34 ec_full ec_exact jit_exact jit_p50 hicache > out/q2_by_gap_c34.out 2>&1
py scripts/write_held.py calibrate ec_full jit_exact jit_p50 ec_exact > out/write_held_calibrate_c34.out 2>&1
py scripts/write_held.py first ec_full jit_exact jit_p50 > out/write_held_first_c32.out 2>&1
py scripts/holds_what.py campaign34 ec_full jit_exact jit_p50 ec_exact > out/holds_what_c34.out 2>&1
py scripts/census_dirty.py campaign34 ${c34} > out/census_dirty_c34.out 2>&1
py scripts/queue_compare.py 3600 4590 campaign32:ec_full campaign34:ec_full campaign32:jit_exact campaign34:jit_exact > out/queue_compare_60_76.out 2>&1
py scripts/queue_compare.py 4590 6600 campaign34:ec_full campaign34:jit_exact campaign34:jit_p50 > out/queue_compare_c34_late.out 2>&1
py scripts/c32_refused_kind.py ec_full jit_exact jit_p50 > out/c32_refused_kind.out 2>&1
py scripts/c32_queue_at.py campaign32 ec_full 4607 > out/c32_queue_at_ec_full_4607.out 2>&1
for a in ec_full jit_exact; do py scripts/join_check.py campaign34 ${a}; done > out/join_check_c34.out 2>&1
{ py scripts/op_cost.py campaign32 ec_full; py scripts/op_cost.py campaign34 ec_full; py scripts/op_cost.py campaign34 jit_exact; } > out/op_cost.out 2>&1
for a in ec_full jit_exact jit_p50 ec_exact hicache; do py scripts/q2_before_done.py campaign34 ${a}; done > out/q2_before_done_c34.out 2>&1
for a in ec_full jit_exact; do py scripts/q2_before_done.py campaign32 ${a}; done > out/q2_before_done_c32.out 2>&1
py scripts/c32_rule.py ec_full jit_exact jit_p50 > out/c32_rule.out 2>&1
py scripts/q1_who_reads.py campaign34 ec_full jit_exact jit_p50 ec_exact > out/q1_who_reads_c34.out 2>&1
py scripts/ec_start_idle.py > out/ec_start_idle.out 2>&1
py scripts/fifo_check.py campaign34 ec_full > out/fifo_check.out 2>&1
echo done
# R=/home/wanhr/sglang/agent_cache/results
# while read label run prog; do py scripts/extract.py ${R}/${run} ${prog} extracts/c34_${label}.tsv; done < ${R}/campaign34/manifest.txt
# while read label run prog; do py scripts/extract.py ${R}/${run} ${prog} extracts/c32_${label}.tsv; done < ${R}/campaign32/manifest.txt
# py scripts/pick_trees.py /home/wanhr/sglang/agent_cache/data/converted/freeinference_16w.programs.jsonl.gz extracts/trees_c32_c34.jsonl.gz <every client.jsonl of the two manifests>
# py scripts/model.py campaign34 ec_full jit_exact jit_p50 hicache ec_exact > out/model_build_c34.out
# py scripts/model.py campaign32 ec_full jit_exact jit_p50 hicache > out/model_build_c32.out
