#!/bin/bash
# Byte check of the figure sets after an edit to the plot scripts: redraws compare_c26_designs and
# compare_c26_designs_c128 into a trial folder and compares with the checksums saved before the edit.
# From the host:  docker exec sglang_hicache bash /sgl-workspace/sglang/agent_cache/results/campaign26/checks/figs_md5/check.sh
# Nothing under results/ is rewritten. PDFs carry their creation date and manifest.txt its folder, so those may differ.
set -uo pipefail
ROOT=/sgl-workspace/sglang/agent_cache
HERE=$ROOT/frozen/oracle_scripts_staging/c26_ec
MD5=$ROOT/results/campaign26/checks/figs_md5
TRIAL=/tmp/figcheck
export CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$ROOT/scripts
rm -rf $TRIAL && mkdir -p $TRIAL/designs $TRIAL/designs_c128

draw() {  # the three plot scripts of one set, into the trial folder
    local set=$1
    export FIGSET=$set FIGSET_OUT=$TRIAL/$set
    cd $ROOT/scripts
    nice -n 15 python3 -B $HERE/plot_timeline.py > $TRIAL/$set.log 2>&1
    nice -n 15 python3 -B $HERE/plot_pricing_vs_jit.py --set $set --out $TRIAL/$set >> $TRIAL/$set.log 2>&1
    nice -n 15 python3 -B $HERE/plot_ec.py >> $TRIAL/$set.log 2>&1
    cd $ROOT
}

draw designs
RULES=""
for arm in ec_r1 ec_r3 ec_r13 ec_r13_rep ec_r134 ec_full; do
    if grep -q "^$arm " results/campaign26/manifest.txt; then RULES="$RULES $arm"; fi
done
STALL_CSV=$TRIAL/designs/ec_wait_split.csv nice -n 15 python3 -I -B $HERE/stall_ec.py baseline oracle_v2 jit_p50_d jit_exact_d $RULES > $TRIAL/stall_ec_designs.out
cd $ROOT/scripts && FIGSET=designs nice -n 15 python3 -B $HERE/plot_stall.py $TRIAL/designs/ec_wait_split.csv >> $TRIAL/designs.log 2>&1

draw designs_c128
RULES=""
for arm in ec_r134_c128 ec_full_c128; do
    if grep -q "^$arm " results/campaign26/manifest.txt; then RULES="$RULES $arm"; fi
done
STALL_LIMIT_S=6900 STALL_ORDER_RUN=20261018_154254_three_tier_wc_oracle_v2_nu_NAT160 STALL_CSV=$TRIAL/designs_c128/ec_wait_split.csv \
    nice -n 15 python3 -I -B $HERE/stall_ec.py baseline_c128 oracle_v2_c128 jit_p50_c128 jit_exact_c128 $RULES > $TRIAL/stall_ec_designs_c128.out
cd $ROOT/scripts && FIGSET=designs_c128 nice -n 15 python3 -B $HERE/plot_stall.py $TRIAL/designs_c128/ec_wait_split.csv >> $TRIAL/designs_c128.log 2>&1

for set in designs designs_c128; do
    echo "== $set: files that differ from the checksums saved before the edit (none listed = all match)"
    (cd $TRIAL/$set && md5sum -c $MD5/${set}_before.md5 2>&1 | grep -v ': OK$')
    echo "   $(cd $TRIAL/$set && md5sum -c $MD5/${set}_before.md5 2>/dev/null | grep -c ': OK$') of $(wc -l < $MD5/${set}_before.md5) match"
done
echo "== stall tables against the kept ones"
cmp $TRIAL/stall_ec_designs.out $ROOT/results/campaign26/stall_ec_designs.out && echo "stall_ec_designs.out matches"
cmp $TRIAL/stall_ec_designs_c128.out $ROOT/results/campaign26/stall_ec_designs_c128.out && echo "stall_ec_designs_c128.out matches"
