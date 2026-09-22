#!/usr/bin/env bash
# The three sub-experiments the conformance audit found NOT_DONE (D16.3):
#   A  Exp 1 second run under --hicache-storage-prefetch-policy timeout   (plan L195)
#   B  Exp 1 background-load variant                                      (plan L207)
#   C  Exp 2 timeout-policy sweep at R in {0,2,8}                         (plan L227)
# Each run gets its own start/stop telemetry and a config.json (plan L54).
source /sgl-workspace/sglang/hicache_eval/scripts/env.sh
cd $WORK/scripts

boot () {  # boot <tag> <policy>
  bash $WORK/scripts/stop_server.sh >/dev/null 2>&1
  find $L3_DIR -mindepth 1 -delete 2>/dev/null
  bash $WORK/scripts/start_server.sh "$1" \
    --enable-hierarchical-cache --hicache-size 100 \
    --hicache-write-policy write_through --hicache-io-backend kernel \
    --hicache-mem-layout page_first \
    --hicache-storage-backend file --hicache-storage-prefetch-policy "$2" \
    --radix-eviction-policy lru
  grep -aoE "existing=[0-9]+ B \([0-9]+ entries\)" $RESULTS/$1/server.log | head -1 \
    > $RESULTS/$1/cold_check.txt
  df -h $L3_DIR | tail -1 > $RESULTS/$1/df.txt
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv > $RESULTS/$1/gpu_procs.txt
}

cfg () { printf '%s\n' "$2" > "$RESULTS/$1/config.json"; }

# ---- A: Exp 1 under the timeout prefetch policy -------------------------
boot exp1_timeout timeout
cfg exp1_timeout '{"experiment":"exp1","prefetch_policy":"timeout","tiers":"recompute,L3","reps":3,"lengths":"512,1024,2048,4096,8192,16384,32512","note":"plan L195 second run; L1/L2 identical across policies per plan, not repeated"}'
bash $WORK/scripts/telemetry.sh start $RESULTS/exp1_timeout
python3 exp1.py --reps 3 --tiers recompute,L3 --policy timeout \
  > $RESULTS/exp1_timeout/run.log 2>&1
bash $WORK/scripts/telemetry.sh stop $RESULTS/exp1_timeout
python3 cachectl.py scrape $RESULTS/exp1_timeout/metrics_after.txt

# ---- B: Exp 1 background-load variant -----------------------------------
boot exp1_bg wait_complete
cfg exp1_bg '{"experiment":"exp1","prefetch_policy":"wait_complete","background_load":"writeload --len 1024 --out 1 --rate 2","tiers":"recompute,L1,L3","lengths":"2048,8192,32512","reps":3,"note":"plan L207"}'
bash $WORK/scripts/telemetry.sh start $RESULTS/exp1_bg
python3 cachectl.py scrape $RESULTS/exp1_bg/metrics_before.txt
python3 exp1.py --reps 3 --tiers recompute,L1,L3 --lengths 2048,8192,32512 \
  --bg rate2 --bg-rate 2 --bg-len 1024 --bg-duration 3000 \
  > $RESULTS/exp1_bg/run.log 2>&1
python3 cachectl.py scrape $RESULTS/exp1_bg/metrics_after.txt
bash $WORK/scripts/telemetry.sh stop $RESULTS/exp1_bg

# ---- C: Exp 2 timeout-policy sweep --------------------------------------
boot exp2_timeout timeout
cfg exp2_timeout '{"experiment":"exp2","prefetch_policy":"timeout","rates":[0,2,8],"reps":3,"probe_len":16384,"writeload":"--len 4096 --out 1 (uniqueness-fixed)","note":"plan L227"}'
bash $WORK/scripts/telemetry.sh start $RESULTS/exp2_timeout
python3 cachectl.py scrape $RESULTS/exp2_timeout/metrics_before.txt
python3 exp2.py --rates 0,2,8 --reps 3 --control timeout_policy --duration 300 \
  > $RESULTS/exp2_timeout/run.log 2>&1
python3 cachectl.py scrape $RESULTS/exp2_timeout/metrics_after.txt
bash $WORK/scripts/telemetry.sh stop $RESULTS/exp2_timeout

echo "ALL THREE MISSING RUNS COMPLETE"
