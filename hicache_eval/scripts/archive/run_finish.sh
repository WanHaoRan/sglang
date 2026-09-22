#!/usr/bin/env bash
# Close the last three plan gaps: Exp 2 timeout R=8, Exp 1 background-load
# L3/L2, and control C3 (py-spy under load).
source /sgl-workspace/sglang/hicache_eval/scripts/env.sh
cd $WORK/scripts
boot () {
  bash $WORK/scripts/stop_server.sh >/dev/null 2>&1
  find $L3_DIR -mindepth 1 -delete 2>/dev/null
  bash $WORK/scripts/start_server.sh "$1" \
    --enable-hierarchical-cache --hicache-size 100 \
    --hicache-write-policy write_through --hicache-io-backend kernel \
    --hicache-mem-layout page_first \
    --hicache-storage-backend file --hicache-storage-prefetch-policy "$2" \
    --radix-eviction-policy lru
  grep -aoE "existing=[0-9]+ B \([0-9]+ entries\)" $RESULTS/$1/server.log | head -1 > $RESULTS/$1/cold_check.txt
}

# ---- Exp 2 timeout sweep, R=8 (completes R in {0,2,8}) ------------------
boot exp2_timeout_r8 timeout
bash $WORK/scripts/telemetry.sh start $RESULTS/exp2_timeout_r8
python3 exp2.py --rates 8 --reps 3 --control timeout_policy --duration 300 \
  > $RESULTS/exp2_timeout_r8/run.log 2>&1 &
E2=$!
sleep 150
# ---- C3: GIL check under load (plan L233) ------------------------------
SPID=$(ps -eo pid,args | grep "[s]glang::scheduler" | awk '{print $1}' | head -1)
{ echo "=== py-spy top, scheduler pid $SPID, 20 s under R=8 write load ==="
  timeout 30 py-spy top --pid "$SPID" --duration 20 --nonblocking 2>&1 | head -30
  echo; echo "=== py-spy dump, top frames per thread ==="
  timeout 40 py-spy dump --pid "$SPID" 2>&1 | head -60
} > $RESULTS/exp2_timeout_r8/C3_gil_check.txt 2>&1
wait $E2
bash $WORK/scripts/telemetry.sh stop $RESULTS/exp2_timeout_r8

# ---- Exp 1 background-load variant: L3 and L2 -------------------------
boot exp1_bg2 wait_complete
bash $WORK/scripts/telemetry.sh start $RESULTS/exp1_bg2
python3 exp1_bg_l3.py > $RESULTS/exp1_bg2/run.log 2>&1
bash $WORK/scripts/telemetry.sh stop $RESULTS/exp1_bg2
echo "FINISH RUNS COMPLETE"
