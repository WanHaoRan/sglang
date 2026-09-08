#!/usr/bin/env bash
# Exp 2 / 3 / 4 for one model on the nixl L3 backend.
#   bash run_exp234.sh <qwen8b|qwen32b|llama70b>
# Each stage is independent: a failure is logged and the next stage still runs.
set -uo pipefail
S=/sgl-workspace/sglang/hicache_eval/scripts
KEY="${1:?usage: run_exp234.sh <model key>}"
source "$S/models.sh" "$KEY"
source "$S/env.sh"

export HICACHE_BACKEND=nixl
export EXP2_SUBDIR="exp2_$KEY"
export EXP3_SUBDIR="exp3_$KEY"
LOG="$RESULTS/run_$KEY.log"
mkdir -p "$RESULTS/$EXP2_SUBDIR" "$RESULTS/$EXP3_SUBDIR"

# Write pressure saturates at P/4096 req/s, which is 3.7 (8B), 0.57 (32B) and
# 0.22 (70B). A fixed rate list would put five of seven points past saturation
# on the big models, so the ladder is scaled per model to span the same
# achieved-rate range. Deviation from the plan's fixed {0,.5,1,2,4,8,16}.
case "$KEY" in
  qwen8b)   RATES="0,0.5,1,2,4,8,16"; TRATES="0,2,8";    CRATE=8 ;;
  qwen32b)  RATES="0,0.25,0.5,1,2,4"; TRATES="0,0.5,2";  CRATE=2 ;;
  llama70b) RATES="0,0.1,0.25,0.5,1,2"; TRATES="0,0.25,1"; CRATE=1 ;;
esac

HI_COMMON="--enable-hierarchical-cache --hicache-size 100 --hicache-io-backend kernel \
--hicache-mem-layout page_first --hicache-storage-backend nixl --radix-eviction-policy lru \
--hicache-write-policy write_through"

say() { echo -e "\n========== [$KEY] $* ==========" | tee -a "$LOG"; }
boot() { # boot <tag> <extra args...>
  local tag="$1"; shift
  bash "$S/stop_server.sh" >>"$LOG" 2>&1
  bash -c "source $S/env.sh && MODEL='$MODEL' REASONING_PARSER='$REASONING_PARSER' \
    bash $S/start_server.sh '$tag' $MODEL_EXTRA_ARGS $*" >>"$LOG" 2>&1
}
wipe() { find "$L3_DIR" -mindepth 1 -delete 2>/dev/null; }

# ---------------------------------------------------------------- Exp 2 ----
say "Exp2 baseline sweep (wait_complete)  rates=$RATES"
wipe
if boot "${EXP2_SUBDIR}_base" $HI_COMMON --hicache-storage-prefetch-policy wait_complete; then
  python3 "$S/exp2.py" --rates "$RATES" --reps 3 --control baseline --duration 150 \
    >>"$RESULTS/$EXP2_SUBDIR/run_baseline.log" 2>&1 || echo "exp2 baseline FAILED" | tee -a "$LOG"
else echo "exp2 baseline server FAILED" | tee -a "$LOG"; fi

say "Exp2 timeout-policy sweep  rates=$TRATES"
wipe
if boot "${EXP2_SUBDIR}_timeout" $HI_COMMON --hicache-storage-prefetch-policy timeout; then
  python3 "$S/exp2.py" --rates "$TRATES" --reps 3 --control timeout_policy --duration 150 \
    >>"$RESULTS/$EXP2_SUBDIR/run_timeout.log" 2>&1 || echo "exp2 timeout FAILED" | tee -a "$LOG"
else echo "exp2 timeout server FAILED" | tee -a "$LOG"; fi

say "Exp2 C1 (L3 on tmpfs)  R=$CRATE"
bash "$S/stop_server.sh" >>"$LOG" 2>&1
mountpoint -q /mnt/tmpfs_l3 && umount /mnt/tmpfs_l3
mkdir -p /mnt/tmpfs_l3 && mount -t tmpfs -o size=60G tmpfs /mnt/tmpfs_l3
if L3_DIR=/mnt/tmpfs_l3 NIXL_DIR=/mnt/tmpfs_l3 boot "${EXP2_SUBDIR}_c1" $HI_COMMON \
     --hicache-storage-prefetch-policy wait_complete; then
  # Page-cache eviction is meaningless on tmpfs; the probe path still calls it.
  L3_DIR=/mnt/tmpfs_l3 python3 "$S/exp2.py" --rates "$CRATE" --reps 3 --control C1_tmpfs \
    --duration 150 >>"$RESULTS/$EXP2_SUBDIR/run_c1.log" 2>&1 || echo "exp2 C1 FAILED" | tee -a "$LOG"
else echo "exp2 C1 server FAILED" | tee -a "$LOG"; fi
bash "$S/stop_server.sh" >>"$LOG" 2>&1; umount /mnt/tmpfs_l3 2>/dev/null

say "Exp2 C2 (backup_skip)  R=$CRATE  -- populate clean, then probe with writes disabled"
wipe
if boot "${EXP2_SUBDIR}_c2pop" $HI_COMMON --hicache-storage-prefetch-policy wait_complete; then
  python3 "$S/exp2.py" --rates 0 --reps 3 --control C2_populate --duration 20 \
    >>"$RESULTS/$EXP2_SUBDIR/run_c2_populate.log" 2>&1
  if HICACHE_EVAL_BACKUP_SKIP=1 boot "${EXP2_SUBDIR}_c2" $HI_COMMON \
       --hicache-storage-prefetch-policy wait_complete; then
    python3 "$S/exp2.py" --rates "$CRATE" --reps 3 --control C2_backup_skip --duration 150 \
      --skip-populate >>"$RESULTS/$EXP2_SUBDIR/run_c2.log" 2>&1 || echo "exp2 C2 FAILED" | tee -a "$LOG"
  else echo "exp2 C2 server FAILED" | tee -a "$LOG"; fi
else echo "exp2 C2 populate server FAILED" | tee -a "$LOG"; fi

say "Exp2 C3 (py-spy under write load R=$CRATE)"
SPID=$(pgrep -f "sglang.launch_server" | head -1)
if [ -n "$SPID" ]; then
  python3 "$S/writeload.py" --len 4096 --out 1 --rate "$CRATE" --duration 60 \
    >"$RESULTS/$EXP2_SUBDIR/c3_writeload.json" 2>/dev/null &
  WPID=$!
  sleep 15
  py-spy top --pid "$SPID" --duration 20 --nonblocking \
    >"$RESULTS/$EXP2_SUBDIR/c3_pyspy_top.txt" 2>&1 || echo "py-spy top failed" | tee -a "$LOG"
  for i in 1 2 3; do
    py-spy dump --pid "$SPID" --nonblocking >>"$RESULTS/$EXP2_SUBDIR/c3_pyspy_dump.txt" 2>&1
    sleep 3
  done
  wait $WPID 2>/dev/null
else echo "C3 skipped: no server pid" | tee -a "$LOG"; fi

# ------------------------------------------------------------- Exp 3 / 4 ----
say "Exp3 five conditions"
bash "$S/stop_server.sh" >>"$LOG" 2>&1
python3 "$S/exp3.py" --conditions wt100,wt30,wts100,wts30,nohicache \
  --client-args "--request-length 1024 --output-length 1024 --num-clients 16 --num-rounds 4 --max-parallel 8 --request-rate 2" \
  >>"$RESULTS/$EXP3_SUBDIR/run_exp3.log" 2>&1 || echo "exp3 FAILED" | tee -a "$LOG"

if [ "$SUPPORTS_THINKING" = "1" ]; then
  say "Exp4 strip-thinking-cache oracle"
  python3 "$S/exp3.py" --conditions oracle100,oracle30 \
    --client-args "--request-length 1024 --output-length 1024 --num-clients 16 --num-rounds 4 --max-parallel 8 --request-rate 2" \
    >>"$RESULTS/$EXP3_SUBDIR/run_exp4.log" 2>&1 || echo "exp4 FAILED" | tee -a "$LOG"
else
  say "Exp4 SKIPPED: strip_thinking_cache is gated on reasoning_tokens>0 (schedule_batch.py:1381) and this model emits none"
  echo "SKIPPED_NO_REASONING_CHANNEL" > "$RESULTS/$EXP3_SUBDIR/exp4_NOT_APPLICABLE.txt"
fi

bash "$S/stop_server.sh" >>"$LOG" 2>&1
say "DONE"
