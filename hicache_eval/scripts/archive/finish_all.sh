#!/usr/bin/env bash
# Post-sweep items from PENDING.md: C3 for all three models (never successfully
# run in-driver), the qwen8b baseline re-run for its rate axis, and the C2
# patch revert.
set -uo pipefail
S=/sgl-workspace/sglang/hicache_eval/scripts
MASTER=${MASTER:-${RESULTS:+$RESULTS/FINISH.log}}
MASTER=${MASTER:-/sgl-workspace/sglang/hicache_eval/results/20260908_nixl_exp234/FINISH.log}
say() { echo -e "\n########## $(date -u +%H:%M) $*" | tee -a "$MASTER"; }

for KEY in qwen8b qwen32b llama70b; do
  ( source "$S/models.sh" "$KEY" >/dev/null
    source "$S/env.sh"
    say "$KEY: boot for C3"
    bash "$S/stop_server.sh" >>"$MASTER" 2>&1
    find "$L3_DIR" -mindepth 1 -delete 2>/dev/null
    bash -c "source $S/env.sh && MODEL='$MODEL' REASONING_PARSER='$REASONING_PARSER' \
      bash $S/start_server.sh finish_${KEY} $MODEL_EXTRA_ARGS \
      --enable-hierarchical-cache --hicache-size 100 --hicache-io-backend kernel \
      --hicache-mem-layout page_first --hicache-storage-backend nixl \
      --radix-eviction-policy lru --hicache-write-policy write_through \
      --hicache-storage-prefetch-policy wait_complete" >>"$MASTER" 2>&1

    if [ "$KEY" = "qwen8b" ]; then
      say "$KEY: Exp2 baseline re-run (rate axis + paired control)"
      EXP2_SUBDIR="exp2_qwen8b_rerun" HICACHE_BACKEND=nixl \
        python3 "$S/exp2.py" --rates 0,0.5,1,2,4,8,16 --reps 3 --control baseline \
        --duration 150 >>"$RESULTS/exp2_qwen8b_rerun_run.log" 2>&1 \
        || echo "rerun FAILED" | tee -a "$MASTER"
    fi

    say "$KEY: C3 py-spy GIL attribution"
    bash "$S/c3_pyspy.sh" "$KEY" >>"$MASTER" 2>&1 || echo "C3 $KEY FAILED" | tee -a "$MASTER"
    bash "$S/stop_server.sh" >>"$MASTER" 2>&1
  )
done

say "reverting the C2 eval patch"
cd /sgl-workspace/sglang && git checkout -- python/sglang/srt/managers/cache_controller.py
grep -c HICACHE_EVAL_BACKUP_SKIP python/sglang/srt/managers/cache_controller.py \
  && echo "!! patch still present" | tee -a "$MASTER" \
  || echo "patch reverted cleanly" | tee -a "$MASTER"
say "FINISH DONE"
