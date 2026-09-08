#!/usr/bin/env bash
# Full Exp 2/3/4 sweep on nixl, smallest model first so the harness is
# validated before the expensive 70B leg.
S=/sgl-workspace/sglang/hicache_eval/scripts
MASTER=/sgl-workspace/sglang/hicache_eval/results/20260908_nixl_exp234/MASTER.log
for k in qwen8b qwen32b llama70b; do
  echo "################ $(date -u +%FT%TZ)  START $k" >> "$MASTER"
  bash "$S/run_exp234.sh" "$k" >> "$MASTER" 2>&1
  echo "################ $(date -u +%FT%TZ)  END   $k (rc=$?)" >> "$MASTER"
done
echo "################ $(date -u +%FT%TZ)  ALL MODELS DONE" >> "$MASTER"
