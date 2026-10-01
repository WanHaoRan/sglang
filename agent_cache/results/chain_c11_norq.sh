#!/usr/bin/env bash
# One-off (2026-10-01): when campaign 11 (compare_20261001_013443) prints ALL DONE, run its fifth arm,
# three_tier_wc_norq (three_tier_wc with the #39283 admission-time SSD re-query disabled), with the same knobs, then
# append that arm's manifest line to campaign 11's manifest. Runs inside sglang_hicache (docker exec -d).
AC=/sgl-workspace/sglang/agent_cache; R=$AC/results; C11=$R/compare_20261001_013443
until grep -qa 'ALL DONE' "$R/compare_driver_c11.log" 2>/dev/null; do sleep 60; done
[ -e "$R/.c11_norq_launched" ] && exit 0
ps -eo args | grep -qE '^bash run_compare.sh' && { echo "a driver is still running, not launching"; exit 1; }
sleep 30
date -u +%FT%TZ > "$R/.c11_norq_launched"
cd "$AC/scripts" && ARMS=three_tier_wc_norq LEVEL=NAT160 MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 CTX=131072 KV_DTYPE=auto \
  TEMPLATE= TRACE=$AC/traces/lmcache_agentic_trace_262k.json L3_CLEANER_PCT=70,60 NCONV=128 C=128 TURNS=40 GAP=10 \
  GAP_CAP=1200 OFFSET=32 SEED=0 K=0 CHECK_IDS=0 CLIENT_TIMEOUT=9000 SGLANG_TIMEOUT_KEEP_ALIVE=3600 \
  bash run_compare.sh > "$R/compare_driver_c11_norq.log" 2>&1
N=$(grep -ao 'compare dir: .*' "$R/compare_driver_c11_norq.log" | head -1 | cut -d' ' -f3)
grep -q '^three_tier_wc_norq ' "$C11/manifest.txt" || cat "$N/manifest.txt" >> "$C11/manifest.txt"
echo "norq arm finished $(date -u +%T)Z in $N; manifest line appended to $C11/manifest.txt" >> "$R/.c11_norq_launched"
