#!/usr/bin/env bash
# One-off (2026-09-22): when campaign 8 (GAP=10) prints ALL DONE, launch campaign 9 (GAP=1, everything else identical),
# then keep an iostat log of the SSD in its compare dir until it prints ALL DONE. Runs inside sglang_hicache (docker exec -d).
AC=/sgl-workspace/sglang/agent_cache; R=$AC/results
until grep -qa 'ALL DONE' "$R/compare_driver_c8.log" 2>/dev/null; do sleep 60; done
[ -e "$R/.c9_launched" ] && exit 0
ps -eo args | grep -qE '^bash run_compare.sh' && { echo "driver still running, not launching"; exit 1; }
sleep 30
date -u +%FT%TZ > "$R/.c9_launched"
cd "$AC/scripts" && ARMS='hbm_host three_tier_to three_tier_wc hbm_lru' LEVEL=NAT160 MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 CTX=131072 \
  KV_DTYPE=auto TEMPLATE= TRACE=$AC/traces/lmcache_agentic_trace_262k.json L3_CLEANER_PCT=70,60 NCONV=128 C=128 TURNS=40 \
  GAP=1 GAP_CAP=1200 OFFSET=32 SEED=0 K=0 CHECK_IDS=0 CLIENT_TIMEOUT=9000 nohup bash run_compare.sh > "$R/compare_driver_c9.log" 2>&1 &
CMP=""; until [ -n "$CMP" ]; do sleep 5; CMP=$(grep -ao 'compare dir: .*' "$R/compare_driver_c9.log" 2>/dev/null | cut -d' ' -f3); done
nohup iostat -dxt vdc 10 > "$CMP/iostat_vdc.log" 2>&1 < /dev/null &
echo "c9 launched into $CMP at $(date -u +%T)Z" >> "$R/.c9_launched"
until grep -qa 'ALL DONE' "$R/compare_driver_c9.log" 2>/dev/null; do sleep 60; done
pkill -x iostat; echo "c9 ALL DONE, iostat stopped $(date -u +%T)Z" >> "$R/.c9_launched"
