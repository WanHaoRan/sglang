#!/usr/bin/env bash
# Run the same client workload against several arms back to back, each on a fresh boot with identical pools.
#   container: bash /sgl-workspace/sglang/agent_cache/scripts/run_compare.sh
#   host:      docker exec -d sglang_hicache bash -c 'cd /sgl-workspace/sglang/agent_cache/scripts && nohup bash run_compare.sh > ../results/compare_driver.log 2>&1'
# Knobs (env): ARMS="hbm_lru hbm_host three_tier_to three_tier_wc"  LEVEL=NAT64  [L1_TOKENS= HOST_GB= only to override the level]
#              MODEL=Qwen/Qwen3-32B  CTX=40960  L3_CLEANER_PCT=70,60  NCONV=48  C=48  TURNS=24  GAP=21  GAP_CAP=600  CHECK_IDS=0
# C18 (2026-09-18, A100) was: ARMS="hbm_lru hbm_host three_tier" LEVEL=PL L1_TOKENS=65536 HOST_GB=12 NCONV=32 C=32 TURNS=12 GAP=10 GAP_CAP=600
# Writes results/compare_<UTC time>/manifest.txt: one line per arm  "<arm> <server run dir> <client dir>"  (input of timeline.py).
# Progress: lines "STAGE START|DONE|FAILED <arm>" in the driver log; "ALL DONE" at the end. Stop cleanly: kill this script, then stop_server.sh.
set -uo pipefail
AC=/sgl-workspace/sglang/agent_cache
cd "$AC/scripts" || { echo "STOP: $AC/scripts missing (run inside the container)"; exit 1; }
ARMS=${ARMS:-"hbm_lru hbm_host three_tier_to three_tier_wc"}; LEVEL=${LEVEL:-NAT64}
# L1_TOKENS / HOST_GB are exported only when the caller set them: setting them unconditionally would turn every level into "custom"
[ -n "${L1_TOKENS:-}" ] && export L1_TOKENS; [ -n "${HOST_GB:-}" ] && export HOST_GB
export MODEL=${MODEL:-Qwen/Qwen3-32B} CTX=${CTX:-40960} L3_CLEANER_PCT=${L3_CLEANER_PCT-70,60}
[ -n "${KV_DTYPE:-}" ] && export KV_DTYPE; [ -n "${TEMPLATE+x}" ] && export TEMPLATE; [ -n "${TRACE:-}" ] && export TRACE
export NCONV=${NCONV:-48} C=${C:-48} TURNS=${TURNS:-24} GAP=${GAP:-21} GAP_CAP=${GAP_CAP:-600} CHECK_IDS=${CHECK_IDS:-0}
# OFFSET rotates the conversation list; 29 is the 48-wide window with the most >=24-turn conversations (42/48), so c holds for the whole run
export OFFSET=${OFFSET:-29} SEED=${SEED:-0} K=${K:-0}
# CLIENT_TIMEOUT: wall cap per arm in seconds (0 = none), enforced inside the client (start_client.sh MAX_SECONDS ->
# replay_agentic.py --max-seconds): conversations stop at their next turn boundary, in-flight requests complete, every
# completed turn is kept and the summary line says "capped": true, which is what prints STAGE CAPPED below. (Campaign 7 used
# `timeout --foreground` here, which only signals the bash wrapper and never reached the client: no arm was ever capped.)
# A no-tiering arm recomputes every evicted session and can run 2-10x longer than the tiered ones; cap it and compare
# over the common window (RUNBOOK 7.3).
export CLIENT_TIMEOUT=${CLIENT_TIMEOUT:-0}
CMP="$AC/results/compare_$(date -u +%Y%m%d_%H%M%S)"; mkdir -p "$CMP"; MANIFEST="$CMP/manifest.txt"
printf 'arms=%s level=%s L1_TOKENS=%s HOST_GB=%s MODEL=%s CTX=%s KV_DTYPE=%s TRACE=%s L3_CLEANER_PCT=%s NCONV=%s C=%s TURNS=%s GAP=%s GAP_CAP=%s OFFSET=%s SEED=%s K=%s CLIENT_TIMEOUT=%s started_utc=%s\n' \
  "$ARMS" "$LEVEL" "${L1_TOKENS:-natural}" "${HOST_GB:-level}" "$MODEL" "$CTX" "${KV_DTYPE:-fp8_e5m2}" "${TRACE:-default}" "$L3_CLEANER_PCT" "$NCONV" "$C" "$TURNS" "$GAP" "$GAP_CAP" "$OFFSET" "$SEED" "$K" "$CLIENT_TIMEOUT" "$(date -u +%FT%TZ)" > "$CMP/config.txt"
echo "compare dir: $CMP"; cat "$CMP/config.txt"

for ARM in $ARMS; do
  echo "STAGE START $ARM $(date -u +%T)"
  bash stop_server.sh
  case "$ARM" in three_tier*)
    # cold L3 for every three-tier arm: wipe the nixl store (find, not rm -rf: rm silently fails past ~10k files, RUNBOOK 10)
    find /mnt/ssd/hicache_l3 -mindepth 1 -delete 2>/dev/null; echo "L3 wiped: $(find /mnt/ssd/hicache_l3 -type f | wc -l) files left" ;;
  esac
  if ! bash start_server.sh "$ARM" "$LEVEL"; then echo "STAGE FAILED $ARM (boot)"; bash stop_server.sh; continue; fi
  RUN=$(cat "$AC/results/.current_run")
  T0=$(date +%s)
  TAG="cmp_${ARM}" MAX_SECONDS="${CLIENT_TIMEOUT}" bash start_client.sh; rc=$?
  CLIENT=$(ls -d "$AC/results/$RUN"/client_*_cmp_"$ARM" 2>/dev/null | tail -1)
  # the run counts when the client wrote its summary line; a non-zero rc after that is start_client.sh's per-turn view, not the run
  if [ -n "$CLIENT" ] && grep -q '"kind": "summary"' "$CLIENT/client.jsonl" 2>/dev/null; then
    grep -q '"capped": true' "$CLIENT/client.jsonl" && echo "STAGE CAPPED $ARM at ${CLIENT_TIMEOUT}s (completed turns kept)"
    [ "$rc" -eq 0 ] || echo "note: start_client.sh exited $rc after the client finished (post-processing); run kept"
    echo "$ARM $RUN $(basename "$CLIENT")" >> "$MANIFEST"
    echo "STAGE DONE $ARM $(date -u +%T) client wall $(( $(date +%s) - T0 )) s -> $CLIENT"
  else
    echo "STAGE FAILED $ARM (client, rc=$rc, no summary line)"
  fi
  bash stop_server.sh
done
echo "ALL DONE $(date -u +%T); manifest: $MANIFEST"; cat "$MANIFEST"
