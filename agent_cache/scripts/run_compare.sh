#!/usr/bin/env bash
# Run the same client workload against several arms back to back, each on a fresh boot with identical pools.
#   container: bash /sgl-workspace/sglang/agent_cache/scripts/run_compare.sh
#   host:      docker exec -d sglang_hicache bash -c 'cd /sgl-workspace/sglang/agent_cache/scripts && nohup bash run_compare.sh > ../results/compare_driver.log 2>&1'
# Knobs (env): ARMS="hbm_lru hbm_host three_tier"  LEVEL=PL  L1_TOKENS=65536  HOST_GB=12  NCONV=32  C=32  TURNS=12  GAP=10  GAP_CAP=0  CHECK_IDS=0
# Writes results/compare_<UTC time>/manifest.txt: one line per arm  "<arm> <server run dir> <client dir>"  (input of timeline.py).
# Progress: lines "STAGE START|DONE|FAILED <arm>" in the driver log; "ALL DONE" at the end. Stop cleanly: kill this script, then stop_server.sh.
set -uo pipefail
AC=/sgl-workspace/sglang/agent_cache
cd "$AC/scripts" || { echo "STOP: $AC/scripts missing (run inside the container)"; exit 1; }
ARMS=${ARMS:-"hbm_lru hbm_host three_tier"}; LEVEL=${LEVEL:-PL}
export L1_TOKENS=${L1_TOKENS:-65536} HOST_GB=${HOST_GB:-12}
export NCONV=${NCONV:-32} C=${C:-32} TURNS=${TURNS:-12} GAP=${GAP:-10} GAP_CAP=${GAP_CAP:-0} CHECK_IDS=${CHECK_IDS:-0}
CMP="$AC/results/compare_$(date -u +%Y%m%d_%H%M%S)"; mkdir -p "$CMP"; MANIFEST="$CMP/manifest.txt"
printf 'arms=%s level=%s L1_TOKENS=%s HOST_GB=%s NCONV=%s C=%s TURNS=%s GAP=%s GAP_CAP=%s started_utc=%s\n' \
  "$ARMS" "$LEVEL" "$L1_TOKENS" "$HOST_GB" "$NCONV" "$C" "$TURNS" "$GAP" "$GAP_CAP" "$(date -u +%FT%TZ)" > "$CMP/config.txt"
echo "compare dir: $CMP"; cat "$CMP/config.txt"

for ARM in $ARMS; do
  echo "STAGE START $ARM $(date -u +%T)"
  bash stop_server.sh
  if [ "$ARM" = three_tier ]; then
    # cold L3 for every three-tier arm: wipe the nixl store (find, not rm -rf: rm silently fails past ~10k files, RUNBOOK 10)
    find /mnt/nvme/hicache_l3 -mindepth 1 -delete 2>/dev/null; echo "L3 wiped: $(find /mnt/nvme/hicache_l3 -type f | wc -l) files left"
  fi
  if ! bash start_server.sh "$ARM" "$LEVEL"; then echo "STAGE FAILED $ARM (boot)"; bash stop_server.sh; continue; fi
  RUN=$(cat "$AC/results/.current_run")
  T0=$(date +%s)
  if TAG="cmp_${ARM}" bash start_client.sh; then
    CLIENT=$(ls -d "$AC/results/$RUN"/client_*_cmp_"$ARM" | tail -1)
    echo "$ARM $RUN $(basename "$CLIENT")" >> "$MANIFEST"
    echo "STAGE DONE $ARM $(date -u +%T) client wall $(( $(date +%s) - T0 )) s -> $CLIENT"
  else
    echo "STAGE FAILED $ARM (client)"
  fi
  bash stop_server.sh
done
echo "ALL DONE $(date -u +%T); manifest: $MANIFEST"; cat "$MANIFEST"
