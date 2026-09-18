#!/usr/bin/env bash
# Boot one SGLang server for the agent KV-tiering study (RUNBOOK 5.2) and wait until it is ready.
#
#   host:      docker exec -it sglang_hicache bash /sgl-workspace/sglang/agent_cache/scripts/start_server.sh [ARM] [LEVEL]
#   container: bash /sgl-workspace/sglang/agent_cache/scripts/start_server.sh [ARM] [LEVEL]
#
#   ARM    hbm_lru (default; single tier, arm a) | hbm_host (L1+L2, arm b) | three_tier (L1+L2+L3 nixl on the NVMe, arm c)
#   LEVEL  P0 (default: L1 262144 tok, host 64 GB) | PH (131072 / 48) | PL (131072 / 18)          (RUNBOOK 5.3)
#
# Writes $RUNDIR/server_<ARM>.log, .pid and _args.txt (RUNDIR = agent_cache/results/<stamp>, the stamp from .current_results).
# Stop the server later with:  kill $(cat $RUNDIR/server_<ARM>.pid)
set -uo pipefail

ARM=${1:-hbm_lru}
LEVEL=${2:-P0}
MODEL=Qwen/Qwen3-32B-FP8
AC=/sgl-workspace/sglang/agent_cache
PORT=30000

[ -d "$AC" ] || { echo "STOP: $AC not found; run inside the container (docker exec -it sglang_hicache bash $0)"; exit 1; }
[ -s "$AC/.current_results" ] || { echo "STOP: $AC/.current_results missing (RUNBOOK 2.4 block 6 writes the stamp)"; exit 1; }
RUNDIR="$AC/results/$(cat "$AC/.current_results")"; mkdir -p "$RUNDIR"

case "$LEVEL" in
  P0) L1=262144; HSIZE=64 ;;
  PH) L1=131072; HSIZE=48 ;;
  PL) L1=131072; HSIZE=18 ;;
  *)  echo "STOP: unknown LEVEL '$LEVEL' (P0 | PH | PL)"; exit 1 ;;
esac

# refuse to double-boot: the port must be free (raw TCP probe: anything listening counts, not only a server with /health) and no previous pid of this arm alive
if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
  echo "STOP: something is already listening on port $PORT; stop it first (kill \$(cat $RUNDIR/server_*.pid))"; exit 1
fi
if [ -f "$RUNDIR/server_$ARM.pid" ] && kill -0 "$(cat "$RUNDIR/server_$ARM.pid")" 2>/dev/null; then
  echo "STOP: $ARM is already running (pid $(cat "$RUNDIR/server_$ARM.pid"))"; exit 1
fi

COMMON=(
  # no --reasoning-parser: under ignore_eos the reply tail can contain <think>, and a parser would move the rest of the
  # text into reasoning_content, which the client does not re-feed (RUNBOOK 4.5; deviation from the measured launch line)
  --model-path "$MODEL" --host 0.0.0.0 --port "$PORT" --context-length 32768
  # replay template (RUNBOOK 4.5): history renders like the generation prompt, so a re-fed reply is a cache hit
  --chat-template "$AC/templates/qwen3_replay_nothink.jinja"
  # KV dtype + attention backend: the only combination measured on this A100 (5.1)
  --kv-cache-dtype fp8_e5m2 --attention-backend triton
  # pools and paging: identical L1 on every arm of a level, 0.85 = measured mem fraction, 64 in-flight requests
  --page-size 64 --chunked-prefill-size 8192 --mem-fraction-static 0.85 --max-total-tokens "$L1"
  --max-running-requests 64 --radix-eviction-policy lru
  # metrics (6.1) + per-response cached-token details (the client reads them)
  --enable-metrics --enable-cache-report
)
HOST=(
  # L2: host pool of HSIZE GB, write-through, kernel io + page_first layout (5.1)
  --enable-hierarchical-cache --hicache-size "$HSIZE"
  --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first
)
L3=(
  # L3: nixl store on the local NVMe, timeout prefetch policy, live-path defaults spelled out (5.1)
  --hicache-storage-backend nixl --hicache-storage-prefetch-policy timeout
  --hicache-storage-backend-extra-config "@$RUNDIR/l3_extra.json"
)
L3ENV=(SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR=/mnt/nvme/hicache_l3 SGLANG_HICACHE_NIXL_BACKEND_PLUGIN=POSIX)

case "$ARM" in
  hbm_lru)    ARGS=("${COMMON[@]}"); ENVV=() ;;
  hbm_host)   ARGS=("${COMMON[@]}" "${HOST[@]}"); ENVV=() ;;
  three_tier)
    [ -d /mnt/nvme/hicache_l3 ] || { echo "STOP: /mnt/nvme/hicache_l3 missing: the NVMe is not mounted (RUNBOOK 2.2, every morning)"; exit 1; }
    printf '{"prefetch_threshold": 256, "prefetch_timeout_base": 1.0, "prefetch_timeout_per_ki_token": 0.25}\n' > "$RUNDIR/l3_extra.json"
    ARGS=("${COMMON[@]}" "${HOST[@]}" "${L3[@]}"); ENVV=("${L3ENV[@]}") ;;
  *) echo "STOP: unknown ARM '$ARM' (hbm_lru | hbm_host | three_tier)"; exit 1 ;;
esac

echo "arm: $ARM  level: $LEVEL (L1=$L1 tokens, host=$HSIZE GB)  log: $RUNDIR/server_$ARM.log"
# setsid + nohup: the server survives the end of this docker exec session
env "${ENVV[@]}" setsid nohup python3 -m sglang.launch_server "${ARGS[@]}" > "$RUNDIR/server_$ARM.log" 2>&1 < /dev/null &
echo $! > "$RUNDIR/server_$ARM.pid"
echo "launched $ARM: pid $(cat "$RUNDIR/server_$ARM.pid")   (watch: tail -f $RUNDIR/server_$ARM.log)"

# wait up to 2400 s (JIT-cold boot ~633 s, warm 278-285 s, RUNBOOK 2.6): /health AND the 'fired up' log line
for i in $(seq 1 2400); do
  if curl -sf -o /dev/null --max-time 3 "http://127.0.0.1:$PORT/health" \
     && grep -q 'fired up and ready to roll' "$RUNDIR/server_$ARM.log"; then
    echo "READY: $ARM after ${i}s"; break
  fi
  if ! kill -0 "$(cat "$RUNDIR/server_$ARM.pid")" 2>/dev/null; then
    echo "STOP: $ARM died during startup; last lines:"; tail -20 "$RUNDIR/server_$ARM.log"; exit 1
  fi
  sleep 1
done

# startup facts (each value labelled; RUNBOOK 5.2 assert block has the full list)
grep -a 'server_args=' "$RUNDIR/server_$ARM.log" | head -1 > "$RUNDIR/server_${ARM}_args.txt"
echo "template loaded:  $(grep -c 'Detected user specified Jinja chat template' "$RUNDIR/server_$ARM.log")   (want 1)"
echo "device pool:      $(grep -aoE 'max_total_num_tokens=[0-9]+' "$RUNDIR/server_$ARM.log" | head -1)   (want $L1)"
echo "profiled warning: $(grep -c 'larger than the profiled value' "$RUNDIR/server_$ARM.log")   (want 0)"
if [ "$ARM" != hbm_lru ]; then
  echo "host pool:        $(grep -aoE 'host pool: [0-9]+ tokens, [0-9.]+ GB' "$RUNDIR/server_$ARM.log" | head -1)   (want $HSIZE GB)"
fi
if [ "$ARM" = three_tier ]; then
  echo "nixl backend:     $(grep -c 'Backend POSIX was instantiated' "$RUNDIR/server_$ARM.log")   (want 1)"
  echo "L3 cleaner dir:   $(grep -aoE "HiCacheL3Cleaner started: dirs=\[[^]]*\]" "$RUNDIR/server_$ARM.log" | head -1)   (want /mnt/nvme/hicache_l3)"
fi
