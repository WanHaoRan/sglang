#!/usr/bin/env bash
# Boot one SGLang server for the agent KV-tiering study (RUNBOOK 5.2) and wait until it is ready.
#
#   host:      docker exec -it sglang_hicache bash /sgl-workspace/sglang/agent_cache/scripts/start_server.sh [ARM] [LEVEL]
#   container: bash /sgl-workspace/sglang/agent_cache/scripts/start_server.sh [ARM] [LEVEL]
#
#   ARM    hbm_lru (default; single tier, arm a) | hbm_host (L1+L2, arm b) | three_tier (L1+L2+L3 nixl on the NVMe, arm c)
#   LEVEL  P0 (default: L1 262144 tok, host 64 GB) | PH (131072 / 48) | PL (131072 / 18)          (RUNBOOK 5.3)
#   Overrides (env): L1_TOKENS=<tokens> and/or HOST_GB=<GB> replace the level's pools; the run dir is then named custom_L1<tokens>_H<GB>.
#   Keep the host pool >= L1 (host tokens = int(GB*1e9/131072)//64*64; 12 GB = 91,520 tokens): a smaller host pool is an untraced path.
#
# Every boot gets its own directory  agent_cache/results/<UTC yyyymmdd_HHMMSS>_<ARM>_<LEVEL>/  holding server.log, server.pid,
# server_args.txt, run.txt (+ l3_extra.json); agent_cache/results/.current_run names the latest one (start_client.sh writes into it).
# Stop the server with:  bash stop_server.sh
set -uo pipefail

ARM=${1:-hbm_lru}
LEVEL=${2:-P0}
MODEL=Qwen/Qwen3-32B-FP8
AC=/sgl-workspace/sglang/agent_cache
PORT=30000

[ -d "$AC" ] || { echo "STOP: $AC not found; run inside the container (docker exec -it sglang_hicache bash $0)"; exit 1; }
RESULTS="$AC/results"; mkdir -p "$RESULTS"

case "$LEVEL" in
  P0) L1=262144; HSIZE=64 ;;
  PH) L1=131072; HSIZE=48 ;;
  PL) L1=131072; HSIZE=18 ;;
  *)  echo "STOP: unknown LEVEL '$LEVEL' (P0 | PH | PL)"; exit 1 ;;
esac
if [ -n "${L1_TOKENS:-}" ] || [ -n "${HOST_GB:-}" ]; then
  L1=${L1_TOKENS:-$L1}; HSIZE=${HOST_GB:-$HSIZE}; LEVEL="custom_L1${L1}_H${HSIZE}"
fi
RUN="$RESULTS/$(date -u +%Y%m%d_%H%M%S)_${ARM}_${LEVEL}"

# refuse to double-boot: the port must be free (raw TCP probe: anything listening counts, not only a server with /health) and no previous pid of this arm alive
if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
  echo "STOP: something is already listening on port $PORT; stop it first (bash stop_server.sh)"; exit 1
fi
if [ -s "$RESULTS/.current_run" ] && [ -f "$RESULTS/$(cat "$RESULTS/.current_run")/server.pid" ] \
   && kill -0 "$(cat "$RESULTS/$(cat "$RESULTS/.current_run")/server.pid")" 2>/dev/null; then
  echo "STOP: a server is already running: $(cat "$RESULTS/.current_run") (bash stop_server.sh first)"; exit 1
fi
mkdir -p "$RUN"; basename "$RUN" > "$RESULTS/.current_run"

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
  --hicache-storage-backend-extra-config "@$RUN/l3_extra.json"
)
L3ENV=(SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR=/mnt/nvme/hicache_l3 SGLANG_HICACHE_NIXL_BACKEND_PLUGIN=POSIX)

case "$ARM" in
  hbm_lru)    ARGS=("${COMMON[@]}"); ENVV=() ;;
  hbm_host)   ARGS=("${COMMON[@]}" "${HOST[@]}"); ENVV=() ;;
  three_tier)
    [ -d /mnt/nvme/hicache_l3 ] || { echo "STOP: /mnt/nvme/hicache_l3 missing: the NVMe is not mounted (RUNBOOK 2.2, every morning)"; exit 1; }
    printf '{"prefetch_threshold": 256, "prefetch_timeout_base": 1.0, "prefetch_timeout_per_ki_token": 0.25}\n' > "$RUN/l3_extra.json"
    ARGS=("${COMMON[@]}" "${HOST[@]}" "${L3[@]}"); ENVV=("${L3ENV[@]}") ;;
  *) echo "STOP: unknown ARM '$ARM' (hbm_lru | hbm_host | three_tier)"; exit 1 ;;
esac

echo "arm: $ARM  level: $LEVEL (L1=$L1 tokens, host=$HSIZE GB)  run dir: $RUN"
# setsid + nohup: the server survives the end of this docker exec session; SGLANG_LOG_MS=1 -> millisecond log timestamps
env SGLANG_LOG_MS=1 "${ENVV[@]}" setsid nohup python3 -m sglang.launch_server "${ARGS[@]}" > "$RUN/server.log" 2>&1 < /dev/null &
echo $! > "$RUN/server.pid"
printf 'arm=%s\nlevel=%s\nL1=%s\nhost_gb=%s\nstarted_utc=%s\n' "$ARM" "$LEVEL" "$L1" "$HSIZE" "$(date -u +%FT%TZ)" > "$RUN/run.txt"
echo "launched $ARM: pid $(cat "$RUN/server.pid")   (watch: tail -f $RUN/server.log)"

# wait up to 2400 s (JIT-cold boot ~633 s, warm 278-285 s, RUNBOOK 2.6): /health AND the 'fired up' log line
for i in $(seq 1 2400); do
  if curl -sf -o /dev/null --max-time 3 "http://127.0.0.1:$PORT/health" \
     && grep -q 'fired up and ready to roll' "$RUN/server.log"; then
    echo "READY: $ARM after ${i}s"; break
  fi
  if ! kill -0 "$(cat "$RUN/server.pid")" 2>/dev/null; then
    echo "STOP: $ARM died during startup; last lines:"; tail -20 "$RUN/server.log"; exit 1
  fi
  sleep 1
done

# startup facts (each value labelled; RUNBOOK 5.2 assert block has the full list)
grep -a 'server_args=' "$RUN/server.log" | head -1 > "$RUN/server_args.txt"
echo "template loaded:  $(grep -c 'Detected user specified Jinja chat template' "$RUN/server.log")   (want 1)"
echo "device pool:      $(grep -aoE 'max_total_num_tokens=[0-9]+' "$RUN/server.log" | head -1)   (want $L1)"
echo "profiled warning: $(grep -c 'larger than the profiled value' "$RUN/server.log")   (want 0)"
if [ "$ARM" != hbm_lru ]; then
  echo "host pool:        $(grep -aoE 'host pool: [0-9]+ tokens, [0-9.]+ GB' "$RUN/server.log" | head -1)   (want $HSIZE GB)"
fi
if [ "$ARM" = three_tier ]; then
  echo "nixl backend:     $(grep -c 'Backend POSIX was instantiated' "$RUN/server.log")   (want 1)"
  echo "L3 cleaner dir:   $(grep -aoE "HiCacheL3Cleaner started: dirs=\[[^]]*\]" "$RUN/server.log" | head -1)   (want /mnt/nvme/hicache_l3)"
fi
