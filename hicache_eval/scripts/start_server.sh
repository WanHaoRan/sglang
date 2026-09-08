#!/usr/bin/env bash
# start_server.sh <tag> [extra server args...]
# Launches the server, waits for readiness, sends and discards one warm-up
# request, and records the pool sizes the plan asks for.
set -uo pipefail
source /sgl-workspace/sglang/hicache_eval/scripts/env.sh
TAG="$1"; shift
OUT="$RESULTS/$TAG"; mkdir -p "$OUT"

export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=$L3_DIR
export SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE=${L3_MAX_SIZE:-380G}
# nixl reads a different env var; exporting both keeps one launcher for both backends
export SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR=${NIXL_DIR:-$L3_DIR}

BASE_ARGS=(
  --model-path "$MODEL"
  --host 0.0.0.0 --port "$PORT"
  --page-size 64
  --context-length 32768
  --chunked-prefill-size 8192
  --mem-fraction-static 0.85
  --enable-cache-report
  --enable-metrics
  --reasoning-parser qwen3
)
printf "%s\n" "${BASE_ARGS[@]}" "$@" > "$OUT/server_args.txt"
python3 -m sglang.launch_server "${BASE_ARGS[@]}" "$@" > "$OUT/server.log" 2>&1 &
SPID=$!
echo "$SPID" > "$OUT/server.pid"

ready=0
for i in $(seq 1 900); do
  if curl -sf -o /dev/null --max-time 3 "$BASE/health" 2>/dev/null \
     && grep -q "The server is fired up and ready to roll" "$OUT/server.log" 2>/dev/null; then
    ready=1; break
  fi
  if ! kill -0 "$SPID" 2>/dev/null; then
    echo "SERVER DIED during startup; tail of log:" >&2; tail -40 "$OUT/server.log" >&2; exit 1
  fi
  sleep 1
done
[ "$ready" -eq 1 ] || { echo "SERVER NOT READY after 900s" >&2; tail -40 "$OUT/server.log" >&2; exit 1; }

# Warm-up request, discarded.
curl -s "$BASE/generate" -H "Content-Type: application/json" \
  -d "{\"text\":\"warm up the engine please\",\"sampling_params\":{\"max_new_tokens\":1,\"temperature\":0}}" >/dev/null 2>&1

{
  echo "tag: $TAG"
  echo "pid: $SPID"
  grep -iE "max_total_num_tokens|KV cache size|Allocating .* host memory|HiCache|storage backend|hicache" "$OUT/server.log" | head -40
} > "$OUT/startup_facts.txt"
echo "READY $TAG (pid $SPID)"
