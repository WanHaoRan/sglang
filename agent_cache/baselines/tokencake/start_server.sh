#!/usr/bin/env bash
# Boot TokenCake (pku-lemonade/TokenCake, the authors' vLLM 0.22.0 fork) on the host and wait until it is ready.
# The modes and server environment are those of its experiment driver (tools/tokencake_experiments/campaign.py):
#   offload-agent (default)  TokenCake scheduling + lifecycle-based preservation over vLLM's native CPU offloading
#   agent                    TokenCake scheduling only
#   offload                  preservation only
#   native                   the same checkout with TokenCake off: its own vLLM 0.22.0 base
# TokenCake has two tiers, GPU and CPU; it has no disk tier.
#
#   host: bash /home/wanhr/sglang/agent_cache/baselines/tokencake/start_server.sh
#
# Knobs (environment): MODE (above), HOST_GB (160: the CPU KV capacity in decimal GB, as SGLang's host pool; passed
#   to --kv-offloading-size in GiB, at most CPU_GIB_MAX: 120, below), GPU_KV_TOKENS (668096), MAX_SEQS (64), BATCH_TOKENS (8192), CTX (262144),
#   MEM_CAP_GB, SERVER_EXTRA, VENV (this directory's .venv, built by setup.sh).
# The client (baselines/common/backends.py, TokenCakeBackend) sends request ids tc-<uuid4>, the metadata in
# vllm_xargs.tokencake and stall_started/stall_finished to /v1/tokencake/events around every wait.
# Stop with: bash agent_cache/baselines/common/stop_server.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../common/lib.sh"
VENV=${VENV:-$HERE/.venv}
# as `source .venv/bin/activate` in the official instructions: FlashInfer's JIT build needs the venv's ninja
[ -x "$VENV/bin/vllm" ] || { echo "STOP: no TokenCake build in $VENV (bash $HERE/setup.sh)"; exit 1; }
MODE=${MODE:-offload-agent}; HOST_GB=${HOST_GB:-160}
HOST_GIB=$(awk -v g="$HOST_GB" 'BEGIN { printf "%.2f", g * 1e9 / 2^30 }')
# vLLM 0.22's native CPU offloading allocates its CPU tier as one pinned tensor (kv_offload/cpu/gpu_worker.py,
# torch.zeros(..., pin_memory=True)); on this host one pinned allocation succeeds at 122 GiB and fails at 130 and 149
# GiB ("CUDA error: out of memory", tested 2026-11-21), so the tier is capped at CPU_GIB_MAX: a deviation from the
# 160 GB the other systems get, which pin in pieces or register memory.
CPU_GIB_MAX=${CPU_GIB_MAX:-120}
HOST_GIB=$(awk -v g="$HOST_GIB" -v m="$CPU_GIB_MAX" 'BEGIN { printf "%.2f", (g > m ? m : g) }')
KV_BYTES=$((BL_GPU_KV_TOKENS * BL_KV_BYTES_PER_TOKEN))

ARGS=(
  serve "$BL_MODEL" --host 0.0.0.0 --port "$BL_PORT" --max-model-len "${CTX:-262144}"
  --tensor-parallel-size 1 --pipeline-parallel-size 1
  # as the other vLLM baselines: the same GPU KV pool in bytes as SGLang's, and its batching limits
  --kv-cache-memory-bytes "$KV_BYTES" --max-num-batched-tokens "${BATCH_TOKENS:-8192}" --max-num-seqs "${MAX_SEQS:-64}"
  --enable-prompt-tokens-details --generation-config vllm
)
case "$MODE" in
  offload-agent) ARGS+=(--additional-config '{"tokencake":{}}') ;;
  agent)         ARGS+=(--additional-config '{"tokencake":{"offload":{"enabled":false}}}') ;;
  offload)       ARGS+=(--additional-config '{"tokencake":{"scheduling":{"enabled":false}}}') ;;
  native)        ;;
  *) echo "STOP: MODE is offload-agent, agent, offload or native (got $MODE)"; exit 1 ;;
esac
case "$MODE" in offload-agent|offload) ARGS+=(--kv-offloading-size "$HOST_GIB" --kv-offloading-backend native) ;; esac
read -r -a EXTRA <<< "${SERVER_EXTRA:-}"
ARGS+=("${EXTRA[@]}")
# the driver's server environment (campaign.py, server_command)
ENVV=(VLLM_LOGGING_LEVEL=INFO PATH="$VENV/bin:$PATH" VIRTUAL_ENV="$VENV" VLLM_USE_SIMPLE_KV_OFFLOAD=0
      VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false VLLM_SERVER_DEV_MODE=1 VLLM_USE_V1=1)

RUN=$(bl_new_run "tokencake_$MODE") || exit 1
VERSIONS=$("$VENV/bin/python" -c "import importlib.metadata as m; print('vllm=' + m.version('vllm')); print('torch=' + m.version('torch')); print('transformers=' + m.version('transformers'))" 2>/dev/null)
FORK=$(git -C "$HERE/src/TokenCake" rev-parse --short HEAD 2>/dev/null)
echo "system: tokencake (fork $FORK)  mode: $MODE  CPU KV: $HOST_GB GB ($HOST_GIB GiB)  run dir: $RUN"
bl_launch "$RUN" "${ENVV[@]}" -- "$VENV/bin/vllm" "${ARGS[@]}"
printf 'system=tokencake\nmode=%s\nfork_commit=%s\n%s\nmodel=%s\ngpu_kv_tokens=%s\ngpu_kv_bytes=%s\nbatch_tokens=%s\nmax_seqs=%s\nhost_gb=%s (%s GiB)\nssd=none\nmem_cap_gb=%s\nstarted_utc=%s\nserver_extra=%s\n' \
  "$MODE" "$FORK" "$VERSIONS" "$BL_MODEL" "$BL_GPU_KV_TOKENS" "$KV_BYTES" "${BATCH_TOKENS:-8192}" "${MAX_SEQS:-64}" \
  "$HOST_GB" "$HOST_GIB" "${MEM_CAP_GB:-none}" "$(date -u +%FT%TZ)" "${SERVER_EXTRA:-}" > "$RUN/run.txt"
echo "launched: pid $(cat "$RUN/server.pid")   (watch: tail -f $RUN/server.log)"
bl_wait_ready "$RUN" 1800 || { bl_stop; exit 1; }

# startup facts, each labelled
echo "versions:         $(echo "$VERSIONS" | tr '\n' ' ')"
echo "GPU KV cache:     $(grep -aoE 'GPU KV cache size: [0-9,]+ tokens' "$RUN/server.log" | head -1)   (want $BL_GPU_KV_TOKENS tokens)"
echo "TokenCake:        $(grep -aoiE "tokencake[^\n]{0,100}(enabled|config|scheduling|offload)[^\n]{0,80}" "$RUN/server.log" | head -1)"
echo "events route:     $(curl -s -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d '{}' "http://127.0.0.1:$BL_PORT/v1/tokencake/events")   (want 400: the route exists and rejects an empty event)"
