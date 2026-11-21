#!/usr/bin/env bash
# Boot stock vLLM (GPU prefix cache only, no lower tier) on the host and wait until it is ready.
#
#   host: bash /home/wanhr/sglang/agent_cache/baselines/vllm_vanilla/start_server.sh
#
# Knobs (environment): MODEL, PORT (30000), GPU_KV_TOKENS (668096, the pool our SGLang arms profile on this H200),
#   MAX_SEQS (64, SGLang's --max-running-requests), BATCH_TOKENS (8192, SGLang's --chunked-prefill-size),
#   CTX (262144), MEM_CAP_GB (unset: no cap on host RAM), SERVER_EXTRA (flags appended last, split on spaces),
#   VENV (this directory's .venv, built by setup.sh).
# Every boot gets agent_cache/results/<UTC>_vllm_vanilla/ with server.log, server.pid, command.txt and run.txt;
# agent_cache/results/.current_run names it. Stop with: bash agent_cache/baselines/common/stop_server.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../common/lib.sh"
VENV=${VENV:-$HERE/.venv}
# as `source .venv/bin/activate` in the official instructions: FlashInfer's JIT build needs the venv's ninja
[ -x "$VENV/bin/vllm" ] || { echo "STOP: no vLLM in $VENV (bash $HERE/setup.sh builds it)"; exit 1; }

KV_BYTES=$((BL_GPU_KV_TOKENS * BL_KV_BYTES_PER_TOKEN))
ARGS=(
  serve "$BL_MODEL" --host 0.0.0.0 --port "$BL_PORT" --max-model-len "${CTX:-262144}"
  # the same GPU KV pool in bytes as SGLang's, and its batching limits
  --kv-cache-memory-bytes "$KV_BYTES" --max-num-batched-tokens "${BATCH_TOKENS:-8192}" --max-num-seqs "${MAX_SEQS:-64}"
  # the client reads usage.prompt_tokens_details.cached_tokens
  --enable-prompt-tokens-details
  # sampling comes from the request alone (temperature 0, forced length), not from the model's generation_config
  --generation-config vllm
)
read -r -a EXTRA <<< "${SERVER_EXTRA:-}"
ARGS+=("${EXTRA[@]}")

RUN=$(bl_new_run "vllm_vanilla") || exit 1
echo "system: vllm_vanilla  model: $BL_MODEL  GPU KV: $BL_GPU_KV_TOKENS tokens ($KV_BYTES bytes)  run dir: $RUN"
bl_launch "$RUN" VLLM_LOGGING_LEVEL=INFO PATH="$VENV/bin:$PATH" VIRTUAL_ENV="$VENV" -- "$VENV/bin/vllm" "${ARGS[@]}"
VERSION=$("$VENV/bin/python" -c 'import vllm; print(vllm.__version__)' 2>/dev/null)
printf 'system=%s\nvllm=%s\nmodel=%s\ngpu_kv_tokens=%s\ngpu_kv_bytes=%s\nbatch_tokens=%s\nmax_seqs=%s\nhost_gb=0\nssd=none\nmem_cap_gb=%s\nstarted_utc=%s\nserver_extra=%s\n' \
  vllm_vanilla "$VERSION" "$BL_MODEL" "$BL_GPU_KV_TOKENS" "$KV_BYTES" "${BATCH_TOKENS:-8192}" "${MAX_SEQS:-64}" \
  "${MEM_CAP_GB:-none}" "$(date -u +%FT%TZ)" "${SERVER_EXTRA:-}" > "$RUN/run.txt"
echo "launched: pid $(cat "$RUN/server.pid")   (watch: tail -f $RUN/server.log)"
bl_wait_ready "$RUN" 1800 || exit 1

# startup facts, each labelled
echo "vllm version:     $VERSION"
echo "GPU KV cache:     $(grep -aoE 'GPU KV cache size: [0-9,]+ tokens' "$RUN/server.log" | head -1)   (want $BL_GPU_KV_TOKENS tokens)"
echo "max concurrency:  $(grep -aoE 'Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x' "$RUN/server.log" | head -1)"
echo "prefix caching:   $(grep -aoE "enable_prefix_caching[=:] ?(True|False)" "$RUN/server.log" | head -1)   (want True)"
echo "attention:        $(grep -aoE 'Using [A-Za-z_ ]+ backend' "$RUN/server.log" | head -1)"
