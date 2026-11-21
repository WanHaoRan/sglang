#!/usr/bin/env bash
# Boot Continuum (vllm-continuum, the authors' vLLM 0.10.2 fork) on the host and wait until it is ready, with
# LMCache 0.3.6 attached in-process for the CPU and SSD tiers, as its README's "Continuum Scheduling Mode ... With
# CPU offload" (LMCacheConnectorV1 with LMCache's environment configuration).
#
#   host: bash /home/wanhr/sglang/agent_cache/baselines/continuum/start_server.sh
#
# Knobs (environment):
#   POLICY (continuum: --scheduling-policy continuum | fcfs: the fork's "Original vLLM mode", its own stock base)
#   LMCACHE (on | off: GPU only), HOST_GB (160: LMCache's CPU tier), SSD_CAP_GB (1800: its disk tier, LRU-evicted;
#   0 = no disk tier), ODIRECT (true), CHUNK (256, LMCache's default), GPU_KV_TOKENS (668096), MAX_SEQS (64),
#   BATCH_TOKENS (8192), CTX (262144), MEM_CAP_GB, SERVER_EXTRA, VENV (this directory's .venv, built by setup.sh).
# The client sends job_id, is_last_step and this_func_call (baselines/common/backends.py, ContinuumBackend).
# Continuum writes its per-job record (scheduler_timestamps) into the run dir at shutdown (RUN_OUTPUT_DIR).
# The disk tier /mnt/ssd/baselines/lmcache_inproc is not wiped here; run_baselines.sh wipes it before each arm.
# Stop with: bash agent_cache/baselines/common/stop_server.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../common/lib.sh"
VENV=${VENV:-$HERE/.venv}
[ -x "$VENV/bin/vllm" ] || { echo "STOP: no Continuum build in $VENV (bash $HERE/setup.sh)"; exit 1; }
POLICY=${POLICY:-continuum}; LMCACHE=${LMCACHE:-on}
HOST_GB=${HOST_GB:-160}; SSD_CAP_GB=${SSD_CAP_GB:-1800}; ODIRECT=${ODIRECT:-true}; CHUNK=${CHUNK:-256}
# HOST_GB is decimal GB, as SGLang's --hicache-size (160 GB = 1,627,648 tokens); LMCache sizes in GiB (1024**3)
HOST_GIB=$(awk -v g="$HOST_GB" 'BEGIN { printf "%.2f", g * 1e9 / 2^30 }')
L2_DIR=/mnt/ssd/baselines/lmcache_inproc
KV_BYTES=$((BL_GPU_KV_TOKENS * BL_KV_BYTES_PER_TOKEN))
case "$POLICY" in continuum|fcfs) ;; *) echo "STOP: POLICY is continuum or fcfs (got $POLICY)"; exit 1 ;; esac

ARGS=(
  serve "$BL_MODEL" --host 0.0.0.0 --port "$BL_PORT" --max-model-len "${CTX:-262144}"
  # as the other vLLM baselines: the same GPU KV pool in bytes as SGLang's, and its batching limits
  --kv-cache-memory-bytes "$KV_BYTES" --max-num-batched-tokens "${BATCH_TOKENS:-8192}" --max-num-seqs "${MAX_SEQS:-64}"
  --enable-prompt-tokens-details --generation-config vllm
  --scheduling-policy "$POLICY"
)
ENVV=(VLLM_LOGGING_LEVEL=INFO PATH="$VENV/bin:$PATH" VIRTUAL_ENV="$VENV")
if [ "$LMCACHE" = on ]; then
  ARGS+=(--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}')
  # LMCache 0.3.6 reads its configuration from LMCACHE_* variables (lmcache/v1/config.py); sizes in GB
  ENVV+=(LMCACHE_CHUNK_SIZE="$CHUNK" LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE="$HOST_GIB")
  if [ "$SSD_CAP_GB" != 0 ]; then
    mkdir -p "$L2_DIR" || { echo "STOP: cannot create $L2_DIR"; exit 1; }
    ENVV+=(LMCACHE_LOCAL_DISK="file://$L2_DIR/" LMCACHE_MAX_LOCAL_DISK_SIZE="$SSD_CAP_GB"
           LMCACHE_EXTRA_CONFIG="{\"use_odirect\": $ODIRECT}")
  fi
fi
read -r -a EXTRA <<< "${SERVER_EXTRA:-}"
ARGS+=("${EXTRA[@]}")

RUN=$(bl_new_run "continuum_${POLICY}$([ "$LMCACHE" = on ] && echo _lmcache)") || exit 1
ENVV+=(RUN_OUTPUT_DIR="$RUN")
VERSIONS=$("$VENV/bin/python" -c 'import importlib.metadata as m; print(f"vllm={m.version(\"vllm\")}\nlmcache={m.version(\"lmcache\")}\ntransformers={m.version(\"transformers\")}")' 2>/dev/null)
FORK=$(git -C "$HERE/src/vllm-continuum" rev-parse --short HEAD 2>/dev/null)
PATCHED=$(git -C "$HERE/src/vllm-continuum" diff --stat 2>/dev/null | tail -1)
echo "system: continuum (fork $FORK; ${PATCHED:-unpatched})  policy: $POLICY  LMCache: $LMCACHE (CPU $HOST_GB GB = $HOST_GIB GiB, disk ${SSD_CAP_GB} GB)  run dir: $RUN"
bl_launch "$RUN" "${ENVV[@]}" -- "$VENV/bin/vllm" "${ARGS[@]}"
printf 'system=continuum\npolicy=%s\nlmcache=%s\nfork_commit=%s\nfork_diff=%s\n%s\nmodel=%s\ngpu_kv_tokens=%s\ngpu_kv_bytes=%s\nbatch_tokens=%s\nmax_seqs=%s\nhost_gb=%s\nssd_cap_gb=%s\nssd_dir=%s\nodirect=%s\nchunk=%s\nmem_cap_gb=%s\nstarted_utc=%s\nserver_extra=%s\n' \
  "$POLICY" "$LMCACHE" "$FORK" "${PATCHED:-none}" "$VERSIONS" "$BL_MODEL" "$BL_GPU_KV_TOKENS" "$KV_BYTES" \
  "${BATCH_TOKENS:-8192}" "${MAX_SEQS:-64}" "$HOST_GB" "$SSD_CAP_GB" "$L2_DIR" "$ODIRECT" "$CHUNK" "${MEM_CAP_GB:-none}" \
  "$(date -u +%FT%TZ)" "${SERVER_EXTRA:-}" > "$RUN/run.txt"
echo "launched: pid $(cat "$RUN/server.pid")   (watch: tail -f $RUN/server.log)"
bl_wait_ready "$RUN" 1800 || { bl_stop; exit 1; }

# startup facts, each labelled
echo "versions:         $(echo "$VERSIONS" | tr '\n' ' ')"
echo "GPU KV cache:     $(grep -aoE 'GPU KV cache size: [0-9,]+ tokens' "$RUN/server.log" | head -1)   (want $BL_GPU_KV_TOKENS tokens)"
echo "policy:           $(grep -aoE "scheduler_config=[^)]*policy='[a-z]+'|'policy': '[a-z]+'|policy=[a-z]+" "$RUN/server.log" | head -1)   (want $POLICY)"
echo "LMCache:          $(grep -aoE "LMCache[^\n]{0,40}(initialized|Initializing|engine)[^\n]{0,60}" "$RUN/server.log" | head -1)"
