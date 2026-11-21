#!/usr/bin/env bash
# Boot vLLM + LMCache on the host in LMCache's recommended multiprocess mode, and wait until both are ready:
# an `lmcache server` holds the CPU tier (L1, host RAM) and a disk tier (L2, the fs_native adapter on /mnt/ssd),
# and vLLM attaches to it through LMCacheMPConnector, the connector LMCache ships
# (docs/source/getting_started/quickstart.rst, "MP mode (recommended)"; compatibility.rst for vLLM >= 0.20).
#
#   host: bash /home/wanhr/sglang/agent_cache/baselines/vllm_lmcache/start_server.sh
#
# Knobs (environment): HOST_GB (160: the L1 size, our host pool), SSD_CAP_GB (1800: L2 size, LRU-evicted;
#   0 = no L2), ODIRECT (true: L2 reads bypass the page cache), CHUNK (256: LMCache's chunk, its default),
#   GPU_KV_TOKENS (668096), MAX_SEQS (64), BATCH_TOKENS (8192), CTX (262144), LMCACHE_PORT (6555),
#   LMCACHE_HTTP_PORT (6580), CONNECTOR (lmcache | bundled, below), MEM_CAP_GB, SERVER_EXTRA, VENV (this directory's .venv, built by setup.sh).
# The L2 directory /mnt/ssd/baselines/lmcache is not wiped here; run_baselines.sh wipes it before each arm.
# Stop with: bash agent_cache/baselines/common/stop_server.sh (vLLM first, then the LMCache server)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../common/lib.sh"
VENV=${VENV:-$HERE/.venv}
# as `source .venv/bin/activate` in the official instructions: FlashInfer's JIT build needs the venv's ninja
[ -x "$VENV/bin/vllm" ] && [ -x "$VENV/bin/lmcache" ] || { echo "STOP: no vLLM + LMCache in $VENV (bash $HERE/setup.sh)"; exit 1; }
HOST_GB=${HOST_GB:-160}; SSD_CAP_GB=${SSD_CAP_GB:-1800}; ODIRECT=${ODIRECT:-true}; CHUNK=${CHUNK:-256}
# HOST_GB is decimal GB, as SGLang's --hicache-size (160 GB = 1,627,648 tokens); LMCache sizes in GiB (1 << 30)
HOST_GIB=$(awk -v g="$HOST_GB" 'BEGIN { printf "%.2f", g * 1e9 / 2^30 }')
LMC_PORT=${LMCACHE_PORT:-6555}; LMC_HTTP=${LMCACHE_HTTP_PORT:-6580}
L2_DIR=/mnt/ssd/baselines/lmcache
KV_BYTES=$((BL_GPU_KV_TOKENS * BL_KV_BYTES_PER_TOKEN))

LMC=(server --host 127.0.0.1 --port "$LMC_PORT" --http-port "$LMC_HTTP" --chunk-size "$CHUNK"
     --l1-size-gb "$HOST_GIB" --eviction-policy LRU)
if [ "$SSD_CAP_GB" != 0 ]; then
  mkdir -p "$L2_DIR" || { echo "STOP: cannot create $L2_DIR"; exit 1; }
  # one L2 adapter on the SSD: a size cap enforced by LRU eviction, O_DIRECT so reads come from the disk
  LMC+=(--l2-adapter "{\"type\":\"fs_native\",\"base_path\":\"$L2_DIR\",\"use_odirect\":$ODIRECT,\"max_capacity_gb\":$SSD_CAP_GB,\"eviction\":{\"eviction_policy\":\"LRU\"}}")
fi
# CONNECTOR=lmcache (default): the LMCacheMPConnector LMCache ships, selected by kv_connector_module_path, as its
# docs recommend for vLLM >= 0.20 (compatibility.rst). CONNECTOR=bundled forces vLLM's builtin copy
# (LMCACHE_USE_UPSTREAM_MP, vllm/.../lmcache_mp_connector.py, _resolve_lmcache_mp_connector).
MODULE="\"kv_connector_module_path\":\"lmcache.integration.vllm.lmcache_mp_connector\","; UPSTREAM=()
if [ "${CONNECTOR:-lmcache}" = bundled ]; then MODULE=""; UPSTREAM=(LMCACHE_USE_UPSTREAM_MP=1); fi
KVT="{\"kv_connector\":\"LMCacheMPConnector\",$MODULE\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://127.0.0.1\",\"lmcache.mp.port\":$LMC_PORT}}"
ARGS=(
  serve "$BL_MODEL" --host 0.0.0.0 --port "$BL_PORT" --max-model-len "${CTX:-262144}"
  # as vllm_vanilla: the same GPU KV pool in bytes as SGLang's, and its batching limits
  --kv-cache-memory-bytes "$KV_BYTES" --max-num-batched-tokens "${BATCH_TOKENS:-8192}" --max-num-seqs "${MAX_SEQS:-64}"
  --enable-prompt-tokens-details --generation-config vllm
  # the lower tiers
  --kv-transfer-config "$KVT"
)
read -r -a EXTRA <<< "${SERVER_EXTRA:-}"
ARGS+=("${EXTRA[@]}")

for p in "$LMC_PORT" "$LMC_HTTP"; do
  (exec 3<>"/dev/tcp/127.0.0.1/$p") 2>/dev/null && { echo "STOP: port $p is already in use; the LMCache server needs it (LMCACHE_PORT, LMCACHE_HTTP_PORT)"; exit 1; }
done
RUN=$(bl_new_run "vllm_lmcache") || exit 1
VERSIONS=$("$VENV/bin/python" -c 'import vllm, lmcache; print(f"vllm={vllm.__version__}\nlmcache={lmcache.__version__}")' 2>/dev/null)
echo "system: vllm_lmcache  model: $BL_MODEL  GPU KV: $BL_GPU_KV_TOKENS tokens  L1: $HOST_GB GB ($HOST_GIB GiB)  L2: ${SSD_CAP_GB} GB on $L2_DIR (O_DIRECT $ODIRECT)  run dir: $RUN"
bl_launch_aux "$RUN" lmcache_server LMCACHE_LOG_LEVEL=INFO PATH="$VENV/bin:$PATH" VIRTUAL_ENV="$VENV" -- "$VENV/bin/lmcache" "${LMC[@]}"
bl_wait_port "$LMC_PORT" 600 "$(awk '$2 == "lmcache_server" {print $1}' "$RUN/aux.pids")" || { bl_stop; exit 1; }
bl_launch "$RUN" VLLM_LOGGING_LEVEL=INFO "${UPSTREAM[@]}" PATH="$VENV/bin:$PATH" VIRTUAL_ENV="$VENV" -- "$VENV/bin/vllm" "${ARGS[@]}"
printf 'system=%s\nconnector=%s\n%s\nmodel=%s\ngpu_kv_tokens=%s\ngpu_kv_bytes=%s\nbatch_tokens=%s\nmax_seqs=%s\nhost_gb=%s (%s GiB)\nssd_cap_gb=%s\nssd_dir=%s\nodirect=%s\nchunk=%s\nmem_cap_gb=%s\nstarted_utc=%s\nserver_extra=%s\n' \
  vllm_lmcache "${CONNECTOR:-lmcache}" "$VERSIONS" "$BL_MODEL" "$BL_GPU_KV_TOKENS" "$KV_BYTES" "${BATCH_TOKENS:-8192}" "${MAX_SEQS:-64}" \
  "$HOST_GB" "$HOST_GIB" "$SSD_CAP_GB" "$L2_DIR" "$ODIRECT" "$CHUNK" "${MEM_CAP_GB:-none}" "$(date -u +%FT%TZ)" "${SERVER_EXTRA:-}" > "$RUN/run.txt"
echo "launched: vLLM pid $(cat "$RUN/server.pid"), LMCache server $(cat "$RUN/aux.pids")   (watch: tail -f $RUN/server.log $RUN/lmcache_server.log)"
bl_wait_ready "$RUN" 1800 || { bl_stop; exit 1; }

# startup facts, each labelled
echo "versions:         $(echo "$VERSIONS" | tr '\n' ' ')"
echo "GPU KV cache:     $(grep -aoE 'GPU KV cache size: [0-9,]+ tokens' "$RUN/server.log" | head -1)   (want $BL_GPU_KV_TOKENS tokens)"
echo "connector:        $(grep -aoE "kv_connector_module_path='[a-z_.]+'" "$RUN/server.log" | head -1)   (want lmcache.integration.vllm.lmcache_mp_connector unless CONNECTOR=bundled)"
echo "LMCache L1:       $(grep -aoE "now total is [0-9]+ MB / [0-9]+ MB" "$RUN/lmcache_server.log" | tail -1)   (want / $(awk -v g="$HOST_GIB" 'BEGIN { printf "%d", g * 1024 }') MB: L1 pins memory as it grows)"
echo "LMCache L2:       $(grep -aoE "Created FS native L2 adapter: [^ ]+ \([^)]*odirect=[A-Za-z]+" "$RUN/lmcache_server.log" | head -1)"
