#!/usr/bin/env bash
# Boot vanilla SGLang (GPU radix cache only, no HiCache) in the sglang_hicache container and wait until it is ready.
# It is arm hbm_lru of agent_cache/scripts/start_server.sh on the engine copy campaigns 28 and 29 ran, with their
# model settings; nothing is new here but the name.
#
#   host: bash /home/wanhr/sglang/agent_cache/baselines/sglang_vanilla/start_server.sh
#
# Knobs (environment): GPU_KV_TOKENS (unset: the pool the server profiles, 668,096 tokens; set it small for
#   check_server.py), ENGINE (engine_ec_20261111d: a folder of agent_cache/frozen/), MODEL, SERVER_EXTRA.
# Stop with: docker exec sglang_hicache bash /sgl-workspace/sglang/agent_cache/scripts/stop_server.sh
set -uo pipefail
CAC=/sgl-workspace/sglang/agent_cache
ENGINE=${ENGINE:-engine_ec_20261111d}
exec docker exec \
  -e MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}" -e CTX=262144 -e KV_DTYPE=auto -e TEMPLATE= \
  -e L1_TOKENS="${GPU_KV_TOKENS:-}" -e SGLANG_TIMEOUT_KEEP_ALIVE=3600 -e ENGINE_PYTHONPATH="$CAC/frozen/$ENGINE/python" -e SERVER_EXTRA="${SERVER_EXTRA:-}" \
  sglang_hicache bash "$CAC/scripts/start_server.sh" hbm_lru NAT160
