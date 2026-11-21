#!/usr/bin/env bash
# Boot SGLang + HiCache (GPU, host and SSD tiers, stock policy) in the sglang_hicache container and wait until ready.
# It is arm three_tier_wc of agent_cache/scripts/start_server.sh, the `hicache` arm of campaigns 28 and 29, on the
# engine copy they ran; nothing is new here but the name.
#
#   host: bash /home/wanhr/sglang/agent_cache/baselines/sglang_hicache/start_server.sh
#
# Knobs (environment): GPU_KV_TOKENS (unset: the pool the server profiles, 668,096 tokens; set it small for
#   check_server.py), HOST_GB (160: the host KV pool), ENGINE (engine_ec_20261111d: a folder of agent_cache/frozen/),
#   L3_CLEANER_PCT (90,80: the SSD cleaner's marks, as campaigns 28 and 29), MODEL, SERVER_EXTRA.
# The SSD store /mnt/ssd/hicache_l3 is not wiped here; the launcher wipes it before each arm.
# Stop with: docker exec sglang_hicache bash /sgl-workspace/sglang/agent_cache/scripts/stop_server.sh
set -uo pipefail
CAC=/sgl-workspace/sglang/agent_cache
ENGINE=${ENGINE:-engine_ec_20261111d}
exec docker exec \
  -e MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}" -e CTX=262144 -e KV_DTYPE=auto -e TEMPLATE= \
  -e HOST_GB="${HOST_GB:-160}" -e L3_CLEANER_PCT="${L3_CLEANER_PCT:-90,80}" \
  -e L1_TOKENS="${GPU_KV_TOKENS:-}" -e SGLANG_TIMEOUT_KEEP_ALIVE=3600 -e ENGINE_PYTHONPATH="$CAC/frozen/$ENGINE/python" -e SERVER_EXTRA="${SERVER_EXTRA:-}" \
  sglang_hicache bash "$CAC/scripts/start_server.sh" three_tier_wc NAT160
