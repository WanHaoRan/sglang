# SGLang + HiCache (GPU, host and SSD tiers, stock policy)

Our engine as campaigns 28 and 29 ran it (`frozen/engine_ec_20261111d`), arm `three_tier_wc` of
`scripts/start_server.sh` (the `hicache` arm of those campaigns): a 160 GB write-through host pool, the nixl SSD
store on `/mnt/ssd/hicache_l3` with prefetch policy `wait_complete`, its cleaner at 90 and 80 % of the volume.

| step | host command (from `agent_cache/baselines/`) | takes |
|---|---|---|
| boot | `bash sglang_hicache/start_server.sh` (in the `sglang_hicache` container) | 3-5 min |
| check | `bash common/run_check.sh sglang_hicache tiers GPU_KV_TOKENS=100000 HOST_GB=30 -- check_server.py --backend openai --backend-opt details=on --prompt-tokens 20000 --block-tokens 64 --stage host:150000:cached --stage ssd:400000:cached` | 8 min |
| stop | `docker exec sglang_hicache bash /sgl-workspace/sglang/agent_cache/scripts/stop_server.sh` | 10 s |

## Checks (2026-11-21, GPU)

- `check_server.py` with GPU 100K tokens and a 30 GB host pool (`results/20261121_183956_three_tier_wc_custom_L1100000_H30/`):
  cold 0 cached (1.371 s); warm 19,968 (0.092 s); extended 19,968; after 150,000 filler tokens 19,968 cached, by
  SGLang's own tier counts all from host memory (0.091 s); after 400,000 more all from the SSD (1.954 s); first-token
  logprobs equal to the warm run's (0.0) at both. 5 of 5 passed. The SSD load takes longer than recomputing (1.371 s).
