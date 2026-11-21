# SGLang, radix cache only

Our engine as campaigns 28 and 29 ran it (`frozen/engine_ec_20261111d`), arm `hbm_lru` of `scripts/start_server.sh`:
no HiCache, so an evicted prefix is recomputed. Model settings as those campaigns: bf16 KV (`KV_DTYPE=auto`), the
stock chat template, context 262,144, Triton attention, 64-token pages, chunked prefill 8,192, 64 running requests.

| step | host command (from `agent_cache/baselines/`) | takes |
|---|---|---|
| boot | `bash sglang_vanilla/start_server.sh` (in the `sglang_hicache` container) | 3 min |
| check | `bash common/run_check.sh sglang_vanilla gpu GPU_KV_TOKENS=100000 -- check_server.py --backend openai --backend-opt details=on --prompt-tokens 20000 --block-tokens 64 --stage gpu_evicted:150000:recomputed` | 6 min |
| stop | `docker exec sglang_hicache bash /sgl-workspace/sglang/agent_cache/scripts/stop_server.sh` | 10 s |

The trace replay drives it with our own client (`scripts/start_programs.sh`, native `/generate`), as campaigns 28 and 29.

## Checks (2026-11-21, GPU)

- `check_server.py` (`results/20261121_183648_hbm_lru_custom_L1100000_H160/`): cold 0 cached, TTFT 1.454 s; warm
  19,968 cached (all full 64-token pages), 0.092 s, logprobs within 0.087 of cold; extended 19,968; after 150,000
  filler tokens recomputed (0 cached) with logprobs equal to the cold run (0.0). 4 of 4 passed.
- 20-minute replay at campaign 28's setting: 907 requests, 0 errors, mean TTFT 24.9 s, 50 % of prompt tokens
  recomputed (`results/baseline_smoke_compare/`).
- The same cold 20,000-token prompt takes 1.454 s here and 0.557 s on vLLM 0.33.0: this engine's prefill is 2.6x
  slower on this box, which outweighs every cache effect in the early minutes of a replay.
