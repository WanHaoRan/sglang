# Stock vLLM (prefix cache only)

vLLM 0.33.0 from PyPI, serving our model with the GPU KV pool, batching limits and context of our SGLang arms. It has
no lower tier: an evicted prefix is recomputed. It is the stock engine the vLLM-based baselines are measured against
(`vllm_lmcache` runs the same vLLM 0.33.0 with LMCache attached).

| step | host command (from `agent_cache/baselines/`) | takes |
|---|---|---|
| build | `bash vllm_vanilla/setup.sh` | 1 min (uv cache warm) |
| boot | `bash vllm_vanilla/start_server.sh` | 1-2 min |
| check | `GPU_KV_TOKENS=100000 CTX=65536 bash vllm_vanilla/start_server.sh`, then the check below | 3 min |
| stop | `bash common/stop_server.sh` | 10 s |

```bash
# container: sglang_hicache (the check runs where the replay client runs; host networking reaches port 30000)
docker exec sglang_hicache bash -c 'cd /sgl-workspace/sglang/agent_cache/baselines/common && python3 check_server.py \
  --backend vllm --prompt-tokens 20000 --block-tokens 64 --stage gpu_evicted:150000:recomputed'
```

## Server flags and why

| flag | value | why |
|---|---|---|
| `--kv-cache-memory-bytes` | 65,676,066,816 (668,096 tokens x 98,304 B) | the GPU KV pool our SGLang arms profile on this H200 |
| `--max-num-batched-tokens` | 8,192 | SGLang's `--chunked-prefill-size` |
| `--max-num-seqs` | 64 | SGLang's `--max-running-requests` |
| `--max-model-len` | 262,144 | the model's window, as SGLang's `--context-length` |
| `--enable-prompt-tokens-details` | on | the client reads `usage.prompt_tokens_details.cached_tokens` |
| `--generation-config` | `vllm` | sampling from the request alone (temperature 0, forced length) |
| everything else | vLLM 0.33.0's defaults | block size 16, FlashAttention, prefix caching on, FCFS |

## Checks (2026-11-21, GPU)

- `check_server.py`, 100K-token pool: a cold 20,000-token prompt has 0 cached; the same prompt again has 19,984
  cached (all but one 16-token block; TTFT 0.041 s against 0.557 s cold) with first-token logprobs within 0.074 of
  the cold run; after 150,000 filler tokens it is recomputed (0 cached) with logprobs equal to the cold run's (0.0).
  On vLLM 0.34.0 the same check passed with 64-token blocks.
- Replay plumbing: 2 minutes of FreeInference and of TraceLab through `common/replay_baseline.py`, 0 errors; on
  TraceLab 83 % of prompt tokens came from the cache against 86 % an infinite cache would hold, so the server's own
  reply ids round-trip; all 132 requests matched campaigns 28 and 29 request by request.
- 20-minute replay at campaign 28's setting (`results/baseline_smoke/`): 1,855 requests, 0 errors, mean TTFT 5.06 s
  (p50 0.13 s, p90 22.3 s), 34 % of prompt tokens recomputed. On the same users SGLang + HiCache sent 1,676 at
  9.19 s with 10 % recomputed, and SGLang without HiCache 907 at 24.9 s: vLLM computes much faster on this box.

## Not the same as our SGLang arms

- Attention runs on FlashAttention; every SGLang arm so far pins Triton (`scripts/start_server.sh`). This, not the
  cache, is the likeliest reason stock vLLM out-serves SGLang + HiCache above.
- Block size 16 (cache granule) against SGLang's 64-token pages.
