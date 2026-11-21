# vLLM + LMCache (multiprocess mode, CPU and SSD tiers)

vLLM 0.33.0 and LMCache 0.5.7.post1 from PyPI, attached the way LMCache's quickstart recommends ("MP mode"): an
`lmcache server` process holds the CPU tier (L1) and the SSD tier (L2), and vLLM reaches it through the
`LMCacheMPConnector` LMCache ships (`kv_connector_module_path`, the documented choice for vLLM >= 0.20). The vLLM side
is exactly `vllm_vanilla`, so the pair differs by LMCache alone.

| step | host command (from `agent_cache/baselines/`) | takes |
|---|---|---|
| build | `bash vllm_lmcache/setup.sh` | 1 min (uv cache warm) |
| boot | `bash vllm_lmcache/start_server.sh` | 2-3 min (LMCache pins its L1 first) |
| check | `GPU_KV_TOKENS=100000 CTX=65536 HOST_GB=30 SSD_CAP_GB=200 bash vllm_lmcache/start_server.sh`, then below | 5 min |
| official benchmark | `bash vllm_lmcache/official/multi_round_qa.sh <label>` against each of the two servers | 10 min each |
| stop | `bash common/stop_server.sh` (vLLM, then the LMCache server) | 20 s |

```bash
# container: sglang_hicache; LMCache's metrics (port 6580) name the tier that served the checked request
docker exec sglang_hicache bash -c 'cd /sgl-workspace/sglang/agent_cache/baselines/common && python3 check_server.py \
  --backend vllm --prompt-tokens 20000 --block-tokens 256 --stage l1_cpu:150000:cached --stage l2_ssd:400000:cached \
  --metrics http://127.0.0.1:6580/metrics --metric-match "lookup_hit_l[12]_tokens|l2_load_completed_requests"'
```

## Configuration

| setting | value | why |
|---|---|---|
| vLLM flags | as `vllm_vanilla` | the same engine and pools |
| `--kv-transfer-config` | `LMCacheMPConnector`, module `lmcache.integration.vllm.lmcache_mp_connector`, server `tcp://127.0.0.1:6555` | LMCache's recommended attachment |
| `lmcache server --l1-size-gb` | 149.01 (GiB) | our 160 GB host pool in bytes (LMCache sizes in GiB) |
| `--eviction-policy` | LRU | LMCache's quickstart |
| `--chunk-size` | 256 | LMCache's default |
| `--l2-adapter` | `fs_native` on `/mnt/ssd/baselines/lmcache`, `use_odirect: true`, `max_capacity_gb: 1800` with LRU eviction | an SSD tier read from the disk, not the page cache, and bounded below the disk's size |
| L2 store policy | `default` (every chunk to L2, kept in L1) | LMCache's default: write-through, as our HiCache arms |

## Version choice

vLLM 0.34.0 (the newest) does not run with LMCache 0.5.7.post1 (the newest LMCache): LMCache's connector reads
`CachedRequestData.resumed_req_ids`, which 0.34.0 removed (AttributeError on the first cached request), and vLLM
0.34.0's builtin connector (`LMCACHE_USE_UPSTREAM_MP=1`) passes `server_url`, which LMCache 0.5.7's scheduler adapter
does not take. LMCache 0.5.7.post1 shipped (2026-11-12) when vLLM 0.33.0 was the newest release; the two resolve
together on torch 2.14.0, the torch LMCache's wheel is built for, with no dependency conflict (`uv pip check`).

## Checks (2026-11-21, GPU)

- `check_server.py` with GPU 100K tokens, L1 30 GB, L2 200 GB: cold 0 cached; warm 19,984 cached (TTFT 0.041 s);
  extended prompt 20,000 cached; after 150,000 filler tokens the prompt came back from L1 (19,968 tokens, LMCache's
  `lookup_hit_l1_tokens_total` +19,968, TTFT 0.083 s); after 400,000 more it was loaded from the SSD
  (`l2_load_completed_requests_total` +1, `lookup_hit_l2_tokens_total` +19,968, TTFT 1.217 s), with first-token
  logprobs identical (difference 0.0) to the L1-restored request: the SSD copy is bit-exact.
- On this box an SSD load of 20,000 tokens (1.96 GB at about 1.8 GB/s) takes longer than recomputing them on vLLM
  (0.56 s cold): the SSD tier pays only where the GPU is busy.
