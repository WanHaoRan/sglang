# TokenCake (agent-aware scheduling + tool-window preservation in CPU memory)

The authors' code (github.com/pku-lemonade/TokenCake, commit `1901ebf`, 2026-09-28; EuroSys '27), a fork of vLLM
0.22.0, run with the modes and server environment of its own experiment driver
(`tools/tokencake_experiments/campaign.py`). TokenCake has two tiers, GPU and CPU; it has no disk tier.

| step | host command (from `agent_cache/baselines/`) | takes |
|---|---|---|
| build | `bash tokencake/setup.sh` | 2 min (clone, 249 MB wheel, uv) |
| boot | `bash tokencake/start_server.sh` (`MODE=offload-agent` by default) | 1 min (35 min the first time: kernel build) |
| checks | `common/run_check.sh` lines below | 5 min each |
| official evaluation | `bash tokencake/official/dag.sh` | 30-60 min |
| stop | `bash common/stop_server.sh` | 20 s |

## What the release does (read from `vllm/tokencake/`, 30 Python files over vLLM v0.22.0; no compiled part changed)

- **Spatial scheduling** (`scheduling.py`): admission reserves for accepted requests' inputs and declared budgets,
  with a hybrid priority from agent importance, DAG structure and runtime state.
- **Temporal scheduling** (`offload_policy.py`, `offloading.py`, `lifecycle.py`): when a client reports a tool stall
  for a finished request, a benefit test (stall estimate against transfer time, GPU pressure, CPU room, importance)
  decides whether its prefix blocks are preserved in CPU memory through vLLM's native CPU offloading; a later
  request that hits them recovers them on demand. The paper's predictive upload (CPU to GPU before the tool returns)
  is not in the release; its README says so.
- **Signals**: metadata in `vllm_xargs.tokencake` (`lifecycle_id` = the request's `request_id`, a `tc-` UUID4;
  `offload_eligible` and `reusable_prefix` gate preservation) and `stall_started` / `stall_finished` events posted
  to `/v1/tokencake/events`, with an optional `estimated_duration_s`.

## What our replay sends

As TokenCake's own client does (`tools/tokencake_experiments/dataset_client.py`): every request's id is a `tc-` UUID4
derived from our request id; its metadata marks every turn the session extends as `offload_eligible` with a
`reusable_prefix` (its datasets mark tool nodes so), `agent_type` the stream's role; after each reply that is not the
stream's last, `stall_started` with `kind` the recorded tool (or `user`) and `estimated_duration_s` the running median
wait after that tool (our dist hints' table; `--backend-opt est=exact` sends the recorded wait, `est=none` lets
TokenCake learn per kind); `stall_finished` when the wait is over, just before the next turn.

## Configuration

| setting | value | why |
|---|---|---|
| vLLM flags | as `vllm_vanilla` (GPU pool in bytes, 8,192, 64, 262,144) | the same pools and batching |
| `--additional-config` | `{"tokencake":{}}` (scheduling and offload on) | its full system |
| `--kv-offloading-size`, `--kv-offloading-backend` | 120 (GiB, 129 GB), `native` | the largest CPU tier this host can give it (below); the backend its driver uses |
| environment | `VLLM_USE_SIMPLE_KV_OFFLOAD=0`, `VLLM_WORKER_MULTIPROC_METHOD=spawn`, `TOKENIZERS_PARALLELISM=false`, `VLLM_SERVER_DEV_MODE=1`, `VLLM_USE_V1=1` | its driver's server environment |
| dependencies | resolved as PyPI stood on 2026-09-29 (torch 2.11.0, transformers 4.57.6) | its commit day |

## Checks (2026-11-21, GPU)

- First boot: vLLM 0.22.0 picks FlashInfer's CUTLASS MoE backend for this model and JIT-builds its kernels (182
  objects, about 35 minutes on this box's 16 cores); they are cached in `~/.cache/flashinfer/0.6.11.post2/` and later
  boots take about a minute. pip's FlashInfer ships no prebuilt kernels; the vLLM Docker image would.
- `check_server.py` (GPU 100K tokens, CPU 30 GB; `results/20261121_192707_tokencake_offload-agent/`): cold 0 cached
  (1.290 s); warm 19,984 (0.035 s); extended 20,000; after 150,000 filler tokens recomputed (0 cached: TokenCake
  keeps nothing in CPU memory without a stall event). 4 of 4 passed with `--recompute-tol 0.25`: the recomputed
  prompt's logprobs differ from the cold run's by 0.06, since TokenCake's scheduler sets each step's prefill budget
  and so chunks the same prompt differently (same tokens, different bf16 arithmetic).
- `check_session.py`, its preservation (GPU 100K tokens; a 20,000-token turn, 8 x 16,000 filler tokens arriving as
  it completes, the next turn after a 10 s wait; `results/20261121_184959_tokencake_offload-agent/`): with stall
  events the benefit test selected the session once the fill raised pressure (`tokencake_decision_total`: 8
  low_pressure, 1 no_waiting_demand, then 1 selected), saved its 1,250 blocks (`tokencake_saved_blocks_total`), and
  the next turn found 20,000 of 20,528 tokens cached, all from CPU memory (`cpu_hit_tokens` and
  `external_prefix_cache_hits_total` +20,000; TTFT 0.123 s). Without events: 0 cached, 0.767 s.

## Deviation: a smaller CPU tier

vLLM 0.22.0's native CPU offloading allocates the whole CPU tier as one pinned tensor
(`vllm/v1/kv_offload/cpu/gpu_worker.py`, `torch.zeros(..., pin_memory=True)`). On this host one pinned allocation
succeeds at 122 GiB and fails at 130 and 149 GiB with "CUDA error: out of memory" (tested 2026-11-21 in TokenCake's
own environment; the full-size boot failed the same way). The other systems pin in pieces (LMCache, 10 GiB at a
time) or as SGLang does, and get 160 GB. TokenCake gets 120 GiB (129 GB), `CPU_GIB_MAX` in `start_server.sh`;
changing vLLM's allocator would no longer be the released code.

## TokenCake's own evaluation, scaled down (2026-11-21, GPU)

`official/dag.sh`: its experiment driver (`tools/tokencake_experiments`) on its fixed multi-agent DAG workload
(`conversation-tools`), launching the server itself with its own sizing (GPU memory utilization 0.5, which leaves a
106,736-token KV pool here; context 32,768; CPU KV 100 GiB), native mode then the full system, at QPS 1.0.

| workload | mode | application end to end | mean application latency | finished preemptions |
|---|---|---|---|---|
| first 8 of its 24 applications (`results/20261121_204138_tokencake_official/`) | native | 68.7 s | 61.5 s | 0 |
| the same | offload-agent | 72.0 s | 65.5 s | 0 |
| all 24 applications, its README's scale (`results/20261121_222828_tokencake_official/`) | native | 189.3 s | 154.6 s | 0 |
| the same | offload-agent | 154.0 s | 115.2 s | 0 |

- The driver's report: E2E reduction -4.9 % (TokenCake slower) at this scale. Its README's gain (-34 % at QPS 1.0 on
  24 applications, Qwen2.5-14B on an A800) comes with memory pressure (41 preemptions natively); 8 applications on
  this H200 create none, so there is nothing for its preservation to save.
- At its full 24 applications TokenCake cuts application end to end by 18.7 % and mean application latency by
  25.5 % against native on this box: the direction of its README's -34 %, at a smaller size, with no preemption in
  either run.
- Two fixes to run the driver here, neither in TokenCake's code: the driver sets `CUDA_VISIBLE_DEVICES` to the GPU's
  UUID when `nvidia-smi` answers, which vLLM 0.22 then fails to parse as an integer, so `official/nosmi/nvidia-smi` (a
  stub that fails) makes it pass the index; and nothing else may hold the GPU, since vLLM's memory profiling aborts
  when free memory changes under it (the first native case failed so).

## The trace replay (2026-11-21, GPU)

20 minutes at campaign 28's setting, full sizes (668,096-token GPU pool, 120 GiB CPU; `results/baseline_smoke2/`,
run `20261121_204835_tokencake_offload-agent`): 2,292 requests, 0 errors, mean TTFT 1.34 s (p50 0.24 s, p90 3.98 s),
10 % of prompt tokens recomputed. Its counters at the end: 1,379 stalled sessions selected for preservation (the
benefit test's other outcomes mostly `no_waiting_demand` and `backoff`), 228,385 blocks saved to CPU memory, 18.0 M
prompt tokens served from CPU (`external_prefix_cache_hits_total`) beside 62.7 M from the GPU cache.
- Against its own base on the same users (`tokencake_native`: the same build with TokenCake and CPU offloading off,
  no metadata or events; run `20261121_220435_tokencake_native`): 1,537 requests, mean TTFT 9.11 s (p90 34.1 s), 42 %
  of prompt tokens recomputed. Paired on the 1,537 requests both sent, its base minus TokenCake: +8.63 s
  [+6.81, +10.97] (`results/baseline_smoke_compare/compare_arms.out`): on this trace TokenCake's preservation and
  scheduling carry it from slower than stock vLLM 0.33.0 (its vLLM 0.22.0 base: +6.18 s [+4.90, +7.84]) to faster.
