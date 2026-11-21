# TokenCake (agent-aware scheduling + tool-window preservation in CPU memory)

The authors' code (github.com/pku-lemonade/TokenCake, commit `1901ebf`, 2026-09-28; EuroSys '27), a fork of vLLM
0.22.0, run with the modes and server environment of its own experiment driver
(`tools/tokencake_experiments/campaign.py`). TokenCake has two tiers, GPU and CPU; it has no disk tier.

| step | host command (from `agent_cache/baselines/`) | takes |
|---|---|---|
| build | `bash tokencake/setup.sh` | 2 min (clone, 249 MB wheel, uv) |
| boot | `bash tokencake/start_server.sh` (`MODE=offload-agent` by default) | 2-3 min |
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
| `--kv-offloading-size`, `--kv-offloading-backend` | 149.01 (GiB), `native` | our host pool in bytes; the backend its driver uses |
| environment | `VLLM_USE_SIMPLE_KV_OFFLOAD=0`, `VLLM_WORKER_MULTIPROC_METHOD=spawn`, `TOKENIZERS_PARALLELISM=false`, `VLLM_SERVER_DEV_MODE=1`, `VLLM_USE_V1=1` | its driver's server environment |
| dependencies | resolved as PyPI stood on 2026-09-29 (torch 2.11.0, transformers 4.57.6) | its commit day |
