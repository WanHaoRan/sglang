# Continuum (released rule: a 2 s pin for short tools), with LMCache's CPU and SSD tiers

The authors' code (github.com/Hanchenli/vllm-continuum, commit `316a587`, 2026-03-19), a fork of vLLM 0.10.2, run as
its README's "Continuum Scheduling Mode ... With CPU offload": `--scheduling-policy continuum` and LMCache 0.3.6
in-process (`LMCacheConnectorV1`). `POLICY=fcfs` runs the same build as its README's "Original vLLM mode", the
fork's own stock base.

| step | host command (from `agent_cache/baselines/`) | takes |
|---|---|---|
| build | `bash continuum/setup.sh` | 2 min (clone, patch, 417 MB wheel, uv) |
| boot | `bash continuum/start_server.sh` (`POLICY=continuum LMCACHE=on` by default) | 2-3 min |
| checks | `common/run_check.sh` lines below | 5 min each |
| official evaluation | `bash continuum/official/swebench.sh` (needs `.venv_agent` and the task images) | 15-30 min |
| stop | `bash common/stop_server.sh` | 20 s |

## What the release does (read from the diff against vLLM v0.10.2: 20 files, `vllm/v1/core/estimate_with_func.py`,
`vllm/v1/core/sched/{scheduler,request_queue}.py`, `vllm/v1/request.py`, `vllm/entrypoints/openai/protocol.py`)

- **Pin.** When a request finishes and is not its job's last step, its KV blocks are kept (not freed) for 2 s if
  the average time its tool has taken so far is 2 s or less (`FIXED_THRESHOLD_CONTINUUM`, `set_up_pin`); a tool not
  seen before counts as 0 s, so it is pinned. A pin is released after 2 s unless a request of the job is waiting, or
  when the job's next request finishes. The paper's TTL estimator is not in the release.
- **Tool name.** Parsed from the reply text: the first word of its single ```` ```bash ```` block (mini-swe-agent's
  format). Requests may also name it in a `this_func_call` field, which the release then overwrites with the parse.
- **Queue.** Pinned jobs first, then job-level FCFS (a job's first arrival); preemption takes the newest job that
  is not on its last step, and frees the newest pin when nothing else is running.
- **Identity.** `job_id` and `is_last_step` request fields (top level), which its own agent sends via `extra_body`.

## What our replay sends, and the one patch

`job_id` = the stream (`<tag>-c<instance>`), so a job never has two requests in flight; `is_last_step` on the stream's
last request; `this_func_call` = the recorded reply's tool as Continuum's parser would name it (`Bash/git` -> `git`,
other tools by name; none after a reply without a tool call). Our replies are synthetic, so the parser finds no bash
block and the release would pin nothing; `patches/0001-keep-the-requests-tool-when-the-reply-names-none.patch`
(one line in `request_finished`) keeps the request's own field when the parse finds none. With `PATCH=0` in setup.sh
the release runs unpatched, and Continuum is then its queue order alone.

## Configuration

| setting | value | why |
|---|---|---|
| vLLM flags | as `vllm_vanilla` (GPU pool in bytes, 8,192, 64, 262,144) | the same pools and batching |
| `--scheduling-policy` | `continuum` | the system |
| LMCache | 0.3.6 in-process, `LMCACHE_MAX_LOCAL_CPU_SIZE=149.01` (GiB), local disk `/mnt/ssd/baselines/lmcache_inproc` capped at 1,800 GiB, `use_odirect` | our host pool in bytes; an SSD tier read from the disk |
| dependencies | resolved as PyPI stood on 2025-09-16 (transformers 4.56.1, torch 2.8.0) | vLLM 0.10.2's time; today's resolution brings transformers 5.x |
| `RUN_OUTPUT_DIR` | the run dir | the scheduler writes its per-job record (`scheduler_timestamps`) there at shutdown |

## Checks (2026-11-21, GPU)

- `check_server.py` with LMCache on (GPU 100K tokens, CPU 30 GB, disk 200 GB; `results/20261121_184200_continuum_continuum_lmcache/`):
  cold 0 cached (TTFT 1.371 s); warm 19,999 cached (0.021 s); extended 20,000; after 150,000 filler tokens 19,999
  cached from LMCache's CPU tier (0.061 s); after 400,000 more 19,999 from its disk ("Retrieved 20000 ... 1.8311 gb,
  cost 3502.9 ms, throughput 0.5227 GB/s" in its log; TTFT 3.527 s); first-token logprobs equal to the warm run's
  (0.0) at both. LMCache 0.3.6's disk tier reads at a third of LMCache 0.5.7's speed on the same SSD.
- `check_session.py`, Continuum's pin, GPU only (40K-token pool, context 32,768; a 12,000-token turn, then 6 x 8,000
  filler tokens arriving as it completes, the next turn 1.5 s later; `results/20261121_184845_continuum_continuum/`):
  with the tool sent (`ls`), the next turn found 12,000 of 12,528 tokens cached, twice (TTFT 0.751 and 0.055 s);
  with `func=off`, 0 cached, twice (0.301 and 0.293 s): the filler evicted the unpinned prefix. The scheduler's own
  record has 2 pins and 2 unpins. A larger pool hides the pin: vLLM's chunked prefill allocates 8,192 tokens of
  blocks a step, so within the 2 s pin a fill never reached the idle prefix in a 100K-token pool.
- Cold prefill of 20,000 tokens takes 1.371 s on this fork's vLLM 0.10.2 against 0.557 s on vLLM 0.33.0.
- 20-minute replay at campaign 28's setting with LMCache (`results/baseline_smoke2/`, run
  `20261121_192849_continuum_continuum_lmcache`): 2,263 requests, 0 errors, mean TTFT 1.66 s (p50 0.12 s, p90
  2.64 s), 7.5 % of prompt tokens recomputed; the scheduler's record (`scheduler_timestamps`) has 1,093 pins and
  1,091 releases over 128 jobs, and 23 preemptions.

## Continuum's own evaluation, scaled down (2026-11-21, GPU)

`official/swebench.sh`: its vendored mini-swe-agent on the first 8 SWE-bench Verified tasks (all astropy), 8 jobs at
once, step limit 50, GPU-only fork with a 100K-token pool, once with `--scheduling-policy fcfs` and once with
`continuum`; job durations by the authors' `continuum_exp/analyze.py` from the scheduler's own record.

| policy | jobs | model calls | average job | median | max | pins | call time mean, p90 |
|---|---|---|---|---|---|---|---|
| fcfs (`results/20261121_202314_continuum_fcfs/`) | 8 | 224 | 39.3 s | 35.1 s | 80.1 s | 0 | 1.19 s, 1.72 s |
| continuum (`results/20261121_202634_continuum_continuum/`) | 8 | 245 | 92.4 s | 72.7 s | 402.6 s | 214 | 2.76 s, 12.30 s |

- The release's parser read the real agent's replies (218 and 224 tool calls extracted), so this run needs no patch,
  and the pin acted (214 pins).
- The authors' shipped result (500 jobs, Llama-3.1-70B, 4 H100: 215.5 s against 465.2 s) does not reproduce at this
  scale. Eight live jobs diverge from run to run (224 against 245 calls, 4 of 8 tasks submitted in each, different
  output lengths), so job and call times mix the policy with the paths the agents took; queue waits were near zero in
  both runs (mean 3 ms and 1 ms), so the pressure the pin is meant to relieve hardly arose.
- The controlled comparison is the trace replay: Continuum against its own FCFS base on the same users
  (`results/baseline_smoke2/`, arms `continuum` and `continuum_fcfs`).
