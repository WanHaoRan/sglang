# Evaluation starter kit: does deadline-driven KV tiering pay off for agent workloads?

*Compiled 14 September 2026. Everything here was verified by opening the source (repo, dataset card, docs, or the SGLang 0.5.19 wheel) unless marked otherwise.*

## 0. The question to answer first

Before building the policy, establish two facts on real traces: how much HBM memory-time is idle behind tool calls at realistic concurrency, and how much of the recompute cost sits behind gaps long enough to hide a host or flash restore. Two independent negative results (Ganjihal's "When Prediction Doesn't Pay"; the `agentic-kv-cache` replay) found that in a capacity-bound single HBM tier, TTL and hazard policies are byte-identical or within 0.5 pp of leaf-LRU, and that a third of recompute comes from requests within 10 s of the previous one. If the gap distribution and tier costs in your target setting don't put substantial memory-time behind ≥1 s gaps, the idea won't pay and you should know that in a week, not a quarter. The plan below is ordered accordingly: characterize and simulate first, then serve for real, then validate with live agents.

## 1. Workload sources

### 1.1 Real traces with recorded tool gaps

**LMCache agentic traces** — `huggingface.co/datasets/sammshen/lmcache-agentic-traces` (CC-BY-4.0, 2.4 GB, parquet). 787 multi-turn agent sessions, 24,881 iterations: SWE-bench Verified via OpenHands CodeAct (669 sessions, median 38 turns), GAIA via Inspect AI (85), WildClaw (10); generated with MiniMax-M2.5, Claude Sonnet 4.6, Claude Opus 4.6, DeepSeek V3.1. Fields: `session_id`, `model`, `input` (the full cumulative OpenAI-format message list, so iteration N embeds the assistant output of N−1), `output_length`, and `pre_gap`, "seconds between the previous iteration's response completing and this iteration's request being sent" (median 0.71 s). Median input 21K tokens, contexts grow from ~14K to ~35K. This is the best single source for a replay with real content and real gaps. A converter to AIPerf's `mooncake_trace` format lives in the author's `sammshen/agentic-dataset` repo; the same data converts trivially to SGLang's `agentic-trace` JSON (per-turn message deltas), see §3.

**SemiAnalysis AgentX (Claude Code) traces** — `huggingface.co/datasets/semianalysisai/cc-traces-weka-062126` (rolling alias `semianalysis_cc_traces_weka_with_subagents`; older `cc-traces-weka-042026`, `cc-traces-weka-no-subagents-051226`; Apache-2.0 per the simulator README). 393 Claude Code sessions, ~68K requests, selected for ≥20 requests and ≤10 concurrent subagents. Content is removed: "inputs become session-scoped chained hashes in 64-token blocks." Retained: request/response timing with end-to-start delays, token counts (median input 142K, median output 444), conversation and subagent IDs, subagent SPAWN/JOIN topology (44% of sessions). Methodology: `inferencex.semianalysis.com/agentx`. This is the standardized benchmark the engines (vLLM, SGLang, TRT-LLM, Dynamo, LMCache, Mooncake) are being tuned against, and its long contexts are what stress tiers. Caveats: hashes are session-local, so there is no cross-session prefix sharing, and tokens are synthetic.

**Mooncake FAST'25 traces** — `github.com/kvcache-ai/Mooncake/tree/main/FAST25-release`. `toolagent_trace.jsonl` (23,608 requests, one hour of production tool/agent traffic), `conversation_trace.jsonl` (12,031), `synthetic_trace.jsonl` (3,993), plus the older `arxiv-trace/mooncake_trace.jsonl`. Fields: `timestamp` (ms), `input_length`, `output_length`, `hash_ids` (remapped 512-token prefix blocks, global scope, so cross-session sharing is visible). No session ID, no content, no explicit gaps; multi-turn structure is implicit in shared hash prefixes. Useful for simulator validation (Mooncake published the hit-rate curve) and for cross-session sharing; SGLang's `bench_serving` downloads these automatically.

**kv-cache-tester traces** — `github.com/callanjfox/kv-cache-tester`: 739 anonymized Claude Code conversation traces plus `trace_replay_tester.py`, which generates synthetic content matching per-turn token counts while preserving the real assistant responses, pre-warms a ~12K-token shared tool/system prefix, and scales concurrency adaptively against a p95-TTFT SLO. Used in LMCache's May 2026 MI300X agentic benchmark (`blog.lmcache.ai`, 2026/05/12), which is a good template for a two-tier comparison (no cache vs HBM-only vs HBM+64 GB DRAM).

Also: `github.com/LMCache/lmcache-agent-trace` ("agent application/benchmark/workload traces should be placed here") is the LMCache team's collection point for further traces.

### 1.2 Live agent generators (for validation and for building your own corpus)

**mini-swe-agent + SWE-bench Verified** is what Continuum used, and their preview code (`github.com/Hanchenli/vllm-continuum`, Apache-2.0) shows the exact recipe: start the server, run `mini-extra swebench` with either fixed workers or a Poisson job-arrival rate, score with `sb-cli`, analyze with `continuum_exp/analyze.py`. No traces are shipped; the workload is generated live against a vLLM fork with `--scheduling-policy continuum` and optional LMCache CPU offload. The same client works against an SGLang OpenAI endpoint.

**OpenHands (CodeAct)** is the source of the LMCache SWE-bench traces and of the "OpenHands / SWE-smith" traces SGLang's loader docstring refers to; pointing it at an SGLang endpoint and recording through a proxy gives you traces in your own format.

**BFCL v4** (`gorilla.cs.berkeley.edu/leaderboard.html`; ICML 2025 paper) has multi-turn, web-search, and memory categories with executable tools, so tool latencies are real and varied; Continuum measured ~2 s mean tool times on BFCL v4 web search with the slowest 10% of `fetch_url` calls accounting for 52.5% of delay.

**τ²-bench** (Sierra; airline/retail/telecom with a simulated user) is what Choi & Joshi used for human-approval-style pauses, and **GAIA via Inspect AI** is the other component of the LMCache traces.

If you need synthetic gaps, Tokencake's tool-latency table is a reasonable prior: file system ~100 ms, git ~100 ms–1 s, short web search ~100 ms–2 s, medium web search 1–5 s, AI generation 5–30 s; Continuum's measured SWE-bench tool times were 0.9–3.5 s mean with heavy tails; CacheWise reports Claude Code `mypy` P99 of 182 s and median session length of 36 min.

## 2. Simulation tools

**`github.com/gauravapiscean/agentic-kv-cache`** (MIT). A block-granular discrete-event prefix-cache simulator with the constraints that matter (prefix-contiguous hits, radix rule that blocks with resident children can't be evicted, in-flight pinning). Loads AgentX (64-token blocks, session-local hashes) and Mooncake (512-token blocks, global hashes) via `make data`; `make repro` runs validation against Mooncake's published curve (note the unexplained 4–6 pp offset), gap characterization, recompute-by-gap, and a policy ablation (LRU-leaf, TTL-300s, LFU-leaf, hazard +H, cost +C, session-coherent +G). It has no GPU timing and a single tier, which is exactly the regime where prediction lost. The natural first experiment is to extend it with tiers and per-tier miss costs (use the TTFT estimator from our earlier discussion: device/host/storage hit lengths, bandwidths, a profiled prefill curve), then re-run the ablation reporting memory-time freed and estimated TTFT rather than hit rate.

**AgentServeSim** (arXiv 2606.09613, UCF). Models HBM/DRAM/CXL, a RadixAttention-style prefix cache, operator-level timing profiles per (device, model), tool durations in "replay mode" (captured durations) or "generative mode" (per-tool distributions), and a policy hook where "the Orchestrator assigns deadlines to the program's KV nodes." Validated within 6% JCT and 2% throughput of vLLM; implements vLLM-FCFS, Autellix, InferCept, Continuum. No public repository was found; worth emailing the authors, since it is the closest existing model of the deadline hook.

## 3. Serving harness: SGLang + HiCache

### 3.1 Client side

SGLang 0.5.19's `python -m sglang.benchmark.serving` (the old `sglang.bench_serving` path is deprecated) has two relevant datasets. `--dataset-name agentic-trace --dataset-path trace.json --backend sglang-oai-chat` replays "pre-built multi-turn agentic traces (e.g. OpenHands / SWE-smith)": each conversation is a list of turns, each turn holding only the new non-assistant messages (`{"messages": [...], "prompt_tokens": N}`), and the client "replays each conversation round by round, feeding the server's real assistant reply back into the next round's history." Flags: `--agentic-max-turns`, `--dataset-offset`, `--sharegpt-output-len` (default 220 tokens per turn, matched to OpenHands-style replies), plus the usual `--max-concurrency` and `--request-rate`. `--dataset-name mooncake --mooncake-workload toolagent --mooncake-num-rounds N --mooncake-slowdown-factor F` replays the Mooncake trace by timestamp.

Two gaps to patch before either is usable for this study. The agentic-trace replay fires rounds back-to-back: `wrap_multi_turn_request_func` in `sglang/benchmark/serving.py` loops over rounds, awaits the response, appends the assistant reply, and immediately sends the next round. Add a per-turn `pre_gap` field to the trace JSON and an `await asyncio.sleep(gap)` between rounds (about twenty lines), otherwise there is no idle window and nothing to evict into. The Mooncake replay fires all rounds of a session "as a burst," which is the opposite of what you want; use it for cross-session sharing experiments, not for gap studies.

Converting the LMCache parquet into the SGLang format is straightforward: group rows by `session_id`, sort by iteration, and for each iteration emit the messages that were appended since the previous iteration's input (drop the assistant message, which the server will regenerate), carrying `output_length` and `pre_gap` alongside. Keep in mind that regenerating replies with a different model changes context lengths and tool-call text; that is fine for a systems study as long as you report it.

**NVIDIA AIPerf** is the alternative that already does the gap-faithful replay. `aiperf profile --scenario inferencex-agentx-mvp --public-dataset semianalysis_cc_traces_weka_with_subagents --endpoint-type chat --concurrency 32 --streaming --cache-bust first_turn_prefix --system-idle-gap-cap-seconds 10 --benchmark-duration 1800` replays AgentX against any OpenAI-compatible endpoint, "preserves the recorded timing within every trace" using end-to-start delays "so the replay retains the captured agent pacing and KV-cache reuse intervals," schedules per session tree (parents wait on subagent SPAWN/JOIN), keeps exactly `--concurrency` trees live, and reports per-turn TTFT/ITL, throughput, and time-weighted session metrics. It works with SGLang directly and through SGLang Model Gateway or Dynamo with routing-key headers. Two things to watch: the "global idle guard" shifts all timers when no request is active so the next arrives within 10 s, which compresses long gaps at low concurrency (raise the cap or keep concurrency high enough that something is always running), and warmup starts each trajectory at a random 25–75% point of the session with a cache-bust tag, so steady-state hit rates are not inflated by repeated plays. Docs: `docs.nvidia.com/aiperf/dev/tutorials/datasets-inputs/inference-x-agent-x-mvp-benchmark`.

### 3.2 Server side

Baseline configurations to sweep, all on the same model and hardware:

- HBM only, LRU (stock SGLang; `--radix-eviction-policy lru`, alternatives `lfu`, `fifo`, `slru`, `priority`).
- HBM + host: `--enable-hierarchical-cache --hicache-ratio {1,2,4}` (host pool as a multiple of the device pool; or `--hicache-size` GB) with `--hicache-write-policy {write_through, write_through_selective, write_back}`, `--hicache-io-backend kernel`, `--hicache-mem-layout page_first_direct`.
- HBM + host + L3: `--hicache-storage-backend file` on a local NVMe is the cheapest three-tier setup; `mooncake` or `hf3fs` for a real shared store; with `--hicache-storage-prefetch-policy {best_effort, wait_complete, timeout}` and `--page-size 64`.
- Memory pressure is the independent variable: shrink `--mem-fraction-static` (or pick contexts like AgentX's 142K median) so that the working set of live sessions exceeds HBM; without pressure LRU keeps everything and every policy looks identical.

For the policy under test, the implementation hooks are the ones identified earlier: a new `EvictionStrategy` returning predicted next-access time (the interface is `get_priority(node)`; `PriorityStrategy` already orders by a per-node priority, which can carry a deadline as a first hack), a timer-driven `load_back` decoupled from admission (today it only fires in the prefill adder), a per-request hint carrying predicted tool duration (Tokencake's `t_req`; SGLang requests already accept extra fields), and the landing-zone budget in the prefill adder.

### 3.3 What to measure

Per-turn TTFT (p50/p95/p99) and ITL from the client; throughput at fixed concurrency (session trees, not requests); prefix hit rate from the benchmark output and from SGLang's Prometheus metrics; HiCache backup/prefetch token counters (the code has metrics collectors for backup and storage prefetch volume). Add two instrumentation points that no harness gives you: per request, log the `MatchResult` split (device hit, host hit, storage hit) so you can compute a *restore-hit rate* (fraction of returning turns whose context was fully device-resident on arrival, versus host-restored, storage-restored, or recomputed), and sample the token-usage gauge over time to compute memory-time (token-seconds) held by idle sessions. Report gains as throughput at a fixed TTFT SLO and memory-time freed, and always include a short-gap regime (median gap ~1 s, like the LMCache traces) as the honest negative case.

## 4. Suggested sequence

Week 1: characterize. Run `agentic-kv-cache`'s characterization on AgentX and Mooncake toolagent; compute `pre_gap` distributions per benchmark and per tool from the LMCache traces; combine with per-token KV size for your model to get memory-time idle behind gaps ≥ {1, 5, 30} s at concurrency {32, 64, 128}. This alone tells you the upper bound on what any parking policy can free.

Weeks 2–3: simulate. Extend the simulator with tiers, per-tier restore cost from the estimator, and the deadline policy; compare LRU-leaf, TTL, Tokencake-style two-tier offload, and the deadline policy on memory-time, estimated TTFT, and restore-hit rate. Decide go/no-go here.

Weeks 4–6: serve. SGLang + HiCache with the patched agentic-trace client on LMCache traces (real content, real gaps), then AIPerf/AgentX for the long-context, standardized view. Sweep HBM pressure and host/L3 configuration; establish the stock-HiCache baselines before touching policy.

Later: validate with live agents (mini-swe-agent on SWE-bench Verified per the Continuum recipe, BFCL v4 web search) and record your own traces through a proxy, which also yields the per-tool latency distributions the predictor needs.

## 5. Links

- LMCache agentic traces: https://huggingface.co/datasets/sammshen/lmcache-agentic-traces
- AgentX methodology: https://inferencex.semianalysis.com/agentx ; datasets: https://huggingface.co/datasets/semianalysisai/cc-traces-weka-042026 , https://huggingface.co/datasets/semianalysisai/cc-traces-weka-no-subagents-051226 (current corpus alias in AIPerf: `semianalysis_cc_traces_weka_with_subagents`, repo `semianalysisai/cc-traces-weka-062126`)
- AIPerf AgentX tutorial: https://docs.nvidia.com/aiperf/dev/tutorials/datasets-inputs/inference-x-agent-x-mvp-benchmark
- vLLM x AgentX blog: https://vllm.ai/blog/2026-09-08-vllm-agentx
- Mooncake traces: https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release
- kv-cache-tester: https://github.com/callanjfox/kv-cache-tester ; LMCache MI300X agentic benchmark: https://blog.lmcache.ai/en/2026/05/12/benchmarking-lmcache-for-multi-turn-agentic-workloads-on-amd-mi300x/
- LMCache agent trace collection: https://github.com/LMCache/lmcache-agent-trace ; LMBench: https://github.com/LMCache/LMBench
- Prefix-cache simulator: https://github.com/gauravapiscean/agentic-kv-cache
- AgentServeSim: https://arxiv.org/abs/2606.09613
- Continuum preview code: https://github.com/Hanchenli/vllm-continuum
- SGLang bench serving guide: https://docs.sglang.io/docs/developer_guide/bench_serving ; HiCache benchmark dir: https://github.com/sgl-project/sglang/tree/main/benchmark/hicache
- BFCL v4: https://gorilla.cs.berkeley.edu/leaderboard.html ; paper: https://proceedings.mlr.press/v267/patil25a.html
