# How ten 2026 papers on agent KV-cache storage, tiering, protocols and pricing evaluate their systems

Reading notes, 17 Nov 2026. Each paper's full text was read from its arXiv HTML page. The abs pages (checked 17 Nov 2026) show that every paper has only a v1. Section, figure and table numbers refer to that v1. Mode labels: (i) live agents; (ii) trace replay on a real engine; (iii) simulation; (iv) synthetic workload. "Inference" marks my own reading; "not stated" means a keyword search of the full text found nothing.

## 1. HEAR (arXiv 2610.06597): evaluation setup

### Takeaway
HEAR has two experiment families.
- **KV-cache experiments (ordering and retention).** They replay recorded request trajectories against vLLM, with a GPU tier and a 96 GiB CPU ("L2") tier.
  - SCBench is a synthetic session workload: Poisson session arrivals and N(5,1) s think times.
  - The production Mooncake trace is replayed at 50/75/100% of the sessions that start in one 15-minute window.
  - Recorded input/output lengths are replayed. There are no tools and no SSD.
- **Role-configuration experiments.** Live LangGraph research agents run end to end. They choose sparse-attention modes per agent role and do not exercise KV tiering.

### Cited Findings
- **Record.**
  - "Can Agent Harnesses and Inference Engines Hear Each Other? The HEAR Protocol for Agentic LLM Serving", Jiaqi Zhao, Haodong Chen, Jitai Hao et al.
  - v1 only, 5 Oct 2026. Comment: "Jiaqi Zhao, Haodong Chen, and Jitai Hao contributed equally".
  - — [abs](https://arxiv.org/abs/2610.06597); full text [HTML v1](https://arxiv.org/html/2610.06597v1)
- **(1) Mode, KV experiments: record once, then replay (App. A.2).**
  - "We first execute and record a complete trajectory, including request arrival times, input and output lengths, session/turn relationships, and prefix information. All configurations then replay the same trajectory."
  - The aim is that "configurations differ only in request scheduling and KV-cache management behavior."
  - — [HTML v1, App. A.2](https://arxiv.org/html/2610.06597v1)
- **SCBench is a synthetic session workload (Sec. 4.2; App. A.2).**
  - "SCBench [13] lacks request timestamps, so we construct a closed-loop multi-turn workload with Poisson session arrivals and sampled inter-turn think times".
  - Each sample becomes "an independent session". "After each response completes, a user think time is sampled". Think time follows N(5,1) s.
  - "all configurations replay the same session-arrival and think-time sequences."
  - — [HTML v1, Sec. 4.2 and App. A.2](https://arxiv.org/html/2610.06597v1)
- **Mooncake is a production trace replay (Sec. 4.2; App. A.2).**
  - The trace "records request timestamps, input/output lengths, and remapped block hashes, preserving real arrivals and inter-turn gaps".
  - "We replay 50%, 75%, and 100% of sessions beginning within a 15-minute window, preserving their start times and complete turn sequences."
  - Load levels come "only by replaying 50%, 75%, and 100% of the sessions whose first requests fall within the same 15-minute window, without altering the original timing relationships between requests."
  - — [HTML v1, Sec. 4.2 and App. A.2](https://arxiv.org/html/2610.06597v1)
- **(1) Mode, role-configuration experiments: live agents (Sec. 4.3).**
  - The runs use "208 BrowseComp-Plus questions [5] and 100 DeepResearchBench instances [7] per arm".
  - "Both retain their native orchestration and retrieval environments within a common LangGraph-based Main–Retriever–Reader/sub-agent harness, with GLM-4.7-Flash serving Main and Reader/sub-agent inference on separate H100 GPUs."
  - — [HTML v1, Sec. 4.3](https://arxiv.org/html/2610.06597v1)
- **Controls (Sec. 4.1).** "all methods use identical models, inputs, prompts, workflow and tool budgets, and hardware. Policies and method-specific parameters are frozen before evaluation, each arm starts from an independently initialized serving state, and reported results include all failures and retries." — [HTML v1, Sec. 4.1](https://arxiv.org/html/2610.06597v1)
- **(2) Tools.**
  - KV experiments have no tools. The gap between turns is a sampled N(5,1) s think time (SCBench) or the trace's recorded gap (Mooncake).
  - Live runs keep their "native ... retrieval environments".
  - — [HTML v1, App. A.2; Sec. 4.3](https://arxiv.org/html/2610.06597v1)
- **(3) Model outputs.**
  - KV experiments replay the recorded input and output lengths (App. A.2).
  - Live BrowseComp-Plus runs: "generation uses temperature zero, top-p=1, and disables model thinking" (App. A.4).
  - Quality judges: Qwen3-32B for BrowseComp-Plus, and the "vendored official RACE evaluator with gpt-5.4-mini" for DeepResearchBench (App. B.4).
  - — [HTML v1](https://arxiv.org/html/2610.06597v1)
- **(4) Workload sources.**
  - SCBench [13].
  - The Mooncake trace [21], cited as "Mooncake: a kvcache-centric disaggregated architecture for llm serving. ACM Transactions on Storage 22 (4)".
  - BrowseComp-Plus (208 questions) and DeepResearchBench (100 instances).
  - — [HTML v1, Sec. 4 and references](https://arxiv.org/html/2610.06597v1)
- **(5) Integration, KV experiments (App. A.2–A.3).**
  - Engine → harness: "Engine KV events update a Harness-maintained residency directory". The harness "matches reusable prefix blocks sequentially from the beginning of the prompt" and "separately counts reusable prefix tokens in the GPU and CPU tiers".
  - Harness → engine, priority: a request is hot if reusable prefixes cover ≥50% of the prompt, and its priority = −N_reuse.
  - Harness → engine, waiting protection: Guard "is submitted by the harness to the engine at fixed times and supports subsequent updates".
  - Harness → engine, return estimate (Session-Aware): a session-return probability p, "updated only from completed conversations and does not use future information from the current replay".
  - Engine-side retention: the engine computes r = p·e^(−idle/400), which sets eviction order only.
  - Outcome report: the engine "reports the realized execution order, cache reuse, retries, and errors."
  - — [HTML v1, App. A.2–A.3](https://arxiv.org/html/2610.06597v1)
- **Integration, live runs.** A frozen role-to-mode mapping is loaded "at workflow initialization", and each model request does "a constant-size lookup". — [HTML v1, App. A.1, A.3](https://arxiv.org/html/2610.06597v1)
- **(6) Load.**
  - SCBench: Poisson session arrivals. Batch completion takes 224–481 s across arms (Table 3).
  - Mooncake: throughput is 86–128 turns/min (Table 4).
  - BrowseComp-Plus: "at most four Main and twelve Reader requests in flight" (App. A.4).
  - — [HTML v1](https://arxiv.org/html/2610.06597v1)
- **(7) Stack.**
  - KV experiments: Qwen3-8B. SCBench runs on one RTX PRO 6000 and Mooncake on one H100, each "with a 96-GiB L2 KV cache" (App. A.2).
  - Engine: the FCFS baseline "uses the default engine scheduler [11]" (Sec. 4.2). Sec. 1 identifies [11] as vLLM ("an inference engine, such as vLLM [11] or SGLang [34]").
  - Live runs: GLM-4.7-Flash. "Dense execution uses vLLM 0.26.0, while OmniKV, H2O, and SnapKV are provided through SparseEngine" (App. A.4).
  - — [HTML v1](https://arxiv.org/html/2610.06597v1)
- **(8) Metrics.**
  - TTFT P50/P95/max.
  - Batch completion, "the elapsed time to complete all users' multi-turn conversations".
  - Reuse, "the KV-cache reuse rate for follow-up prompt tokens" (Table 3).
  - Session time, "the total response-waiting time per multi-turn conversation", and throughput in turns per minute (Table 4).
  - Timelines of GPU KV-pool occupancy, cumulative KV restoration and waiting requests (App. B.1, Fig. 5).
  - Live runs: wall time or makespan, P95 end-to-end latency, Judge accuracy, RACE (Table 5).
  - — [HTML v1](https://arxiv.org/html/2610.06597v1)
- **(9) Baselines and ablations.**
  - FCFS, and Cache-Aware ("Inspired by KVFlow").
  - Cache-Aware+Guard-W, with W ∈ {40, 60} s on SCBench and 40 s on Mooncake.
  - Session-Aware ("inspired by prediction-guided methods"), alone and combined with Cache-Aware+Guard.
  - Live runs: six arms, {vLLM, OmniKV} × {vLLM, H2O, SnapKV}.
  - — [HTML v1, Sec. 4.2–4.3](https://arxiv.org/html/2610.06597v1)
- **Headline numbers.**
  - SCBench: Cache-Aware raises reuse from 19.2% to 87.6%, cuts median TTFT from 63.1 to 4.6 s, and cuts batch completion from 481 to 224 s. Guard-40 keeps "a 1.61× batch speedup".
  - Mooncake: at 50% load, Session-Aware raises reuse from 19.5% to 31.9%.
  - At 100% load, the combined policy's throughput is 0.89× FCFS, and "no policy dominates across loads".
  - — [HTML v1, Sec. 4.2, Tables 3–4](https://arxiv.org/html/2610.06597v1)
- **(10) Released artifacts.**
  - The reproducibility statement points to Appendix A.
  - "complete launch commands, environment manifests, and source hashes are included in the supplementary artifact" (App. A.4).
  - Neither the text nor the abs page gives a public URL.
  - — [HTML v1](https://arxiv.org/html/2610.06597v1); [abs](https://arxiv.org/abs/2610.06597)
- **(11) Threats the authors state.**
  - Judge provenance: "Arms A, B, and F are judged on an RTX PRO 6000, whereas Arms C, D, and E are judged on an H100 ... re-evaluating all arms in a single Judge environment would further remove this environmental difference" (App. B.4).
  - The conclusion says future-use-aware retention "must account for live queue state under overload" (Sec. 5).
  - — [HTML v1](https://arxiv.org/html/2610.06597v1)

### Inferences
- **Classification.**
  - SCBench is a synthetic workload (iv), recorded once and replayed on a real engine (ii).
  - Mooncake is a trace replay on a real engine (ii).
  - The BrowseComp-Plus and DeepResearchBench results are live agents (i), but none of them tests KV tiering.
- **Tiers (inference).** The KV tiers are GPU plus CPU DRAM only, based on "GPU and CPU tiers" and the "L2" label. No SSD tier is used.
- **Threats I notice.**
  - **Think time.** A 5 s think time is short next to real agent tool times. SCBench's reuse loss therefore comes mainly from queueing and ordering, not from long idle gaps.
  - **Fixed constants.** The 400 s decay and the 50% hot cutoff are fixed constants.
  - **No repeats.** No repeated runs or variance are reported.
  - **Scale.** There is one 8B model on one GPU.
  - **Prompt content.** Replay keeps lengths, but token content from Mooncake's hashed blocks must be synthesized (inference).

### Gaps
- **Counts and rates.** The number of SCBench sessions, the Poisson rate, and the number of Mooncake sessions in the window are not stated.
- **Mooncake trace details.**
  - Which Mooncake trace file is used, and how sessions are identified in it, are not stated.
  - Whether recorded gaps are timed from response completion or replayed as absolute timestamps is not stated.
- **Replay and engine details.**
  - How output lengths are forced is not stated.
  - The vLLM version for the KV experiments is not stated.
  - How the 96 GiB L2 tier is implemented (vLLM offload connector, LMCache or other) is not stated.

## 2. KVTether (arXiv 2609.39819): evaluation setup

### Takeaway
KVTether replays agent traces on a real cluster: SGLang 0.5.8 with prefill-decode disaggregation, GLM-4.7 (358B) on 12 GB200 GPUs, LMCache/Mooncake, and a 400 GB host KV pool per node.
- **Traces.** One is collected live from the Codex harness on TerminalBench. Two are production coding-agent traces (400 tasks each).
- **Replay.** Prompts are random tokens matched to recorded input counts and reusable-prefix lengths, and recorded output counts are enforced. Tool calls are sleeps of recorded duration. Lifecycle events are replayed in order. Tasks arrive by Poisson at five rates.
- **Not tested.** There is no SSD tier and no load-dependent tool time.

### Cited Findings
- **Record.** "Capture the lifecycle: KV Cache management in ReAct Agents with KVTether", Kaihua Fu, Yukun Zhou, Chaokun Chang et al. v1 only, 30 Sep 2026. — [abs](https://arxiv.org/abs/2609.39819); [HTML v1](https://arxiv.org/html/2609.39819v1)
- **(1) Mode: replay on the real cluster (Sec. 8.1, Workloads).**
  - "we replay each task through LLM inference on the real cluster, using random prompts with recorded per-round input token counts and enforcing recorded output counts."
  - "We adjust prompt content across rounds to exactly match each request's potentially reusable prefix length in the trace."
  - "We simulate tool calls by waiting for their recorded durations, eliminating variability in tool execution time across experiments."
  - — [HTML v1, Sec. 8.1](https://arxiv.org/html/2609.39819v1)
- **Lifecycle events are replayed too (Sec. 8.1).**
  - "we replay lifecycle events at their original execution points and in order relative to LLM calls. This preserves message boundaries, prefix sharing, fork inheritance, and suffix invalidation from message replacement or removal."
  - For Codex, KVTether's programming model is integrated "into the harness to record events online during trace collection".
  - For Agent-X and Agent-G, "we reconstruct events offline from logged message updates, context relationships, and termination records, using only information available at that execution point."
  - — [HTML v1, Sec. 8.1](https://arxiv.org/html/2609.39819v1)
- **(2) Tools.** Tool calls are replayed as sleeps of recorded duration, not executed (Sec. 8.1). Nothing ties tool time to load. — [HTML v1, Sec. 8.1](https://arxiv.org/html/2609.39819v1)
- **(3) Model outputs.** Recorded output counts are enforced on random prompts (Sec. 8.1). Output charges are therefore "fixed by replay" (Sec. 8.2). — [HTML v1](https://arxiv.org/html/2609.39819v1)
- **(4) Workload sources.**
  - Codex: "we collect trajectories by running the Codex agent harness (OpenAI, 2026a) on TerminalBench (Merrill et al., 2026) tasks, with all LLM requests served by DeepSeek-V4-Pro" (Sec. 8.1).
  - Agent-X and Agent-G: "execution traces from two internal coding agents deployed by a large cloud provider. Each trace contains 400 tasks" (Sec. 8.1). Sec. 2.2 says the two agents "have a combined peak of over 27,000 active sessions".
  - Per task ("request"): "Requests span 3–127 LLM rounds and accumulate 9K–374K prefill tokens", and they use 1–47 distinct tool types (Sec. 2.2, Table 1).
  - Sub-agents: they account for up to 86.5% of LLM rounds in Agent-X, 17.7% in Codex and 25.0% in Agent-G (Sec. 2.2).
  - — [HTML v1](https://arxiv.org/html/2609.39819v1)
- **(5) Integration (Sec. 7).**
  - An "in-harness framework exposes the eight primitives". It records invocations synchronously and exports them asynchronously in order.
  - A server-side lifecycle manager "reproduces the LLM engine's prompt-formatting and tokenization pipeline for independent message-to-KV translation, and interposes on LLM calls".
  - "Before forwarding each call, it passes the call ID and context ID to the cache engine".
  - What crosses the boundary: lifecycle events (append, replace, remove, fork, spawn, one_off, pin, unpin), call ID and context ID. The engine does not parse tool calls.
  - Integration takes "at most 216 lines of code per harness" across six harnesses (Sec. 9, Table 5).
  - — [HTML v1, Sec. 7–9](https://arxiv.org/html/2609.39819v1)
- **(6) Load.**
  - "We evaluate each trace at five RPS settings with Poisson arrivals ... Task samples and arrival sequences are identical across methods" (Sec. 8.1).
  - "Requests" are tasks, each spanning many LLM rounds (Sec. 2.2).
  - A representative load point uses "0.06, 0.05, and 0.06 requests per second (RPS)" (Sec. 2.3).
  - Figs. 13–14 analyze "one-hour windows from workload replay" (Sec. 8.3).
  - — [HTML v1](https://arxiv.org/html/2609.39819v1)
- **(7) Stack (Sec. 8.1, Table 4).**
  - Hardware: 3 nodes, each with 2 Grace CPUs, 4 GB200 GPUs and 4×200 Gb/s RoCE.
  - Software: SGLang 0.5.8, LMCache 0.5.5, MoonCake 0.3.8, CUDA 12.9.
  - Model: GLM-4.7 (358B), PD disaggregation (1 prefill + 2 decode instances, TP=4).
  - KV capacity: "400 GB of host memory to the KV pool (1.2 TB total)", with cross-node KV transfer through the MoonCake transfer engine.
  - Tiers in the design: "GPU HBM, host memory, and remote memory" (Sec. 7). No SSD.
  - — [HTML v1](https://arxiv.org/html/2609.39819v1)
- **(8) Metrics (Sec. 8.1).**
  - KV hit ratio: "total number of cache-hit input tokens divided by the total number of input tokens".
  - Cumulative prefill time.
  - End-to-end latency, from task start to completion.
  - Average task pricing at DeepSeek-V4-Pro peak rates of "$0.044, $1.32, and $3.96 per million cache-hit input, cache-miss input, and output tokens".
  - Also zombie-page occupancy (Fig. 13), wrong-direction eviction ratio (Fig. 14) and overheads (Sec. 8.5).
  - — [HTML v1](https://arxiv.org/html/2609.39819v1)
- **(9) Baselines and ablations.**
  - LMCache, "LRU eviction based on page-access recency".
  - MORI, which "ranks agent programs by idleness".
  - Ablations: w/o LT (lifecycle tracking off, MRU kept) and w/o RO (LRU inside the Suspended tier).
  - — [HTML v1, Sec. 8.1, 8.3](https://arxiv.org/html/2609.39819v1)
- **Headline numbers (Sec. 8.2).**
  - Average absolute hit-ratio gains are 33.3% over LMCache and 24.5% over MORI.
  - End-to-end latency falls by 16.5% and 12.4%, and estimated task pricing by 40.0% and 33.2%.
  - The "theoretical maximum" hit ratio (82.8% average) "is measured by replaying each workload's requests offline, one at a time" (Sec. 2.3).
  - — [HTML v1](https://arxiv.org/html/2609.39819v1)
- **(10) Released artifacts.** The text has no code or trace release statement; its only URLs point to third-party harnesses. The abs page lists no code link. — [HTML v1](https://arxiv.org/html/2609.39819v1); [abs](https://arxiv.org/abs/2609.39819)
- **(11) Threats the authors state.**
  - "The latency gains are smaller than the prefill reductions because decoding remains necessary" (Sec. 8.2).
  - The overheads are called negligible: about 350 ms of KV-scope resolution for a 256K-token input, against 19.96 s average calls (Sec. 8.5).
  - — [HTML v1](https://arxiv.org/html/2609.39819v1)

### Inferences
- **Classification.** This is trace replay on a real engine (ii). Only the Codex traces come from live agent runs.
- **Content-free replay.** Random prompts with matched reusable-prefix lengths keep the KV access pattern but drop real token content. Prefix structure is set by construction, which suits a cache study but cannot show tokenization or content effects.
- **Model mismatch.** Traces collected with DeepSeek-V4-Pro (Codex) and unknown production models are replayed on GLM-4.7. Recorded tool times and arrival patterns therefore reflect a different serving speed.
- **Arrivals.** Poisson task arrivals replace the recorded arrival times. Tool durations stay fixed whatever the load.
- **Event reconstruction.** Agent-X and Agent-G lifecycle events are rebuilt offline, which may differ from what a live harness would emit.
- **Reporting.** No variance or repeated runs are reported.

### Gaps
- **Codex trace.** The number of Codex/TerminalBench tasks is not stated.
- **Rates and runs.** The five RPS values for each trace (shown only in Fig. 11) and the total replay duration per run are not stated in the text.
- **Engine path.** Whether experiments used KVTether's "native engine" or its "compatibility path" onto LMCache/MoonCake Store is not stated.
- **Client path.** Whether the replay client speaks an OpenAI-compatible API is not stated.

## 3. The Storage View of KV Cache (arXiv 2611.10178): evaluation setup

### Takeaway
This is a characterization study with no serving engine in the loop.
- **Traces.** It normalizes 11 traces (15 workloads) into 16-token prefix-chained block streams.
- **Policy questions.** It answers them by offline counterfactual accounting, with "causal counts and no capacity limit": per-workload p90 TTL, completion-time reclamation and session-ID flagging. A fixed-concurrency replay estimates resident state.
- **Live input.** One trace (Open Deep Research on DeepResearch Bench, 50 tasks) was collected live on vLLM with real Tavily web search. The six Exgentic agent traces carry real tool latency.

### Cited Findings
- **Record.**
  - "The Storage View Of KV Cache: Characterizing Cache Behavior Across LLM Workloads", Talia Q. Chen, Daniel Waddington, Narangerelt Batsoyol et al.
  - v1 only, 6 Nov 2026.
  - — [abs](https://arxiv.org/abs/2611.10178); [HTML v1](https://arxiv.org/html/2611.10178v1)
- **(1) Mode: trace-driven characterization (Sec. 5).**
  - "Our trace-driven analysis characterizes demand presented to KV storage rather than a complete serving stack, and does not model transfer costs, scheduling, or a particular finite-capacity policy."
  - "Our results quantify the information exposed by workload, session, and lifecycle context, but do not predict the end-to-end performance of a specific storage policy."
  - — [HTML v1, Sec. 5](https://arxiv.org/html/2611.10178v1)
- **Policy experiments are counterfactual accounting (Sec. 4.4, Fig. 6).**
  - Fig. 6a sets the "idle TTL set to the 90th percentile of observed within-session gaps, estimated over all seven workloads or over each workload's own sessions".
  - Fig. 6b compares an "idle TTL of 1, 5, and 30 minutes" with "reclaiming state at session completion".
  - Fig. 6c counts blocks "with causal counts and no capacity limit".
  - — [HTML v1, Sec. 4.4](https://arxiv.org/html/2611.10178v1)
- **Fixed-concurrency replay (App. A.5).**
  - "Each of N slots replays randomly chosen recorded sessions from the trace back to back, starting a new one as soon as the previous one ends. Each session keeps its recorded call times and context sizes."
  - Bytes are computed at "144 KiB per token, the bf16 KV size of Qwen3-8B".
  - — [HTML v1, App. A.5](https://arxiv.org/html/2611.10178v1)
- **Block classification uses the whole trace (Sec. 3.1).** "Block classification is retrospective ... we classify blocks using their complete observed trace, while the storage implications we derive use only information available at each access." — [HTML v1, Sec. 3.1](https://arxiv.org/html/2611.10178v1)
- **(2) Tools.**
  - The Exgentic traces "combine runs from four to five models; their timestamps come from agent execution and therefore include real tool latency" (App. A.1).
  - The ODR collection used live web search: "web search uses Tavily" (App. A.1).
  - No tool is replayed, because no engine is driven.
  - — [HTML v1](https://arxiv.org/html/2611.10178v1)
- **(3) Model outputs.** No generation takes place in the analysis. "Blocks generated during decode are stored and become reusable from the session's next call; outputs of a session's final call are never read and count only toward footprint" (Sec. 3.3). — [HTML v1](https://arxiv.org/html/2611.10178v1)
- **(4) Workload sources (Sec. 3.2; Table 1, App. A.1).** Sessions, requests and timestamp origin:

  | Trace | Sessions | Requests | Timestamps |
  |---|---|---|---|
  | WildChat | 838K | 1.96M | real, 13 months |
  | AppWorld* | 1,500 | 48K | agent run |
  | BrowseComp+* | 1,948 | 38K | agent run |
  | SWE-bench* | 1,959 | 92K | agent run |
  | τ²-bench* Airline / Retail / Telecom | 957 / 1,848 / 1,844 | 11K / 22K / 30K | agent run |
  | RAGBench | 67K | 67K | synthetic |
  | Open Deep Research (own) | 50 | 1,973 | agent run, 5.6 h |
  | Qwen-Bailian To-C / To-B / Code | 23K / 173K / 26K | 43K / 173K / 43K | real, 2 h |
  | RAGPulse | 19K | 35K | real, 42 days |
  | TraceLab | 4,281 | 357K | real, 8 months |
  | BurstGPT v2 | 55K | 232K | real, 110 days |

  - The starred traces are "Agent runs released by Exgentic".
  - — [HTML v1, Table 1](https://arxiv.org/html/2611.10178v1)
- **Normalization (Sec. 3.3).**
  - Raw-text traces are tokenized with "the Qwen3-8B tokenizer", divided "into 16-token blocks", with "vLLM-style prefix-chained hashes".
  - These are "normalized replays of request structure rather than exact reconstructions of the original KV layout".
  - Metadata-only traces (BurstGPT, TraceLab) are used "only for idle-time, concurrency, and projected footprint analyses".
  - For those traces, reused state is estimated as min(input_{k+1}, input_k + output_k) (App. A.1).
  - — [HTML v1](https://arxiv.org/html/2611.10178v1)
- **Live collection of the ODR trace (App. A.1).**
  - Setup: Open Deep Research (a LangGraph application) runs "on 50 English tasks (IDs 51–100) from DeepResearch Bench ..., one task at a time, with user clarification disabled".
  - Serving: "All model roles use Qwen3-8B ... served by vLLM ... on one NVIDIA A100 GPU with a 32,768-token context window".
  - Reduced limits: "we reduce several upstream limits, including the number of concurrent researchers, supervisor rounds, tool-calling rounds, and output lengths".
  - Recording: calls are captured "at the client's HTTP transport, capturing the exact request messages, response, and wall-clock start and end times".
  - Concurrency caveat: "Because tasks run sequentially, the timestamps capture concurrency among researchers within a task but not across concurrent users."
  - — [HTML v1, App. A.1](https://arxiv.org/html/2611.10178v1)
- **(5) Integration.** The paper proposes three signals for storage: workload ID, session ID and session completion. It tests them only through the accounting above, not in an implementation (Sec. 4.4). — [HTML v1](https://arxiv.org/html/2611.10178v1)
- **(6) Load.** The fixed-concurrency replay keeps "exactly N sessions live at all times" (App. A.5, Fig. 4). There is no other arrival process. — [HTML v1](https://arxiv.org/html/2611.10178v1)
- **(7) Stack.** No engine is used for the results. ODR was collected on vLLM, Qwen3-8B and one A100 (App. A.1). Byte figures assume the Qwen3-8B bf16 KV geometry (Sec. 4.3). — [HTML v1](https://arxiv.org/html/2611.10178v1)
- **(8) Metrics.**
  - Reuse scope (within- versus cross-session hits), recurrence gaps, lifetimes, and footprint in GiB (Sec. 4).
  - "Dead state", defined as "the share of resident block-seconds occupied by blocks whose holding sessions have all completed and that are never accessed again" (App. A.6).
  - — [HTML v1](https://arxiv.org/html/2611.10178v1)
- **(9) Baselines and comparisons.**
  - Pooled versus per-workload TTL. "The pooled TTL of 253 s is too short for Qwen To-C and Qwen Code" (Sec. 4.4).
  - Fixed TTLs versus completion reclamation, and per-request versus per-session counting.
  - A five-minute TTL versus completion with shared blocks (App. A.6).
  - — [HTML v1](https://arxiv.org/html/2611.10178v1)
- **(10) Released artifacts.** The full text has no code or trace release statement. The abs page lists no code link. — [HTML v1](https://arxiv.org/html/2611.10178v1); [abs](https://arxiv.org/abs/2611.10178)
- **(11) Threats the authors state.**
  - No transfer costs, scheduling or finite-capacity policy are modeled.
  - "Some statistics are measured offline rather than learned incrementally".
  - Normalization uses one tokenizer and 16-token blocks; App. A.2 checks block-size sensitivity.
  - — [HTML v1, Sec. 5](https://arxiv.org/html/2611.10178v1)

### Inferences
- **Classification.** This is offline analysis. It has no engine and no timing or bandwidth model, so it is closest to (iii) without a simulator of serving. The agent traces came from live runs (i) made by other groups (Exgentic) and, for ODR, by the authors.
- **Use for a prefix-cache paper.** It is a source of real inter-turn gap distributions with real tool latency (Exgentic, TraceLab, ODR). Its TTL results ignore read cost and capacity, so they bound information value, not latency.
- **Threats I notice.**
  - The Exgentic traces' tool latencies come from unknown serving conditions.
  - ODR runs one task at a time with reduced limits.
  - The TraceLab production timing comes from a provider's cache, which the authors flag (App. A.1).

### Gaps
- **Trace release.** Whether the ODR trace or the normalization scripts are released is not stated.
- **Session reconstruction.** How sessions are delimited in WildChat and RAGBench beyond "session structure" is described only briefly.

## 4. EfficientAgent (arXiv 2609.33762): evaluation setup

### Takeaway
EfficientAgent is the most controlled design of the ten.
- **Live runs.** Live OpenHands CodeActAgent runs on SWE-bench Verified (16 concurrent workers, real tools in per-task sandboxes) produce trajectories.
- **Replay.** One trajectory set (4,427 calls, 147.1M prompt tokens) is replayed as a closed loop on vLLM 0.13.0 + LMCache 0.3.12. The replay uses exact prompt token IDs. A vLLM logits processor forces the recorded output tokens. Each task's next call waits for the previous response plus the recorded agent and tool interval.
- **Tiers and cross-hardware runs.** Tiers are GPU plus host DRAM (3–80 GiB per rank), with no SSD. The cross-hardware results (RTX 3090, H800) come from additional runs whose mode is not explicitly labeled.

### Cited Findings
- **Record.**
  - "EfficientAgent: What Makes KV Cache Offloading Work for Concurrent Agents?", Kunming Shao, Jierun Chen, Jiangnan Yu et al.
  - v1 only, 27 Sep 2026. Comment: "25 pages, 8 figures, 13 tables. Code: this https URL".
  - — [abs](https://arxiv.org/abs/2609.33762); [HTML v1](https://arxiv.org/html/2609.33762v1)
- **(1) Mode, live runs (Sec. 5.1; App. A).**
  - "The H20 experiments use OpenHands v0.56.0 with CodeActAgent and Qwen3-Coder-30B-A3B-Instruct in BF16 on SWE-bench Verified".
  - Settings: "Both live runs use an active pool of A=16 tasks (16 OpenHands workers), at most 100 agent iterations, temperature zero, top-p=1, a 245,760-token input limit, and a 16,384-token output limit."
  - "Each task has a separate project sandbox with two CPU cores and an 8 GiB memory limit."
  - "Workflow time spans the first task dispatch through completion of agent inference; patch grading runs separately."
  - — [HTML v1, App. A](https://arxiv.org/html/2609.33762v1)
- **(1) Mode, dependency-preserving replay (Sec. 5.1; App. B).**
  - "we replay the SWE-bench Verified trajectories of the live H20 run without a host tier: 4,427 model calls with 147.1M prompt tokens."
  - "Replay submits the exact prompt token IDs and constrains decoding, through a vLLM logits processor, to emit the recorded output tokens."
  - "A task's next call becomes eligible after its preceding response finishes and its recorded inter-call interval elapses."
  - "Each replay starts a fresh serving process and a cold cache."
  - — [HTML v1, App. B](https://arxiv.org/html/2609.33762v1)
- **(2) Tools.**
  - Live runs execute real tools in per-task sandboxes (App. A).
  - The replay waits for recorded intervals. An inter-call interval "includes agent computation, tool execution, transport, and any retries in that interval" (App. A, Timing boundary).
  - "All recorded intervals, including long waits, remain in the reported makespan" (App. B).
  - The intervals do not change with load.
  - — [HTML v1](https://arxiv.org/html/2609.33762v1)
- **(3) Model outputs.**
  - Outputs are teacher-forced: "Each completed run returns all 4,427 recorded sequences, every output token as recorded, without HTTP or prompt-length errors."
  - Quality is measured separately: "free-generation task quality is assessed separately in live execution".
  - Failed calls: "115 calls without a usable successful completion; their elapsed time remains in the intervals between retained requests."
  - — [HTML v1, App. B](https://arxiv.org/html/2609.33762v1)
- **(4) Workload sources (App. B).**
  - Totals: "4,427 successful recorded model calls, 147,124,984 prompt tokens, and 1,835,698 output tokens."
  - Per task: "56.0 calls (median 51, range 32–100), 1.86M prompt tokens, and 23.2K output tokens; the longest prompt has 238,105 tokens".
  - The cache-stable prompt length covers 98.1% of prompt tokens.
  - Prompts are not rewritten: "Our replay keeps every recorded prompt unchanged" (Sec. 2).
  - — [HTML v1](https://arxiv.org/html/2609.33762v1)
- **(5) Integration.**
  - OpenHands sends requests to vLLM. The replay submits token IDs over HTTP (App. B).
  - No agent hints cross the boundary. Admission is engine-side: "The runtime estimates A and N̄ of Equation 5 over a recent window, from the tasks with recent requests and their prompt lengths, and the host tier reports whether it is full and evicting" (Sec. 4).
  - — [HTML v1](https://arxiv.org/html/2609.33762v1)
- **(6) Load.**
  - The workload is closed loop with an active pool of A = 16 (or 8) tasks. The engine cap is "16 (max_num_seqs)" (App. A, Table 2).
  - Runs take 124–212 min of makespan (Sec. 5.2–5.3).
  - "Tasks enter the active pool in the recorded order" (App. B).
  - — [HTML v1](https://arxiv.org/html/2609.33762v1)
- **(7) Stack (Sec. 5.1; App. A, Table 2).**
  - Engine and hardware: vLLM 0.13.0 with prefix caching and LMCache 0.3.12, on 8×H20 with TP8.
  - GPU KV: 343,408 tokens (7.86 GiB per GPU).
  - Host tier: 3, 4.5, 5, 10, 20, 40 or 80 GiB per rank, in "1,024 tokens / sha256_cbor" chunks of 24 MiB per rank.
  - Other deployments: 8×RTX 3090, 2×H800 and 8×H800. The dense Qwen2.5-Coder-32B-Instruct runs on the RTX 3090 and 2×H800.
  - There is no SSD tier.
  - — [HTML v1](https://arxiv.org/html/2609.33762v1)
- **(8) Metrics (App. B).**
  - Computed prefill comes from "vLLM's request_prefill_kv_computed_tokens_sum"; preemptions from num_preemptions_total.
  - Host read/write volumes come from LMCache transfer-event token counts.
  - Also mean queue delay, makespan and prefix-hit rate.
  - — [HTML v1](https://arxiv.org/html/2609.33762v1)
- **(9) Baselines and ablations (App. B).**
  - Recompute: vLLM GPU prefix caching.
  - Offload: "the LMCache host tier without write admission".
  - Fixed write admission, and capacity-conditioned admission.
  - Sweeps cover host capacity, active pool (16 vs 8), GPU memory fraction (Fig. 7), prefix caching on or off (Table 9) and history condensers (App. G).
  - — [HTML v1](https://arxiv.org/html/2609.33762v1)
- **Repeats.** "Figures plot every run, including independent repeats of recomputation and of 40 GiB offload; the two recompute runs reproduce computed prefill to 0.2%" (Sec. 5.1). — [HTML v1](https://arxiv.org/html/2609.33762v1)
- **Headline numbers.**
  - From 5 to 20 GiB per rank, "computed prefill falls from 76.7M to 23.5M and 5.3M tokens (93.1%), and makespan ... from 209 to 161 and 128 min (38.7%)" (Sec. 5.2).
  - On H800, "offload lengthens every run we measured (1.08–1.87)" (Sec. 5.7).
  - — [HTML v1](https://arxiv.org/html/2609.33762v1)
- **Stack-distance predictions were recorded before the runs.** "We recorded stack-distance predictions before running 10, 20, and 80 GiB per rank at A=16 (Table 4) and 0, 3, and 4.5 GiB at A=8 (Table 5)" (Sec. 5.4). — [HTML v1](https://arxiv.org/html/2609.33762v1)
- **(10) Released artifacts.** "The write-admission connector, the dependency-preserving replay framework, and the analysis tools, including the stack-distance capacity model, are available at https://github.com/KunmingSHAO/efficientagent_release" (Reproducibility Statement). — [HTML v1](https://arxiv.org/html/2609.33762v1)
- **(11) Threats the authors state.**
  - "summing engine times over requests or ranks does not yield makespan" (App. A).
  - The replay "measures serving performance on fixed trajectories" (App. B).
  - Transfer-event counts "include repeated movements of the same token" (App. B).
  - — [HTML v1](https://arxiv.org/html/2609.33762v1)

### Inferences
- **Classification.** Live agents (i) produce the trajectories and the quality checks. The policy results come from teacher-forced, closed-loop trace replay on a real engine (ii).
- **Run labeling.** The cross-hardware rows of App. E are grouped by "workflow wall-clock, engine window, and reported inference time", and App. E mentions a "subset" and a "Complete SWE-bench Verified benchmark". This suggests those were live runs, not replays (inference). If so, offload and recompute arms there may differ in trajectories.
- **Task count (inference).** About 79 tasks (4,427 calls ÷ 56.0 calls per task).
- **Threats I notice.**
  - **Load-specific timing.** The replayed intervals were recorded under the no-host-tier run's load.
  - **Narrow coverage.** There is one benchmark, one agent framework and one temperature-0 model.
  - **No SSD.**
  - **Admission signal.** The working-set estimate needs task identity ("tasks with recent requests"), and the paper does not say how the engine gets it.

### Gaps
- **Task count.** The exact number of SWE-bench Verified instances is not stated.
- **Engine-side task identity.** How the engine learns which task a request belongs to is not stated.
- **Cross-hardware runs.** Whether they were replays or live is not stated.
- **Trace release.** Whether the recorded trajectories are released alongside the replay framework is not stated.

## 5. CacheScout (arXiv 2608.14624): evaluation setup

### Takeaway
CacheScout runs four multi-agent workloads (GSM8K, MT-Bench, GAIA, SWE-bench) on one six-agent AutoGen SelectorGroupChat supervisor framework "with tools". It runs them against patched vLLM v0.11 (Llama-3.1-8B; Qwen3-235B-A22B-FP8 at scale) and sweeps session arrival rates from 0.2 to 50 sessions/s.
- **Mode not stated.** The paper does not say whether sessions are live or replayed, whether tools execute, how many sessions run, or what the arrival process is.
- **Tiers.** The evaluation is about the GPU prefix cache under small block budgets. No SSD tier is evaluated.

### Cited Findings
- **Record.** "Learning Agent Execution for KV-Cache Management in Agentic Serving", Rui Zhang, Chaeeun Kim, Shaoting Feng et al. v1 only, 16 Jul 2026. — [abs](https://arxiv.org/abs/2608.14624); [HTML v1](https://arxiv.org/html/2608.14624v1)
- **(1) Mode (Sec. 5.1).**
  - "We evaluate CacheScout on four representative multi-agent workloads: GSM8K ..., MT-Bench ..., GAIA ..., and SWE-bench .... All four run on the same six-agent supervisor framework with tools and AutoGen's SelectorGroupChat for dynamic LLM-selected routing."
  - The motivation study is offline: "Figure 4(b) replays each workload's execution trace through an online first-order transition counter" (Sec. 2.2.2).
  - — [HTML v1](https://arxiv.org/html/2608.14624v1)
- **(2) Tools.** The framework is "with tools" (Sec. 5.1). Whether tools are executed, and their latency, are not stated. The Continuum baseline's TTL is set "to 0.3 s" "Following the observed inter-turn latency in AutoGen" (Sec. 5.1). — [HTML v1](https://arxiv.org/html/2608.14624v1)
- **(3) Model outputs.** The text does not say. Routing is "dynamic LLM-selected" (Sec. 5.1).
- **(4) Workload sources.** GSM8K, MT-Bench, GAIA and SWE-bench (Sec. 5.1). The motivation study also covers Pipeline, Debate, SelectorGroupChat and Random topologies (Sec. 2.2.2). Versions, subsets and counts are not stated. — [HTML v1](https://arxiv.org/html/2608.14624v1)
- **(5) Integration (Sec. 4).**
  - No framework cooperation is needed: "each request is fingerprinted from its own prefix block hashes, so a fingerprint change signals an agent dispatch and updates transition statistics and the block-to-agent mapping without any framework cooperation".
  - Prefetch is a normal request: Warmup "issues an inference request containing only the predicted agent's anchor with max_tokens=1, which the engine recognizes and excludes from transition learning".
  - It is "executed between agent turns through the standard serving API".
  - — [HTML v1, Sec. 4](https://arxiv.org/html/2608.14624v1)
- **(6) Load.**
  - "across the full arrival-rate sweep of Figure 11 (0.2–50 sessions/s)". For Qwen3-235B, "sweeping the arrival rate from 0.2 to 2.0 sessions/s" (Sec. 5.4).
  - GPU cache budget "varies from 100 to 200 blocks" in Fig. 14 (Sec. 5.4).
  - — [HTML v1](https://arxiv.org/html/2608.14624v1)
- **(7) Stack.**
  - Patch: vLLM v0.11 (V1 engine), with about 2,300 lines of runtime logic and an 800-line patch. The patch touches "the block pool, the engine core, the scheduler, and the scheduler- and worker-side halves of the CPU-offload connector" (Sec. 4).
  - Hardware and models: 8× RTX PRO 6000 Blackwell (96 GB) with Llama-3.1-8B-Instruct; Qwen3-235B-A22B-FP8 with TP over 4× H200 (Sec. 5.1).
  - CPU tier: CacheScout "reuses vLLM's existing LMCache and CPU offloading infrastructure", and eviction prefers "blocks whose contents are already resident in the CPU tier" (Sec. 4).
  - — [HTML v1](https://arxiv.org/html/2608.14624v1)
- **(8) Metrics (Sec. 5.1).** KV-cache hit rate ("total_cached_tokens / total_prompt_tokens"), TTFT (mean, median, P99, CDF), per-turn latency, and throughput in turns/s. Overheads are measured by "in-process microbenchmarks" (Sec. 5.5). — [HTML v1](https://arxiv.org/html/2608.14624v1)
- **(9) Baselines and ablations.**
  - Vanilla vLLM, "reactive prefix caching with LRU eviction" (Sec. 5.1).
  - Continuum, "TTL-based pinning around tool-call boundaries" (Sec. 5.1).
  - The Qwen3-235B study uses vanilla vLLM only, "as Continuum's vLLM fork does not support this model" (Sec. 5.1).
  - Ablations: eviction-only versus full (Sec. 5.3), and the adaptive prefetch gate on or off (Sec. 5.4).
  - — [HTML v1](https://arxiv.org/html/2608.14624v1)
- **Headline numbers.**
  - Hit rate rises "by 10–18 percentage points" over vanilla vLLM, and mean TTFT falls by 18–45% (Sec. 5.2).
  - Prefetch alone "is inert on these workloads", adding "at most one percentage point" (Sec. 5.3).
  - — [HTML v1](https://arxiv.org/html/2608.14624v1)
- **(10) Released artifacts.** "CacheScout has been implemented on top of vLLM and will be open-sourced along with the benchmark suite" (Sec. 7). No link is given. — [HTML v1](https://arxiv.org/html/2608.14624v1)
- **(11) Threats the authors state.** Budgets "below the workload's ~93-block maximum request footprint cannot admit every request and are excluded" (Sec. 5.4). There is no limitations section. — [HTML v1](https://arxiv.org/html/2608.14624v1)

### Inferences
- **Classification.** This is likely live multi-agent sessions on a real engine (i), with dataset-driven synthetic session arrivals (iv). Inference: the "LLM-selected routing" and the per-arm Continuum fork imply free generation. If so, each arm's agent trajectories can differ.
- **Short tool gaps.** A 0.3 s observed inter-turn latency means tool gaps are almost negligible. The benefit therefore comes from protecting shared agent anchors (system prompts and tool definitions) under small GPU budgets, not from surviving long tool waits.
- **No variance.** No repeats or variance are reported.

### Gaps
- **Unstated basics.**
  - Number of sessions per workload.
  - Session arrival process (Poisson or otherwise).
  - Run length.
  - Whether tools are real, mocked or replayed.
  - Output control (sampling, max tokens).
  - Whether CPU offload is enabled in the evaluated configurations.

## 6. TempoKV (arXiv 2609.35065): evaluation setup

### Takeaway
TempoKV uses a synthetic long-document QA workload on real hardware.
- **Stack.** Modified vLLM v0.23.0 + LMCache v0.5.1 run on one H100. The fast tier is a CXL memory-semantic flash device (XCENA MX1P: 128 GiB device DRAM backed by a 15.36 TB NVMe), with precomputed KV staged from SSD.
- **Workload.** It is tiny and has no agents or sessions: 12 NarrativeQA documents at 0.5 s fixed intervals, or 16+16 NarrativeQA/LooGLE contexts at 0.2–4 req/s.
- **Engine limits.** At most four active sequences, and GPU prefix caching disabled.

### Cited Findings
- **Record.** "TempoKV: Timely Staging of LLM KV Caches for Memory-Semantic Flash", Jay H. Park, Hyungjun Kim, Dong Kim et al. v1 only, 28 Sep 2026. — [abs](https://arxiv.org/abs/2609.35065); [HTML v1](https://arxiv.org/html/2609.35065v1)
- **(1) Mode, main policy experiments (Sec. 5.1).**
  - "We use 12 full NarrativeQA documents, four from each of three length groups, with one fixed question per document."
  - Prefix cache ratio is varied "across 50%, 75%, and 100%".
  - "Requests arrive at fixed 0.5-s intervals, independently of response completion, without cache resets between requests."
  - — [HTML v1, Sec. 5.1](https://arxiv.org/html/2609.35065v1)
- **(1) Mode, DAX comparison (Sec. 5.3).**
  - "We randomly sample 16 contexts from each of NarrativeQA and LooGLE, using one associated question per context."
  - "requests arrive at fixed intervals of 1/r s, independently of response completion, with the offered request rate r ranging from 0.2 to 4 requests/s."
  - — [HTML v1, Sec. 5.3](https://arxiv.org/html/2609.35065v1)
- **(2) Tools.** None. No agent or multi-turn workload is used. — [HTML v1, Sec. 5](https://arxiv.org/html/2609.35065v1)
- **(3) Model outputs.** "Generation is capped at 128 output tokens with natural end-of-sequence termination" (Sec. 5.1); the cap is 32 tokens in Sec. 5.3. — [HTML v1](https://arxiv.org/html/2609.35065v1)
- **(4) Workload sources.** NarrativeQA and LooGLE. Input lengths span "16,078–67,744 tokens across the two tokenizers" (Sec. 5.1). — [HTML v1](https://arxiv.org/html/2609.35065v1)
- **Pre-run state.**
  - "Before each run, we retain precomputed KV on SSD, reset the fast tier, and initialize predictors from the same model-specific calibration" (Sec. 5.1).
  - KV is kept "in complete 256-token LMCache chunks" (Sec. 5.1).
  - — [HTML v1](https://arxiv.org/html/2609.35065v1)
- **(5) Integration (Sec. 4).**
  - "After each scheduling iteration, a runtime adapter computes TTU from a bounded scheduler snapshot and sends only claim metadata to the controller in LMCache's multiprocess service."
  - TTU uses "calibrated execution costs", and "Output-length estimates come from recent completed requests" (Sec. 3.1).
  - No client-side hints are used.
  - — [HTML v1](https://arxiv.org/html/2609.35065v1)
- **(6) Load.** Arrivals are open loop at fixed intervals (above). "We use vLLM's FCFS with up to four active sequences and GPU prefix caching disabled" (Sec. 5). — [HTML v1](https://arxiv.org/html/2609.35065v1)
- **(7) Stack (Sec. 5).**
  - Host and GPU: a 72-core Xeon Granite Rapids with one H100 PCIe (80 GB).
  - Flash device: 128 GiB DRAM "backed by a 15.36 TB Samsung PM1753 NVMe SSD", connected over CXL ×8.
  - Fast-tier budget: "a 100 GiB device-DRAM fast tier with a protected-capacity budget of half that capacity". The sensitivity runs use 100, 50 and 25 GiB.
  - Models: Qwen2.5-14B-Instruct (YaRN ×4) and Llama-3.1-8B-Instruct, in BF16.
  - — [HTML v1](https://arxiv.org/html/2609.35065v1)
- **(8) Metrics.** p95 TTFT, output throughput, and protected-capacity cost C = (∫P(t)dt)/N in GiB·s/request. "C measures reservation, not physical DRAM occupancy" (Sec. 5). — [HTML v1](https://arxiv.org/html/2609.35065v1)
- **(9) Baselines and ablations.**
  - Demand, Immediate, Queue-K (K ∈ {1, 4}), TempoKV-Static (a fixed "approximately 11 GiB/s" staging rate) and TempoKV (Sec. 5.1).
  - LMCache-DAX, "unmodified LMCache without our runtime adapter, controller, or staging provider" (Sec. 5.3).
  - — [HTML v1](https://arxiv.org/html/2609.35065v1)
- **Headline numbers.**
  - C falls by 63–91% versus Immediate (Sec. 5.1).
  - Against LMCache-DAX, p95 TTFT falls by up to 48.0% and throughput rises by up to 27.8% on NarrativeQA (Sec. 5.3).
  - — [HTML v1](https://arxiv.org/html/2609.35065v1)
- **(10) Released artifacts.** No code release statement was found. The only URL is the device vendor's documentation. — [HTML v1](https://arxiv.org/html/2609.35065v1)
- **(11) Threats the authors state.** There is no limitations section.

### Inferences
- **Classification.** This is a synthetic workload (iv) on a real engine and a real SSD-backed device. With FlatCake, it is one of only two papers of the ten whose results come from measured SSD reads.
- **Threats I notice.**
  - Each run issues 12–32 requests.
  - Four active sequences is a very low concurrency.
  - Prefix caching is off on the GPU.
  - Arrivals are deterministic and open loop.
  - There are no idle gaps, so TTU is pure queueing time, not tool or think time.
  - No repeated runs are reported.
  - It transfers poorly to agent sessions, where time-to-use is dominated by tool duration.

### Gaps
- **Unstated basics.** Number of runs per point, variance, and any code or artifact availability.

## 7. FlatCake (arXiv 2610.38355): evaluation setup

### Takeaway
FlatCake replays a public Claude Code agentic trace (393 sessions, 98,827 requests, 21.6B input tokens) on SGLang v0.5.19 HiCache.
- **Stack.** DeepSeek-V4-Flash on 8×B200, with 1 TiB host DRAM and 4 TiB NVMe budgets, against HiCache-L2 and Mooncake-L3 baselines.
- **Replay.** Requests are sent "in recorded arrival order" at fixed rates of 0.1–1.0 QPS with no warm-up. Recorded inter-arrival timing, session think time and tool gaps are therefore not reproduced (inference).
- **Unstated.** Prompt synthesis, output control and how much of the trace each run covers are not stated.

### Cited Findings
- **Record.** "FlatCake: A Flat KVCache Storage Architecture for LLM Serving", Xin Huang, Teng Ma, Zhuoran Li et al. v1 only, 28 Oct 2026. Comment: "10 figures". — [abs](https://arxiv.org/abs/2610.38355); [HTML v1](https://arxiv.org/html/2610.38355v1)
- **(1) Mode (Sec. 5.1).**
  - "We replay the production claude-code-weka-0621 agentic coding trace (SemiAnalysis, 2026), comprising 98,827 requests and 21.6B input tokens across 393 multi-turn sessions (Table 1)."
  - "Requests are replayed in recorded arrival order across arrival rates from 0.1 to 1.0 QPS without cache warm-up, following the SGLang HiCache benchmark methodology".
  - The Llama-3.1-70B study sweeps "input lengths from 8K to 128K tokens" (Sec. 5.4).
  - — [HTML v1](https://arxiv.org/html/2610.38355v1)
- **Trace provenance.** The reference reads "CC Traces — Weka, With Subagents (v7)", https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126, "Claude Code agentic coding traces; accessed 2026-07-28". The trace's native block size is 64 tokens (Sec. 2). — [HTML v1, references and Sec. 2](https://arxiv.org/html/2610.38355v1)
- **(2) Tools.** Not stated. — [HTML v1](https://arxiv.org/html/2610.38355v1)
- **(3) Model outputs.** Not stated. — [HTML v1](https://arxiv.org/html/2610.38355v1)
- **(4) Workload sources.**
  - Main workload: the Claude Code trace above.
  - Characterization: eight production traces (Mooncake Conversation, Synthetic, ToolAgent; Qwen-Bailian to-C, to-B API, Thinking, Coder; the agentic trace) "under an infinite-capacity cache" (Table 1).
  - — [HTML v1](https://arxiv.org/html/2610.38355v1)
- **(5) Integration.**
  - FlatCake sits "behind the standard HiCacheStorage interface (exists, get, put) without modifying model runner or attention kernels" (Sec. 4).
  - Write-time placement uses engine-observable signals: radix-tree reference count, "session turn > 1 with prior output length < τ tokens", and "session context ≥ L_agent" (Sec. 3.2, Table 2).
  - The block key includes the session ID (Sec. 3.2).
  - — [HTML v1](https://arxiv.org/html/2610.38355v1)
- **(6) Load.** QPS points are 0.1, 0.25, 0.5, 0.75 and 1.0 (Fig. 8 caption), with no warm-up (Sec. 5.1). — [HTML v1](https://arxiv.org/html/2610.38355v1)
- **(7) Stack (Sec. 5.1).**
  - Hardware: 8-GPU B200 server, "3 TB of host DRAM and an 8-drive Samsung PM9A3 NVMe SSD RAID 0 array".
  - Budget: "an identical KV cache budget of 1 TiB host DRAM and 4 TiB NVMe SSD".
  - Software: "SGLang v0.5.19, Mooncake v0.3.13, CUDA 13.1, and NVIDIA GPUDirect Storage (GDS) 1.16.1".
  - Model: DeepSeek-V4-Flash-0731 with "an fp8_e4m3 KV cache with 256-token pages and a per-rank footprint of 7.5 KiB/token".
  - Second setup: Llama-3.1-70B, TP4 on 4×A100.
  - — [HTML v1](https://arxiv.org/html/2610.38355v1)
- **(8) Metrics (Sec. 5.1).**
  - Mean and median TTFT, and total token throughput.
  - "delivered cache hit rate (counting only blocks transferred to GPU HBM before prefill)".
  - SSD read bandwidth, SSD prefetch latency, and normalized SSD read/write amplification.
  - Per-medium hit attribution (Fig. 7), and P99 TTFT in the FC-MAS ablation (Sec. 5.3).
  - — [HTML v1](https://arxiv.org/html/2610.38355v1)
- **(9) Baselines and ablations.**
  - "+L2 Host DRAM": "SGLang HiCache's two-tier HBM–DRAM LRU cache without SSD offloading".
  - "+L3 Mooncake": "SSD-resident KV blocks are staged through host DRAM buffers before H2D transfer".
  - The Llama sweep adds Recompute-only and Mooncake-GDS (Sec. 5.1).
  - Ablation: FlatCake-w/o-FC-MAS (Sec. 5.3).
  - — [HTML v1](https://arxiv.org/html/2610.38355v1)
- **Headline numbers (Sec. 5.2).**
  - FlatCake's delivered hit rate is "94.16–94.57%" against a "94.72% prefix-reuse ceiling".
  - Mooncake falls from 87.9% at 0.1 QPS to 64.26% at ≥0.5 QPS. L2 plateaus at 53.30%.
  - Mechanism claimed: "When staging memory saturates, SGLang revokes the blocked prefetch and falls back to recomputing the prefix on the GPU".
  - — [HTML v1](https://arxiv.org/html/2610.38355v1)
- **(10) Released artifacts.** No code release statement was found in the text, and the abs page has no code link. — [HTML v1](https://arxiv.org/html/2610.38355v1); [abs](https://arxiv.org/abs/2610.38355)
- **(11) Threats the authors state.** None stated. There is no limitations section.

### Inferences
- **Classification.** This is trace replay on a real engine (ii), but rate-scaled. Recorded arrival order is kept while timing is replaced by a fixed QPS. Session think and tool gaps, which set how long a prefix cache must survive, are therefore not reproduced (inference).
- **Replay length (inference).** Replaying all 98,827 requests would take about 27 h at 1.0 QPS and about 11 days at 0.1 QPS. Each run probably replays a subset, but the text does not say so.
- **Baseline artifact.** The Mooncake collapse is attributed to SGLang's bounded staging budget and prefetch revocation. That is a configuration-sensitive baseline behavior, and the HiCache prefetch settings (policy, timeout) are not reported.
- **Prompt content.** Prompts must be synthesized from the trace's hashed blocks (inference).
- **No variance.** No repeats or variance are reported.

### Gaps
- **Unstated basics.**
  - Prompt construction from the trace.
  - Output-length control.
  - Number of requests or sessions replayed per QPS point, and run duration.
  - Whether inter-turn gaps or session concurrency limits are applied.
  - How session IDs reach the engine.
  - HiCache and Mooncake baseline settings.

## 8. Where Should the KV Cache Live? (arXiv 2609.16215): evaluation setup

### Takeaway
This is a simulator study, run at batch one on a single GPU.
- **Workloads.** Three synthetic generators: long chat, an "agent loop", and document QA.
- **Hardware, model, calibration.** None of them is stated, although the conclusion calls the simulator "calibrated".
- **Cost.** It uses assumed $/GB prices.
- **Prefetch result.** The finding that prefetch does not pay comes from a model that charges speculative fetches on the same contended link.

### Cited Findings
- **Record.** "Where Should the KV Cache Live? Placement Policies Across GPU, CPU, and SSD for Long-Lived Sessions", Srikanta Datta Tumkur, Jay Iyer, Mehar Simhadri et al. v1 only, 14 Sep 2026. The acknowledgment thanks "Vizuara AI Labs for mentorship and computational resources". — [abs](https://arxiv.org/abs/2609.16215); [HTML v1](https://arxiv.org/html/2609.16215v1)
- **(1) Mode.**
  - "All numbers in this section are measured by the simulator" (Sec. V).
  - "The policy comparison runs on synthetic generators whose reuse structure is, by construction, favourable to the policy that matches it" (Sec. VI).
  - "The simulator runs at batch one on a single GPU" (Sec. VI).
  - The conclusion says "built a calibrated tiered simulator" (Sec. VII).
  - — [HTML v1](https://arxiv.org/html/2609.16215v1)
- **(2)–(3) Tools and outputs.** There are no tools and no generation. The "agent loop" is an access pattern that "revisits recent context" (Table I). — [HTML v1](https://arxiv.org/html/2609.16215v1)
- **(4) Workloads.**
  - "long multi-turn chat, an agent loop that revisits recent context, and repeated document QA with heavy prefix reuse" (Sec. IV-A).
  - "The policy grid runs chat as six concurrent sessions of 4,000 accesses rather than the canonical single 24,000-access session" (Sec. IV-B).
  - — [HTML v1](https://arxiv.org/html/2609.16215v1)
- **(5) Integration.** None. Policies run inside the simulator. — [HTML v1](https://arxiv.org/html/2609.16215v1)
- **(6) Load.** Batch one. Six concurrent chat sessions in the policy grid (Sec. IV-B). — [HTML v1](https://arxiv.org/html/2609.16215v1)
- **(7) Stack.**
  - The default tier shape is 1:8:64 (Sec. V). Budgets are in blocks: T = 192 for the frontier; HBM 32, DRAM 64 and SSD 8–1024 for the SLO study (Sec. V-F, V-G).
  - Compute is "roughly 13.2445 ms/token" (Sec. V-D).
  - Costs are "$187.50 / $3.50 / $0.08 for HBM/DRAM/SSD", "assumptions carried from cost_model.py's default tiers, not a market quote" (Sec. V-F).
  - — [HTML v1](https://arxiv.org/html/2609.16215v1)
- **(8) Metrics.** Per-tier hit rate, TTFT (P99), TPOT and its stall component, sessions per GPU, throughput, cost per session (Sec. IV-A), and PCIe migration traffic in GB (Fig. 5). — [HTML v1](https://arxiv.org/html/2609.16215v1)
- **(9) Baselines and ablations.**
  - Baselines: "GPU-only (recompute on resume), full-CPU-offload (FlexGen style), and prefix-reuse (RadixAttention style)".
  - Policies: recency, reuse frequency, "predicted-reuse" (found "byte-identical to recency") and an EWMA "predictive" policy (Sec. IV-B, V-C).
  - Prefetch arms: "none, a causal predictor that uses only past accesses, and a future-reading oracle ... on a bandwidth-charged model where a speculative fetch competes with demand traffic on the same link" (Sec. V-E).
  - — [HTML v1](https://arxiv.org/html/2609.16215v1)
- **Headline numbers.**
  - Tiering yields "73.02× more sessions per GPU, 62.04× lower cost", which the authors call "mechanism wins" (Sec. VI).
  - Prefetch results are "uniformly negative" (Sec. V-E).
  - — [HTML v1](https://arxiv.org/html/2609.16215v1)
- **(10) Released artifacts.** The text names repository files (results/metrics_all.csv, paper/make_figures.py, cost_model.py) but gives no URL. — [HTML v1](https://arxiv.org/html/2609.16215v1)
- **(11) Threats the authors state (Sec. VI).**
  - The synthetic separations are "upper bounds".
  - At batch one, "fetch stall is under one percent of TPOT".
  - Sessions per GPU and cost are "capacity-model outputs from fixed tier prices".
  - Stall attribution "is attributed by hit share, not directly measured".
  - — [HTML v1](https://arxiv.org/html/2609.16215v1)

### Inferences
- **Classification.** Simulation (iii) on synthetic workloads (iv). The sessions-per-GPU gain equals the tier capacity multiple by construction, as the authors note.
- **What "calibrated" means is unknown.** Nothing in the text says what was calibrated or against what: no GPU, model, bandwidths or engine.
- **Untraceable baseline.** The "roadmap" decision table being tested is not identified in the text.
- **Weak counter-evidence.** The paper is weak as counter-evidence against prefetch for agent sessions. Its finding rests on batch one, a single contended link and synthetic reuse.

### Gaps
- **Unstated basics.**
  - Simulator calibration source.
  - Tier bandwidths and latencies.
  - Model, block size and session lengths in tokens.
  - Repository location.
  - Source of the "roadmap".

## 9. Cheng et al., critical-fractile KV reservation (arXiv 2607.16892): evaluation setup

### Takeaway
This is an analytic simulation with no engine and no agents.
- **Scale.** A simulated 8×A100-80GB cluster serves a 70B model.
- **Queueing model.** Queueing delay comes from M/G/1 Pollaczek–Khinchin, and preemption is charged as a 2.5× service-time penalty.
- **Workload.** Output-length distributions come from BurstGPT, Azure LLM 2024 and ShareGPT (10,000 requests per class).
- **Not modeled.** KV here is reserved per request for decoding. Multi-turn or session retention is not modeled.

### Cited Findings
- **Record.** "Robust KV Cache Management for LLM Serving under Output Token Length Uncertainty", Jiaming Cheng, Duong The Do, Duong Tung Nguyen. v1 only, 18 Jul 2026; "10 figures, 10 pages". — [abs](https://arxiv.org/abs/2607.16892); [HTML v1](https://arxiv.org/html/2607.16892v1)
- **(1) Mode.**
  - "We simulate J=8 A100-80GB GPUs serving a 70B model with configurations: k1: (TP=2, PP=1), k2: (TP=4, PP=1), k3: (TP=2, PP=2), k4: (TP=4, PP=2)" (Sec. V-A).
  - "The simulator processes 10,000 requests per class, computes per-group queueing delay using the M/G/1 Pollaczek–Khinchin model under realized memory utilization, and applies a 2.5× service-time penalty for preempted requests to capture KV reload and decode replay overhead" (Sec. V-D).
  - — [HTML v1](https://arxiv.org/html/2607.16892v1)
- **Service-time model.** It uses "α, β ... model-specific constants", plus layer count and all-reduce latency (Sec. III-A). Values and calibration are not given. — [HTML v1](https://arxiv.org/html/2607.16892v1)
- **(2)–(3) Tools and outputs.** None. Output lengths are drawn from trace distributions. Shift experiments scale them by 1.0–3.0× (Sec. V-E). — [HTML v1](https://arxiv.org/html/2607.16892v1)
- **(4) Workload sources (Sec. V-A).**
  - BurstGPT: "1.4M requests over 61 days"; output mean 125, P90 276, P99 1,586.
  - Azure LLM 2024: "44M requests over 7 days", with Code and Conversation classes.
  - ShareGPT: "368K assistant responses".
  - — [HTML v1](https://arxiv.org/html/2607.16892v1)
- **(5) Integration.** None. Decisions are precomputed: "The online phase is O(1) per request" (Sec. IV-C). — [HTML v1](https://arxiv.org/html/2607.16892v1)
- **(6) Load.** Arrival rate is scaled by α_λ (Fig. 3a). The ablation uses "load α_λ=0.6". The production-scale solve uses "15 request classes, 12 candidate parallelism configurations, and N=2000 historical samples per class on a 48-GPU cluster" (Sec. V-B, V-E). — [HTML v1](https://arxiv.org/html/2607.16892v1)
- **(7) Stack.** Simulated only. Cost knobs: "c_p/c_w=10, c_slo/c_w=5, c_rej/c_w=5000, κ=0.2", and ε = 0.15·E[ξ] (Sec. V-A). — [HTML v1](https://arxiv.org/html/2607.16892v1)
- **(8) Metrics.** Per-arrival preemption/waste cost, solve time, P99 latency, and SLO violation rate (Sec. V-B to V-E). — [HTML v1](https://arxiv.org/html/2607.16892v1)
- **(9) Baselines and ablations.**
  - Baselines: Max, Mean, P90, P95, P99, SAA (ε = 0) and LP + kσ (k ∈ {1, 2}); direct MIBLP for scalability.
  - Ablations: uniform routing, no DRO, no prefix caching.
  - Table V reports "mean ± std over 5 seeds".
  - — [HTML v1](https://arxiv.org/html/2607.16892v1)
- **(10) Released artifacts.** No code statement was found. — [HTML v1](https://arxiv.org/html/2607.16892v1)
- **(11) Threats the authors state.**
  - At ρ = 10, "the empirical-quantile baselines and DRO produce near-indistinguishable simulator results on mild temporal splits of BurstGPT and Azure-Conv" (Sec. V-C).
  - Future work includes "integration with production serving runtimes" (Sec. VI).
  - — [HTML v1](https://arxiv.org/html/2607.16892v1)

### Inferences
- **Classification.** Simulation (iii). The arrival process is not named, but M/G/1 implies Poisson arrivals in the queueing model (inference).
- **Narrow relevance to agent KV tiering.** Only the newsvendor ratio carries over. Workload, metric and engine behavior do not.

### Gaps
- **Unstated basics.** Values or fits of α and β, the simulator's code, and the arrival process.

## 10. Online LP routing with KV-memory shadow prices (arXiv 2607.03948): evaluation setup

### Takeaway
This is a simulation in Vidur, chosen "Due to lack of GPU resources".
- **Setup.** Four simulated A100s. Requests are sampled from LMSYS-Chat-1M with Poisson arrivals (λ = 0.4 or 0.5), plus synthetic prefill/decode mixes and a rate-shift test.
- **Decode lengths.** These come from the data. Predictions are exact or ±20%.
- **Not modeled.** There are no agents, sessions, prefix reuse or KV tiers.

### Cited Findings
- **Record.** "Online Linear Programming for Multi-Objective Routing in LLM Serving", Zixi Chen, Yinyu Ye, Zijie Zhou. v1 only, 4 Jul 2026. — [abs](https://arxiv.org/abs/2607.03948); [HTML v1](https://arxiv.org/html/2607.03948v1)
- **(1) Mode (Sec. 4).**
  - "Due to lack of GPU resources, we conducted our experiments using the Vidur simulator ..., which models realistic LLM serving behavior including continuous batching and KV-cache management."
  - "We simulate a scenario with four parallel NVIDIA A100 GPUs" (Sec. 4.1).
  - — [HTML v1](https://arxiv.org/html/2607.03948v1)
- **(2)–(3) Tools and outputs.**
  - There are no tools.
  - Predicted decode lengths are either "Perfect Prediction ... ô_j = o_j" or "20% prediction error, where ô_j ~ Unif(0.8·o_j, 1.2·o_j)" (Sec. 4.2).
  - — [HTML v1](https://arxiv.org/html/2607.03948v1)
- **(4) Workload sources.**
  - LMSYS-Chat-1M; "Each arriving request is randomly sampled from the dataset" (Sec. 4.1).
  - Synthetic "P/D ratio 1:4" and "4:1" (Sec. 4.1, App. C.2).
  - — [HTML v1](https://arxiv.org/html/2607.03948v1)
- **(6) Load.**
  - "a stationary Poisson process, with an arrival rate λ ∈ {0.4, 0.5}". λ = 0.4 is stable and 0.5 is "a slight overload" (Sec. 4.1).
  - The non-stationary test switches "from λ=0.4 to λ=0.6 at the simulation midpoint" (Sec. 4.3).
  - — [HTML v1](https://arxiv.org/html/2607.03948v1)
- **Model scope (Sec. 2).** Requests are routed "to parallel decode workers under tight batch-size and KV-cache constraints". The KV constraint is a per-device budget M: "otherwise, the system must swap requests out and stall compute". — [HTML v1](https://arxiv.org/html/2607.03948v1)
- **(8) Metrics.** Average and tail TTFT, average and tail end-to-end latency, throughput and QPS (Sec. 2, 4.3). — [HTML v1](https://arxiv.org/html/2607.03948v1)
- **(9) Baselines.** "Round Robin", "Least Outstanding Request (LOR)", "Random" and "Power-of-2 Choices". Objective weights are varied (Sec. 4.2). — [HTML v1](https://arxiv.org/html/2607.03948v1)
- **(10) Released artifacts.** "Our routing algorithm has been integrated into the Vidur simulator, and the code is available at https://github.com/qqwetidx/Online-Linear-Programming-for-Vidur" (Sec. 4). — [HTML v1](https://arxiv.org/html/2607.03948v1)
- **(11) Threats the authors state.** "Our validation relies on simulation using the Vidur framework; real system implementation would face additional challenges including integration with serving engines (e.g., vLLM), handling of preemption and KV-cache swapping, and coordination overhead in distributed deployments" (Sec. 6). — [HTML v1](https://arxiv.org/html/2607.03948v1)

### Inferences
- **Classification.** Simulation (iii) with a synthetic arrival process over a chat dataset (iv).
- **Calibration rests on Vidur.** Vidur is profiling-based in general, but the paper does not name the profiled model.
- **Narrow transfer.** The KV-memory "shadow price" is a decode-memory congestion price for routing new requests. It says nothing about retained prefix caches.

### Gaps
- **Unstated basics.**
  - Model simulated in Vidur.
  - Value of M.
  - Number of requests per run.
  - Whether the Vidur configuration models PD disaggregation as Sec. 2 assumes.
  - Number of repeated runs.

## 11. Comparison across the ten papers

### Takeaway
- **Live agents for KV-cache results.** Only EfficientAgent clearly produces them, and even there the policy comparisons use a teacher-forced replay of one live run. CacheScout may run live agents, but the paper does not say.
- **The common real-engine design.** Trace replay with tool time injected as recorded sleeps (KVTether, EfficientAgent, HEAR's Mooncake runs).
- **The SSD tier.**
  - Only FlatCake measures an SSD tier under an agentic trace, and it drops the trace's timing.
  - TempoKV measures SSD staging, but on non-agent synthetic requests.
  - The simulation papers (Where Should the KV Cache Live?, Cheng et al., LP routing) have no real engine.
- **Tool latency.** No paper makes tool latency depend on load.

### Cited Findings

| # | Paper | Mode (which results) | Engine, model, KV tiers | Tools / gaps | Output control | Workload and size | Load | Main metrics | Released |
|---|---|---|---|---|---|---|---|---|---|
| 1 | HEAR | (iv)+(ii): SCBench sessions recorded once, then replayed; (ii): Mooncake trace replay; (i): live LangGraph agents for role-mode choice only | vLLM (version unstated for KV runs; 0.26.0 for live runs); Qwen3-8B on 1 GPU; GPU + 96 GiB CPU "L2"; no SSD | No tools; N(5,1) s think time or recorded gaps; live runs use native retrieval | Recorded input/output lengths; live runs at temperature 0 | SCBench (count unstated); Mooncake sessions starting in one 15-min window (50/75/100%); 208 BCP + 100 DRB live tasks | Poisson session arrivals; recorded starts | TTFT P50/P95/max, batch completion, session time, turns/min, reuse % | "supplementary artifact"; no URL |
| 2 | KVTether | (ii): replay on real cluster; Codex traces collected live | SGLang 0.5.8 PD-disagg, LMCache 0.5.5, Mooncake 0.3.8; GLM-4.7 358B on 12 GB200; 400 GB host pool per node + remote; no SSD | Sleeps of recorded duration; lifecycle events replayed in order | Random prompts with recorded input counts and matched reusable-prefix lengths; recorded output counts enforced | Codex/TerminalBench (count unstated); Agent-X and Agent-G, 400 tasks each | Poisson task arrivals at 5 RPS points (~0.05–0.06 RPS representative) | Hit ratio, cumulative prefill, task end-to-end latency, $ per task | None found |
| 3 | Storage View | Offline characterization and counterfactual TTL accounting; no engine; ODR trace collected live | None (ODR collected on vLLM, Qwen3-8B, 1 A100); 144 KiB/token for bytes | Exgentic runs carry real tool latency; ODR uses real Tavily | n/a | 11 traces / 15 workloads (e.g., SWE-bench 1,959 sessions, TraceLab 4,281, ODR 50) | Fixed-concurrency replay of N live sessions | Reuse scope, recurrence gaps, footprint, retained hits, dead state | None found |
| 4 | EfficientAgent | (i): live OpenHands on SWE-bench Verified; (ii): token-exact closed-loop replay of one live run | vLLM 0.13.0 + LMCache 0.3.12; Qwen3-Coder-30B-A3B on 8×H20 TP8 (+RTX 3090, H800); GPU + host 3–80 GiB/rank; no SSD | Live: real tools in sandboxes; replay: recorded inter-call interval after each response | Exact prompt token IDs; logits processor forces recorded outputs | 4,427 calls, 147.1M prompt tokens, ~56 calls per task | Closed loop, A = 16 or 8 active tasks; engine cap 16 | Computed prefill tokens, makespan, host writes/restores, preemptions | Code, replay framework, model on GitHub |
| 5 | CacheScout | Multi-agent runs on vLLM; live vs replayed not stated | Patched vLLM v0.11; Llama-3.1-8B on RTX PRO 6000; Qwen3-235B-A22B-FP8 on 4×H200; GPU prefix cache (CPU offload reuse unclear) | "with tools"; execution unstated; ~0.3 s inter-turn latency | Not stated | GSM8K, MT-Bench, GAIA, SWE-bench (counts unstated) | 0.2–50 sessions/s (0.2–2.0 at 235B) | Hit rate, TTFT mean/P50/P99, per-turn latency, turns/s | "will be open-sourced" |
| 6 | TempoKV | (iv): synthetic long-document QA on real engine and CXL flash | Modified vLLM 0.23.0 + LMCache 0.5.1; 1 H100; CXL device with 128 GiB DRAM over 15.36 TB NVMe | None | Capped 128 or 32 tokens, natural EOS | 12 NarrativeQA docs; 16 NarrativeQA + 16 LooGLE | Open loop, fixed intervals (0.5 s; 0.2–4 req/s); ≤4 active sequences | p95 TTFT, output throughput, GiB·s/request protected | None found |
| 7 | FlatCake | (ii): rate-scaled replay of Claude Code trace; (iv): context-length sweep | SGLang 0.5.19 HiCache + Mooncake 0.3.13; DeepSeek-V4-Flash on 8×B200; 1 TiB DRAM + 4 TiB NVMe RAID0; GDS | Not reproduced (fixed QPS; inference) | Not stated | 393 sessions, 98,827 requests, 21.6B tokens (amount per run unstated) | Recorded order at 0.1–1.0 QPS; no warm-up | Mean/P50/P99 TTFT, token throughput, delivered hit rate, SSD BW, prefetch latency, R/W amplification | None found |
| 8 | Where Should the KV Cache Live? | (iii)+(iv): simulator with synthetic generators | None; batch one, single GPU; tiers 1:8:64; hardware and model unstated | None | n/a | 3 synthetic workloads; chat as 6 × 4,000 accesses | 6 concurrent sessions in grid | Per-tier hits, P99 TTFT, TPOT stall, sessions/GPU, $/session, PCIe GB | Repo files named; no URL |
| 9 | Cheng et al. | (iii): analytic M/G/1 simulator with trace output-length distributions | Simulated 8×A100-80GB, 70B; GPU KV reservation only | None | Sampled lengths | BurstGPT, Azure 2024, ShareGPT; 10,000 requests per class | Arrival-rate scaling α_λ | Preemption/waste cost, P99, SLO violation, solve time | None found |
| 10 | LP routing | (iii): Vidur simulator | Simulated 4×A100 (model unstated); decode KV budget M | None | Dataset lengths; predictions exact or ±20% | LMSYS-Chat-1M samples; synthetic P/D mixes | Poisson λ = 0.4 or 0.5; shift 0.4→0.6 | Avg/tail TTFT, avg/tail end-to-end latency, throughput | GitHub (Vidur integration) |

- **Table sources.** Each row draws on that paper's sections above:
  - HEAR — [HTML v1](https://arxiv.org/html/2610.06597v1)
  - KVTether — [HTML v1](https://arxiv.org/html/2609.39819v1)
  - Storage View — [HTML v1](https://arxiv.org/html/2611.10178v1)
  - EfficientAgent — [HTML v1](https://arxiv.org/html/2609.33762v1)
  - CacheScout — [HTML v1](https://arxiv.org/html/2608.14624v1)
  - TempoKV — [HTML v1](https://arxiv.org/html/2609.35065v1)
  - FlatCake — [HTML v1](https://arxiv.org/html/2610.38355v1)
  - Where Should the KV Cache Live? — [HTML v1](https://arxiv.org/html/2609.16215v1)
  - Cheng et al. — [HTML v1](https://arxiv.org/html/2607.16892v1)
  - LP routing — [HTML v1](https://arxiv.org/html/2607.03948v1)
- **Hit-rate definitions differ.**
  - HEAR: "KV-cache reuse rate for follow-up prompt tokens" — [HEAR](https://arxiv.org/html/2610.06597v1)
  - KVTether: "cache-hit input tokens divided by the total number of input tokens" — [KVTether](https://arxiv.org/html/2609.39819v1)
  - CacheScout: "total_cached_tokens / total_prompt_tokens" — [CacheScout](https://arxiv.org/html/2608.14624v1)
  - FlatCake: "delivered" hits, "counting only blocks transferred to GPU HBM before prefill" — [FlatCake](https://arxiv.org/html/2610.38355v1)
  - EfficientAgent: reports computed prefill tokens from a vLLM counter instead — [EfficientAgent](https://arxiv.org/html/2609.33762v1)
- **Repeats.**
  - EfficientAgent: "independent repeats of recomputation and of 40 GiB offload" — [EfficientAgent](https://arxiv.org/html/2609.33762v1)
  - Cheng et al.: "5 seeds" in one ablation — [Cheng et al.](https://arxiv.org/html/2607.16892v1)
  - None found in the other eight.

### Inferences
- **Nobody combines the four properties our setting needs.** None of the ten combines all of:
  - a real engine;
  - agent sessions;
  - an SSD tier;
  - preserved inter-turn (tool) timing.

  Each paper misses at least one:
  - FlatCake has the engine, agent trace and SSD, but replaces timing with a fixed QPS.
  - TempoKV has SSD staging, but no sessions.
  - KVTether, EfficientAgent and HEAR's Mooncake runs keep tool or think timing, but stop at host or remote DRAM.

  A paper that replays sessions with recorded tool gaps across GPU, DRAM and SSD would fill a methodological gap, not just a mechanism gap.
- **Output control, from strongest to weakest.**
  1. Token-exact teacher forcing (EfficientAgent's vLLM logits processor).
  2. Random prompts with recorded counts and matched reusable-prefix lengths (KVTether).
  3. Recorded lengths (HEAR).
  4. Capped natural generation (TempoKV).
  5. Unstated (FlatCake, CacheScout).

  A systems paper should say where its replay sits on this list and how divergence is handled. EfficientAgent's statement that every replay returns "all 4,427 recorded sequences" is the bar for an explicit check.
- **Tool time.** Wherever tools are replayed, they are fixed sleeps independent of load. No paper models tool time that grows under contention.
- **Closed versus open loop.** EfficientAgent (closed loop, next call after response + recorded interval) and HEAR's SCBench (think time after response) differ from KVTether (Poisson task arrivals, recorded within-task tool durations) and FlatCake (open-loop fixed QPS). The choice determines whether queueing delay feeds back into reuse distance, which matters for prefetch timing.
- **Baseline vocabulary reviewers will expect.**
  - LMCache with LRU (KVTether, EfficientAgent, TempoKV-DAX).
  - SGLang HiCache L2 and Mooncake L3 (FlatCake).
  - vLLM prefix caching (CacheScout, EfficientAgent).
  - TTL pinning, à la Continuum (CacheScout).
  - Idleness ranking, à la MORI (KVTether).
  - Recency versus frequency versus predictive placement (Where Should the KV Cache Live?).
- **Use these with care.**
  - The "prefetch does not pay" result (Where Should the KV Cache Live?) is a batch-one simulator with an uncalibrated, contended single-link model.
  - CacheScout's "prefetch alone is inert" comes from workloads with sub-second inter-turn gaps.
  - Neither speaks to multi-second tool gaps with an SSD tier.

### Gaps
- **Questions only the authors can answer.**
  - Mode of CacheScout's runs (live or replayed).
  - FlatCake's per-run replay length and prompt synthesis.
  - HEAR's L2 implementation and session counts.
  - Where Should the KV Cache Live?'s calibration source.
- **What I did not do.**
  - I did not inspect linked repositories (EfficientAgent, LP routing) or the Claude Code dataset to confirm driver behavior.
  - Only the papers' own text was used.
