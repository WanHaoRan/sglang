# KV cache across tool calls in agentic LLM serving: closest related work (2023 to November 2026)

*Scope and method.* The notes compare each paper with the four parts of our design: **(a)** per-tool gap distributions sent by the agent framework and conditioned on idle time; **(b)** a newsvendor fetch threshold q = λL/(1+λL) for SSD reads; **(c)** hazard-rate retention and the eviction index (1+λL)·max_H F(H|t)/τ(H), plus preemption of staged reads; **(d)** a Lagrangian price λ on host memory, learned by multiplicative dual ascent, with a weak-duality lower bound. Every listed paper was opened on arXiv, either as an API record (`export.arxiv.org`), as its abstract page, or as its arXiv HTML full text. Each entry says which. From mid-session on, the arXiv API returned HTTP 429, so the later 2026 papers were verified through their arXiv HTML or abstract pages, or through arXiv's own search-listing page (marked "listing only"). Mechanism details come from arXiv HTML full texts read through a summarizing fetch tool. Short quotes are verbatim; everything else is paraphrased. Search terms were generic throughout.

---

## 1. Ranked list: the closest papers and how each relates to our design

### Takeaway
The closest prior work is **Continuum** (per-tool empirical CDF of tool durations, with the tool name taken as the first word of the bash command, turned into a GPU pin TTL) and **CacheWise** (per-tool and argument-cluster duration distributions, expected remaining time, Belady-style eviction). Next come **Ask the Tool, Don't Guess** (runtime progress signals that drive keep, evict and prefetch-lead decisions, and an explicit argument against pre-call estimates), **TokenCake** (per-function EWMA plus a developer-declared duration, offload during calls and predictive upload) and **Adaptive KV Retention** (a load-indexed, mean-field price on host-memory slots). None of them combines framework-sent distributions, conditioning on idle time, size- and price-dependent fetch risk from SSD, and a price learned online by dual ascent. Nearly all of them stop at GPU HBM or GPU↔CPU DRAM.

### Cited Findings

#### Rank 1. Continuum (closest for (a) and (c))
- **Record:** arXiv [2511.02230](https://arxiv.org/abs/2511.02230), "Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live", Hanchen Li et al. (11 authors, incl. J. Gonzalez and I. Stoica). v1 Nov 2025, v7 Sep 2026. Verified by API record. The HTML title of v7 is spelled "Continnum", and an earlier version is titled "CacheTTL" ([v4 HTML](https://arxiv.org/html/2511.02230v4)). Venue: ICLR 2026 workshop poster, "Lifelong Agents: Learning, Aligning, Evolving" ([ICLR page](https://iclr.cc/virtual/2026/10012473)). The v7 HTML uses a PVLDB template with placeholder volume and pages; we found no acceptance record — [arXiv HTML v7](https://arxiv.org/html/2511.02230v7)
- **Mechanism:**
  - **Duration records.** The engine records the inter-request interval inside a program as that tool call's duration. It estimates P(τ, f), "the probability that the tool call f completes within time τ", with "the empirical CDF derived from historical tool-call records S[f]" — [arXiv HTML v7](https://arxiv.org/html/2511.02230v7)
  - **Key.** On SWE-Bench: "We extract the string within the bash block and use the first word afterwards as the tool call name." With |S[f]| ≤ K = 100 it falls back to global records, and otherwise to a default TTL derived from an exponential assumption — [arXiv HTML v7](https://arxiv.org/html/2511.02230v7)
  - **TTL rule.** τ* = argmax_τ P(τ,f)·(T·η + Prefill-Reload(r)) − τ. T·η is the queueing delay that eviction induces, η is a "memoryfulness" factor, and Prefill-Reload is either the recompute time or the CPU reload time (LMCache). Candidate τ are the observed durations plus 0. The KV stays pinned in GPU memory until the TTL expires or "the program terminates" — [arXiv HTML v7](https://arxiv.org/html/2511.02230v7)
  - **Results.** The abstract reports an average job completion time improvement of more than 8× on SWE-Bench, BFCL and OpenHands agents, with models up to GLM-4.5 355B — [arXiv API/abs](https://arxiv.org/abs/2511.02230)
- **Relation to our design:**
  - **Parallels:** (a) and (c).
  - **Shared:**
    - per-tool *distributions* learned online from observed gaps;
    - a key that splits a shell tool by the first word of its command (bash/pytest-style);
    - a fallback from sparse keys to pooled records;
    - a holding horizon chosen by maximizing expected benefit over an empirical CDF;
    - a program-termination signal that releases memory.
  - **Differences:**
    - The engine learns the CDF itself; the framework sends nothing.
    - No conditioning on elapsed idle time: the TTL is fixed when the gap starts.
    - Cost is charged for the full τ, not E[min(A−t, H)].
    - GPU HBM pinning only: no SSD fetch timing, no prefetch before the tool returns, no preemption order for staged reads.
    - The memory term MemUsage/M cancels in the simplified rule, so there is no price on host memory and no capacity coupling.
  - **Strongest prior-art candidate for (a); partial for (c).**

#### Rank 2. CacheWise (closest for conditional per-tool forecasts driving eviction)
- **Record:** arXiv [2606.16824](https://arxiv.org/abs/2606.16824), "CacheWise: Understanding Workloads and Optimizing KVCache Management for Efficiently Serving LLM Coding Agents", Shubham Tiwari et al. (6 authors), June 2026. No venue stated. Verified by API record and [arXiv HTML v1](https://arxiv.org/html/2606.16824v1)
- **Mechanism:**
  - **Trace study.** Real coding-assistant traces show that "grep has short and tightly clustered durations, whereas bash exhibits longer and more variable execution times". Bash-command P50s are about 0.1 s for Python one-liners, 10 s for mypy and 22 s for docker compose — [arXiv HTML v1](https://arxiv.org/html/2606.16824v1)
  - **Prediction.** CacheWise "estimates the expected remaining time for the tool call to complete". It uses the empirical duration distribution of the tool, refined by KMeans clusters over TF-IDF vectors of the tool arguments, because "relying solely on execution duration distributions of tools with the same names is insufficient". Predictors are trained offline on 80% of sessions — [arXiv HTML v1](https://arxiv.org/html/2606.16824v1)
  - **Eviction.** Unreferenced blocks go into a heap keyed by the predicted next-use time, and eviction approximates Belady by evicting the largest first. The heap is rebuilt every 3 engine iterations. There is no TTL and no prefetch. One configuration offloads evicted blocks to CPU DRAM with swap-in on demand — [arXiv HTML v1](https://arxiv.org/html/2606.16824v1)
  - **Hints and results.** The engine parses `tool_name` and `tool_args` itself; no agent changes are needed. The abstract reports 2–2.6× fewer evictions and up to 3.5× better session completion time — [arXiv abs](https://arxiv.org/abs/2606.16824)
- **Relation to our design:**
  - **Parallels:** (a) and (c).
  - **Shared:**
    - per-tool distributions refined below the tool name (argument clusters, where we use the first word);
    - conditioning on time already elapsed;
    - eviction ordered by forecast next use, explicitly as a Belady approximation, which our index reduces to under exact forecasts.
  - **Differences:**
    - Offline-trained and engine-parsed, not framework-sent and learned online.
    - Ranks by a point summary (conditional mean) rather than F(H|t)/τ(H).
    - Ignores prefix size and memory price.
    - GPU-centric: no fetch timing, no SSD, no host scarcity.
  - **Prior-art candidate for (a), conditional per-tool forecasts, and for the Belady limit of (c).**

#### Rank 3. Ask the Tool, Don't Guess (closest for (b)'s prefetch lead, and a direct critique of (a))
- **Record:** arXiv [2609.18849](https://arxiv.org/abs/2609.18849), "Ask the Tool, Don't Guess: Agent Tool Calls Hold Their Progress, and the Serving System Should Read It", Yipeng Liu et al. (4 authors), Sep 2026. No venue stated. Verified by API record and [arXiv HTML v1](https://arxiv.org/html/2609.18849v1)
- **Mechanism:**
  - **Claim.** It argues that "no estimate fixed before a call starts can know its duration, and such estimates may not even rank the calls" — [arXiv abs](https://arxiv.org/abs/2609.18849)
  - **Signal.** Tools instead report progress, either a fraction of the work remaining or a near-end signal. A harness taps it on a side channel keyed by call id, with events carrying work done, total if known, phase and timestamp — [arXiv HTML v1](https://arxiv.org/html/2609.18849v1)
  - **Policy.** In a vLLM patch, HBM-only eviction removes the context whose call "has the longest to run". With host DRAM, the controller "keeps a context while the reported remaining time is under a threshold" with hysteresis, and "refreshes the context one lead time before the predicted return" — [arXiv HTML v1](https://arxiv.org/html/2609.18849v1)
  - **Results.** p90 post-tool TTFT falls 20.7% (HBM only) and 20.8% (HBM + DRAM) against LRU, "close to an oracle" — [arXiv abs](https://arxiv.org/abs/2609.18849)
  - **Taxonomy of existing estimators** — [arXiv HTML v1](https://arxiv.org/html/2609.18849v1):
    - fixed TTL: CachedAttention, SGLang, Anthropic;
    - elapsed time: InferCept, LAMPS;
    - per-tool statistics: Continuum;
    - learned model on name and arguments: CacheWise;
    - engine occupancy: MORI, ConServe, ThunderAgent, Adaptive KV Retention;
    - declared duration: TokenCake, SGLang, Mooncake, vLLM, TensorRT-LLM.
  - **Elapsed-time baseline.** It reimplements an elapsed-time-conditioned history estimator "in the manner of 3Sigma" and finds history predictors weak near the end of a call — [arXiv HTML v1](https://arxiv.org/html/2609.18849v1)
- **Relation to our design:**
  - **Parallels:** (a) hints, (b) a prefetch start at predicted return minus lead time, and (c) a retention threshold.
  - **Shared:** a tool-to-engine hint channel, a retain/evict ordering by time-to-return, and a refresh started ahead of the return.
  - **Differences:**
    - Point estimates of remaining time, with tuned thresholds and hysteresis.
    - No stall-risk threshold that depends on size or price.
    - No price; HBM + DRAM only.
  - **Main risk:** it is the paper most likely to be raised against (a). Our conditional F(x|t) could take its progress signal as one more conditioning variable (our inference, not the paper's).

#### Rank 4. TokenCake (closest for offload plus predictive upload around function calls)
- **Record:** arXiv [2510.18586](https://arxiv.org/abs/2510.18586), "TokenCake: A KV-Cache-centric Serving Framework for LLM-based Multi-Agent Applications" (v1 title "Tokencake"), Zhuohang Bian et al. (5 authors). v1 Oct 2025, v5 Sep 2026. The arXiv comment says "Accepted at EuroSys '27". Verified by API record and [arXiv HTML v5](https://arxiv.org/html/2510.18586v5)
- **Mechanism:**
  - **Prediction.** The Temporal Scheduler keeps "a historical estimate t_history for each function type", updated as an EWMA with α = 0.5. A user-supplied estimate is blended in with β = 0.5 — [arXiv HTML v5](https://arxiv.org/html/2510.18586v5)
  - **Offload rule.** An offload is rejected when the window Tfc − Ttransfer is non-positive ("the stall is too short"). It also requires GPU usage above 0.60 and a composite score above 1.0 — [arXiv HTML v5](https://arxiv.org/html/2510.18586v5)
  - **Upload.** Predictive upload ranks candidates by importance plus urgency (closeness to predicted completion) under an upload budget, and uploads immediately if the tool returns early — [arXiv HTML v5](https://arxiv.org/html/2510.18586v5)
  - **Tiers.** GPU↔CPU only. Transfer time is linear in block count, and CPU capacity is a hard check. The Spatial Scheduler reserves 5–30% of GPU memory for critical-path agents using hand-set weights (2:1:0.5:1) — [arXiv HTML v5](https://arxiv.org/html/2510.18586v5)
  - **Results.** Up to 47.06% lower end-to-end latency than vLLM — [arXiv abs](https://arxiv.org/abs/2510.18586)
- **Relation to our design:**
  - **Parallels:** (a) a per-tool history plus a declared hint, (b) an upload timed against predicted completion, and (c) an offload decision.
  - **Shared:** tool-aware offload during waits and a prefetch back before the return.
  - **Differences:**
    - Point estimate (EWMA), not a distribution, and no conditioning on idle time.
    - The thresholds and weights are tuned, not derived.
    - GPU→CPU rather than host→SSD.
    - No price or dual.

#### Rank 5. Adaptive KV Retention for LLM Agents at Human-Approval Timescales (closest for (d))
- **Record:** arXiv [2608.30830](https://arxiv.org/abs/2608.30830), "Adaptive KV Retention for LLM Agents at Human-Approval Timescales", Minseo Choi and Ananya Joshi, Aug 2026. Verified through the arXiv search listing and [arXiv HTML](https://arxiv.org/html/2608.30830). The HTML states no venue. A NeurIPS 2026 page appeared in search results but was not opened ([neurips.cc listing](https://neurips.cc/virtual/2026/166749)).
- **Mechanism:**
  - **Cost of each state.** Each state j (HBM, host DRAM, discard) costs C_j(ℓ) = α_j·ℓ + β_j in GPU-seconds, an "opportunity cost". The HBM→DRAM break-even is about 1.1 s, so suspended requests move to host DRAM at once — [arXiv HTML](https://arxiv.org/html/2608.30830)
  - **Load-indexed expiration.** Host retention is governed by α₂(λ) = max(0, λ − λ_crit)·β₃/C₂, where λ is the *offered load* and C₂ the number of host slots. The TTL is t₂ = (β₃ − β₂)/α₂(λ), and it is infinite below λ_crit — [arXiv HTML](https://arxiv.org/html/2608.30830)
  - **What it does not do.** There is no per-request wait prediction ("without requiring per-request wait prediction"), no online dual update (load is configured, and online estimation is left to future work), no SSD tier and no prefetch. The TTL is one-shot — [arXiv HTML](https://arxiv.org/html/2608.30830)
  - **Results.** The abstract reports 22–29% more active goodput than MORI and 41–52% more than Continuum on approval workloads — [arXiv search listing](https://arxiv.org/search/?query=human+approval+KV+retention&searchtype=abstract&abstracts=show&order=-announced_date_first&size=50)
- **Relation to our design:**
  - **Parallels:** (c) and (d).
  - **Shared:** a per-slot, mean-field *price on host memory* that rises with load; TTLs from break-even cost reasoning; GPU↔host↔recompute tiers.
  - **Differences:**
    - The price is an analytic function of configured offered load, not a Lagrange multiplier learned from demand minus capacity.
    - One TTL for all suspended requests: no per-session forecasts, no size dependence, no conditioning.
    - Timescales of minutes to hours.
    - No SSD reads.
  - **Closest prior art for (d); not dual ascent and not a lower bound.**

#### Rank 6. MORI (closest prediction-free policy for host-tier retention under scarcity)
- **Record:** arXiv [2606.00866](https://arxiv.org/abs/2606.00866), "Idleness is Relative: Exploiting Tool-Call Idle Windows for Offloading in Agentic Systems with MORI", Tian Xia et al. (8 authors, incl. Hanchen Li and Ion Stoica). v1 May 2026, v2 Nov 11, 2026. No venue stated. Verified by API record and [arXiv HTML v2](https://arxiv.org/html/2606.00866v2)
- **Mechanism:**
  - **Problem.** Existing schedulers "leave DRAM to LRU, so offloaded programs thrash", and "tool-call durations vary by orders of magnitude and are unknown in advance" — [arXiv abs](https://arxiv.org/abs/2606.00866)
  - **Idleness.** ι = T_acting/(T_reasoning + T_acting) over the last k = 5 cycles. "The least idle programs fill GPU HBM, the next fill CPU DRAM", and the rest are dropped — [arXiv HTML v2](https://arxiv.org/html/2606.00866v2)
  - **Control.** Admission control works per tier. A program is proactively demoted if it is acting and ι⁽¹⁾ ≥ 1/2. The control loop runs every 5 s. There is no per-tool prediction, TTL, cost model or price. SSD is future work — [arXiv HTML v2](https://arxiv.org/html/2606.00866v2)
  - **Results.** On average 64% higher throughput and 72% lower p90 TTFT than the best offloading baseline. The baselines are Continuum ported to SGLang, ThunderAgent, and SGLang with its hierarchical cache's LRU — [arXiv abs](https://arxiv.org/abs/2606.00866), [arXiv HTML v2](https://arxiv.org/html/2606.00866v2)
- **Relation to our design:**
  - **Parallels:** (c), and host scarcity through admission control at both tiers.
  - **Shared:** keep, offload or drop decisions over HBM and DRAM, and a critique of LRU in host memory.
  - **Differences:**
    - A relative idleness rank replaces forecasts. It resembles our "one pooled distribution → LRU" limit, but per program.
    - No read time, no SSD, no fetch timing and no price.
    - The thresholds are tuned.

#### Rank 7. InferCept (the original preserve/discard/swap paper)
- **Record:** arXiv [2402.01869](https://arxiv.org/abs/2402.01869), "InferCept: Efficient Intercept Support for Augmented Large Language Model Inference", Reyna Abhyankar et al. (5 authors), 2024. ICML 2024, listed in PMLR vol. 235 ([PMLR listing, from search results](https://proceedings.mlr.press/v235/abhyankar24a.html); [ICML poster page](https://icml.cc/virtual/2024/poster/32755)). Verified by API record and [arXiv HTML v2](https://arxiv.org/html/2402.01869v2)
- **Mechanism:**
  - **Cost model.** It defines GPU memory "waste" (memory × time) for each option. Preserve wastes T_INT·C·M. Discard and chunked discard pay the recompute time over the whole batch. Swap pays 2·T_swap·C_batch·M. Each paused request takes min(Preserve, chunked Discard) — [arXiv HTML v2](https://arxiv.org/html/2402.01869v2)
  - **Swap.** Swap-out runs within a per-iteration budget in descending order of waste. It is bounded by free CPU memory, which is a modeled constraint — [arXiv HTML v2](https://arxiv.org/html/2402.01869v2)
  - **Duration estimate.** The interception duration is estimated dynamically as T̂_INT = t_now − t_call, reaching 93% of an oracle — [arXiv HTML v2](https://arxiv.org/html/2402.01869v2)
  - **Swap-in.** Swap-in happens after the call returns (FCFS). There is no lookahead prefetch — [arXiv HTML v2](https://arxiv.org/html/2402.01869v2)
  - **Results.** Recomputation is 37–40% of forward time in baselines, and InferCept improves throughput 1.6–2× — [arXiv abs](https://arxiv.org/abs/2402.01869)
- **Relation to our design:**
  - **Parallels:** (c), a keep-or-discard rule derived from a cost model, and partly (a).
  - **Shared:** a memory-time currency for holding KV during a call.
  - **Differences:**
    - The elapsed-time estimate is a point estimate.
    - GPU memory, not host memory, is the scarce resource. CPU is a swap target.
    - No SSD, no prefetch, no price.

#### Rank 8. LAMPS (NeurIPS 2025 version names the system MARS)
- **Record:** arXiv [2410.18248](https://arxiv.org/abs/2410.18248), "Fast Inference for Augmented Large Language Models", Rana Shahout et al. (7 authors, incl. M. Mitzenmacher), 2024. Verified by API record and [arXiv HTML v2](https://arxiv.org/html/2410.18248v2). Venue: NeurIPS 2025 under the same title; the proceedings abstract calls the system "MARS" ([MLAnthology](https://mlanthology.org/neurips/2025/shahout2025neurips-fast), linking [OpenReview](https://openreview.net/forum?id=uNqTxj5brQ)). Do not confuse it with the unrelated arXiv 2604.26963, also called "MARS".
- **Mechanism:**
  - **API duration.** API durations are tied to API classes ("execution times within the same API type have low variance") — [arXiv HTML v2](https://arxiv.org/html/2410.18248v2)
  - **Response and output length.** The API response length is the class average. The pre-API output length comes from an OPT-125M classifier over 50 bins — [arXiv HTML v2](https://arxiv.org/html/2410.18248v2)
  - **Handling and ranking.** Each request gets the InferCept handling strategy that minimizes waste. Requests are ranked by the area of memory over time, with a starvation counter of 100 iterations — [arXiv HTML v2](https://arxiv.org/html/2410.18248v2)
  - **Results.** 27–85% lower end-to-end latency and 4–96% lower TTFT than "the existing augmented-LLM system" (presumably InferCept, whose waste equations LAMPS reuses) — [arXiv abs](https://arxiv.org/abs/2410.18248)
- **Relation to our design:**
  - **Parallels:** (a) per-API point estimates and (c) via the waste rule; its memory-over-time ranking resembles our memory-time cost but drives *scheduling*.
  - **Differences:** point estimates, GPU/CPU only, no fetch timing, no price.
  - **Conflict:** Ask the Tool groups LAMPS with elapsed-time estimators ([arXiv HTML](https://arxiv.org/html/2609.18849v1)), whereas the LAMPS text we read describes per-API-class durations. The exact estimator was not specified on the page we read.

#### Rank 9. Astraea
- **Record:** arXiv [2512.14142](https://arxiv.org/abs/2512.14142), "Astraea: A State-Aware Scheduling Engine for LLM-Powered Agents", Hongqiu Ni et al. (7 authors), Dec 2025. No venue stated. Verified through the arXiv search listing and [arXiv HTML v1](https://arxiv.org/html/2512.14142)
- **Mechanism:**
  - **During I/O waits.** Below a GPU-memory high-watermark, all I/O-bound requests are preserved. Above it, the engine takes the argmin of InferCept-style waste over {Preserve, Discard, Swap} — [arXiv HTML](https://arxiv.org/html/2512.14142)
  - **Prediction.** Tool duration is "the category's mean latency" — [arXiv HTML](https://arxiv.org/html/2512.14142)
  - **Scheduling and tiers.** Requests are reclassified reactively as I/O- or compute-intensive for MLFQ-style scheduling. Tiers are GPU and CPU only — [arXiv HTML](https://arxiv.org/html/2512.14142)
- **Relation to our design:**
  - **Parallels:** (a) and (c).
  - **Differences:** a per-category point estimate, a tuned watermark, a GPU-memory currency, and no SSD or price.

#### Rank 10. SYMPHONY (hints with an expected arrival; tiers down to disk)
- **Record:** arXiv [2412.16434](https://arxiv.org/abs/2412.16434), "SYMPHONY: Improving Memory Management for LLM Inference Workloads", Saurabh Agarwal et al. (4 authors), Dec 2024. No venue found. Verified by API record and [arXiv HTML v1](https://arxiv.org/html/2412.16434v1)
- **Mechanism:**
  - **Advisory requests.** These carry `session_id`, `model_id`, `expected_arrival`, `ordered` and `priority`. They are sent "as soon as the user starts typing" (chat), or by an upstream agent using offline-profiled lower bounds of arrival time — [arXiv HTML v1](https://arxiv.org/html/2412.16434v1)
  - **Response to a hint.** The node manager moves the session's K,V "to the fastest memory tier possible" (HBM, DRAM or disk), and always keeps one copy on disk — [arXiv HTML v1](https://arxiv.org/html/2412.16434v1)
  - **Gaps in the policy.** There is no explicit timing algorithm: transfers are greedy when memory is free. There is no host or disk eviction policy and no tool-duration model — [arXiv HTML v1](https://arxiv.org/html/2412.16434v1)
  - **Results.** More than 8× the requests of baselines at similar latency — [arXiv abs](https://arxiv.org/abs/2412.16434)
- **Relation to our design:**
  - **Parallels:** (a) client hints and (b) prefetch across tiers, including disk.
  - **Differences:**
    - The arrival is a point estimate or lower bound.
    - No risk threshold, no host-memory price and no tool-keyed forecasts.

#### Rank 11. KVFlow (framework hints → eviction plus CPU→GPU prefetch)
- **Record:** arXiv [2507.07400](https://arxiv.org/abs/2507.07400), "KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows", Zaifeng Pan et al. (9 authors), Jul 2025. NeurIPS 2025 per search results ([NeurIPS poster page](https://neurips.cc/virtual/2025/poster/119883), [proceedings](https://papers.neurips.cc/paper_files/paper/2025/hash/b7971d31a7d5eb0f1eed2f8f6f368195-Abstract-Conference.html); neither opened). Verified by API record and [arXiv HTML v1](https://arxiv.org/html/2507.07400v1)
- **Mechanism:**
  - **Hints.** Each request carries "the identity of the current agent and the steps-to-execution of all agents", computed on an Agent Step Graph (max or min over predecessors, plus 1) — [arXiv HTML v1](https://arxiv.org/html/2507.07400v1)
  - **Eviction.** It evicts agents with larger steps-to-execution first. Varying suffixes go first, and shared nodes take the minimum — [arXiv HTML v1](https://arxiv.org/html/2507.07400v1)
  - **Prefetch.** It prefetches next-step agents' fixed prompts from CPU to GPU in the background — [arXiv HTML v1](https://arxiv.org/html/2507.07400v1)
  - **What it lacks.** No wall-clock or tool-duration prediction, no SSD, no cost model — [arXiv HTML v1](https://arxiv.org/html/2507.07400v1)
  - **Results.** Up to 1.83× (single workflow) and 2.19× (concurrent workflows) speedup over SGLang's hierarchical radix cache — [arXiv abs](https://arxiv.org/abs/2507.07400)
- **Relation to our design:**
  - **Parallels:** (a) framework hints, (b) prefetch, (c) eviction by future use, a step-count version of Belady.
  - **Differences:**
    - Order-only forecasts.
    - Shared agent prompts rather than per-session prefix caches.
    - CPU→GPU, with no timing risk or price.

#### Rank 12. HEAR protocol (bidirectional harness↔engine hints)
- **Record:** arXiv [2610.06597](https://arxiv.org/abs/2610.06597), "Can Agent Harnesses and Inference Engines Hear Each Other? The HEAR Protocol for Agentic LLM Serving", Jiaqi Zhao et al. (7 authors), Oct 2026. No venue stated. Verified through the arXiv search listing and [arXiv HTML v1](https://arxiv.org/html/2610.06597)
- **Mechanism:**
  - **Harness → engine.** "known future uses and explicitly marked predictions", "scheduling preferences and waiting-time constraints", and cache preparation, retention and release requests — [arXiv HTML](https://arxiv.org/html/2610.06597)
  - **Engine → harness.** "reusable prefix ranges and residency tiers", capacity, queue state and cost estimates — [arXiv HTML](https://arxiv.org/html/2610.06597)
  - **Session-Aware retention.** It scores r = p·e^(−idle/400), with p a return probability estimated from session depth and prior inter-turn intervals. The score sets eviction order only. Tiers are GPU and CPU (96 GiB L2), and the thresholds are tuned — [arXiv HTML](https://arxiv.org/html/2610.06597)
- **Relation to our design:**
  - **Parallels:** (a) a protocol for framework hints and (c) a retention score that decays with idle time.
  - **Differences:**
    - A heuristic exponential decay with a fixed constant, not a hazard computed from a per-key distribution.
    - No size or price, no SSD timing.
    - Prediction format unspecified ("explicitly marked predictions").

#### Rank 13. CacheScout (online-learned agent transitions → survival eviction + prefetch)
- **Record:** arXiv [2608.14624](https://arxiv.org/abs/2608.14624), "Learning Agent Execution for KV-Cache Management in Agentic Serving", Rui Zhang et al. (9 authors, incl. Junchen Jiang), Jul 2026. Verified by API record and [arXiv HTML v1](https://arxiv.org/html/2608.14624v1)
- **Mechanism:**
  - **Model.** An "online first-order Markov chain" over agents, identified from prompt-prefix fingerprints with no framework cooperation — [arXiv HTML v1](https://arxiv.org/html/2608.14624v1)
  - **Eviction score.** Score(b) = (p̃_surv + δ)·(e^(−λ·age) + δ)·|b|, where p̃_surv comes from graph hop distance — [arXiv HTML v1](https://arxiv.org/html/2608.14624v1)
  - **Prefetch.** A warmup of the argmax successor's anchor during idle periods, gated on a predictability metric. It uses vLLM's LMCache CPU path. No tool-duration or idle-gap modeling — [arXiv HTML v1](https://arxiv.org/html/2608.14624v1)
  - **Results.** 18–45% lower mean TTFT — [arXiv abs](https://arxiv.org/abs/2608.14624)
  - **Generalization.** The same authors describe a general observe/score/predict/act runtime layer with CacheScout as one instance (arXiv 2605.27744; listing only) — [arXiv search listing](https://arxiv.org/search/?query=agentic+KV+cache+eviction&searchtype=abstract&abstracts=show&order=-announced_date_first&size=50)
- **Relation to our design:**
  - **Parallels:** (a) learned online, (b), (c).
  - **Differences:**
    - It predicts *which* agent runs next, not *when*.
    - Heuristic constants rather than a derived threshold.
    - No price.

#### Rank 14. The Storage View of KV Cache (session signals; a fixed-quantile TTL)
- **Record:** arXiv [2611.10178](https://arxiv.org/abs/2611.10178), "The Storage View Of KV Cache: Characterizing Cache Behavior Across LLM Workloads", Talia Q. Chen et al. (4 authors, UT Austin and IBM Research), Nov 2026. Verified through the arXiv search listing and [arXiv HTML v1](https://arxiv.org/html/2611.10178)
- **Mechanism:**
  - **Signals.** It proposes three signals: "an opaque workload identifier, a session identifier, and session completion signal". They should guide demotion of "paused-but-live state when pressure rises" and reclamation on completion — [arXiv HTML](https://arxiv.org/html/2611.10178)
  - **Policy experiment.** It uses an "idle TTL set to the 90th percentile of observed within-session gaps", and per-workload TTLs beat pooled ones — [arXiv HTML](https://arxiv.org/html/2611.10178)
  - **Idle timers fail both ways.** "BrowseComp+ retains 88% with a one-minute timer" while "TraceLab retains 48% with a ten-second timer". Transfer costs are not modeled — [arXiv HTML](https://arxiv.org/html/2611.10178)
- **Relation to our design:**
  - **Parallels:** (a) session-end and workload hints, and (c).
  - **Fixed quantile.** Its p90 TTL is exactly the "one fixed quantile for every read" case our design generalizes (one fixed q regardless of size and price).
  - **Differences:** no per-tool keys, no conditioning, no fetch timing, no price.

#### Rank 15. ThunderAgent (program-aware serving, also for RL rollouts)
- **Record:** arXiv [2602.13692](https://arxiv.org/abs/2602.13692), "ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System", Hao Kang et al. (10 authors), Feb 2026 (v3 Jun 2026). Verified by API record and [arXiv HTML v3](https://arxiv.org/html/2602.13692v3)
- **Mechanism:**
  - **During tool waits.** Acting programs stay on the GPU and are discounted in the memory check by a decay f(t_q) of elapsed tool time — [arXiv HTML v3](https://arxiv.org/html/2602.13692v3)
  - **On thrashing.** It "greedily pause[s] and evict[s] programs with the shortest context lengths" — [arXiv HTML v3](https://arxiv.org/html/2602.13692v3)
  - **Cost and tiers.** The cost model is a "Space-Time Product" with a fixed token capacity. There is no CPU or SSD KV tier and no price — [arXiv HTML v3](https://arxiv.org/html/2602.13692v3)
  - **Results.** 1.5–3.6× serving throughput and 1.8–3.9× RL-rollout throughput — [arXiv abs](https://arxiv.org/abs/2602.13692)
- **Relation to our design:**
  - **Parallels:** (c) and a memory-time cost.
  - **Differences:** no forecasts, GPU only, no price.
  - **RL rollouts:** it is the main verified agentic RL-rollout system that manages KV across tool calls.

#### Rank 16. KVTether (harness lifecycle primitives)
- **Record:** arXiv [2609.39819](https://arxiv.org/abs/2609.39819), "Capture the lifecycle: KV Cache management in ReAct Agents with KVTether", Kaihua Fu et al. (9 authors), Sep 2026. Verified through the arXiv search listing and [arXiv HTML v1](https://arxiv.org/html/2609.39819)
- **Mechanism:**
  - **Primitives.** The harness calls eight primitives (append, replace, remove, fork, spawn, one_off, pin, unpin) that set message lifecycle states (Active, Suspended, Expired, Discarded, Pinned, Temporary) — [arXiv HTML](https://arxiv.org/html/2609.39819)
  - **Eviction order.** Expired first, then Suspended in MRU order ("evicts the most recently suspended KV first"), then Active, then Pinned — [arXiv HTML](https://arxiv.org/html/2609.39819)
  - **What it lacks.** No duration prediction. It favors a "stable directional trend... rather than precise per-entry next-use prediction". GPU/host/remote tiers are mentioned; SSD is not — [arXiv HTML](https://arxiv.org/html/2609.39819)
- **Relation to our design:** it parallels (a)'s end-of-session hint and (c)'s ordering. It has no timing model or price.

#### Rank 17. Sutradhara (orchestrator semantic hints; HBM only)
- **Record:** arXiv [2601.12967](https://arxiv.org/abs/2601.12967), "Sutradhara: An Intelligent Orchestrator-Engine Co-design for Tool-based Agentic Inference", Anish Biswas et al. (8 authors, Microsoft), Jan 2026 (v3 Apr 2026). Verified through the arXiv search listing and [arXiv HTML v3](https://arxiv.org/html/2601.12967)
- **Mechanism:**
  - **APIs.** `submit_partial_prefill`, `extend_prefill`, `register_streaming_callback`, `tag_kv_blocks` and `set_reuse_priority` — [arXiv HTML](https://arxiv.org/html/2601.12967)
  - **Eviction.** By tag class (RESPONSE < TOOL_OUTPUT < USER_QUERY < SYSTEM_PROMPT < PARTIAL_PREFILL), with LRU inside a class. Partial-prefill blocks stay pinned while tools run — [arXiv HTML](https://arxiv.org/html/2601.12967)
  - **Stance and tiers.** It states that "tool execution time remains inherently difficult to predict" and assumes HBM only, so evicted blocks are recomputed — [arXiv HTML](https://arxiv.org/html/2601.12967)
  - **Results.** The abstract says "tool calls account for 30-85% of FTR latency" (FTR is the paper's latency metric; not expanded on the listing) — [arXiv search listing](https://arxiv.org/search/?query=tool+call+KV+cache&searchtype=abstract&abstracts=show&order=-announced_date_first&size=50)
- **Relation to our design:** it parallels (a) framework hints. It adopts the opposite stance on forecasts, and has no tiers or price.

#### Rank 18. Lachesis (harness boundaries; HBM vs high-bandwidth flash)
- **Record:** arXiv [2610.08378](https://arxiv.org/abs/2610.08378), "Lachesis: Lifetime-Aware KV Cache Placement for Agent Serving across HBM and High-Bandwidth Flash", Jaehoon Yang et al. (6 authors, incl. Nam Sung Kim), Oct 2026. Verified through the arXiv search listing and [arXiv HTML v1](https://arxiv.org/html/2610.08378)
- **Mechanism:**
  - **Placement.** Segments are placed at write time by class. The harness reports boundaries ("predetermined and known to the harness") — [arXiv HTML](https://arxiv.org/html/2610.08378)
  - **Workers.** Workers are ranked by a ridge-regression model "read only for the order it induces", and the top k go to HBM, with k set by free capacity — [arXiv HTML](https://arxiv.org/html/2610.08378)
  - **What it lacks.** No prefetch, no read-time model (both tiers have equal read bandwidth), no host DRAM — [arXiv HTML](https://arxiv.org/html/2610.08378)
- **Relation to our design:** it parallels (a) harness signals and flash tiers. Placement is by lifetime class, with no timing risk or price.

#### Rank 19. Autellix (program-level scheduling; KV not retained across tool calls)
- **Record:** arXiv [2502.13965](https://arxiv.org/abs/2502.13965), "Autellix: An Efficient Serving Engine for LLM Agents as General Programs", Michael Luo et al. (11 authors), Feb 2025. A web search found no venue record. Verified by API record and [arXiv HTML v1](https://arxiv.org/html/2502.13965v1)
- **Mechanism:**
  - **Scheduling.** PLAS/ATLAS prioritize calls by the program's attained service or critical path. The system is "non-clairvoyant". The load balancer pins long requests to their program's engine — [arXiv HTML v1](https://arxiv.org/html/2502.13965v1)
  - **KV handling.** GPU↔CPU swapping applies to preempted calls. The paper gives no keep/evict policy for tool gaps — [arXiv HTML v1](https://arxiv.org/html/2502.13965v1)
  - **Results.** 4–15× program throughput over vLLM — [arXiv abs](https://arxiv.org/abs/2502.13965)
- **Relation to our design:** program identity is used only for scheduling. There is no retention across tool calls, no forecasts, no tiers beyond CPU swap, and no price.

#### Rank 20. Parrot (application-level hints via Semantic Variables)
- **Record:** arXiv [2405.19888](https://arxiv.org/abs/2405.19888), "Parrot: Efficient Serving of LLM-based Applications with Semantic Variable", Chaofan Lin et al. (7 authors), 2024. The arXiv comment says "To appear on USENIX OSDI 2024". Verified by API record.
- **Mechanism:** Semantic Variables annotate the inputs and outputs of requests to "expose application-level knowledge to public LLM services". The service then runs data-flow analysis across requests — [arXiv abs](https://arxiv.org/abs/2405.19888)
- **Relation to our design:** a precedent for (a) as an interface: the application tells the service its structure. It has no tool-duration timing, KV tiers or price.

#### Rank 21. Conveyor (tool-aware serving through partial tool execution)
- **Record:** arXiv [2406.00059](https://arxiv.org/abs/2406.00059), "Conveyor: Efficient Tool-aware LLM Serving with Tool Partial Execution", Yechen Xu et al. (4 authors), 2024. A web search found no venue record. Verified by API record.
- **Mechanism:** Tools start executing while the LLM is still decoding. An interface lets tool developers expose partial-execution opportunities, and a scheduler supports it. This improves request completion latency by up to 38.8% — [arXiv abs](https://arxiv.org/abs/2406.00059)
- **Relation to our design:** it is tool-aware but does not address KV placement during tool waits. It is a complementary latency lever.

#### Rank 22. Pie (programmable KV control)
- **Record:** arXiv [2510.24051](https://arxiv.org/abs/2510.24051), "Pie: A Programmable Serving System for Emerging LLM Applications", In Gim et al. (4 authors), 2025. The arXiv comment says "SOSP 2025". Verified by API record.
- **Mechanism:** User programs ("inferlets") control generation and can "implement new KV cache strategies" and "integrate computation and I/O" without engine changes. The paper reports 1.3–3.4× gains on agentic workflows — [arXiv abs](https://arxiv.org/abs/2510.24051)
- **Relation to our design:** an alternative division of labor. The application decides KV policy, whereas we send forecasts and let the engine decide. No forecast or price model.

#### Further verified work, mentioned in passing (one line each)
- **EfficientAgent** (arXiv [2609.33762](https://arxiv.org/abs/2609.33762), Kunming Shao et al., Sep 2026; abstract page). It sizes the host tier to the agent pool's "reuse working set" with a stack-distance model, and stops writing large refills when the tier is small. This is host scarcity handled by sizing, not pricing.
- **Where Should the KV Cache Live?** (arXiv [2609.16215](https://arxiv.org/abs/2609.16215), Srikanta Datta Tumkur et al., Sep 2026; [HTML](https://arxiv.org/html/2609.16215)). A GPU/DRAM/SSD placement simulator. It finds "Prefetching does not justify its bandwidth cost" unless reuse is tightly predictable, and it uses $/GB tier prices, which are monetary rather than shadow prices.
- **MORTIS** (arXiv [2611.08638](https://arxiv.org/abs/2611.08638), Xiaoyu Ma et al., Nov 2026; abstract page). Compiler analysis of typed LLM function calls predicts each prompt component's reuse and lifetime, and these predictions drive retention, eviction and prefetching. Tiers are not specified in the abstract.
- **PBKV** (arXiv 2605.06472, Haoyu Zheng et al., May 2026; listing only). It forecasts upcoming agent invocations from historical workflows plus current context, and keeps high-reuse entries in GPU memory with conservative eviction and prefetch — [arXiv search listing](https://arxiv.org/search/?query=prediction-based+KV-cache+management+agent+workflows&searchtype=abstract&abstracts=show&order=-announced_date_first&size=25)
- **Agentic Coding in the Wild** (arXiv [2608.00101](https://arxiv.org/abs/2608.00101), Banruo Liu et al., Jul 2026; abstract page). A GitHub Copilot trace study (761M LLM calls). KV hit rates drop across turn boundaries (about 90% within a turn, 55% across). A "lightweight idle-time predictor" "captures 86–90% of total idle time"; its features are not given in the abstract.
- **TraceLab** (arXiv 2606.30560, Kan Zhu et al., Jun 2026; listing only). A coding-agent workload study that finds heavy-tailed tool calls and only *suggests* "semantic-aware tool-latency prediction" — [arXiv search listing](https://arxiv.org/search/?query=tool+latency+prediction+agent+serving&searchtype=abstract&abstracts=show&order=-announced_date_first&size=50)
- **Tool Waiting and Re-arrival in Compile-Time-Static LLM Serving** (arXiv [2609.34663](https://arxiv.org/abs/2609.34663), Dongkyeom Jang et al., Sep 2026; [HTML](https://arxiv.org/html/2609.34663)). It draws per-tool latencies from TraceLab quantiles, studies KV survival in device memory under FIFO eviction, and selects configurations by simulation. It has no prediction or tiers.
- **Inside the Trap** (arXiv [2611.08902](https://arxiv.org/abs/2611.08902), Akshat Valse and Ali Jannesari, Nov 2026; [HTML](https://arxiv.org/html/2611.08902)). A renewal recursion explains self-sustaining prefix-cache collapse under LRU and bursts. The remedies are admission-side (shedding, a session cap), not retention.
- **Understanding KV Residency in Production-Scale Agentic LLM Workloads** (arXiv 2610.22863, Mayank Arya et al., Oct 2026; listing only). An M365 agent trace replay of a one-hour idle cap, LRU, clairvoyant furthest-next-use and int4 demotion — [arXiv search listing](https://arxiv.org/search/?query=agentic+KV+cache+eviction&searchtype=abstract&abstracts=show&order=-announced_date_first&size=50)
- **RR-Evict** (arXiv 2609.32278; listing only). Round-robin tail-chunk eviction across idle agents that "requires no prediction of future arrivals or tool latency", implemented in SGLang. **Penelope** (arXiv 2611.00391; listing only) evicts from the start of the remaining history and keeps the suffix — [arXiv search listing](https://arxiv.org/search/?query=agentic+KV+cache+eviction&searchtype=abstract&abstracts=show&order=-announced_date_first&size=50)
- **SARIA** (arXiv 2610.31169; listing only). It persists to SSD the KV blocks with reuse "beyond their DRAM residency" and promotes SSD hits to DRAM, with SSD endurance as the cost. It is agent-oriented but has no tool timing — [arXiv search listing](https://arxiv.org/search/?query=agent+KV+cache+SSD&searchtype=abstract&abstracts=show&order=-announced_date_first&size=50)
- **ConServe, "Observation, Not Prediction"** (arXiv 2606.01839; listing only). It pins a conversation to one decoder using observed occupancy, with no learned predictor. **SAGA** (arXiv 2605.00528; listing only) uses Agent Execution Graphs to predict KV reuse across tool-call boundaries from structure, not timing — [arXiv search listing](https://arxiv.org/search/?query=tool+call+KV+cache&searchtype=abstract&abstracts=show&order=-announced_date_first&size=50)
- **Leyline** (arXiv [2606.01065](https://arxiv.org/abs/2606.01065); API record), **Cache Engineering for Agents** (arXiv 2610.28577; listing only) and **PACE** (arXiv 2611.10240; listing only). These are agent→engine *content* directives (span edits, drop/reposition, "the agent's planner tells the cache which context future steps will read"), not timing hints — [arXiv search listing](https://arxiv.org/search/?query=agentic+KV+cache+eviction&searchtype=abstract&abstracts=show&order=-announced_date_first&size=50)
- **TVCache** (arXiv [2602.10986](https://arxiv.org/abs/2602.10986); API record). A tool-*value* cache for RL post-training; it does not manage KV. **VenusRL** (arXiv 2610.03286; listing only) is an agentic RL system whose scheduler covers "KV Cache residency", with no tool-duration prediction stated — [arXiv search listing](https://arxiv.org/search/?query=reinforcement+learning+rollout+multi-turn+tool+KV+cache&searchtype=abstract&abstracts=show&order=-announced_date_first&size=50)
- **Outside the agentic scope, covered by others.** Each was verified by API record:
  - **CachedAttention** (arXiv [2403.19708](https://arxiv.org/abs/2403.19708), USENIX ATC 2024): multi-turn DRAM+SSD cache with "scheduler-aware fetching and eviction ... based on the hints from the inference job scheduler", the queue-aware precedent for our x_i = r_i − Ŵ_i.
  - **Pensieve** (arXiv [2312.05516](https://arxiv.org/abs/2312.05516)): multi-turn GPU/CPU cache.
  - **KVCache Cache in the Wild** (arXiv [2506.02634](https://arxiv.org/abs/2506.02634), USENIX ATC'25): reuse "for a specific request category ... tends to be predictable", with a workload-aware eviction policy.
  - **AsyncLM** (arXiv [2412.07017](https://arxiv.org/abs/2412.07017)): asynchronous function calls with interrupts.
  - **AIOS** (arXiv [2403.16971](https://arxiv.org/abs/2403.16971), COLM 2025): an agent OS with context management.

### Inferences
- The 2025–2026 literature splits into three camps on tool-wait KV:
  - **forecast-based:** Continuum, CacheWise, TokenCake, LAMPS, Astraea, Symphony;
  - **forecast-free:** MORI, RR-Evict, Sutradhara, KVTether, Adaptive KV Retention, ThunderAgent, InferCept's elapsed-time rule;
  - **runtime-progress:** Ask the Tool.

  Our design is forecast-based. It answers the forecast-free camp's main objection, that durations are unpredictable, with distributions, conditioning on idle time, and a risk threshold that degrades to on-demand reads and LRU when forecasts are uninformative.
- On (a), reviewers will most likely cite Continuum: same per-tool empirical CDF, same first-word key for bash, same fallback to pooled records. CacheWise is the second most likely, for conditional remaining time and the Belady approximation. Our defensible differences are:
  - the framework sends the table, which leaves the risk choice with the engine;
  - conditioning on idle time is used for fetch timing, not only for eviction rank;
  - the size- and price-dependent q;
  - SSD reads under host scarcity.
- No verified paper times SSD→host reads around tool gaps. The tool-aware systems stop at HBM or HBM+DRAM. Symphony (disk, hint-driven, not tool-keyed) and CachedAttention (queue look-ahead, not tool-aware) are the nearest tier-wise precedents.

### Gaps
- The arXiv API rate-limited this session (HTTP 429). About 20 2026 papers were therefore verified through arXiv HTML, abstract or search-listing pages rather than API records. "Listing only" entries were not opened individually.
- Mechanism details come from a summarizing fetch of arXiv HTML. Two details were not resolved:
  - CacheWise's conditional-expectation formula did not render in one reading. Its text says "expected remaining time", but the conditioning term should be checked in the PDF.
  - Continuum's v7 mentions SSD offloading in its evaluation, but we did not verify that configuration.
- Venues were not found for Autellix, Conveyor, SYMPHONY, MORI, CacheWise, Ask the Tool and most 2026 papers. KVFlow's NeurIPS 2025 status and Adaptive KV Retention's NeurIPS 2026 page come from search results that were not opened.
- **Unverified leads** (not checked on arXiv, or pointers only):
  - An earlier InferCept title "APIServe", suggested only by a URL slug on sotaverified.org.
  - A Berkeley technical report version of Continuum (EECS-2026-234), seen only in search results.
  - Systems named in HEAR's related work that we did not look up: CONCUR, Agentix, Pythia, Helium.
  - 3Sigma (runtime-distribution scheduling), referenced by Ask the Tool.
  - The vLLM "AgentX" blog (Sep 2026, "offloading KV caches between agent turns"), the llm-d agentic-serving guide, the PSRL KV-offload docs and the Fireworks rollout docs. These are project pages, usable only as pointers.

---

## 2. Prior art: could any paper be read as prior art for our four main claims?

### Takeaway
- **(a) per-tool distributional gap forecasts:** strong partial prior art exists (Continuum, CacheWise).
- **(c) hazard-rate retention index:** partial prior art only (Continuum's horizon argmax, CacheWise's conditional mean, HEAR's idle decay).
- **(d) a price on host memory:** one partial precedent (Adaptive KV Retention's load-indexed mean-field price), not learned by dual ascent.
- **(b) the newsvendor fetch threshold:** no prior art found.
- No paper derives a weak-duality lower bound.

### Cited Findings
- **(a) Per-tool distributional gap forecasts. Flag: Continuum (strong).**
  - It keeps a per-tool empirical CDF of observed inter-request intervals, P(τ, f) from S[f], and names the tool by "the first word afterwards" inside the bash block on SWE-Bench. With ≤ K = 100 records it falls back to global records. Our differences: it is engine-learned, not framework-sent, and it is not conditioned on idle time — [Continuum HTML v7](https://arxiv.org/html/2511.02230v7)
- **(a) Flag: CacheWise (strong on conditioning).**
  - It uses per-tool duration distributions refined by argument clusters (TF-IDF + KMeans) and estimates the "expected remaining time for the tool call to complete". Our differences: trained offline, engine-parsed metadata, no framework table — [CacheWise HTML v1](https://arxiv.org/html/2606.16824v1)
- **(a) Weaker precedents.** All of these are point estimates or quantiles:
  - per-function EWMA blended with a user-declared estimate (TokenCake) — [TokenCake HTML v5](https://arxiv.org/html/2510.18586v5);
  - a per-category mean (Astraea) — [Astraea HTML](https://arxiv.org/html/2512.14142);
  - per-API class durations (LAMPS) — [LAMPS HTML v2](https://arxiv.org/html/2410.18248v2);
  - a p90-of-gaps TTL per workload (Storage View) — [Storage View HTML](https://arxiv.org/html/2611.10178);
  - `expected_arrival` hints (SYMPHONY) — [SYMPHONY HTML v1](https://arxiv.org/html/2412.16434v1).
- **(b) Newsvendor fetch threshold. No prior art found.**
  - An arXiv abstract search for "newsvendor cache" returned no results — [arXiv search](https://arxiv.org/search/?query=newsvendor+cache&searchtype=abstract&abstracts=show&order=-announced_date_first&size=50).
  - The closest prefetch-timing rules are point-estimate lead times: "refreshes the context one lead time before the predicted return" ([Ask the Tool HTML](https://arxiv.org/html/2609.18849v1)); TokenCake's urgency-ranked predictive upload ([TokenCake HTML v5](https://arxiv.org/html/2510.18586v5)); SYMPHONY's greedy transfer on a hint ([SYMPHONY HTML v1](https://arxiv.org/html/2412.16434v1)); and CachedAttention's queue-aware fetching ([arXiv abs](https://arxiv.org/abs/2403.19708)).
  - Continuum's TTL argmax_τ P(τ,f)·Benefit − τ is newsvendor-like in structure, but it decides *retention* on the GPU, not when to start a read — [Continuum HTML v7](https://arxiv.org/html/2511.02230v7).
- **(c) Hazard-rate retention index. Partial precedents, none with a hazard rate or F(H|t)/τ(H).**
  - "hazard rate KV cache" returned no arXiv abstract results — [arXiv search](https://arxiv.org/search/?query=hazard+rate+KV+cache&searchtype=abstract&abstracts=show&order=-announced_date_first&size=50).
  - Closest precedents:
    - Continuum optimizes a holding horizon over an empirical CDF — [HTML v7](https://arxiv.org/html/2511.02230v7);
    - CacheWise evicts by the largest conditional expected remaining time, "approximat[ing] Belady" — [HTML v1](https://arxiv.org/html/2606.16824v1);
    - HEAR's Session-Aware score r = p·e^(−idle/400) — [HTML](https://arxiv.org/html/2610.06597);
    - MORI's idleness rank — [HTML v2](https://arxiv.org/html/2606.00866v2);
    - ThunderAgent's decay of elapsed tool time — [HTML v3](https://arxiv.org/html/2602.13692v3);
    - Adaptive KV Retention's break-even TTLs — [HTML](https://arxiv.org/html/2608.30830).
  - Using furthest-next-use (Belady) as an oracle is common: CacheWise, and the M365 residency study — [arXiv search listing](https://arxiv.org/search/?query=agentic+KV+cache+eviction&searchtype=abstract&abstracts=show&order=-announced_date_first&size=50).
- **(c) Preempting staged reads by the probability they are used in time. No prior art found.** The nearest relatives:
  - Continuum unpins "victims from pinned_requests with the latest program arrival time" — [HTML v7](https://arxiv.org/html/2511.02230v7);
  - MORI demotes not-yet-ready CPU programs, most idle first — [HTML v2](https://arxiv.org/html/2606.00866v2);
  - TokenCake's upload budget protects critical waiting requests — [HTML v5](https://arxiv.org/html/2510.18586v5).
- **(d) A price on host memory learned by dual ascent. Flag: Adaptive KV Retention (partial).**
  - Its α₂(λ) = max(0, λ − λ_crit)·β₃/C₂ is a mean-field price on host slots, indexed by *configured* offered load. The paper says a deployment "would require online load estimation and periodic recalibration", and it has no dual update and no per-session decoupling — [HTML](https://arxiv.org/html/2608.30830).
  - Other memory-time currencies are cost terms, not prices that adapt to capacity: InferCept's waste ([HTML v2](https://arxiv.org/html/2402.01869v2)), LAMPS's memory-over-time ([HTML v2](https://arxiv.org/html/2410.18248v2)), ThunderAgent's Space-Time Product ([HTML v3](https://arxiv.org/html/2602.13692v3)) and Continuum's MemUsage/M ([HTML v7](https://arxiv.org/html/2511.02230v7)).
  - Abstract searches for "KV cache memory price Lagrangian", "Whittle index LLM" and "shadow price LLM serving" found nothing relevant — [search 1](https://arxiv.org/search/?query=KV+cache+memory+price+Lagrangian&searchtype=abstract&abstracts=show&order=-announced_date_first&size=50), [search 2](https://arxiv.org/search/?query=Whittle+index+LLM&searchtype=abstract&abstracts=show&order=-announced_date_first&size=50), [search 3](https://arxiv.org/search/?query=shadow+price+LLM+serving&searchtype=abstract&abstracts=show&order=-announced_date_first&size=50).

### Inferences
- The related-work section should credit Continuum for per-tool empirical CDFs with first-word keys, and CacheWise for conditioning on elapsed time with Belady-style eviction. It can then claim novelty for:
  - the framework-sent distribution table and the division of labor ("framework reports, engine decides the risk");
  - the size- and price-dependent critical ratio for *starting reads*;
  - the retention index's horizon-optimal ratio with expected holding time, and its reductions to Belady and LRU;
  - λ learned online from demand minus capacity, with a weak-duality bound.
- Adaptive KV Retention should be cited as the nearest pricing precedent, and contrasted: configured load versus a learned multiplier; one TTL versus per-session, size-aware decisions.

### Gaps
- Prior-art checks used arXiv *abstract* search only. Full texts may use such terms (newsvendor, hazard, Lagrangian) without the abstract doing so. Non-arXiv venues (OSDI, SOSP, NSDI, EuroSys papers without an arXiv version) were not searched separately.
- The general caching-theory and prefetching literature (integrated prefetching and caching, ski rental, Whittle indices) and serverless keep-alive were out of scope by assignment.

---

## 3. Keep, offload, discard or prefetch: who decides what during a tool call?

### Takeaway
Most systems decide between *keep on GPU*, *offload to CPU DRAM* and *drop and recompute*. Only a few prefetch back before the tool returns: TokenCake, Ask the Tool, SYMPHONY, KVFlow and CacheScout. None of the tool-aware systems read from SSD on a forecast.

### Cited Findings
- InferCept chooses per interception between preserve and (chunked) discard, swapping within a budget. Swap-in happens after the call returns — [HTML v2](https://arxiv.org/html/2402.01869v2)
- LAMPS picks the waste-minimizing strategy per request before scheduling — [HTML v2](https://arxiv.org/html/2410.18248v2). Astraea preserves below a GPU watermark and takes the waste argmin above it — [HTML](https://arxiv.org/html/2512.14142)
- Continuum pins on the GPU for a TTL and then evicts; reload from CPU uses LMCache. It has no prefetch — [HTML v7](https://arxiv.org/html/2511.02230v7)
- TokenCake offloads to CPU during long calls and uploads predictively before the predicted return — [HTML v5](https://arxiv.org/html/2510.18586v5)
- MORI keeps the least idle programs in HBM and the next in DRAM, and drops the rest — [HTML v2](https://arxiv.org/html/2606.00866v2)
- Adaptive KV Retention offloads suspended requests to host DRAM at once and expires them after a load-indexed TTL — [HTML](https://arxiv.org/html/2608.30830)
- Ask the Tool keeps while remaining time is under a threshold and refreshes one lead time before the return — [HTML v1](https://arxiv.org/html/2609.18849v1)
- CacheWise evicts by predicted time-to-reuse, with swap-in on demand when CPU offload is enabled — [HTML v1](https://arxiv.org/html/2606.16824v1)
- ThunderAgent keeps acting programs on the GPU and pauses the shortest-context programs under thrashing — [HTML v3](https://arxiv.org/html/2602.13692v3)
- Sutradhara pins partial-prefill blocks during tool runs (HBM only) — [HTML](https://arxiv.org/html/2601.12967)
- KVFlow and CacheScout prefetch *agent prompts* (CPU→GPU) for the predicted next step or agent — [KVFlow HTML](https://arxiv.org/html/2507.07400v1), [CacheScout HTML](https://arxiv.org/html/2608.14624v1)

### Inferences
- "Read back from SSD in time for the next request" is not a decision any tool-aware paper we found makes. The literature's "fetch" is CPU→GPU, with a point-estimate lead time at best.

### Gaps
- We found no RL-rollout system, other than ThunderAgent, that describes how it keeps KV across tool calls (VenusRL's abstract mentions "KV Cache residency" only).
- We found no MCP-aware serving-engine paper. An abstract search for MCP and serving latency found none that passes MCP tool metadata to the engine — [arXiv search](https://arxiv.org/search/?query=Model+Context+Protocol+serving+latency&searchtype=abstract&abstracts=show&order=-announced_date_first&size=50)

---

## 4. Prediction: how does each system predict tool duration or the next request's time?

### Takeaway
- **Full distributions:** only Continuum (unconditional empirical CDF) and CacheWise (per-tool and argument-cluster distributions summarized as conditional expected remaining time).
- **Point estimates:** TokenCake (EWMA plus declared), Astraea (category mean), LAMPS (per API class), InferCept (elapsed time), SYMPHONY (`expected_arrival`).
- **Learned models predict *which* agent comes next,** not *when*: CacheScout, PBKV, KVFlow.
- **No prediction:** MORI, RR-Evict, Sutradhara, KVTether, Adaptive KV Retention.
- **Runtime progress instead of pre-call estimates:** Ask the Tool.

### Cited Findings
- **Distribution, unconditional, per tool.** Continuum uses an empirical CDF over S[f], with first-word keys and a global fallback — [HTML v7](https://arxiv.org/html/2511.02230v7)
- **Distribution per tool and argument cluster, conditional remaining time.** CacheWise, offline-trained — [HTML v1](https://arxiv.org/html/2606.16824v1)
- **Point estimates:**
  - TokenCake: EWMA (α = 0.5) blended with a user estimate (β = 0.5) — [HTML v5](https://arxiv.org/html/2510.18586v5);
  - Astraea: "the category's mean latency" — [HTML](https://arxiv.org/html/2512.14142);
  - LAMPS: per-API-class durations, with an OPT-125M classifier for output length — [HTML v2](https://arxiv.org/html/2410.18248v2);
  - InferCept: T̂_INT = t_now − t_call — [HTML v2](https://arxiv.org/html/2402.01869v2).
- **Return probability with idle decay.** HEAR's p is estimated "from its depth and prior inter-turn intervals" — [HTML](https://arxiv.org/html/2610.06597)
- **No per-request prediction:**
  - MORI's idleness over the last 5 cycles — [HTML v2](https://arxiv.org/html/2606.00866v2);
  - Adaptive KV Retention ("without requiring per-request wait prediction") — [HTML](https://arxiv.org/html/2608.30830);
  - Sutradhara ("inherently difficult to predict") — [HTML](https://arxiv.org/html/2601.12967);
  - RR-Evict ("requires no prediction of future arrivals or tool latency") — [arXiv search listing](https://arxiv.org/search/?query=tool+call+KV+cache&searchtype=abstract&abstracts=show&order=-announced_date_first&size=50).
- **Runtime progress.** Ask the Tool reports median error of about a fifth of the call's duration at its midpoint, against about a third for the best history predictor. At 90% of the call, error is under 10% against 80% — [HTML v1](https://arxiv.org/html/2609.18849v1)
- **Workload evidence that per-command keys matter.** CacheWise's bash-command P50s range from 0.1 s (Python one-liners) to 22 s (docker compose) — [HTML v1](https://arxiv.org/html/2606.16824v1). TraceLab reports heavy-tailed tool calls — [arXiv search listing](https://arxiv.org/search/?query=tool+latency+prediction+agent+serving&searchtype=abstract&abstracts=show&order=-announced_date_first&size=50). The Copilot trace study's idle-time predictor captures 86–90% of idle time — [arXiv abs](https://arxiv.org/abs/2608.00101)

### Inferences
- No verified paper has the *agent framework* send distributions. The forecasts are learned engine-side (Continuum, CacheWise) or arrive as point hints (SYMPHONY, TokenCake).
- Ask the Tool's evidence that history predictors degrade late in a call is a potential objection. It also motivates conditioning on idle time, which our design does and Continuum does not.

### Gaps
- We could not see the features of the Copilot idle-time predictor (abstract only).
- Ask the Tool's comparison set ("best published predictors") was not checked against conditional distributions like ours.

---

## 5. Decision: how does the prediction become a TTL, prefetch start, eviction order or priority? Derived or tuned?

### Takeaway
Rules derived from a cost model exist for keep versus discard (InferCept, LAMPS, Astraea), for the TTL (Continuum's expected-benefit argmax), and for break-even TTLs (Adaptive KV Retention). Prefetch starts and eviction scores are mostly tuned heuristics or point-estimate lead times. None derives a risk threshold that depends on the prefix size and a memory price.

### Cited Findings
- **TTL:**
  - Continuum: argmax_τ P(τ,f)·(T·η + Prefill-Reload) − τ, derived — [HTML v7](https://arxiv.org/html/2511.02230v7);
  - Adaptive KV Retention: t₂ = (β₃ − β₂)/α₂(λ), derived from calibrated costs — [HTML](https://arxiv.org/html/2608.30830);
  - Storage View: p90 of gaps, a fixed quantile — [HTML](https://arxiv.org/html/2611.10178).
- **Prefetch start:**
  - Ask the Tool: one lead time before the predicted return — [HTML v1](https://arxiv.org/html/2609.18849v1);
  - TokenCake: urgency-ranked under a budget, with weights set by hand — [HTML v5](https://arxiv.org/html/2510.18586v5);
  - KVFlow: next step — [HTML v1](https://arxiv.org/html/2507.07400v1);
  - CacheScout: argmax successor, gated by R ≥ R_min — [HTML v1](https://arxiv.org/html/2608.14624v1);
  - SYMPHONY: greedy on a hint — [HTML v1](https://arxiv.org/html/2412.16434v1).
- **Eviction order:**
  - CacheWise: largest predicted reuse time (Belady approximation) — [HTML v1](https://arxiv.org/html/2606.16824v1);
  - KVFlow: steps-to-execution — [HTML v1](https://arxiv.org/html/2507.07400v1);
  - MORI: idleness — [HTML v2](https://arxiv.org/html/2606.00866v2);
  - HEAR: p·e^(−idle/400) — [HTML](https://arxiv.org/html/2610.06597);
  - KVTether: lifecycle state, MRU among suspended — [HTML](https://arxiv.org/html/2609.39819);
  - Sutradhara: tag class then LRU — [HTML](https://arxiv.org/html/2601.12967);
  - ThunderAgent: shortest context first — [HTML v3](https://arxiv.org/html/2602.13692v3).
- **Scheduling priority:**
  - LAMPS: memory-over-time area — [HTML v2](https://arxiv.org/html/2410.18248v2);
  - Autellix: attained program service or critical path — [HTML v1](https://arxiv.org/html/2502.13965v1);
  - TokenCake: spatial priority with weights 2:1:0.5:1 — [HTML v5](https://arxiv.org/html/2510.18586v5).
- **Tuned constants:**
  - MORI: k = 5, ι ≥ 1/2, 5 s tick — [HTML v2](https://arxiv.org/html/2606.00866v2);
  - HEAR: 400 s decay, 50% hot cutoff — [HTML](https://arxiv.org/html/2610.06597);
  - TokenCake: 0.60 GPU usage, score > 1.0 — [HTML v5](https://arxiv.org/html/2510.18586v5);
  - Continuum: K = 100 — [HTML v7](https://arxiv.org/html/2511.02230v7).

### Inferences
- Our fetch rule is the only one found whose threshold rises with prefix size and memory price: q = λL/(1+λL). All the point-estimate lead-time rules correspond to a single fixed q, the special case our discussion section already names.

### Gaps
- Ask the Tool's thresholds and hysteresis values sit in an appendix that we did not read (Appendix D).

---

## 6. Tiers and scarcity: who models host DRAM and SSD, read time, and memory scarcity?

### Takeaway
Tool-aware systems model GPU HBM, or HBM plus CPU DRAM with linear transfer times. SSD appears in SYMPHONY (disk), in Lachesis (high-bandwidth flash), and in non-tool-aware tiering papers. Host scarcity is handled with capacity checks or admission (InferCept, TokenCake, MORI), with sizing (EfficientAgent), or with a configured mean-field price (Adaptive KV Retention). Nobody uses a learned Lagrangian price.

### Cited Findings
- **GPU and CPU DRAM with a transfer model or capacity checks:**
  - InferCept: CPU-capacity constraint; T_swap from bandwidth — [HTML v2](https://arxiv.org/html/2402.01869v2);
  - TokenCake: linear transfer time, CPU capacity as a hard check — [HTML v5](https://arxiv.org/html/2510.18586v5);
  - MORI: per-tier admission, no read-time term, SSD as future work — [HTML v2](https://arxiv.org/html/2606.00866v2);
  - Ask the Tool: HBM and HBM+DRAM — [HTML v1](https://arxiv.org/html/2609.18849v1);
  - Adaptive KV Retention: host DRAM budget of 375–380 GB, no SSD — [HTML](https://arxiv.org/html/2608.30830).
- **GPU only:**
  - ThunderAgent — [HTML v3](https://arxiv.org/html/2602.13692v3);
  - Sutradhara — [HTML](https://arxiv.org/html/2601.12967);
  - Continuum: GPU pin, with reload from CPU as a cost term — [HTML v7](https://arxiv.org/html/2511.02230v7).
- **SSD or flash:**
  - SYMPHONY: HBM/DRAM/disk, always keeping a disk copy — [HTML v1](https://arxiv.org/html/2412.16434v1);
  - Lachesis: HBM and high-bandwidth flash with equal read bandwidth — [HTML](https://arxiv.org/html/2610.08378);
  - SARIA: DRAM–SSD for agents, with endurance as the cost — [arXiv search listing](https://arxiv.org/search/?query=agent+KV+cache+SSD&searchtype=abstract&abstracts=show&order=-announced_date_first&size=50);
  - Where Should the KV Cache Live? (simulator: prefetching rarely pays for its bandwidth) — [HTML](https://arxiv.org/html/2609.16215);
  - CachedAttention (multi-turn DRAM+SSD) — [arXiv abs](https://arxiv.org/abs/2403.19708).
- **Scarcity as sizing.** EfficientAgent finds offloading pays only when the host tier holds the pool's "reuse working set". A tier sized to it cut recomputed tokens by 93% and end-to-end time by 39% on SWE-bench Verified — [arXiv abs](https://arxiv.org/abs/2609.33762)
- **Scarcity as a configured price.** Adaptive KV Retention's α₂(λ) — [HTML](https://arxiv.org/html/2608.30830)

### Inferences
- Our setting has scarce host memory, SSD reads of tens of µs per token, and reads that hold host memory while in flight. The tool-aware literature has not studied it. EfficientAgent's working-set result and the simulator's "prefetch does not pay unless predictable" result are the empirical findings we have to address: our price and our q are designed to suppress early reads in exactly those regimes.

### Gaps
- We did not verify whether any 2026 tool-aware paper counts storage bandwidth shared between reads and writes, as our constraint does.

---

## 7. Hints from the framework: what do agent frameworks or protocols pass to the engine?

### Takeaway
The hints passed today are identity (program or session id), structure (steps-to-execution, lifecycle states, tags), point timing (`expected_arrival`, a declared duration), content directives, and session completion. None of the verified papers passes a *distribution* of the gap. HEAR's "explicitly marked predictions" is the most general channel, but it leaves the format open. Ask the Tool passes runtime progress.

### Cited Findings
- **SYMPHONY:** `session_id`, `model_id`, `expected_arrival`, `ordered`, `priority` — [HTML v1](https://arxiv.org/html/2412.16434v1)
- **KVFlow:** current agent identity plus steps-to-execution of all agents, sent in request metadata — [HTML v1](https://arxiv.org/html/2507.07400v1)
- **TokenCake:** a user-supplied duration blended with history — [HTML v5](https://arxiv.org/html/2510.18586v5)
- **Continuum:** `program_id`; tool names are parsed by the engine; program termination unpins — [HTML v7](https://arxiv.org/html/2511.02230v7)
- **MORI:** `program_id` only — "MORI needs to know only which program a tool call belongs to" — [HTML v2](https://arxiv.org/html/2606.00866v2)
- **HEAR, harness → engine:** "known future uses and explicitly marked predictions", "waiting-time constraints", and retention or release requests — [HTML](https://arxiv.org/html/2610.06597)
- **HEAR, engine → harness:** residency tiers, capacity, queue state and cost estimates — [HTML](https://arxiv.org/html/2610.06597)
- **KVTether:** append, replace, remove, fork, spawn, one_off, pin, unpin — [HTML](https://arxiv.org/html/2609.39819)
- **Sutradhara:** `tag_kv_blocks`, `set_reuse_priority`, partial-prefill APIs — [HTML](https://arxiv.org/html/2601.12967)
- **Storage View:** workload id, session id, session completion — [HTML](https://arxiv.org/html/2611.10178)
- **Lachesis:** harness-reported turn and sub-agent boundaries — [HTML](https://arxiv.org/html/2610.08378)
- **Ask the Tool:** progress events keyed by call id over a side channel — [HTML v1](https://arxiv.org/html/2609.18849v1)
- **Engine-level "declared duration" hints.** Ask the Tool lists declared-duration mechanisms in SGLang, Mooncake, vLLM and TensorRT-LLM (its classification; not checked in engine docs) — [HTML v1](https://arxiv.org/html/2609.18849v1)
- **Semantic Variables (Parrot)** expose application dataflow — [arXiv abs](https://arxiv.org/abs/2405.19888)
- **Inferlets (Pie)** move KV policy into the application — [arXiv abs](https://arxiv.org/abs/2510.24051)

### Inferences
- A per-key distribution table, plus an explicit session-end entry, is a strictly richer timing hint than any verified interface. HEAR's protocol could carry it. Our design's argument for a distribution rather than a point estimate is not made in any of these papers: the engine, not the framework, knows the size and the price.

### Gaps
- We did not verify the exact engine APIs that Ask the Tool cites (SGLang, Mooncake, vLLM, TensorRT-LLM), nor whether MCP or agent SDKs (LangGraph, AutoGen) forward tool-timing metadata to engines. No arXiv paper on this was found.
