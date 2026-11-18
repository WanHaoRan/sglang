# Multi-tier KV and prefix caching (GPU / host DRAM / SSD / remote): prefetch, eviction, load-vs-recompute, priced memory, and session-gap traces (2023 – Nov 2026)

How to read these notes. Design shorthand: (a) per-tool distributional gap forecasts sent by the agent framework and conditioned on idle time; (b) SSD fetch started when F(x|t) >= q = λL/(1+λL) (newsvendor critical ratio); (c) retain while hazard × read time >= q, evict by (1+λL)·max_H F(H|t)/τ(H), which reduces to Belady/LRU; preempt staged reads in increasing F(r|t); (d) Lagrangian price λ on host memory learned by multiplicative dual ascent, with a weak-duality lower bound. Prior-art scale: HIGH = a reviewer could cite it against one of our main claims; MEDIUM = overlaps one ingredient; LOW = context or baseline. Verification: every arXiv paper below was resolved through the export.arxiv.org API (id_list or search records) or through its arxiv.org abstract page on 2026-11-17. Mechanism details come from the arXiv HTML or PDF full text where those links are cited. Searches used only generic technical terms.

## Q0. Which arXiv papers are closest to the design, and could any be read as prior art? (Ranked, closest first)

### Takeaway
No verified paper combines the four parts. None uses a newsvendor critical ratio to time SSD reads, a hazard × read-time retention rule with Belady/LRU limits, preemption of staged reads by F(r|t), or a host-memory shadow price learned by dual ascent with a weak-duality bound. Five recent papers each overlap one main claim and must be cited and distinguished:
- **Continuum** chooses a GPU-pinning TTL from per-tool empirical CDFs and charges memory in seconds of latency (HIGH).
- **Alibaba's "KVCache Cache in the Wild"** ranks evictions by per-category reuse-time distributions evaluated at time since last access (MEDIUM-HIGH).
- **Intent-Aware Caching** learns per-(intent, turn) log-normal gaps. It reports that a full conditional-probability score did worse than a median threshold (MEDIUM-HIGH).
- **The GitHub Copilot trace study** predicts a survival curve of idle time and conditions it on elapsed idle (MEDIUM).
- **KVLearn** puts a per-byte-second memory cost into a retention score, with a PI-controlled threshold (MEDIUM).

### Cited Findings

**R1. Continuum (closest prior art; agent-specific, covered in depth by another researcher)**
- Record: [arXiv:2511.02230](https://arxiv.org/abs/2511.02230), "Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live". Hanchen Li et al., v1 Nov 2025, v7 Sep 2026. The arXiv record lists no venue ([abs](https://arxiv.org/abs/2511.02230)). A later paper's reference list cites it as an ICLR 2026 Workshop on Lifelong Agents paper ([2610.22863 PDF](https://arxiv.org/pdf/2610.22863)).
- Mechanism: It pins a finished request's KV in GPU memory for a TTL τ* = argmax_τ P(τ,f)·(T·η + Prefill-Reload(r)) − τ ([HTML](https://arxiv.org/html/2511.02230)).
  - P(τ,f) is "the empirical CDF derived from historical tool-call records S[f]" for tool f.
  - T is the sliding-window average queueing delay of evicted requests, and η is a "memoryfulness" factor.
  - Prefill-Reload(r) is the prefill time, or the reload time when CPU offloading is on.
- Cost model: pinning costs τ·MemUsage(r)/M, where M is the average GPU footprint of active requests: pinning r "adds τ latency to approximately k other requests". Benefit and cost are "measured in units of time" ([HTML](https://arxiv.org/html/2511.02230)).
- Prefill-Reload comes from offline profiling: average GPU–CPU offload throughput plus "a quadratic curve" fitted to prefill time vs context length. A minimal variant uses a fixed TTL derived by assuming Exp(1) tool durations and η = 1 ([HTML](https://arxiv.org/html/2511.02230)).
- Relation to our design: parallels (a) and (c), and partly (d).
  - Shares: per-tool duration distributions used by the engine; a retain/release decision that weighs expected avoided reload or prefill against memory occupancy, in time units; a quadratic prefill model.
  - Differs:
    - It retains in GPU memory only, not host memory or SSD.
    - It has no SSD fetch timing and no staged-read preemption.
    - The tool is the key, without the command's first word, and the CDF is the engine's own history rather than a table sent by the agent framework.
    - It sets a TTL from the unconditional CDF when the tool call starts. I found no re-conditioning on elapsed idle in the sections read.
    - Its memory cost is a fixed blocking model (MemUsage/M) rather than a learned shadow price, and it gives no lower bound.
- Prior-art flag: **HIGH** for "per-tool distributional gap forecasts used by the serving engine" and for "memory occupancy priced in seconds of latency".

**R2. KVCache Cache in the Wild (Alibaba)**
- Record: [arXiv:2506.02634](https://arxiv.org/abs/2506.02634), "KVCache Cache in the Wild: Characterizing and Optimizing KVCache Cache at a Large Cloud Provider". Jiahao Wang et al., 2025 (v5 Feb 2026). USENIX ATC '25, per the arXiv comment ([abs](https://arxiv.org/abs/2506.02634)).
- Characterization: "for a specific request category, the pattern tends to be predictable" ([abs](https://arxiv.org/abs/2506.02634)). "Given a time period and request category, the probability distribution of KV$ reuse time follows an exponential distribution", where a category is the request type plus the turn number ([PDF](https://arxiv.org/pdf/2506.02634)).
- Eviction policy: a workload-aware policy ranks blocks lexicographically by (ReuseProb_w(t), offset) ([PDF](https://arxiv.org/pdf/2506.02634)).
  - ReuseProb_w(t) comes from an exponential fit to recently sampled reuse data of workload w, evaluated at t, the time since the block's last access.
  - Head blocks get priority, frequency is dropped, and the time span of the probability calculation is limited.
- Relation to our design: parallels (a) and (c).
  - Shares: per-category reuse-time distributions, learned online and evaluated at the elapsed idle time, drive eviction.
  - Differs:
    - Categories are provider request types and turns, not tool calls.
    - The exponential fit cannot represent multi-modal tool durations (inference).
    - The objective is hit ratio, with no read-time, size or price weighting, no fetch rule and no price.
- Prior-art flag: **MEDIUM-HIGH** for "category-keyed reuse-time distributions conditioned on time since last access, driving eviction". This is the natural citation for "per-category reuse is predictable".

**R3. Intent-Aware Caching (IAC)**
- Record: [arXiv:2611.09715](https://arxiv.org/abs/2611.09715), "Intent-Aware Caching for Efficient LLM Serving". Kazi Hasan Ibn Arif et al., Nov 2026. NeurIPS 2026, per the arXiv comment ([abs](https://arxiv.org/abs/2611.09715)).
- Mechanism: for each state (intent i, turn t), IAC learns two statistics from recent logs ([HTML](https://arxiv.org/html/2611.09715)):
  - the continuation probability P(t+1|i,t) = C(i,t+1)/C(i,t);
  - log-normal inter-turn delays, log D ~ N(μ, σ²).
- It evicts by the lexicographic priority (1[age < median delay exp(μ)], P(t+1|i,t), size), mimicking the TB+S (Threshold-Belady with size) oracle ([HTML](https://arxiv.org/html/2611.09715)).
- A "Full-Horizon" variant scores entries by "the probability that the next turn arrives within the next X seconds given the current age of the entry". It "stays below IAC at every capacity on ShareChat": the continuous score dominates the ranking, and the full per-state distribution is noisy while the median is robust ([HTML](https://arxiv.org/html/2611.09715)).
- It is implemented in SGLang and evaluated on WildChat and ShareChat with their real timestamps ([HTML](https://arxiv.org/html/2611.09715)).
- Relation to our design: parallels (a) and (c), including the Belady-family target.
  - Differs: single-tier GPU radix cache; the objective is hit rate; the median is a fixed threshold rather than hazard × read time against a size- and price-dependent q; no fetch or price; categories are chat intents, not tools.
- Prior-art flag: **MEDIUM-HIGH** for per-category distributional gap forecasts in eviction. It also tested an age-conditioned arrival-probability score and found it inferior, which reviewers may raise against our use of F(H|t).

**R4. Agentic Coding in the Wild (GitHub Copilot production traces)**
- Record: [arXiv:2608.00101](https://arxiv.org/abs/2608.00101), "Agentic Coding in the Wild: Characterizing GitHub Copilot Traces at Production Scale". Banruo Liu et al., Jul 2026. No venue listed ([abs](https://arxiv.org/abs/2608.00101)).
- Mechanism: the paper trains LightGBM quantile regressors (12 quantiles on log idle time) whose output is a survival curve S(t) = Pr(idle > t) at each turn boundary ([HTML](https://arxiv.org/html/2608.00101)).
  - After t0 seconds of idleness it uses S(t)/S(t0), "requiring no additional efforts".
  - The intended use is "evict KV cache once Pr(idle > 60 s) drops below some threshold".
  - Results: ROC-AUC 0.73 for "> 60 s"; it captures 86–90% of idle time.
- Relation to our design: parallels (a), a distributional forecast with conditioning on time already idle, and (c), threshold-based eviction.
  - Differs: it forecasts user idle time at turn boundaries from session features, not tool-call gaps; the operator picks the threshold; there is no fetch timing, price or engine policy.
- Prior-art flag: **MEDIUM** for "distributional gap forecast conditioned on elapsed idle time for KV offloading".

**R5. KVLearn ("To Keep or Not to Keep")**
- Record: [arXiv:2610.28637](https://arxiv.org/abs/2610.28637), "To Keep or Not to Keep: Learning KV Cache Retention in Disaggregated LLM Serving Systems". Dong Liu et al., Aug 2026. ACM SYSTOR 2026, per the arXiv comment ([abs](https://arxiv.org/abs/2610.28637)).
- Mechanism: an online MLP predicts the reuse probability P̂(b). The retention score is CARS(b) = P̂(b)·[R(b) − T(b)] − U(b, Δt_est), with R the recompute cost and T the transfer cost ([PDF](https://arxiv.org/pdf/2610.28637)).
  - U = γ·|b|·Δt "encodes the per-byte-per-second opportunity cost of occupying pool DRAM", with γ = R̄·λ/M_S (running-average recompute cost × block arrival rate / pool capacity).
  - Blocks are admitted if CARS > θ. A PI controller (ATC) adjusts θ "from observed hit rates and memory pressure" to hold pool utilization near ρ* = 0.85. Eviction is in ascending CARS order ([PDF](https://arxiv.org/pdf/2610.28637)).
- Relation to our design: parallels (c) and (d).
  - Shares: memory-time charged in the same units as miss cost; online adaptation of a threshold to memory pressure.
  - Differs:
    - γ is a fixed heuristic and θ tracks a utilization target. It is not a Lagrange multiplier learned by dual ascent on demand minus capacity, and there is no duality bound.
    - It uses a horizon reuse probability rather than a time distribution or hazard.
    - It manages admission into a P/D-disaggregated pool, not SSD fetch timing, and has no agent forecasts.
- Prior-art flag: **MEDIUM** for "memory-time price inside a retention score, adapted online".

**R6. TempoKV**
- Record: [arXiv:2609.35065](https://arxiv.org/abs/2609.35065), "TempoKV: Timely Staging of LLM KV Caches for Memory-Semantic Flash". Jay H. Park et al., Sep 2026. No venue listed ([abs](https://arxiv.org/abs/2609.35065)).
- Mechanism: a reusable-KV hit becomes a metadata-only "claim". The claim is committed (staged from SSD to a protected fast tier) when the runtime-estimated time-to-use (TTU, the time until retrieval begins) falls to the provider-estimated time-to-ready (TTR). The rule uses laxity L̂ = TTU − TTR ([HTML](https://arxiv.org/html/2609.35065)).
  - If protected capacity is lacking, the claim stays "Planned". Once committed, a claim is never revoked or preempted.
  - It cuts protected fast-tier byte-time per request by 63–91% vs immediate staging (vLLM + LMCache) ([abs](https://arxiv.org/abs/2609.35065)).
- Relation to our design: parallels (b), and its byte-time metric matches our token-seconds held by reads.
  - Differs: it uses deterministic point estimates for requests already in the runtime's queue, not pre-arrival gap distributions; it has no risk threshold that grows with size or price. Its irrevocable commitments contrast with our preemption of staged reads by F(r|t).
- Prior-art flag: **MEDIUM** for "just-in-time SSD staging that minimizes held memory-time"; **LOW** for a distributional threshold.

**R7. TokenCake (agent-specific)**
- Record: [arXiv:2510.18586](https://arxiv.org/abs/2510.18586), "TokenCake: A KV-Cache-centric Serving Framework for LLM-based Multi-Agent Applications". Zhuohang Bian et al., Oct 2025 (v5 Sep 2026). EuroSys '27, per the arXiv comment ([abs](https://arxiv.org/abs/2510.18586)).
- Mechanism: its Temporal Scheduler offloads a stalled agent's KV to CPU during function calls and "as the predicted completion time approaches … begins a gradual upload" ([HTML](https://arxiv.org/html/2510.18586)).
  - The predicted duration is an EWMA of observed execution times, blended with a user-supplied estimate.
  - It offloads only if the predicted stall covers a round-trip transfer, a waiting request fits in the freed blocks, and GPU memory pressure exceeds a threshold.
- Relation to our design: parallels (b) and (c) between GPU and CPU.
  - Differs: point estimates, fixed thresholds, no distributions or conditioning, and no host-memory price.
- Prior-art flag: **MEDIUM** for "reload timed ahead of a learned tool-duration forecast".

**R8. SAECache ("Not All Tokens Are Worth Caching")**
- Record: [arXiv:2605.18825](https://arxiv.org/abs/2605.18825), "Not All Tokens Are Worth Caching: Learning Semantic-Aware Eviction for LLM Prefix Caches". Shaoke Fang et al., May 2026. No venue listed ([abs](https://arxiv.org/abs/2605.18825)).
- Mechanism: a multi-queue prefix-cache eviction policy with a continuation classifier and per-token-type weights learned from miss-after-eviction feedback ([HTML](https://arxiv.org/html/2605.18825)).
  - "inter-turn intervals in each session can be well captured by log-normal distributions, with their parameters varying across different sessions".
  - μ and σ are re-estimated online by maximum likelihood from elapsed times at hits.
- Relation to our design: parallels (a) and (c), with a timing distribution inside eviction priority.
  - Differs: GPU hit rate only; keyed by session or queue type, not tool; no fetch and no price.
- Prior-art flag: **LOW-MEDIUM**.

**R9. CachedAttention (its multi-tier KV store is called AttentionStore)**
- Record: [arXiv:2403.19708](https://arxiv.org/abs/2403.19708), "Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention". Bin Gao et al., 2024. USENIX ATC 2024 ([abs](https://arxiv.org/abs/2403.19708)).
- Mechanism: KV of multi-turn sessions is kept in host DRAM and disks, with layer-wise pre-loading and asynchronous saving ([abs](https://arxiv.org/abs/2403.19708)).
  - Scheduler-aware fetching: a "look-ahead prefetching window" over the job queue, sized from the host memory available for prefetching and the average session KV size, prefetches disk-resident KV of waiting jobs into host memory ([HTML](https://arxiv.org/html/2403.19708)).
  - Scheduler-aware eviction: a "look-ahead eviction window" prefers evicting KV whose jobs sit near the tail of the queue or outside the window ([HTML](https://arxiv.org/html/2403.19708)).
- Relation to our design: parallels (b) and (c) driven by queue hints.
  - Differs: only requests that have already arrived are visible, so nothing is read during the tool gap; window sizes come from capacity rather than risk; no probabilities, size-dependent thresholds or prices.
- Prior-art flag: **LOW**. It is the canonical "scheduler-aware prefetch/eviction" citation.

**R10. MORI (agent-specific)**
- Record: [arXiv:2606.00866](https://arxiv.org/abs/2606.00866), "Idleness is Relative: Exploiting Tool-Call Idle Windows for Offloading in Agentic Systems with MORI". Tian Xia et al., May 2026. No venue listed ([abs](https://arxiv.org/abs/2606.00866)).
- Mechanism: it ranks agentic programs by "idleness", the recent fraction of time spent in tool calls. The least idle are kept in HBM, the next in DRAM, and the rest are dropped (admission control at both tiers). Placement is sticky, and promotion is proactive ([HTML](https://arxiv.org/html/2606.00866)).
  - Premise: "tool-call durations vary by orders of magnitude and are unknown in advance", and existing schedulers leave DRAM to LRU, so offloaded programs thrash ([HTML](https://arxiv.org/html/2606.00866)).
- Relation to our design: parallels (c) across GPU and host.
  - Differs: it explicitly avoids "a model of individual tool-call durations", ranking by each program's own recent history. No price and no SSD.
- Prior-art flag: **LOW-MEDIUM**. It is a strong baseline for two-tier retention across tool gaps.

**R11. Pensieve**
- Record: [arXiv:2312.05516](https://arxiv.org/abs/2312.05516), "Stateful Large Language Model Serving with Pensieve". Lingfan Yu et al., Dec 2023 (v3 Oct 2024). EuroSys '25, per the paper's ACM reference block ([HTML](https://arxiv.org/html/2312.05516)).
- Mechanism: a two-tier GPU–CPU conversation cache that evicts token chunks (configurable; 32 tokens in their experiments) in ascending "retention value". The value is a chunk's recomputation cost divided by the time since the conversation was last active ([HTML](https://arxiv.org/html/2312.05516)).
  - Leading-end chunks are preferred for eviction because they are cheaper to recompute.
  - It uses ahead-of-time swapping and pipelined transfer; dropped context is recomputed.
  - Experiments assume an average user think time of 60 s (varied up to 600 s) ([HTML](https://arxiv.org/html/2312.05516)).
- Relation to our design: parallels (c), as a cost-per-idle-time index, and recomputation of a lost prefix.
  - Differs: recency is the proxy for return rate; there are no forecasts, no SSD and no price.
- Prior-art flag: **LOW**.

**R12. EfficientAgent**
- Record: [arXiv:2609.33762](https://arxiv.org/abs/2609.33762), "EfficientAgent: What Makes KV Cache Offloading Work for Concurrent Agents?". Kunming Shao et al., Sep 2026. No venue listed ([abs](https://arxiv.org/abs/2609.33762)).
- Mechanism: offloading pays only when the host tier holds the agent pool's "reuse working set". A stack-distance model estimates that working set to size the tier. When the tier is too small, a runtime policy stops writing large refills of evicted context ([abs](https://arxiv.org/abs/2609.33762)).
- Relation to our design: (c)/(d) capacity reasoning about host-memory scarcity during tool waits.
  - Differs: sizing and write-admission heuristics, with no forecasts or prices.
- Prior-art flag: **LOW**.

**R13. FlatCake**
- Record: [arXiv:2610.38355](https://arxiv.org/abs/2610.38355), "FlatCake: A Flat KVCache Storage Architecture for LLM Serving". Xin Huang et al., Oct 2026. No venue listed ([abs](https://arxiv.org/abs/2610.38355)).
- Mechanism: "holding host staging buffers for in-flight prefixes causes contention under load that aborts prefetches" ([abs](https://arxiv.org/abs/2610.38355)).
  - FlatCake treats DRAM and SSD as peers with session-contiguous placement and GDS reads.
  - On a production Claude Code trace it reports a 94% hit rate vs 64% for Mooncake ([abs](https://arxiv.org/abs/2610.38355)).
  - It attributes Mooncake's SSD read amplification partly to "revoked prefetches" ([HTML](https://arxiv.org/html/2610.38355)).
- Relation to our design: it is empirical motivation for counting reads in progress against host capacity and for preempting staged reads. It solves the problem architecturally rather than by policy.
- Prior-art flag: **LOW**.

**R14. Strata (closest arXiv paper to the HiCache line)**
- Record: [arXiv:2508.18572](https://arxiv.org/abs/2508.18572), "Strata: Hierarchical Context Caching for Long Context Language Model Serving". Zhiqiang Xie et al., Aug 2025. The arXiv comment says "under peer review" ([abs](https://arxiv.org/abs/2508.18572)). TempoKV's reference list cites it as OSDI ([TempoKV HTML](https://arxiv.org/html/2609.35065)).
- Mechanism:
  - GPU-assisted I/O counters fragmented paged layouts ([abs](https://arxiv.org/abs/2508.18572)).
  - Cache-aware batch formation caps a "loading-bound" ratio of aggregated load to compute, and fills unavoidable stalls with decode work ([HTML](https://arxiv.org/html/2508.18572)).
  - It reports up to 3.2×, 2.6× and 1.9× throughput at equal TTFT vs SGLang-HiCache, vLLM-LMCache and TensorRT-HiCache ([HTML](https://arxiv.org/html/2508.18572)).
- Relation to our design: it acts after arrival, on the read cost r_i and the queue wait Ŵ_i. There is no pre-arrival fetch, retention forecast or price.
- Prior-art flag: **LOW**.

**R15. LMCache**
- Record: [arXiv:2510.09665](https://arxiv.org/abs/2510.09665), "LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference". Yuhan Liu et al., Oct 2025. No venue listed ([abs](https://arxiv.org/abs/2510.09665)).
- Mechanism: an engine-agnostic KV layer for vLLM and SGLang with many storage backends ([HTML](https://arxiv.org/html/2510.09665)).
  - It "exploits this idle interval [queueing] to prefetch the queued queries' KV cache from slower storage tiers into faster ones", with a configurable target tier.
  - A controller aggregates batched_admit and batched_evict events for KV-aware routing.
- Relation to our design: parallels (b), but only for queued requests. No forecasts or price.
- Prior-art flag: **LOW**.

**R16. Mooncake**
- Record: [arXiv:2407.00079](https://arxiv.org/abs/2407.00079), "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving". Ruoyu Qin et al., Jun 2024 (v5 Oct 2026) ([abs](https://arxiv.org/abs/2407.00079)).
  - Published at FAST '25 (Best Paper) as "Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot" ([USENIX](https://usenix.org/conference/fast25/presentation/qin)).
- Mechanism: P/D disaggregation over a cluster-wide CPU DRAM/SSD KV pool. Prefill-node selection balances reuse, load and the TTFT SLO, and a prediction-based early rejection policy handles overload ([abs](https://arxiv.org/abs/2407.00079); [HTML](https://arxiv.org/html/2407.00079)).
  - The host cache "can use … LRU, LFU, or algorithms based on request characteristics".
  - On its trace LRU beat LFU and LengthAwareCache, and growing capacity from 1,000 to 50,000 blocks raised the hit ratio from 34% to 55% ([HTML](https://arxiv.org/html/2407.00079)).
- Relation to our design: the multi-tier substrate. No forecasting or pricing.
- Prior-art flag: **LOW**.

**R17. MemServe**
- Record: [arXiv:2406.17565](https://arxiv.org/abs/2406.17565), "MemServe: Context Caching for Disaggregated LLM Serving with Elastic Memory Pool". Cunchen Hu et al., Jun 2024. No venue listed ([abs](https://arxiv.org/abs/2406.17565)).
- Mechanism: MemPool manages distributed memory and KV across instances. A global scheduler routes requests using prompt-tree locality and a cost model. Its global prompt trees carry a TTL, "commonly in minutes", because the scheduler cannot see local evictions ([HTML](https://arxiv.org/html/2406.17565)).
- Relation to our design: substrate and routing. The TTL serves metadata staleness, not retention.
- Prior-art flag: **LOW**.

**R18. Marconi**
- Record: [arXiv:2411.19379](https://arxiv.org/abs/2411.19379), "Marconi: Prefix Caching for the Era of Hybrid LLMs". Rui Pan et al., Nov 2024. MLSys 2025 ([abs](https://arxiv.org/abs/2411.19379)).
- Mechanism: admission is based on "forecasts of their reuse likelihood across a taxonomy of different hit scenarios" ([abs](https://arxiv.org/abs/2411.19379)).
  - FLOP-aware eviction combines recency with "compute savings … normalized against the space they consume". The paper discusses its relation to GDSF ([HTML](https://arxiv.org/html/2411.19379)).
- Relation to our design: (c). A savings-per-byte index, like our normalization by memory held.
  - Differs: no time forecasts, a tuned recency weight, single tier.
- Prior-art flag: **LOW**.

**R19. RAGCache**
- Record: [arXiv:2404.12457](https://arxiv.org/abs/2404.12457), "RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation". Chao Jin et al., Apr 2024 ([abs](https://arxiv.org/abs/2404.12457)).
  - Published in ACM TOCS 44(1), Nov 2025 ([author PDF](https://chaojin0310.github.io/files/papers/TOCS25-RAGCache.pdf)).
- Mechanism: a knowledge tree spanning GPU and host memory, managed by PGDSF ([PDF](https://arxiv.org/pdf/2404.12457)).
  - PGDSF uses a GreedyDual-style priority: Clock + AvgCost × Frequency in the listing, with node size also accounted for. Clock rises to the maximum priority of evicted nodes.
  - Nodes are swapped out to host only once, and requests are reordered for cache awareness.
  - In its ablation, PGDSF gave lower average TTFT than GDSF, LRU and LFU at every host-memory size tested (8–128 GiB) ([PDF](https://arxiv.org/pdf/2404.12457)).
- Relation to our design: (c). A cost-aware GreedyDual index whose aging clock acts like a threshold.
- Prior-art flag: **LOW**.

**R20. LARU ("Toward Robust and Efficient ML-Based GPU Caching")**
- Record: [arXiv:2509.20979](https://arxiv.org/abs/2509.20979), "Toward Robust and Efficient ML-Based GPU Caching for Modern Inference". Peng Chen et al., Sep 2025. No venue listed ([abs](https://arxiv.org/abs/2509.20979)).
- Mechanism: among the top-k LRU candidates, LARU evicts the one with the largest predicted next-request time (LightGBM). After a misprediction-induced miss it falls back to safe LRU evictions and halves its confidence ([HTML](https://arxiv.org/html/2509.20979)).
- Relation to our design: (c) with a point estimate of reuse time and robustness to bad predictions. No distribution, price or tiers.
- Prior-art flag: **LOW**.

**R21–R35 (context and baselines; all LOW unless noted)**
- **R21. When Fancy Eviction Fails.** [arXiv:2609.28870](https://arxiv.org/abs/2609.28870), Yiyu Liu et al., 2026. It evaluates 14 eviction algorithms on production traces of more than 20B tokens ([abs](https://arxiv.org/abs/2609.28870)).
  - LRU is "surprisingly robust" because "prefix reuse is dominated by the steady pacing of active sessions", but headroom to Belady remains.
  - It recommends quick demotion, compute-aware eviction and capacity-adaptive granularity. A compute-aware "BeladyCompute" approximates the ILP optimum ([HTML](https://arxiv.org/html/2609.28870)).
  - Relation: a baseline and counterpoint for (c). Our index reduces to LRU with one pooled distribution.
- **R22. RR-Evict.** [arXiv:2609.32278](https://arxiv.org/abs/2609.32278), Zaifeng Pan et al., 2026. It identifies "recency synchronization", where LRU concentrates evictions on a few agents' whole histories ([abs](https://arxiv.org/abs/2609.32278)).
  - Fix: round-robin tail-chunk eviction across idle agent trajectories. It "requires no prediction of future arrivals or tool latency"; implemented in SGLang; up to −75.4% P99 TTFT vs LRU.
  - Relation: a forecast-free alternative to (c).
- **R23. WAKMan.** [arXiv:2610.36355](https://arxiv.org/abs/2610.36355), Suhas Kumar Morisetty et al., 2026. "agentic steps return within seconds and favor recency", while popular prompts and long conversations favor frequency ([abs](https://arxiv.org/abs/2610.36355)).
  - It picks the eviction policy from live traffic, for +5 pp hit rate over LRU.
- **R24. Where Should the KV Cache Live?** [arXiv:2609.16215](https://arxiv.org/abs/2609.16215), Srikanta Datta Tumkur et al., 2026. A simulator over HBM, DRAM and SSD ([HTML](https://arxiv.org/html/2609.16215)).
  - Prefetch was "uniformly negative" on a bandwidth-charged link: even an oracle prefetcher was slower than no prefetch in over half the cells, because "a speculative fetch delays the demand fetch".
  - Their "predicted reuse" policy was "byte-identical to recency" as shipped.
  - Relation: counter-evidence that (b) must address.
- **R25. Predictive Multi-Tier Memory Management.** [arXiv:2604.26968](https://arxiv.org/abs/2604.26968), Sanjeev Rao Ganjihal (sole author), 2026. A Bayesian reuse predictor (Beta priors over 16 block/transition-type pairs) drives EMA-scored eviction and prefetch over a six-tier hierarchy ([abs](https://arxiv.org/abs/2604.26968)).
  - Caution: v2's comment says it corrects seven bibliography entries and narrows a claim ([abs](https://arxiv.org/abs/2604.26968)).
- **R26. PCR.** [arXiv:2603.23049](https://arxiv.org/abs/2603.23049), Wenfeng Wang et al., 2026 (RAG). A "look-ahead LRU replacement policy that uses pending requests in the scheduler queue", queue-based SSD prefetching, and layer-wise overlap of loading and compute ([abs](https://arxiv.org/abs/2603.23049)).
- **R27. PEEK.** [arXiv:2607.02525](https://arxiv.org/abs/2607.02525), Bing Xie et al., 2026. A radix tree over the pending queue, with an eviction hook that "protects blocks ancestral to queued demand" ([abs](https://arxiv.org/abs/2607.02525)).
- **R28. SARIA.** [arXiv:2610.31169](https://arxiv.org/abs/2610.31169), Hui Li et al., 2026. Agent KV across DRAM and SSD ([abs](https://arxiv.org/abs/2610.31169)):
  - It persists to SSD only blocks with reuse beyond their DRAM residency, and promotes SSD-hit blocks into DRAM after prefill.
  - It paces DRAM reads while SSD transfers are active. Built on LMCache MP, evaluated with TraceLab workloads.
- **R29. Cake.** [arXiv:2410.03065](https://arxiv.org/abs/2410.03065), "Compute Or Load KV Cache? Why Not Both?", Shuowei Jin et al., 2024. See Q3.
- **R30. HCache.** [arXiv:2410.05004](https://arxiv.org/abs/2410.05004), Shiwei Gao et al., EuroSys 2025. See Q3.
- **R31. CacheFlow.** [arXiv:2604.25080](https://arxiv.org/abs/2604.25080), Sean Nian et al., 2026. See Q3.
- **R32. EVICPRESS.** [arXiv:2512.14946](https://arxiv.org/abs/2512.14946), Shaoting Feng et al., 2025. See Q4.
- **R33. AdaptCache.** [arXiv:2509.00105](https://arxiv.org/abs/2509.00105), Shaoting Feng et al., 2025; SOSP 2025 BigMem workshop. See Q4.
- **R34. Towards More Economical Context-Augmented LLM Generation by Reusing Stored KV Cache.** [arXiv:2503.14647](https://arxiv.org/abs/2503.14647), Hanchen Li et al., 2025. See Q4.
- **R35. FastServe.** [arXiv:2305.05920](https://arxiv.org/abs/2305.05920), "Fast Distributed Inference Serving for Large Language Models", Bingyang Wu et al., 2023. A skip-join MLFQ scheduler with a mechanism that "proactively offloads" and uploads KV between GPU and host memory ([abs](https://arxiv.org/abs/2305.05920)). Queue-driven swapping.

**Mentioned in passing (out of scope or covered by other researchers)**
- Agent-specific, beyond R1/R7/R10/R12/R22:
  - InferCept [2402.01869](https://arxiv.org/abs/2402.01869) (Reyna Abhyankar et al.). Cited as ICML 2024 by [TokenCake](https://arxiv.org/html/2510.18586). It "decides per tool call whether to preserve, swap, or discard" KV, per MORI's related work ([MORI HTML](https://arxiv.org/html/2606.00866)).
  - KVFlow [2507.07400](https://arxiv.org/abs/2507.07400) (Zaifeng Pan et al.; cited as NeurIPS in [TempoKV](https://arxiv.org/html/2609.35065)). A steps-to-execution value from an Agent Step Graph guides eviction and prefetch ([abs](https://arxiv.org/abs/2507.07400)).
  - Autellix [2502.13965](https://arxiv.org/abs/2502.13965) (Michael Luo et al.).
  - ScaleSim [2601.21473](https://arxiv.org/abs/2601.21473) (Zaifeng Pan et al.). "Invocation distance" drives proactive prefetching and priority eviction ([abs](https://arxiv.org/abs/2601.21473)).
  - CacheScout, "Learning Agent Execution for KV-Cache Management in Agentic Serving", [2608.14624](https://arxiv.org/abs/2608.14624) (Rui Zhang et al.). Learned agent transitions guide eviction and proactive prefetch ([abs](https://arxiv.org/abs/2608.14624)).
  - Leyline [2606.01065](https://arxiv.org/abs/2606.01065) (Bole Ma et al.). A policy-to-engine directive primitive for KV edits ([abs](https://arxiv.org/abs/2606.01065)).
- Theory: "Online Scheduling for LLM Inference with KV Cache Constraints", [2502.07115](https://arxiv.org/abs/2502.07115), Patrick Jaillet et al.
- Token-level offloading inside one long request: out of scope.
  - InstInfer [2409.04992](https://arxiv.org/abs/2409.04992), KVDrive [2605.18071](https://arxiv.org/abs/2605.18071), HillInfer [2602.18750](https://arxiv.org/abs/2602.18750), NEO [2411.01142](https://arxiv.org/abs/2411.01142), FlexGen [2303.06865](https://arxiv.org/abs/2303.06865).
- Prefix-cache baseline: SGLang's RadixAttention, [2312.07104](https://arxiv.org/abs/2312.07104), Lianmin Zheng et al. Cited as NeurIPS by [Strata](https://arxiv.org/html/2508.18572). It "uses a least recently used (LRU) cache within a radix tree" ([Mooncake HTML](https://arxiv.org/html/2407.00079)).

### Inferences
- Claim-by-claim prior-art status, from the papers verified above:
  - **Per-tool distributional gap forecasts: not novel by itself.** Continuum (R1) uses per-tool empirical CDFs inside the engine; Alibaba (R2) and IAC (R3) use per-category fitted distributions; the Copilot study (R4) conditions a survival curve on elapsed idle. Defensible novelty is narrower: the agent framework sends the table, keyed by tool plus command word with fallback; the engine conditions on idle time; and the forecast drives SSD fetch, host retention and preemption under a price, not GPU pinning or eviction alone.
  - **Newsvendor fetch threshold q = λL/(1+λL): no match found.** Nearest are TempoKV (deterministic laxity TTU − TTR ≤ 0, R6) and TokenCake (predicted stall ≥ transfer time, R7). Continuum's argmax_τ P(τ,f)·B − τ is a related expected-benefit-minus-holding-cost optimization, but for retention TTLs.
  - **Hazard-rate retention index with Belady/LRU limits: no match found.** IAC mimics a Threshold-Belady oracle with a median-age test and reports that a conditional-probability score underperformed (R3). Alibaba evaluates a reuse probability at time since last access (R2). Pensieve uses cost per idle time (R11). Our index uses hazard × read time vs q plus a best-horizon maximum; the Belady/LRU reductions are, as far as I found, not stated in any verified paper.
  - **Price on host memory learned by dual ascent: no match found.** Closest:
    - KVLearn: a per-byte-second γ set heuristically, plus a PI controller on a utilization target (R5).
    - Continuum: memory cost τ·MemUsage/M, an implicit fixed price of about 1/M seconds per token-second, valid only if "the waiting queue always contains enough requests" ([HTML](https://arxiv.org/html/2511.02230)). Unlike our λ, it does not fall to zero when memory is plentiful.
    - Neither uses Lagrangian decoupling or reports a weak-duality bound.
- In Continuum's displayed objective, the request's memory share multiplies both cost and benefit, so the TTL depends on the tool CDF and reconstruct/queueing time, not on the prefix's size relative to price. Our q_i rises with λL_i. This contrast is my reading of their formula and is worth stating explicitly in related work.
- The staged-read and bandwidth issues our design handles (reads hold host memory; speculative reads compete with demand reads) are independently reported as real problems: FlatCake's aborted and revoked prefetches (R13), "Where Should the KV Cache Live?" (R24), and TempoKV's protected byte-time (R6).

### Gaps
- Unverified leads (not found on arXiv; do not cite as arXiv):
  - **IMPRESS (FAST '25).** Weijian Chen et al., "IMPRESS: An Importance-Informed Multi-Tier Prefix KV Storage System for Large Language Model Inference". Only the [USENIX page](https://usenix.org/conference/fast25/presentation/chen-weijian-impress) was found; an arxiv.org search for its key terms returned 0 results. A search snippet says it loads only important KVs and keeps chunks with important KVs in faster tiers, but I did not read the paper.
  - **LPC, "Learned Prefix Caching for Efficient LLM Inference" (NeurIPS 2025).** [NeurIPS page](https://neurips.cc/virtual/2025/poster/117662), [OpenReview PDF](https://openreview.net/pdf/a340edd38ffafcfd1843a7f71d85464d9fb3e3df.pdf); not on arXiv per my search. IAC describes LPC as an MLP predicting conversation-continuation probability, with "an exponential time-decay" lowering the probability while a prefix is inactive ([IAC HTML](https://arxiv.org/html/2611.09715)).
  - **HotPrefix.** Yuhang Li et al., 2025: hotness-based admission and CPU/GPU promotion, per secondary summaries ([catalog](https://rmarcus.info/dbscholar/papers/hdef9b04805eb320a)). No arXiv record found.
  - **SGLang HiCache.** I found no arXiv paper whose subject is HiCache; an arxiv.org search on radix-tree + host-memory hierarchical KV terms returned nothing. It appears in arXiv papers only as a baseline or implementation substrate (Strata R14; IAC R3; RR-Evict R22). Pointer only: [LMSYS blog, Sept 2025](https://lmsys.org/blog/2025-09-10-sglang-hicache).
- Secondary-only venue claims: Strata as OSDI and KVFlow as NeurIPS (from [TempoKV's references](https://arxiv.org/html/2609.35065)); Continuum as an ICLR 2026 workshop paper (from [2610.22863](https://arxiv.org/pdf/2610.22863)). These are not on the arXiv records.
- Most 2026 entries (R3–R8, R10, R12, R13, R20–R28) are unreviewed preprints, except IAC (NeurIPS 2026), KVLearn (SYSTOR 2026) and TokenCake (EuroSys '27).
- I could not confirm from the extracted text whether Alibaba's ReuseProb_w(t) is the fitted density or a conditional probability. I also could not confirm whether Continuum's latest version (v7) re-plans TTLs as idle time elapses.

## Q1. Prefetch and eviction: how do multi-tier KV systems decide what to bring up and what to evict?

### Takeaway
Prefetch in the multi-tier literature is triggered almost exclusively by requests that have already arrived (queue look-ahead, hit discovery). Pre-arrival prefetch appears only in agent-specific work (TokenCake's predicted tool completion) and in multi-agent workflow graphs (KVFlow, ScaleSim, CacheScout). Eviction ranges from LRU and LFU through cost/size GreedyDual variants to learned reuse probability. Recent 2026 trace studies find LRU hard to beat on paced sessions, but far from Belady for agents.

### Cited Findings
- Queue-driven prefetch:
  - CachedAttention prefetches disk-resident KV of waiting jobs within a look-ahead window sized by free host memory ([HTML](https://arxiv.org/html/2403.19708)).
  - LMCache prefetches queued queries' KV to a faster tier during the queueing interval ([HTML](https://arxiv.org/html/2510.09665)).
  - PCR uses "queue-based prefetching" from SSD ([abs](https://arxiv.org/abs/2603.23049)).
  - TempoKV stages from SSD when TTU ≤ TTR ([HTML](https://arxiv.org/html/2609.35065)).
  - FastServe proactively swaps KV between GPU and host under its scheduler ([abs](https://arxiv.org/abs/2305.05920)).
- Forecast-driven prefetch:
  - TokenCake uploads offloaded KV as the predicted (EWMA) function completion approaches ([HTML](https://arxiv.org/html/2510.18586)).
  - KVFlow prefetches by steps-to-execution ([abs](https://arxiv.org/abs/2507.07400)), ScaleSim by "invocation distance" ([abs](https://arxiv.org/abs/2601.21473)), and CacheScout by a learned agent-transition model ([abs](https://arxiv.org/abs/2608.14624)).
  - HyMCache uses "request-level prefix prefetching" to stage reads in CXL device DRAM ([arXiv:2607.18141](https://arxiv.org/abs/2607.18141), Hakbeom Jang et al., 2026).
- Scheduling instead of prefetch: Strata forms batches whose aggregated load-to-compute ratio stays under a "loading-bound" limit ([HTML](https://arxiv.org/html/2508.18572)).
- Recency and frequency eviction:
  - Mooncake found LRU best on its trace ([HTML](https://arxiv.org/html/2407.00079)).
  - "When Fancy Eviction Fails" finds LRU robust across 14 algorithms because sessions are paced steadily, but well short of Belady ([abs](https://arxiv.org/abs/2609.28870)).
  - WAKMan finds no single policy always best and switches policies online ([abs](https://arxiv.org/abs/2610.36355)).
- Cost-aware eviction:
  - Pensieve: recompute cost / idle time ([HTML](https://arxiv.org/html/2312.05516)).
  - Marconi: recency + FLOP savings per byte ([HTML](https://arxiv.org/html/2411.19379)).
  - RAGCache: prefix-aware GDSF ([PDF](https://arxiv.org/pdf/2404.12457)).
  - "When Fancy Eviction Fails": compute-aware Belady ([HTML](https://arxiv.org/html/2609.28870)).
- Learned or predicted eviction: Alibaba's exponential per-category reuse probability ([PDF](https://arxiv.org/pdf/2506.02634)); IAC ([HTML](https://arxiv.org/html/2611.09715)); SAECache ([HTML](https://arxiv.org/html/2605.18825)); KVLearn ([PDF](https://arxiv.org/pdf/2610.28637)); LARU ([HTML](https://arxiv.org/html/2509.20979)); Bayesian Beta reuse predictor ([abs](https://arxiv.org/abs/2604.26968)).
- Agent-trajectory eviction without prediction:
  - RR-Evict spreads tail-chunk eviction round-robin across idle agents ([abs](https://arxiv.org/abs/2609.32278)).
  - MORI keeps the least idle programs in HBM and the next in DRAM, and drops the rest ([HTML](https://arxiv.org/html/2606.00866)).
- Write and admission side:
  - SARIA persists only blocks that are likely to be reused beyond DRAM residency ([abs](https://arxiv.org/abs/2610.31169)).
  - EfficientAgent stops writing refills when the host tier is smaller than the reuse working set ([abs](https://arxiv.org/abs/2609.33762)).
  - KVLearn admits on CARS > θ ([PDF](https://arxiv.org/pdf/2610.28637)).
- Substrates (context):
  - MemServe's MemPool with TTL'd global prompt trees ([HTML](https://arxiv.org/html/2406.17565)).
  - The I/O path for r_i: Tutti's GPU-centric SSD object store with slack-aware I/O scheduling ([arXiv:2605.03375](https://arxiv.org/abs/2605.03375), Shi Qiu et al.); Range's per-byte rather than per-object storage costs ([arXiv:2610.21241](https://arxiv.org/abs/2610.21241), Dingsen Shi et al.); KVStream's multi-tier aggregate-bandwidth scheduling ([arXiv:2610.31827](https://arxiv.org/abs/2610.31827), Avinash Maurya et al.); Unified KV Pooling with SPDK ([arXiv:2606.14779](https://arxiv.org/abs/2606.14779), Minchul Kang et al.).
  - Bounded-State Restoration's "restoration working set" of local staging memory during restore ([arXiv:2608.17826](https://arxiv.org/abs/2608.17826), Zixuan Li).
  - TraCT's CXL shared-memory KV pool ([arXiv:2512.18194](https://arxiv.org/abs/2512.18194), Dongha Yoon et al.).
- Counter-evidence on prefetch: in a calibrated simulator, prefetch on a contended link cost more than it saved ([HTML](https://arxiv.org/html/2609.16215)).

### Inferences
- Prefetching during the tool gap is the gap our design fills relative to the multi-tier systems (CachedAttention, LMCache, Strata, Mooncake, TempoKV). In those systems a read can start only once the request is visible to the engine.
- Bounded-State Restoration's local staging memory during restore is the same resource as our "reads in progress hold host memory" term. Citing it alongside FlatCake supports the capacity constraint Σ(h_i + p_i) ≤ R.

### Gaps
- I did not read Mooncake v5's (Oct 2026) changes. Its eviction description comes from the HTML available on 2026-11-17.
- IMPRESS (FAST '25) could not be verified on arXiv. See the Q0 Gaps.

## Q2. Reuse signals: which systems use the waiting queue, reuse probability, inter-arrival distributions, or predicted reuse time? (incl. the Alibaba study)

### Takeaway
All four signal types appear in the literature, but distributional time signals are recent (2025–2026) and nearly all target single-tier GPU eviction for hit rate. The Alibaba ATC'25 study found reuse times short (80% under 10 min for to-C, within 10 s for API traffic) and exponential within a request category and time window. It uses the fitted distribution, evaluated at time since last access, as the primary eviction key.

### Cited Findings
- **Alibaba (ATC '25):**
  - Trace A (to-C) carries the request type, user ID and parent chat ID, so sessions can be reconstructed. Text is 78% of its workload; multi-turn ratios are 47–51% for all types except multimodal ([PDF](https://arxiv.org/pdf/2506.02634)).
  - Trace B (to-B API) has a multi-turn ratio below 0.1%, yet "single-turn requests contribute 97% of the total cache hits" ([PDF](https://arxiv.org/pdf/2506.02634)).
  - "In Trace A, 80% of the reuse time falls within less than 10 minutes, while in Trace B it falls within 10 seconds." Search requests show "the second-longest reuse time" ([PDF](https://arxiv.org/pdf/2506.02634)).
  - The reuse-time distribution is exponential given a time period and category. Next-day per-category reuse probability can be predicted from the previous day's log, except that image requests are harder ([PDF](https://arxiv.org/pdf/2506.02634)).
  - Policy: priority (ReuseProb_w(t), offset), fitted on a background sample ([PDF](https://arxiv.org/pdf/2506.02634)).
- **Inter-arrival distributions:**
  - IAC: log-normal per (intent, turn); median as reuse window ([HTML](https://arxiv.org/html/2611.09715)).
  - SAECache: log-normal per session, online MLE ([HTML](https://arxiv.org/html/2605.18825)).
  - Continuum: per-tool empirical CDF ([HTML](https://arxiv.org/html/2511.02230)).
  - Copilot study: a survival curve S(t), conditioned as S(t)/S(t0) ([HTML](https://arxiv.org/html/2608.00101)).
- **Reuse probability:**
  - KVLearn's PRP features recency, decayed frequencies, time of day and session age ([PDF](https://arxiv.org/pdf/2610.28637)).
  - LPC (not on arXiv) predicts continuation, with exponential time decay ([IAC HTML](https://arxiv.org/html/2611.09715)).
  - Marconi forecasts reuse likelihood by hit-scenario taxonomy ([abs](https://arxiv.org/abs/2411.19379)).
  - A Bayesian Beta predictor works over block and transition types ([abs](https://arxiv.org/abs/2604.26968)).
- **Predicted reuse time (point):**
  - LARU uses LightGBM next-request time ([HTML](https://arxiv.org/html/2509.20979)).
  - TokenCake uses an EWMA of function durations ([HTML](https://arxiv.org/html/2510.18586)).
  - TempoKV uses TTU estimates ([HTML](https://arxiv.org/html/2609.35065)).
  - ScaleSim uses invocation distance ([abs](https://arxiv.org/abs/2601.21473)); KVFlow uses steps-to-execution ([abs](https://arxiv.org/abs/2507.07400)).
  - "Where Should the KV Cache Live?" tested an EWMA next-touch predictor; it ranked behind reuse frequency on agent loops ([HTML](https://arxiv.org/html/2609.16215)).
- **Waiting queue:** CachedAttention ([HTML](https://arxiv.org/html/2403.19708)), LMCache ([HTML](https://arxiv.org/html/2510.09665)), PCR ([abs](https://arxiv.org/abs/2603.23049)), PEEK ([abs](https://arxiv.org/abs/2607.02525)), Strata ([HTML](https://arxiv.org/html/2508.18572)).
- **Idleness history instead of forecasts:** MORI ([HTML](https://arxiv.org/html/2606.00866)). Pensieve uses 1/idle time ([HTML](https://arxiv.org/html/2312.05516)).
- **Conflicting distribution shapes:** Alibaba reports exponential reuse times per category and time window ([PDF](https://arxiv.org/pdf/2506.02634)). IAC and SAECache report long-tailed log-normal inter-turn delays ([IAC HTML](https://arxiv.org/html/2611.09715); [SAECache HTML](https://arxiv.org/html/2605.18825)).

### Inferences
- The exponential vs log-normal conflict likely reflects granularity: block reuse within a category and time window vs per-session turn gaps. Either way, it supports our choice to ship empirical per-key distributions rather than a parametric family.
- With an exponential (memoryless) fit, conditioning on idle time adds nothing. Our conditioning matters only for non-exponential, multi-modal tool durations, which TraceLab's per-tool latency spread suggests are common (see Q5).
- IAC's negative result on a full conditional-probability score (R3) is the most direct empirical challenge to an index built on F(H|t). Our differences may answer it: weighting by read time and size, and comparing against a price. But a reviewer will ask for an ablation against a median-threshold rule.

### Gaps
- I did not find a per-category reuse-time analysis in Mooncake's or ServeGen's arXiv text. ServeGen's abstract ([abs](https://arxiv.org/abs/2505.09999)) does not mention sessions or inter-turn gaps.

## Q3. Load vs recompute: which works choose between reading KV from slower storage and recomputing it, or split the work, and with what cost models?

### Takeaway
Most works split restoration between loading and recomputing rather than choosing one. Cake meets in the middle; HCache, Krul and CacheFlow use schedulers or DP planners; CacheBlend pipelines selective recompute with loading. Cost models are profiled: per-layer compute vs I/O time, quadratic prefill fits, and bandwidth averages. Only Continuum and KVLearn fold load-vs-recompute costs into a retention decision, the same coupling our design uses.

### Cited Findings
- **Cake:** "Computation starts from the beginning of the sequence, while I/O loading starts from the end … and merge in the middle" ([HTML](https://arxiv.org/html/2410.03065)). It reports a 2.6× average TTFT reduction vs compute-only and I/O-only ([abs](https://arxiv.org/abs/2410.03065)).
- **HCache (EuroSys 2025):** restores state from stored hidden states.
  - Its bubble-free restoration scheduler "integrates resource-complementary methods to optimize the balance between computation and IO tasks" ([abs](https://arxiv.org/abs/2410.05004)).
  - At long contexts "the scheduler detects that token computation is very expensive and falls back to the HCache-only approach" ([HTML](https://arxiv.org/html/2410.05004)).
  - Up to 1.93× lower TTFT than KV offload, with 1.92–2.40× less storage ([abs](https://arxiv.org/abs/2410.05004)).
- **CachedAttention:** layer-wise pre-loading overlaps KV loading with prefill. A larger pre-loading buffer lets loading start earlier when per-layer loading exceeds per-layer compute ([HTML](https://arxiv.org/html/2403.19708)).
- **Pensieve:** evicts leading-end chunks first, because their recomputation is cheaper, and recomputes dropped context ([HTML](https://arxiv.org/html/2312.05516)).
- **Continuum:** Prefill-Reload(r) comes from offline profiling of average CPU offload throughput and a quadratic prefill-vs-context fit; profiling takes "less than 10 minutes" per hardware–model pair ([HTML](https://arxiv.org/html/2511.02230)). Its OutofOrderCost term (queueing delay after eviction) is "absent from prior retention policies such as InferCept, which only considers the reload cost" ([HTML](https://arxiv.org/html/2511.02230)).
- **KVLearn:** recompute cost R(b) is calibrated offline on the prefill GPU, across two FlashAttention regimes; transfer cost is T(b) ≈ |b|/β. Its example: T ≈ 2.7 ms vs R ≈ 6.8 ms for a 67 MB block ([PDF](https://arxiv.org/pdf/2610.28637)).
- **CacheFlow:** "A batch-aware dynamic-programming planner jointly determines what to recompute and what to load across requests and layer blocks, balancing GPU computation against shared I/O resources". Restoration latency is 2.24–3.00× lower ([abs](https://arxiv.org/abs/2604.25080)).
- **Krul:** [arXiv:2507.08045](https://arxiv.org/abs/2507.08045), Junyi Wen et al., 2025. A "recomputation-loading pipeline" with a "bubble-free restoration scheduler" for multi-turn state restoration under per-conversation cross-layer KV compression ([abs](https://arxiv.org/abs/2507.08045)).
- **KVPR:** [arXiv:2411.17089](https://arxiv.org/abs/2411.17089), Chaoyi Jiang et al., ACL Findings 2025. I/O-aware partial recomputation for CPU-offloaded KV, aimed at PCIe bottlenecks in decoding ([abs](https://arxiv.org/abs/2411.17089)).
- **CacheBlend:** [arXiv:2405.16444](https://arxiv.org/abs/2405.16444), Jiayi Yao et al. It "selectively recomputes the KV values of a small subset of tokens", and the extra delay "can be pipelined with the retrieval of KV caches" ([abs](https://arxiv.org/abs/2405.16444)).
  - Non-prefix descendants with adaptive recompute ratios: CacheTune ([arXiv:2605.24022](https://arxiv.org/abs/2605.24022), Fei Li et al.) and CacheSmith ([arXiv:2610.18925](https://arxiv.org/abs/2610.18925), Xinyue Ma et al.) ([abs](https://arxiv.org/abs/2605.24022); [abs](https://arxiv.org/abs/2610.18925)).
- **Recompute or Restore?:** [arXiv:2611.15126](https://arxiv.org/abs/2611.15126), Sungwon Woo et al., 2026. On a static-graph NPU, a host-DRAM tier with "a scheduler-aware event model whose predictions are frozen before each test cohort" decides when restoring beats recomputing. SLO attainment rose from 22.8% to 100% in one overload condition ([abs](https://arxiv.org/abs/2611.15126)).
- **VAMP:** [arXiv:2609.13537](https://arxiv.org/abs/2609.13537), Jinpyo Kim et al., 2026. It compares "the estimated future work" of staging expert weights from host, evicting prefixes that may need re-prefill, or preempting ([abs](https://arxiv.org/abs/2609.13537)).
- **EfficientAgent:** offloading gives inconsistent results "even where loading a token back is several times cheaper than recomputing it", because the host tier must hold the reuse working set ([abs](https://arxiv.org/abs/2609.33762)).
- **"Can I Buy Your KV Cache?":** [arXiv:2606.13361](https://arxiv.org/abs/2606.13361), Luoyuan Zhang et al., 2026. Reuse is "9-50x cheaper in compute than prefill, and the gap widens with length (prefill's attention scales with L^2)" ([abs](https://arxiv.org/abs/2606.13361)).
- **HiState:** [arXiv:2610.41508](https://arxiv.org/abs/2610.41508), Zaifeng Pan et al., 2026. For hybrid models, adaptive checkpoint intervals trade checkpoint storage against recomputation, and historical checkpoints sit on SSD ([abs](https://arxiv.org/abs/2610.41508)).
- **LayerKV:** [arXiv:2410.00428](https://arxiv.org/abs/2410.00428), Yi Xiong et al. Layer-wise KV allocation and offloading cut queueing-driven TTFT ([abs](https://arxiv.org/abs/2410.00428)).

### Inferences
- Our break-even rule is equivalent in spirit to the profiled models in Continuum and KVLearn: reading beats recomputation beyond position b/(2ρβ), with ρ fitted offline. The quadratic prefill fit appears in Continuum.
- Our design treats recomputation as a failure cost rather than co-scheduling it with loading. Hybrid restoration (Cake, HCache, CacheFlow) is complementary and could shorten r_i, which would lower q-relevant read times. Related work should present these as orthogonal mechanisms.

### Gaps
- I did not extract Cake's or KVPR's internal scheduling formulas. KVPR's method is cited from its abstract only.

## Q4. Priced KV memory: does any work price KV memory or use utility maximization, Lagrangian methods or dual ascent to allocate it across tenants or sessions?

### Takeaway
Several works charge memory occupancy in time or utility units inside a per-object score. Continuum uses τ·MemUsage/M in seconds; KVLearn uses γ·|b|·Δt; TempoKV measures byte-time; Marconi and RAGCache use savings-per-byte or GreedyDual priorities. Others optimize dollar, carbon or Pareto costs offline. I found no serving system that relaxes the capacity constraint with a Lagrange multiplier, learns it by dual ascent on demand minus capacity, and reports a weak-duality lower bound.

### Cited Findings
- **Continuum:** "The cost of pinning a request's KV cache comes from the opportunity cost of occupying GPU memory". Cost = τ·MemUsage(r)/M, and it assumes "the waiting queue always contains enough requests for this blocking effect to occur" ([HTML](https://arxiv.org/html/2511.02230)).
- **KVLearn:** "U(b, Δt) = γ·|b|·Δt: encodes the per-byte-per-second opportunity cost of occupying pool DRAM", with γ = R̄·λ/M_S derived from a capacity target ρ* = 0.85. A PI controller (ATC) adjusts the admission threshold θ ([PDF](https://arxiv.org/pdf/2610.28637)).
- **TempoKV:** the objective is "protected fast-tier byte-time per request" ([abs](https://arxiv.org/abs/2609.35065)).
- **Utility functions across tiers:**
  - EVICPRESS: "a unified utility function that quantifies the effect of quality and delay of the lossy compression or eviction". Placement uses a heuristic "with the goal of maximizing the utility function scores on each storage tier" ([abs](https://arxiv.org/abs/2512.14946)).
  - AdaptCache: decides "the compression algorithm, compression rate and device placement for each KV cache entry to maximise DRAM hits and minimise loading delay" ([abs](https://arxiv.org/abs/2509.00105)).
- **Savings-per-memory indices:** Marconi ([HTML](https://arxiv.org/html/2411.19379)); RAGCache PGDSF ([PDF](https://arxiv.org/pdf/2404.12457)).
- **Dollar, carbon and Pareto pricing (offline or provisioning):**
  - An analytical cloud-cost model of storing and reusing KV vs recomputing, as a function of reuse frequency ([arXiv:2503.14647](https://arxiv.org/abs/2503.14647)).
  - GreenCache trades SSD embodied carbon against compute carbon under SLOs ([arXiv:2505.23970](https://arxiv.org/abs/2505.23970), Yuyang Tian et al.).
  - Kareto searches the Pareto frontier of cost, throughput and latency over tiered-storage configurations ([arXiv:2603.08739](https://arxiv.org/abs/2603.08739), Xianzhe Zheng et al.).
  - A market in which agents "buy the right to load" a publisher's precomputed KV ([arXiv:2606.13361](https://arxiv.org/abs/2606.13361)).
- **Multi-tenant accounting:** PrefixShield "meters newly materialized full KV blocks, carries responsibility across requests" and uses "projected debt to select the group supplying eviction candidates" ([arXiv:2608.01657](https://arxiv.org/abs/2608.01657), Zhiyu Wang et al.). PrefixShield does not price memory. Fairness of serving across clients is the subject of "Fairness in Serving Large Language Models" ([arXiv:2401.00588](https://arxiv.org/abs/2401.00588), Ying Sheng et al.). I verified only its record and did not examine its mechanism.
- **Theory (in passing):** a hindsight-optimal integer-program benchmark and online algorithms for scheduling under KV memory constraints ([arXiv:2502.07115](https://arxiv.org/abs/2502.07115)).

### Inferences
- The defensible novelty for (d) is threefold:
  1. Deriving the per-session score from a Lagrangian relaxation, so that λ is a shadow price and sessions decouple.
  2. Learning λ by multiplicative dual ascent on observed demand minus capacity, where refused and preempted reads count as demand.
  3. The weak-duality lower bound.
- KVLearn's γ and PI threshold, and Continuum's 1/M, are the closest heuristics and should be contrasted directly.

### Gaps
- Pricing from caching theory and networking (e.g., congestion pricing) was out of scope and not searched.
- An arXiv paper framing cache eviction as constrained utility maximization with a shadow price appeared in search ([arXiv:2609.20068](https://arxiv.org/abs/2609.20068)). It concerns geo-mining document extraction, not KV serving, so I did not record or verify its authors.

## Q5. Traces: which public traces or characterizations of multi-turn or agentic sessions report inter-turn gaps, tool durations or KV reuse times that could support per-tool gap distributions?

### Takeaway
Per-tool latency distributions are now measured. TraceLab, a released trace of about 4,300 Claude Code and Codex sessions, shows tool latency spanning milliseconds to hours, with large spread within a tool. The production studies (GitHub Copilot, Microsoft M365) report bimodal gaps: within-turn gaps of seconds and cross-turn gaps of minutes, with heavy tails. Neither production trace is released. The Alibaba traces (released) give parent-chat links and reuse times but almost no agentic traffic.

### Cited Findings
- **TraceLab** ([arXiv:2606.30560](https://arxiv.org/abs/2606.30560), Kan Zhu et al., 2026; dataset released per the abstract).
  - Contents: about 4,300 sessions, 350,000 LLM steps and 430,000 tool calls from Claude Code and Codex ([abs](https://arxiv.org/abs/2606.30560)).
  - Average tool latency is 16.8 s, and "latency depends heavily on tool type": Read/Edit take milliseconds to seconds, Agent/AskUserQuestion take minutes, and Claude's Bash "ranges from milliseconds to minutes, although its median is below one second" ([HTML](https://arxiv.org/html/2606.30560)).
  - For Claude, calls under 1 s are 70% of calls but under 1% of tool time, while calls over 1 min are 4.9% of calls but 92% of tool time. For Codex, calls over 1 min are 3.1% of calls but 61% of time ([HTML](https://arxiv.org/html/2606.30560)).
  - Positive human waits: median 1.4 min, p90 20.6 min. Positive tool latencies: median 0.3 s, p90 13.6 s. Human thinking is 92.3% of session wall-clock ([HTML](https://arxiv.org/html/2606.30560)).
  - It names "semantic-aware tool-latency prediction" and "KV-cache management around human-paced gaps" as opportunities ([abs](https://arxiv.org/abs/2606.30560)).
- **GitHub Copilot** ([arXiv:2608.00101](https://arxiv.org/abs/2608.00101)).
  - Scale: 3.2M users, 13M sessions, 761M LLM calls ([abs](https://arxiv.org/abs/2608.00101)).
  - KV-cache idle, the time between consecutive LLM calls, has a median of 5.0 s. Over 90% of idle intervals fall within a turn, with a KV median of 1.2 s; cross-turn intervals (8–9%) have a KV median of 172 s. The median user idle time is 1,512 s ([HTML](https://arxiv.org/html/2608.00101)).
  - KV hit rate averages 90% within a turn but 55% across turn boundaries ([abs](https://arxiv.org/abs/2608.00101)).
  - Release of the traces is not stated in the parts I read.
- **Microsoft M365** ([arXiv:2610.22863](https://arxiv.org/abs/2610.22863), Mayank Arya et al., 2026).
  - Four production traces with more than 121M model calls ([abs](https://arxiv.org/abs/2610.22863)).
  - Within-session idle gaps have medians of 8.38 s (Research), 9.87 s (Assist) and 7.87 s (Chat), and P99s of about 20, 50 and 93 minutes ([PDF](https://arxiv.org/pdf/2610.22863)).
  - In Research, inter-turn gaps are 7.0% of gaps but 92.6% of idle time (median 125.0 s vs 7.5 s within a turn) ([PDF](https://arxiv.org/pdf/2610.22863)).
  - A one-hour idle cap lowers the hold-all-live HBM ceiling 3.6× (501.8 to 140.8 TB), while the replayed reconstruction rate rises only from 4.07% to 4.53% ([PDF](https://arxiv.org/pdf/2610.22863)).
  - The traces cannot be released. The authors plan a parametric generator, AgenticTraceSim, that models idle gaps ([PDF](https://arxiv.org/pdf/2610.22863)).
- **Storage view of KV cache** ([arXiv:2611.10178](https://arxiv.org/abs/2611.10178), Talia Q. Chen et al., 2026).
  - It characterizes reuse, recurrence timing and lifetimes at block granularity across 11 production and public traces, including coding and tool-using agents ([abs](https://arxiv.org/abs/2611.10178)).
  - Single-session state "can remain useful through long idle periods but loses value at session completion" ([abs](https://arxiv.org/abs/2611.10178)).
  - It reports recurrence-gap shares, e.g., Qwen To-C gaps above one minute ([HTML](https://arxiv.org/html/2611.10178)).
- **When Fancy Eviction Fails:** releases FreeInference and Chutes traces with block identity across sessions and multi-turn agentic traffic ([HTML](https://arxiv.org/html/2609.28870); [abs comment](https://arxiv.org/abs/2609.28870)). It notes that the Qwen Bailian (Alibaba) traces' "session information reveals very little agentic workload" ([HTML](https://arxiv.org/html/2609.28870)).
- **Alibaba traces:** time, type, user ID, turn (parent chat ID) and anonymized content ([PDF](https://arxiv.org/pdf/2506.02634)). IAC used "the cloud trace released by the authors" ([IAC HTML](https://arxiv.org/html/2611.09715)).
- **Mooncake trace:** "only the timing of request arrivals, the number of input tokens, and the number of output tokens, the remaped block hash" ([HTML](https://arxiv.org/html/2407.00079)). Session IDs are not listed among its fields.
- **BurstGPT:** [arXiv:2401.17644](https://arxiv.org/abs/2401.17644), Yuxin Wang et al. 10.31M Azure OpenAI traces, including "user conversation patterns: counts and intervals within conversations" ([abs](https://arxiv.org/abs/2401.17644)).
- **WildChat:** [arXiv:2405.01470](https://arxiv.org/abs/2405.01470), Wenting Zhao et al., ICLR 2024. Used with "real timestamps" for inter-turn delays by IAC, which finds delays vary by intent: code_generation is longer than chitchat ([IAC HTML](https://arxiv.org/html/2611.09715)).
- **ServeGen:** [arXiv:2505.09999](https://arxiv.org/abs/2505.09999), Yuxing Xiang et al., NSDI '26. Production characterization across language, multimodal and reasoning models ([abs](https://arxiv.org/abs/2505.09999)).
- **Agentic characterizations without gap distributions (supporting context):**
  - AgentCgroup: tool calls plus container and agent initialization are 55–60% of end-to-end task latency on SWE-rebench ([arXiv:2602.09345](https://arxiv.org/abs/2602.09345), Yusheng Zheng et al.).
  - Agentic AI Workload Characteristics (IISWC 2026): agents shift "from read/explore behavior early in execution to execute/write behavior later" ([arXiv:2605.26297](https://arxiv.org/abs/2605.26297), Yichao Yuan et al.).
  - GAIATrace (IISWC 2026): token-level agent traces ([arXiv:2606.01725](https://arxiv.org/abs/2606.01725), Donghwan Kim et al.).
  - A two-week production trace of 11.7M requests from a general-purpose agent platform ([arXiv:2609.34432](https://arxiv.org/abs/2609.34432), Yihao Zheng et al.).
  - AgentSysBench: non-LLM components dominate latency in 5 of 10 applications ([arXiv:2608.15127](https://arxiv.org/abs/2608.15127), Chaokun Chang et al.).
  - HPCA 2026 agent cost analysis ([arXiv:2506.04301](https://arxiv.org/abs/2506.04301), Jiin Kim et al.).
  - LLMRunner generates closed-loop multi-turn load with configurable think-time distributions ([arXiv:2611.15372](https://arxiv.org/abs/2611.15372), Arush Upadhyaya et al.).
- **Older assumption:** Pensieve assumed a 60 s mean think time ([HTML](https://arxiv.org/html/2312.05516)).

### Inferences
- TraceLab is the only verified public source with per-tool latency distributions for coding agents. It shows large spread within a tool (Bash from milliseconds to minutes), which directly motivates keying by tool plus command word.
- The Copilot and M365 studies motivate conditioning on idle time and separating within-turn from cross-turn gaps: bimodal gaps with heavy tails, and inter-turn gaps that dominate idle time.

### Gaps
- FlatCake evaluates on a production Claude Code trace published as a Hugging Face dataset (SemiAnalysis "cc-traces-weka-062126") ([FlatCake HTML](https://arxiv.org/html/2610.38355)). I did not inspect the dataset.
- ShareChat (used by IAC) was not verified.
- I did not check whether Continuum's tool-duration records are public.
