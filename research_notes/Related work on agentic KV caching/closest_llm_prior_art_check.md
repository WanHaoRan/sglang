# Closest LLM-serving prior art, checked against the full texts: Continuum, CacheWise, "Ask the Tool, Don't Guess", Intent-Aware Caching (checked 17 Nov 2026)

*Scope and method.* This note settles what four papers actually say, so that a conference paper can state its novelty precisely.

Design shorthand:
- **(a) Forecasts.** The agent framework sends per-tool idle-gap distributions, keyed by tool plus the first word of the command. The engine conditions each on the idle time so far.
- **(b) Fetch.** An SSD read starts once P(arrival before the read completes | idle so far) ≥ q = λL/(1+λL).
- **(c) Retain.** An idle prefix cache stays in host memory while hazard × read time ≥ q. Eviction follows a Gittins-type index.
- **(d) Price.** λ prices host memory in TTFT seconds per token-second. It is learned by multiplicative dual ascent, with a weak-duality lower bound.

Sources and how they were read:
- **Pages fetched.** For each paper I downloaded the arXiv abstract page and the latest arXiv HTML once. The HTML served Continuum v7, CacheWise v1, Ask the Tool v1 and IAC v1. I also fetched Continuum's v1 and v4 HTML to trace how its formula changed.
- **Request budget.** That is 10 arxiv.org page loads in total, with no export-API calls and no web searches.
- **Formulas.** They are quoted from the LaTeX `alttext` that arXiv's HTML stores for every equation, so they are verbatim.
- **Prose quotes.** They are verbatim from the HTML text, with straight apostrophes.
- **Code.** I downloaded the three code repositories the papers themselves link, as archives of each default branch (commit hashes given). I only read them, never ran them, to check whether each implements the paper's formula.
- **Negative claims.** Statements such as "contains no X" rest on a term scan of each paper's main body, cut at "References". Where noted, I also read appendices in full.
- **Inference.** Anything derived rather than read is marked "inference".

The two earlier notes are cited as:
- **Note A:** `agentic_tool_call_kv.md`
- **Note B:** `multi_tier_kv_cache.md`

Line numbers refer to those files as of 17 Nov 2026.

---

## 1. Continuum (arXiv 2511.02230): what is the exact TTL objective, and which earlier note is right?

### Takeaway
**The disagreement.** Both earlier notes quote the rule Continuum actually solves, Eq. (2), correctly. Each is right about a different equation:
- **Eq. (1)** charges pinned GPU memory in seconds of latency. The charge is Cost(τ,r) = MemUsage(r)/M × τ, a fixed conversion rate justified by assuming the waiting queue is always full.
- **Eq. (2).** The benefit carries the same factor, and the paper divides it out ("By eliminating the shared MemUsage(r)/M"). The rule it solves, Eq. (2), therefore has no memory term, no capacity constraint and no price. The request's size enters only through Prefill-Reload(r).
- **The hybrid formula.** "P(τ,f)·(T·η + Prefill − Reload) − τ·MemUsage/M" appears in no version I read (v1, v4, v7). "Prefill-Reload(r)" is one quantity, the prefill *or* reload time, not a difference.

**The specific questions:**
- **Conditioning.** P(τ,f) is an unconditional empirical CDF, and the TTL is fixed once, when the tool call starts.
- **Tiers.** The method pins in GPU memory only; SSD appears only as an evaluation variant run on LMCache's SSD layer.
- **Keys.** Tools are keyed by the first word of the bash command on SWE-Bench, learned online. The minimum is K = 100 samples, with a fallback to global records and then to a default derived from Exp(1).
- **Prefill.** It is modelled as a fitted quadratic in context length.

**Released code.** The code at HEAD implements neither equation. It uses a fixed 2-second TTL, applied only when the tool's mean duration is at most 2 s.

### Cited Findings

#### Record (verified)
- **Title and authors.** The arXiv metadata title is "Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live". The 11 authors are Hanchen Li, Runyuan He, Qiuyang Mang, Qizheng Zhang, Huanzhi Mao, Xiaokun Chen, Hangrui Zhou, Huanchen Zhang, Alvin Cheung, Joseph Gonzalez and Ion Stoica — [arXiv abs](https://arxiv.org/abs/2511.02230)
- **Versions.** v1 Tue 4 Nov 2025; v2 20 Dec 2025; v3 30 Jan 2026; v4 4 May 2026; v5 11 May 2026; v6 25 May 2026; v7 Tue 8 Sep 2026 (latest). The HTML watermark reads "arXiv:2511.02230v7 [cs.OS] 08 Sep 2026" — [arXiv abs](https://arxiv.org/abs/2511.02230), [v7 HTML](https://arxiv.org/html/2511.02230v7)
- **Venue.** The abstract page has no Comments or Journal-ref field. The v7 HTML carries an unfilled VLDB template ("PVLDB, 20(1): XXX-XXX, 2027", "doi:XX.XX/XXX.XX"), which is not an acceptance record — [v7 HTML](https://arxiv.org/html/2511.02230v7)
- **Title spellings.** The v7 HTML spells the title "Continnum"; the v4 HTML title is "CacheTTL: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live" — [v7 HTML](https://arxiv.org/html/2511.02230v7), [v4 HTML](https://arxiv.org/html/2511.02230v4)
- **Code.** §9: "The implementation and artifacts are available at: https://github.com/Hanchenli/vllm-continuum" — [v7 HTML](https://arxiv.org/html/2511.02230v7)
- **Results claimed.** The abstract claims "improves the average job completion times by over 8x while improving throughput". This is on "SWE-Bench, BFCL, OpenHand" with "Llama-3.1 8B/70B, Gemma-3 12B, and GLM-4.5 355B" — [v7 HTML](https://arxiv.org/html/2511.02230v7)
- **Shared author.** Huanchen Zhang is an author of both Continuum and Ask the Tool — [Continuum abs](https://arxiv.org/abs/2511.02230), [Ask the Tool abs](https://arxiv.org/abs/2609.18849)

#### The decision rule and cost function, verbatim (v7 §4.1 "Utility Model", §4.2 "Setting the TTL Value")
- **Units (§4.1).** "Both the benefit and the cost are measured in units of time, since they ultimately translate into changes in the total job completion latency across all programs." Also: "For simplicity, Benefit(r) assumes that the next request arrives within the TTL window. The case where TTL expires before the tool call returns is addressed in Sec. 4.2." — [v7 HTML](https://arxiv.org/html/2511.02230v7)
- **Cost (§4.1 "Cost Estimation", unnumbered display).**
  - The lead-in: "The cost of pinning a request's KV cache comes from the opportunity cost of occupying GPU memory that could otherwise be used to serve other requests:"
  - The display: `\mathsf{Cost}(\tau,r)=\frac{\mathsf{MemUsage}(r)}{\mathcal{M}}\times\tau,`
  - The explanation: "The ratio MemUsage(r)/M represents how many average requests are blocked when r is pinned. In other words, if pinning r occupies the same memory as k requests, then pinning r adds τ latency to approximately k other requests. We assume that the waiting queue always contains enough requests for this blocking effect to occur when KV retention is necessary."
  - — [v7 HTML](https://arxiv.org/html/2511.02230v7)
- **Benefit (§4.1 "Benefit Estimation", unnumbered displays).**
  - `\mathsf{Benefit}(r) =\mathsf{CacheMissCost}(r)+\mathsf{OutofOrderCost}(r)`
  - `\mathsf{CacheMissCost}(r)=\frac{\mathsf{MemUsage}(r)\times\mathsf{Prefill\text{-}Reload}(r)}{\mathcal{M}}`
  - `\eta=-\mathrm{Corr}(k,N-k)`
  - `\mathsf{OutofOrderCost}(r)=\frac{\mathcal{T}}{\mathcal{M}}\times\mathsf{MemUsage}(r)\times\eta.`
  - The paper's framing: "This OutofOrderCost component is the key term absent from prior retention policies such as InferCept (Abhyankar et al., 2024), which only considers the reload cost."
  - — [v7 HTML](https://arxiv.org/html/2511.02230v7)
- **Eq. (1).**
  - `\tau^{*}=\text{argmax}_{\tau}\ \mathcal{P}(\tau,f)\times\mathsf{Benefit}(r)-\mathsf{Cost}(\tau,r),`
  - The definition that follows: "where P(τ,f) estimates the probability that the tool call f completes within time τ. This formula captures the expected net benefit, in terms of total job latency, of retaining the KV cache of r for a duration of τ"
  - — [v7 HTML](https://arxiv.org/html/2511.02230v7)
- **Eq. (2).**
  - The lead-in: "By eliminating the shared MemUsage(r)/M, the formula above can be transformed to"
  - The display: `\text{argmax}_{\tau}\ \mathcal{P}(\tau,f)\times\big(\mathcal{T}\cdot\eta+\mathsf{Prefill\text{-}Reload(r)}\big)-\tau,`
  - The follow-up: "indicating that we only need to additionally compute T and P(τ,f) in our implementation."
  - — [v7 HTML](https://arxiv.org/html/2511.02230v7)
- **The estimator and the solver (unnumbered).**
  - `\mathcal{P}(\tau,f)=\frac{1}{{|S[f]|}}\cdot{\sum_{t\in S[f]}\mathbb{I}[t\leq\tau]}`, "where I[·] is the indicator function."
  - "Finally, we solve Equation (2) by enumerating all unique tool-call durations recorded in S[f] as candidates (including τ=0) and selecting the one with the highest expected reward."
  - — [v7 HTML](https://arxiv.org/html/2511.02230v7)

#### Every term, as the paper defines it (v7 Table 3, §4.1–4.2, Algorithm 1)
- **τ:** "TTL". The paper says "To set an effective TTL value (in seconds)" — [v7 HTML](https://arxiv.org/html/2511.02230v7)
- **MemUsage(r):** "GPU memory occupied by r" (Table 3); "the amount of GPU memory used by the KV cache of request r" (§4.1) — [v7 HTML](https://arxiv.org/html/2511.02230v7)
- **M (calligraphic `\mathcal{M}`):** "Average memory occupied by the seen requests" (Table 3); "the average GPU memory footprint of active requests" (§4.1). The two definitions differ: *seen* versus *active* requests — [v7 HTML](https://arxiv.org/html/2511.02230v7)
- **Prefill-Reload(r) (`\mathsf{Prefill\text{-}Reload}(r)`, a hyphen inside a name, not a minus):**
  - Table 3: "Time for reconstructing KV cache in GPU".
  - §4.1: "Prefill-Reload(r) is the time cost for prefill or reloading depending on whether CPU offloading is turned on. This is based on a quick offline profiling described in Sec 5.2."
  - — [v7 HTML](https://arxiv.org/html/2511.02230v7)
- **CacheMissCost(r):** "Cost of reloading r" (Table 3). Also "the cost of reloading or prefilling the KV cache for request r" (§4.1) — [v7 HTML](https://arxiv.org/html/2511.02230v7)
- **OutofOrderCost(r):** "Cost of out-of-order for request r" (Table 3). Also "the expected queueing delay for the request due to waiting for other requests to free GPU memory" (§4.1) — [v7 HTML](https://arxiv.org/html/2511.02230v7)
- **η:**
  - "Memoryfulness factor of the workload" (Table 3).
  - "Let N be the total number of requests in a program and k the number of requests that have already been served."
  - η = 0 "when the workload is fully memoryless"; η = 1 when "all programs have the same fixed number of requests" (§4.1).
  - — [v7 HTML](https://arxiv.org/html/2511.02230v7)
- **T (calligraphic `\mathcal{T}`):**
  - "Average waiting time" (Table 3).
  - "where T is the average queueing delay for previous requests", with T/M "the average waiting time per unit context size" (§4.1).
  - "T can be estimated as the sliding window average for queueing delay experienced by requests who was evicted" (§4.2).
  - "initialize T as zero" (§4.2 "Cold-start Handling").
  - — [v7 HTML](https://arxiv.org/html/2511.02230v7)
- **P(τ,f):** "Estimated finish-within-TTL probability for f" (Table 3) — [v7 HTML](https://arxiv.org/html/2511.02230v7)
- **S[f] and f:** S[f] "denotes the recorded tool-call information for tool f" (Algorithm 1, global state); f is "Next tool to be called after finishing r" (Algorithm 1, line 10) — [v7 HTML](https://arxiv.org/html/2511.02230v7)

#### Is P(τ,f) conditioned on elapsed time? No: it is unconditional, and the TTL is fixed when the call starts
- **The estimator.** P(τ,f) is the empirical CDF over all recorded durations of f (formula above) — [v7 HTML](https://arxiv.org/html/2511.02230v7)
- **When the TTL is computed.**
  - It is computed once, at request finish. Algorithm 1, `OnRequestFinish`: "P[id] ← CalcTTL(r, S[f])".
  - `Schedule` then only compares against the clock: "If current time > P[id] and id ∉ Q.programs then Free KV cache used by id's last request".
  - §5.2: pin_request "records a pair of request and its expiration time current_timestamp+τ* in a dictionary pinned_requests and deliberately skips freeing the request's KV blocks."
  - — [v7 HTML](https://arxiv.org/html/2511.02230v7)
- **Term scan.** The v7 main body contains none of "elapsed", "conditional", "survival", "hazard", "prefetch", "price" or "Lagrang" — [v7 HTML](https://arxiv.org/html/2511.02230v7)

#### CPU or SSD? The method pins in GPU memory; offloading is LMCache's, and SSD appears only in the evaluation
- **Abstract.** "Continnum selectively pins the KV cache in GPU memory with a time-to-live value determined by the reload cost and potential queueing delay induced by eviction." — [v7 HTML](https://arxiv.org/html/2511.02230v7)
- **Where CPU offload enters.** Only through Prefill-Reload(r). The §5.2 "Offline Profile" measures "(1) GPU-CPU bandwidth for CPU offloading cases. We measure by taking the average CPU offloading throughput." No SSD bandwidth is profiled, and "SSD" does not occur in §4 or §5 — [v7 HTML](https://arxiv.org/html/2511.02230v7)
- **Evaluation setup (§6.1).** "CPU DRAM offloading We use vllm 0.10.2 with LMCache 0.3.7 (Cheng et al., 2025b). For A100 GPUs, we set the DRAM size used in offloading to be 100GB; For B200 and H100 GPUs, we set the DRAM size used in offloading to be 200GB per GPU." — [v7 HTML](https://arxiv.org/html/2511.02230v7)
- **The SSD runs (§6.3 "Sensitivity Analysis", paragraph "SSD Offloading").**
  - "Similar to CPU offloading, SSD offloading offers bigger space but slower loading. We evaluate Continnum with extended SSD storage layer beyond CPU offloading using LMCache on SWE-bench workload with llama-8B on B200. As shown in Figure 15, Continnum consistently improves average delay compared with baselines when also utilizing disks of different sizes."
  - Figure 15: "(a) SSD Size = 400G", "(b) SSD Size = 800G", "Continnum reduces delay when we extend offloading devices to SSDs beyond CPU offloading."
  - — [v7 HTML](https://arxiv.org/html/2511.02230v7)
- **Releasing memory under pressure (§5.2 "Prevention of deadlocks").** "we iteratively selects victims from pinned_requests with the latest program arrival time to unpin and free the space until the first request can be scheduled to run." — [v7 HTML](https://arxiv.org/html/2511.02230v7)
- **Unpinning (§5.2).** "It scans pinned_requests and unpins entries whose TTL have expired and whose program_id does not currently appear in the waiting queue." — [v7 HTML](https://arxiv.org/html/2511.02230v7)

#### How the per-tool distributions are keyed and learned
- **Key (§5.1).**
  - "the handler extracts the call's name and uses this as the tool call type."
  - "In SWE-Bench, it is guaranteed that each LLM's response containing a function call will include exactly one bash function call. We extract the string within the bash block and use the first word afterwards as the tool call name."
  - Other formats "can be handled by extending this parser with the same schema-specific extraction logic."
  - — [v7 HTML](https://arxiv.org/html/2511.02230v7)
- **Who parses: the engine.**
  - "The handler parses tool calls from LLM outputs, tracks per-tool latency using observed inter-request intervals within the same program_id, and returns TTL to the scheduler."
  - The client contributes only an identifier: "On the client side, we attach a program identifier (program_id) to every inference request".
  - — [v7 HTML](https://arxiv.org/html/2511.02230v7)
- **Learned online from server-side timestamps (§5.1).**
  - "compute the inter-request interval t_arrive^{p,i+1} − t_finish^{p,i}. We record this interval as the execution time of the tool call this time to store for TTL computation in the future."
  - Algorithm 1, `OnRequestArrive`: "S[f] ← S[f] ∪ {t}".
  - — [v7 HTML](https://arxiv.org/html/2511.02230v7)
- **Minimum sample count and fallback (§4.2 "Cold-start Handling").**
  - Global fallback: "we first try to use the global tool-call information to estimate P(τ, f_any)".
  - Default: "we design a minimal version of Continnum that uses a fixed TTL threshold T_default, derived from the same cost model by assuming that the tool-call duration follows an exponential distribution with unit mean, i.e., ToolCallDuration∼Exp(1); and the workload is fully memoryful, i.e., η=1."
  - Thresholds: "we use T_default when |S| ≤ K; otherwise, we use the global records when |S[f]| ≤ K, and use the fine-grained TTL setting for the remaining cases. In our implementation, we set K=100 and initialize T as zero."
  - The sentence before this calls the threshold "M", a notation slip, since M also denotes average memory.
  - An offline alternative is mentioned: "users can also obtain these cost-model statistics during training."
  - — [v7 HTML](https://arxiv.org/html/2511.02230v7)

#### Is prefill modelled as quadratic in context length? Yes: an empirical quadratic fit
- §5.2 "Offline Profile": "(2) Prefill vs context length curve for estimating prefill cost. We measure this by doing prefill for chunk sizes {1000,2000,4000,...max_context_length} and fit a quadratic curve on the data." — [v7 HTML](https://arxiv.org/html/2511.02230v7)
- On pages still resident: "there could be some pages for the request remaining in GPU memory that does not need recomputation. But these remaining pages are usually small when memory is contended and we approximate by the full prefill time with little error." — [v7 HTML](https://arxiv.org/html/2511.02230v7)
- On cost: "Profiling takes less than 10 minutes for each hardware model pair." — [v7 HTML](https://arxiv.org/html/2511.02230v7)

#### Version history of the rule
- **v1 (4 Nov 2025) used a confidence bound, not a cost-benefit objective.**
  - "we set the timeout based on the predicted upper bounds B(δ) and B_f(δ) of the tool-call intervals, with a confidence level of 1−δ."
  - "When a tool f has been invoked at least N times (i.e., |S_f| ≥ N), we use its per-tool bound B_f(δ); otherwise, we fall back to the global bound B(δ)."
  - "The parameters δ, N, and T_default are tunable." v1 also feeds in "Number of turns".
  - v1's key: "use the first word as the tool call name".
  - — [v1 HTML](https://arxiv.org/html/2511.02230v1)
- **v4 (4 May 2026, titled "CacheTTL") matches v7.** It has the same Cost(τ,r), CacheMissCost and OutofOrderCost; the same Eq. (1)/(2) text including "By eliminating the shared MemUsage(r)/M"; K=100; and the same SSD sensitivity experiment — [v4 HTML](https://arxiv.org/html/2511.02230v4)

#### Released code (default branch HEAD, commit 316a587)
- **Fixed TTL.** `vllm/v1/core/estimate_with_func.py` defines `FIXED_THRESHOLD_CONTINUUM = 2.0  # seconds` — [code](https://github.com/Hanchenli/vllm-continuum/blob/316a58794a6ff86b216e579b74fd56ed0c5a911f/vllm/v1/core/estimate_with_func.py)
- **Per-tool mean.** `update_func_call_exec_time` stores a per-tool running mean of server-side intervals, under the comment "#TODO Hanchen This is currently just an average" — [code](https://github.com/Hanchenli/vllm-continuum/blob/316a58794a6ff86b216e579b74fd56ed0c5a911f/vllm/v1/core/estimate_with_func.py)
- **Pin rule.** `set_up_pin` returns 0 if the tool's mean exceeds 2.0 s and otherwise returns 2.0 — [code](https://github.com/Hanchenli/vllm-continuum/blob/316a58794a6ff86b216e579b74fd56ed0c5a911f/vllm/v1/core/estimate_with_func.py)
- **Key.** `ToolCallParser` returns `words[0]` of the single bash block ("Extract the first word (command) from the action"), using the "Same regex pattern as mini-swe-agent" — [code](https://github.com/Hanchenli/vllm-continuum/blob/316a58794a6ff86b216e579b74fd56ed0c5a911f/vllm/v1/core/estimate_with_func.py)
- **What is absent.** The file contains no empirical CDF, η, T or K — [code](https://github.com/Hanchenli/vllm-continuum/blob/316a58794a6ff86b216e579b74fd56ed0c5a911f/vllm/v1/core/estimate_with_func.py)

#### Corrections to the earlier notes
1. **The hybrid formula is in neither note.** Neither note literally contains "P(τ,f)·(T·η + Prefill − Reload) − τ·MemUsage/M". Note A line 19 and Note B line 19 both quote Eq. (2) correctly, and the hybrid is in none of v1, v4 or v7. "Prefill − Reload" misreads `\mathsf{Prefill\text{-}Reload}(r)`, "the time cost for prefill or reloading" — [v7 HTML](https://arxiv.org/html/2511.02230v7)
2. **Note B line 28 is wrong.** It says "The tool is the key, without the command's first word". In fact:
   - v7 §5.1 uses "the first word afterwards as the tool call name" on SWE-Bench — [v7 HTML](https://arxiv.org/html/2511.02230v7)
   - v1 says "use the first word as the tool call name" — [v1 HTML](https://arxiv.org/html/2511.02230v1)
   - The code returns `words[0]` — [code](https://github.com/Hanchenli/vllm-continuum/blob/316a58794a6ff86b216e579b74fd56ed0c5a911f/vllm/v1/core/estimate_with_func.py)
3. **Note B line 259 is right about Eq. (1), with two qualifications.** The line reads "an implicit fixed price of about 1/M seconds per token-second ... it does not fall to zero when memory is plentiful".
   - MemUsage is "GPU memory", so "per token-second" holds only if memory is counted in tokens.
   - The factor cancels in Eq. (2), which the paper states itself ("By eliminating the shared MemUsage(r)/M"). Note B line 261 is therefore a statement of the paper, not merely "my reading".
   - — [v7 HTML](https://arxiv.org/html/2511.02230v7)
4. **Note A line 33 is right but incomplete.** It reads "The memory term MemUsage/M cancels in the simplified rule, so there is no price on host memory and no capacity coupling". It should add two points:
   - Eq. (1) does charge memory-time in latency units before the cancellation.
   - T, the sliding-window queueing delay of evicted requests, is the only load-dependent input to the rule.
   - — [v7 HTML](https://arxiv.org/html/2511.02230v7)
5. **Note A line 18 has the fallback order slightly wrong.** It reads "With |S[f]| ≤ K = 100 it falls back to global records, and otherwise to a default TTL". The default TTL is used when the *global* count |S| ≤ K, not "otherwise" — [v7 HTML](https://arxiv.org/html/2511.02230v7)
6. **Note A line 19 omits two release paths.** It says the KV "stays pinned ... until the TTL expires or 'the program terminates'". Unpinning also requires that the program_id is not in the waiting queue, and a deadlock path unpins by "latest program arrival time" — [v7 HTML](https://arxiv.org/html/2511.02230v7)
7. **Two open gaps are now closed.** Note A line 335 and Note B line 272 left these open:
   - The SSD runs are a §6.3 sensitivity experiment on LMCache's SSD layer.
   - v7 does not re-plan TTLs as idle time elapses.
   - — [v7 HTML](https://arxiv.org/html/2511.02230v7)

#### What Continuum does and does not contain, relative to (a)–(d)
- **(a) Forecasts.**
  - *Contains:* per-tool empirical CDFs, learned online by the engine from server-side inter-request intervals; the first word of the bash command as the key (SWE-Bench); and a fallback to global records below K = 100, then to an Exp(1)-derived default.
  - *Does not contain:* distributions sent by the agent framework (the client sends only program_id), or any conditioning on elapsed idle time.
  - — [v7 HTML](https://arxiv.org/html/2511.02230v7)
- **(b) Fetch.** *Does not contain* any rule for starting a read: no prefetch, no SSD read decision, no critical ratio — [v7 HTML](https://arxiv.org/html/2511.02230v7)
- **(c) Retain.**
  - *Contains:* a GPU retention horizon chosen by expected benefit minus holding cost over an unconditional CDF, fixed at the start of the call; and unpinning by latest program arrival under deadlock.
  - *Does not contain:* a hazard rate, an eviction index, or re-evaluation as idle time passes.
  - — [v7 HTML](https://arxiv.org/html/2511.02230v7)
- **(d) Price.**
  - *Contains:* a fixed latency charge for memory-time in Eq. (1), under a saturated-queue assumption, which cancels in Eq. (2).
  - *Does not contain:* a capacity constraint, a multiplier, dual ascent, a learned price, or a lower bound.
  - — [v7 HTML](https://arxiv.org/html/2511.02230v7)

### Inferences
- **Suggested related-work sentence (inference).** "Continuum chooses a GPU-pinning TTL by maximizing P(τ,f)·B − τ over an unconditional per-tool empirical CDF. B is the avoided prefill or reload time plus a memoryfulness-weighted queueing delay. Memory occupancy enters its derivation at a fixed conversion rate, the request's share of the average footprint, which cancels from the rule it solves."
- **Implied price (inference).** Read against (d), Eq. (1) prices memory-time at a constant λ = 1/M, in seconds per memory-unit-second. The rate is set by the measured average footprint, not by scarcity. Because the benefit carries the same factor, the solved rule is λ-free: each second of pinning costs exactly one second, for any request size and any memory headroom.
- **Contrast with (c), from Eq. (2) (inference).**
  - Continuum charges the full τ even when the tool returns early, so its interior optimum satisfies p(τ)·B = 1, where p is the *unconditional* density.
  - Charging the expected occupancy E[min(D,τ)] = ∫₀^τ S(x)dx instead gives derivative S(τ)·(h(τ)·B − 1), i.e. the hazard condition h(τ)·B = 1. Our rule generalizes that condition with a size- and price-dependent threshold.
  - Since p = h·S ≤ h, under single-crossing conditions Continuum stops pinning no later than a hazard rule with the same B would.
  - Its τ* is also chosen once at call start, while a hazard rule re-evaluated at each idle time stays consistent with conditioning.
- **Opposite size directions (inference).** In Eq. (2) size enters only through Prefill-Reload(r), which grows (quadratically) with context, so larger contexts get *longer* TTLs. Under our q = λL/(1+λL), a larger prefix must clear a *higher* threshold. Unless read time dominates, the two rules move in opposite directions with size.
- **The Exp(1) default (inference).** With P(τ) = 1 − e^(−τ) and η = 1, Eq. (2) gives τ* = ln(T + Prefill-Reload(r)) when the bracket exceeds 1 (in the Exp's time unit), and 0 otherwise. The "fixed" T_default therefore depends on B unless B is fixed. The paper reports no value.
- **What "free" means (inference).** This rests on vLLM's prefix-caching semantics, not on the paper. A zero or expired TTL returns blocks to vLLM's free queue, where they remain reusable until reallocated. Continuum's pin therefore protects against LRU eviction rather than deciding whether the cache survives.
- **Released code versus paper (inference).** The released code (a 2 s fixed TTL with a mean gate) resembles the paper's "minimal version" rather than Eq. (2). A reviewer or baseline may use either. Ask the Tool reimplemented the paper's "per-tool time-to-live from the empirical distribution of past durations" (§4.1 of [Ask the Tool HTML](https://arxiv.org/html/2609.18849v1)). If we run Continuum as a baseline, we should state which variant we ran.

### Gaps
- **Venue.** The ICLR 2026 workshop venue claimed in Notes A and B was not re-verified; the arXiv record states no venue.
- **Code version.** I do not know which code commit or branch produced v7's experiments; I read only the default branch HEAD.
- **Unreported values.** T_default and the experimental values of T are not reported. The paper does not say whether Prefill-Reload(r) used CPU or SSD reload time in the SSD runs.
- **Versions not read.** I did not read v2, v3, v5 or v6. The cost-benefit objective therefore appeared somewhere between v1 (Nov 2025) and v4 (May 2026).

---

## 2. CacheWise (arXiv 2606.16824): what exactly is its "expected remaining time", and how does eviction use it?

### Takeaway
**The predictor.** Verbatim, it is E[τ_i(t) | τ_i(t) − T_i > t − T_i], "the expected remaining time for the tool call to complete". It is computed from per-tool empirical duration distributions or, "in such cases" where names are insufficient, from per-argument-cluster distributions (TF-IDF + KMeans). The released predictor computes exactly E[T − t | T > t] = ∫ₜ S / S(t).

**Eviction.** It removes blocks "in decreasing order of τ_i(t)", described as an attempt "to approximate the optimal eviction strategy" that "follows Belady's rule".

**What it lacks.** There is no size weighting, no price, no prefetch and no SSD. One configuration offloads to CPU DRAM and swaps back on demand. The predictor is trained offline on 80% of sessions, and the vLLM engine parses `tool_name` and `tool_args` itself.

### Cited Findings

#### Record (verified)
- **Title and authors.** "CacheWise: Understanding Workloads and Optimizing KVCache Management for Efficiently Serving LLM Coding Agents". The 6 authors are Shubham Tiwari, Tapan Chugh, Nash Rickert, Simon Peter, Ratul Mahajan and Haiying Shen — [arXiv abs](https://arxiv.org/abs/2606.16824)
- **Version and venue.** v1 (Mon 15 Jun 2026) is the only version ("[cs.DC]"). There is no Comments or Journal-ref field, so no venue is stated — [arXiv abs](https://arxiv.org/abs/2606.16824), [v1 HTML](https://arxiv.org/html/2606.16824v1)
- **Results claimed.** The abstract says CacheWise "reduces KVCache evictions by up to 2–2.6× and improves total agent session completion time by up to ~3.5×". It gives the code at github.com/cachewise-project/cachewise-coding-traces — [v1 HTML](https://arxiv.org/html/2606.16824v1)

#### Model and decision rule, verbatim (no numbered equations; §4 and §5.2)
- **Model (§4).**
  - "Let M denote the total KVCache block capacity of an inference node".
  - "Each session S_i has a contiguous, monotonically growing prefix of KVCache blocks d_i, out of which k_i(t) are resident in XPU memory at time t."
  - "the number of additional blocks that must be allocated is a_i(t)=d_i−k_i(t). If W(t)+a_i(t) ≤ M, the request is admitted immediately. Otherwise, the system must evict a_i(t) blocks from one or more other sessions before admission."
  - — [v1 HTML](https://arxiv.org/html/2606.16824v1)
- **Implication #2 (§4).**
  - "Let τ_i denote the time to next reuse of session S_i's resident blocks, i.e., the time until S_i issues its next request and those blocks are accessed again."
  - "An ideal eviction policy therefore selects `j^{*}=\operatorname*{arg\,max}_{j\,\in\,\mathcal{S}_{t},\;j\neq i}\tau_{j}`, retaining sessions whose KVCache will be needed soonest."
  - — [v1 HTML](https://arxiv.org/html/2606.16824v1)
- **The rule (§5.2).**
  - "Ideally, the optimal eviction strategy follows Belady's rule (Belady, 1966), i.e., at time t, evict blocks whose next access time τ_i(t) is furthest into the future".
  - "CacheWise proposes a predictive KVCache eviction policy that attempts to approximate the optimal eviction strategy by evicting blocks in decreasing order of τ_i(t)."
  - "CacheWise only needs to predict the relative order of τ(t) across sessions to identify which blocks to evict."
  - — [v1 HTML](https://arxiv.org/html/2606.16824v1)
- **The predictor (§5.2).** "For a session S_i that generated a tool call with metadata m=(tool_name, tool_args) at time T_i, our predictor estimates it's next expected use time E[τ_i(t)] using historical information ... Specifically, at time t, CacheWise estimates the expected remaining time for the tool call to complete, i.e., `\mathbb{E}[\tau_{i}(t)\mid\tau_{i}(t)-T_{i}>t-T_{i}]` by analyzing distributions of tool execution durations with similar metadata m from historical sessions with similar characteristics e.g., same user, same project." — [v1 HTML](https://arxiv.org/html/2606.16824v1)
- **Terms.**
  - τ_i(t) is "the time to next reuse" (§4) and also the "next access time" (§5.2).
  - T_i is the "time when the tool call was generated" (§5.3); t is the current time; m = (tool_name, tool_args).
  - The conditioning event τ_i(t) − T_i > t − T_i says the call's duration exceeds the time elapsed since it was issued.
  - — [v1 HTML](https://arxiv.org/html/2606.16824v1)
- **Argument clusters (§5.2).**
  - "relying solely on execution duration distributions of tools with the same names is insufficient ... CacheWise semantically clusters historical samples based on the TF-IDF embeddings of their arguments, and uses these per-cluster distributions instead of per-tool distributions in such cases."
  - Footnote 2: "We use KMeans ... We upper bound the number of clusters to a large value to achieve maximum granularity".
  - Footnote 3: "a maximum size of 5000 lexicons".
  - — [v1 HTML](https://arxiv.org/html/2606.16824v1)
- **Table 3 (bash clusters by argument type; P50/P90/P99, n).**
  - python -c: 0.1/0.2/0.4 s, n=9
  - mypy: 10/56/182 s, n=46
  - docker compose: 22/89/152 s, n=30
  - git add/status: 0.1/34/97 s, n=34
  - — [v1 HTML](https://arxiv.org/html/2606.16824v1)
- **Granularity results (§6.5).**
  - "Point Estimates uses a single point estimate of τ̂ derived from the global mean tool duration across all tool types and sessions."
  - "Tool Name Only conditions the estimate on tool name alone, using the per-tool empirical duration distribution (§5). C20, C50, and C100 further refine the estimate by clustering tool calls based on tool arguments".
  - "C100 achieves the highest performance, with upto 19% improvement in session completion time."
  - — [v1 HTML](https://arxiv.org/html/2606.16824v1)

#### Heap and re-evaluation (§5.3)
- **Implementation.** "~2,500 lines of Python". Each block carries "tool_name, tool_args and T_i (time when the tool call was generated), which can be parsed by the vLLM engine itself." — [v1 HTML](https://arxiv.org/html/2606.16824v1)
- **Heap.** "When a request completes and its blocks become unreferenced ..., those blocks retain their metadata and are inserted into an eviction heap for ordered eviction. CacheWise replaces the default LRU eviction policy: ... adds the block to the eviction heap with the prediction as its priority value." — [v1 HTML](https://arxiv.org/html/2606.16824v1)
- **Re-evaluation.** "CacheWise periodically re-evaluates all unreferenced blocks through the predictor and rebuilds the eviction heap with updated E[τ_i(t)] estimates", with "N_rebuild=3" engine iterations — [v1 HTML](https://arxiv.org/html/2606.16824v1)

#### Size, price, tiers, prefetch
- **Size and price.** The eviction priority is the prediction alone (§5.3). Size enters only scheduling: "CacheWise selects request r_i which requires fewest additional blocks a_i(t) to be allocated as the next request to dispatch" (§5.1). The main body contains none of "price", "prefetch", "hazard" or "Lagrang" (term scan) — [v1 HTML](https://arxiv.org/html/2606.16824v1)
- **CPU offload (§6.4).** "evicted blocks are transferred from GPU memory to CPU DRAM and swapped back on demand when the corresponding session resumes". Result: "CacheWise achieves ~1.19x lower session completion times than vLLM." — [v1 HTML](https://arxiv.org/html/2606.16824v1)
- **Tiers (§7).** "Tiered storage is complementary to CacheWise's approach." SSD appears only in background and related work — [v1 HTML](https://arxiv.org/html/2606.16824v1)
- **Human pauses (§6.1).** "we always evict KVCache from idle sessions before evicting from sessions with toolcall induced delays." — [v1 HTML](https://arxiv.org/html/2606.16824v1)

#### How distributions are obtained, and who parses tool metadata
- **Offline training (§6.1).** "We randomly sample 80% of the sessions for offline training of the predictor, and use the remaining traces for our experiments. We assume that the system deploying CacheWise routinely logs request traces which can be used as the training data." — [v1 HTML](https://arxiv.org/html/2606.16824v1)
- **Drift.** "CacheWise's estimators can be periodically retrained on recent traces to adapt to distribution shift, though we leave a thorough study of drift robustness to future work." — [v1 HTML](https://arxiv.org/html/2606.16824v1)
- **The engine parses (§5.3, above).** §7: CacheWise "only modifies the serving system". — [v1 HTML](https://arxiv.org/html/2606.16824v1)

#### Released predictor code (commit 181c435)
- **The formula.** `tool_duration_prediction/infer.py` has the docstring "Inference for expected remaining tool-call duration E[T - t | T > t]." — [code](https://github.com/cachewise-project/cachewise-coding-traces/blob/181c435a090d328d00bbbee4c8eeb27d32f3abd2/tool_duration_prediction/infer.py)
- **How it is computed.** `expected_remaining_time(curve, elapsed_ms)` integrates the "prob still running" curve from the elapsed time to the curve's last point by the trapezoid rule, then divides by the curve's value at the elapsed time. It returns 0 when the elapsed time is past the last point — [code](https://github.com/cachewise-project/cachewise-coding-traces/blob/181c435a090d328d00bbbee4c8eeb27d32f3abd2/tool_duration_prediction/infer.py)
- **Curve choice.** `select_curve` takes the nearest TF-IDF cluster, but uses the tool's overall curve when cosine similarity < `SIM_THRESHOLD = 0.3` — [code](https://github.com/cachewise-project/cachewise-coding-traces/blob/181c435a090d328d00bbbee4c8eeb27d32f3abd2/tool_duration_prediction/infer.py)
- **What the repo holds.** Its file list shows trace parsing, workload analysis, and predictor training and inference. I saw no vLLM integration — [repo](https://github.com/cachewise-project/cachewise-coding-traces/tree/181c435a090d328d00bbbee4c8eeb27d32f3abd2)

#### Corrections to the earlier notes
- **Resolved gap.** Note A line 334 ("conditional-expectation formula did not render") is resolved: the formula is E[τ_i(t) | τ_i(t) − T_i > t − T_i], and the code computes E[T − t | T > t] — [v1 HTML](https://arxiv.org/html/2606.16824v1), [code](https://github.com/cachewise-project/cachewise-coding-traces/blob/181c435a090d328d00bbbee4c8eeb27d32f3abd2/tool_duration_prediction/infer.py)
- **Otherwise accurate.** Note A lines 41–43 match the text: offline training on 80%, a heap keyed by the prediction, rebuilds every 3 iterations, no prefetch, and on-demand swap-in from CPU — [v1 HTML](https://arxiv.org/html/2606.16824v1)
- **Minor omission.** Note A line 433 lists "Full distributions: only Continuum ... and CacheWise". IAC also fits a full log-normal per (intent, turn) (Section 4 below) — [IAC v1 HTML](https://arxiv.org/html/2611.09715v1)

#### What CacheWise does and does not contain, relative to (a)–(d)
- **(a) Forecasts.**
  - *Contains:* per-tool and per-argument-cluster empirical duration distributions, conditioned on elapsed time and re-evaluated every 3 iterations; engine-parsed; offline-trained.
  - *Does not contain:* framework-sent distributions, online learning, or first-word keys (it uses argument clusters instead). It also uses nothing from the conditional distribution beyond its mean.
  - — [v1 HTML](https://arxiv.org/html/2606.16824v1)
- **(b) Fetch.** *Does not contain* any read-start rule: reloads are on demand — [v1 HTML](https://arxiv.org/html/2606.16824v1)
- **(c) Retain.**
  - *Contains:* Belady-approximating eviction by the largest predicted (conditional) remaining time.
  - *Does not contain:* size, read time, a price, a threshold, a hazard rate, or any normalization by holding time.
  - — [v1 HTML](https://arxiv.org/html/2606.16824v1)
- **(d) Price.** *Does not contain* a price: M is a hard capacity — [v1 HTML](https://arxiv.org/html/2606.16824v1)

### Inferences
- **Notation (inference).** If τ_i is an absolute time, E[τ_i(t) | ...] is an expected next-use time. Subtracting t gives E[D − a | D > a], with D = τ_i − T_i and a = t − T_i. After each rebuild every entry is scored at the same t, so ranking by either quantity is identical. Between rebuilds, entries inserted at different times are compared on predictions made at different times.
- **One summary statistic (inference).** Ranking by conditional mean remaining time uses one summary of the distribution. Two sessions with equal mean remaining time but different chances of returning within the next read time get the same rank. Our index, (1+λL)·max_H F(H|t)/τ(H), separates them and also weights by size and price.
- **Truncation in the released code (inference).** The predictor truncates the survival curve at the largest observed duration. A call that outlives all of its history therefore gets remaining time 0, i.e. is ranked as about to return. Under heavy tails that is the opposite of the right ranking. I do not know whether the vLLM integration does the same.
- **Credit (inference).** CacheWise is the citation for conditioning on elapsed time and for eviction framed as a Belady approximation. It should be credited for both and not contradicted.

### Gaps
- **Engine integration.** The vLLM integration code was not found. How the engine computes E[τ] at runtime, and how the "same user, same project" conditioning is implemented, are unconfirmed.
- **Cluster count.** The main results' cluster count is given only as "upper bound ... to a large value"; §6.5 reports C100 as best.

---

## 3. "Ask the Tool, Don't Guess" (arXiv 2609.18849): what is its claim against pre-call estimates, and how are return times and lead times formed?

### Takeaway
**The claim.** "no estimate fixed before a call starts can know its duration, and such estimates may not even rank the calls". The stated reason: "A long call's duration is set by the load on the machine, a neighbour in the sandbox, or the state of a remote API".

**The evidence:**
- **Replays.** Recorded commands replayed on an idle sandbox ran an order of magnitude faster, busy neighbours made calls several times slower, and long calls ranked with no correlation between production load and the idle machine.
- **Concentration.** Fewer than 3% of calls hold almost two thirds of tool time.
- **Sample counts.** Continuum's roughly 100-sample requirement is met by "only a handful of tool identities".
- **Accuracy.** Median error at 50% and 90% of each call is compared with four reimplemented predictors plus a 3Sigma-style estimator conditioned on elapsed time. At 90%, progress errs by under 10% and the best history predictor by 80%.

**Its own estimate.** It is a point estimate: progress is "converted linearly into a remaining-time estimate", or a near-end announcement that carries no amount.

**The controller.** It keeps a context while "the reported remaining time is under a threshold", adds hysteresis when a host tier exists, and "refreshes the context one lead time before the predicted return". The paper gives no value or selection rule for the lead time or the threshold.

**Memory.** Memory shortage is modelled as an exogenous, random moment. There is no price, no size dependence and no SSD.

### Cited Findings

#### Record (verified)
- **Title and authors.** "Ask the Tool, Don't Guess: Agent Tool Calls Hold Their Progress, and the Serving System Should Read It". The 4 authors are Yipeng Liu, Yingqiang Zhang, Feifei Li and Huanchen Zhang — [arXiv abs](https://arxiv.org/abs/2609.18849)
- **Version and venue.** v1 (Wed 16 Sep 2026) is the only version ("[cs.DC]"). There is no Comments or Journal-ref field, so no venue is stated — [arXiv abs](https://arxiv.org/abs/2609.18849), [v1 HTML](https://arxiv.org/html/2609.18849v1)
- **No code.** The text links no repository of its own; its GitHub links point to other projects — [v1 HTML](https://arxiv.org/html/2609.18849v1)

#### The claim, verbatim
- **Abstract.** "Serving systems decide whether that cache stays, leaves, or comes back by guessing how long the tool will run, from the tool's name, its history, a duration declared before the call, or the engine's own occupancy. We show that no estimate fixed before a call starts can know its duration, and such estimates may not even rank the calls." — [v1 HTML](https://arxiv.org/html/2609.18849v1)
- **§1.** "The guesses fail for a reason that no better model will remove: the information is not available before the call. A long call's duration is set by the load on the machine, a neighbour in the sandbox, or the state of a remote API ..., none of which the command line shows. Predictors built from a call's name, arguments, or history cannot even rank the long calls. History helps only when it was collected in the environment the call now runs in, and a history that is both large enough to trust (Li et al., 2025; Tiwari et al., 2026) and collected under one load rarely exists (Section 2.1)." — [v1 HTML](https://arxiv.org/html/2609.18849v1)
- **§2.1.** "None of them reads the running call: they all keep predicting its duration implicitly, from what lies around it. Every rung fixes its estimate before the call runs, or never sees inside it." — [v1 HTML](https://arxiv.org/html/2609.18849v1)
- **Table 1 (the "ladder" of timing signals).**
  - Fixed TTL (CachedAttention, SGLang, Anthropic): sees "nothing".
  - Elapsed time (InferCept, LAMPS): "elapsed time".
  - Per-tool statistics (Continuum): "name".
  - Learned model (CacheWise): "name, arguments".
  - Engine-side occupancy (MORI, ConServe, ThunderAgent, Adaptive KV Retention): "engine state, not the call".
  - Declared duration (TokenCake, SGLang, Mooncake, vLLM, TensorRT-LLM): "fixed estimate".
  - Reported progress (this paper): "live progress".
  - The text adds: "Elapsed-time estimates keep a call as long again as it has already run".
  - — [v1 HTML](https://arxiv.org/html/2609.18849v1)
- **Self-scoping (§6).** "Our claim is about the decisions that need a remaining time, and about the calls whose duration is set by the environment, which no call-time feature can see." — [v1 HTML](https://arxiv.org/html/2609.18849v1)

#### The evidence
- **Concentration (§2.1).** "fewer than three percent of tool calls hold almost two thirds of all tool time (Figure 2)". The data (footnote 1): "these leaderboard runs (six submissions, about three thousand trajectories), the OpenHands evaluation runs (Wang et al., 2025), and our own runs on a hundred random SWE-bench Verified tasks using Qwen 3.8 Flash" — [v1 HTML](https://arxiv.org/html/2609.18849v1)
- **Environment replays (§2.1).** "We replayed the commands recorded in leaderboard runs on an idle sandbox. The same command finished an order of magnitude faster (Figure 3a). Restricting the sandbox's CPU quota changed almost nothing. Adding busy neighbours, however, made most calls several times slower (Figure 3b). The order of the long calls recorded under production load has no correlation with their order on an idle machine, and when the idle machine is oversubscribed, much of its own ordering is lost as well (Figure 3c)." — [v1 HTML](https://arxiv.org/html/2609.18849v1)
- **History size and match (§2.1).**
  - "A history pooled across environments is an order of magnitude worse near the end of the call than the matching history (Section 4)".
  - "Continuum, for one, asks for about a hundred past calls per tool before it trusts a distribution (Li et al., 2025). In our corpus only a handful of tool identities reach that count."
  - — [v1 HTML](https://arxiv.org/html/2609.18849v1)
- **Competitors (§4.1).** "We compare against four published predictors, each reimplemented as its paper defines it: Continuum's (Li et al., 2025) per-tool time-to-live from the empirical distribution of past durations, CacheWise's (Tiwari et al., 2026) clusters of tool names and arguments, TokenCake's (Bian et al., 2027) running average of recent calls, and InferCept's (Abhyankar et al., 2024; Shahout et al., 2025) two rules, a profiled duration per tool and the elapsed rule that a call has as long left to run as it has already run. Progress signals from our instruments are converted linearly into a remaining-time estimate. We also add our own in-flight history estimator in the manner of 3Sigma's conditioning on elapsed time (Park et al., 2018)." — [v1 HTML](https://arxiv.org/html/2609.18849v1)
- **Midpoint accuracy (§4.2).**
  - "As a courtesy, we fix the sandbox configuration for the predictors, where their history matches the current machine."
  - "Across all the methods at 50% of each call, the progress stream's median error is about a fifth of the call's duration, against a third for the best history predictor (Figure 7a)."
  - — [v1 HTML](https://arxiv.org/html/2609.18849v1)
- **End-of-call accuracy (§4.2).** "Near the end, the progress stream is far more accurate than any predictor, because the history predictors are reading the far end of a distribution that no longer describes this call. A call still running at that point has outlasted most of its class, and the history has little left to say about it. The progress stream's median error at 90% of each call is under 10%, against 80% for the best history predictor (Figure 7b)." — [v1 HTML](https://arxiv.org/html/2609.18849v1)
- **Trigger timeliness (§4.2, Appendix C).**
  - Definition: "The recovery budget is defined as the time it takes to get a context back, and an estimator triggers when it first reports a return within that budget. The trigger is timely if it lands just enough budget before the return."
  - Headline: "The progress stream is timely in about a quarter of the runs with a two-second budget and a sixth with a half-second budget, far better than the baselines."
  - Figure 13 (2 s budget): progress is timely in "31 % of test runs, 15 % of builds and 50 % of scripts; the best baseline in 0, 0 and 13 %".
  - Figure 14 (0.5 s budget): "21 % of test runs, 2 % of builds and 50 % of scripts; no baseline triggers in time on any run."
  - — [v1 HTML](https://arxiv.org/html/2609.18849v1)
- **A concession that pre-call sorting works for short calls (§4.1).**
  - "Most tool time can be sent down the right path before the call starts." This means sorting calls into too short to matter, self-declaring, needs progress, or unparseable.
  - Table 3 gives precision of 98/97/96% and recall of 80/83/58% on mini-SWE-agent, OpenHands and the authors' own runs.
  - — [v1 HTML](https://arxiv.org/html/2609.18849v1)

#### How the predicted return time is formed: a point estimate, not a distribution
- **Linear conversion (§4.1).** "Progress signals from our instruments are converted linearly into a remaining-time estimate." — [v1 HTML](https://arxiv.org/html/2609.18849v1)
- **Events (§3.2).** "Every event has the same shape: a call id, the work done, the total if known, the phase, and a timestamp." For multi-phase calls: "Where earlier runs of the same command show how much of the call each phase takes, the wrapper folds the bars into one fraction with those shares as weights". — [v1 HTML](https://arxiv.org/html/2609.18849v1)
- **Rate (Appendix C).** "The whole-run rate carries the old load for tens of seconds. A windowed rate recovers within about ten seconds, at the price of overshooting." — [v1 HTML](https://arxiv.org/html/2609.18849v1)
- **Weak signals (§2.2).** "A weak signal cannot say how much is left, but in most cases its announcement of the final phase comes early enough to start a reload or a recompute in time." — [v1 HTML](https://arxiv.org/html/2609.18849v1)
- **Declared durations (§2.2).** "Calls that declare their own duration give the scheduler a deadline rather than an estimate: a sleep is exact, and so is a timeout wrapping a command that would never return on its own; around any other command the timeout is an upper bound, which is what a prefetch needs." — [v1 HTML](https://arxiv.org/html/2609.18849v1)
- **No distribution.** The main body contains no "quantile", "percentile", "survival", "hazard" or "conditional" (term scan). No distributional form of its own estimate is described — [v1 HTML](https://arxiv.org/html/2609.18849v1)

#### The refresh rule and its "lead time"
- **§5.1.** "Our method based on progress keeps a context while the reported remaining time is under a threshold. With a host tier it adds hysteresis, letting go only once the estimate has moved well past the threshold, and refreshes the context one lead time before the predicted return." — [v1 HTML](https://arxiv.org/html/2609.18849v1)
- **Appendix D, "The controller".** "While the reported remaining time is under the keep threshold, it refreshes the session's context; once the estimate has moved well past the threshold, it stops, and the context ages out under LRU or moves to the host tier; if the estimate falls back, the refreshes resume. With a host tier, the controller also refreshes the context one lead time before the predicted return and reloads it if it has gone. The oracle runs the same controller with the true remaining time; LRU never hints." — [v1 HTML](https://arxiv.org/html/2609.18849v1)
- **Engine patch (Appendix D).** "refresh moves its resident blocks to the safe end of the eviction queue at no cost, release moves them to the front while keeping them addressable, and query reports how much of it is resident." Also: "The scheduler, its eviction rule, and the model are untouched". — [v1 HTML](https://arxiv.org/html/2609.18849v1)
- **No numbers for the controller.** No numeric value or selection rule is given for "lead time", "keep threshold" or the hysteresis band in §5 or Appendix D. The two settings, 2 s and 0.5 s, are *recovery budgets* used to score trigger timeliness in §4.2 and Appendix C; Figure 8's x-axis is the "extra lead allowed" — [v1 HTML](https://arxiv.org/html/2609.18849v1)

#### Memory scarcity and cost
- **Shortage as exogenous (§4.2).** "Memory runs short for reasons outside any one call. From the call's point of view the moment is random: uniform over the call, and on average halfway through." — [v1 HTML](https://arxiv.org/html/2609.18849v1)
- **Cost of mistakes, stated qualitatively (§5.1).** "With GPU memory alone, a wrongly kept context costs someone else a recompute and a wrongly evicted one costs this call a recompute, so the decision is a pure ordering: who stays. With a host tier, an evicted context is copied to DRAM rather than dropped, so a wrong let-go costs only a reload, and the decision becomes when to let go and when to fetch back, paid for mainly in latency and DRAM traffic." — [v1 HTML](https://arxiv.org/html/2609.18849v1)
- **Trust, not price (§5.3).** "Two ledgers exist, one for memory kept on a report's word and one for latency risked on it, so a lie can burn only the ledger it draws on." These ledgers are trust accounting, not a memory price — [v1 HTML](https://arxiv.org/html/2609.18849v1)
- **Term scan.** The main body has no "SSD", "hazard" or "Lagrang". Its single "price" is "at the price of output tokens" — [v1 HTML](https://arxiv.org/html/2609.18849v1)
- **Results (§5.2).**
  - "With a host tier, progress cuts the p90 by 21% and the oracle by 23%, and both reload a third of what LRU does."
  - On the baselines: "Continuum's per-tool time-to-live does not expire for many calls, where it issues no hint and falls back to LRU. CacheWise, TokenCake and InferCept estimate each call from the short-dominated history of its kind, so their estimates fall below the keep threshold for most calls and they keep nearly everything."
  - The abstract gives 20.7% (HBM only) and 20.8% (HBM + DRAM) against LRU.
  - — [v1 HTML](https://arxiv.org/html/2609.18849v1)

#### Corrections to the earlier notes
- **Note A line 62 conflates evaluation with implementation.** It says: "In a vLLM patch, HBM-only eviction removes the context whose call 'has the longest to run'."
  - That phrase is the decision *judged* in §4.2 ("Among the idle contexts resident in GPU memory, the serving system evicts the one whose call has the longest to run").
  - In the implemented patch, "The scheduler, its eviction rule, and the model are untouched". The controller refreshes contexts whose reported remaining time is under the keep threshold, and the others "age out under LRU".
  - — [v1 HTML](https://arxiv.org/html/2609.18849v1)
- **Otherwise accurate.** Note A lines 60, 63, 71 and 453 match the text — [v1 HTML](https://arxiv.org/html/2609.18849v1)

#### What Ask the Tool does and does not contain, relative to (a)–(d)
- **(a) Forecasts.**
  - *Contains:* a side channel from the harness to the engine, keyed by call id, carrying progress events; and explicit evidence against pre-call and history-based estimates, including one conditioned on elapsed time.
  - *Does not contain:* distributions sent by the framework, or any distribution in its own method.
  - — [v1 HTML](https://arxiv.org/html/2609.18849v1)
- **(b) Fetch.**
  - *Contains:* a host-tier refresh "one lead time before the predicted return", driven by a point estimate.
  - *Does not contain:* a probability threshold, any dependence on size or price, or SSD.
  - — [v1 HTML](https://arxiv.org/html/2609.18849v1)
- **(c) Retain.**
  - *Contains:* keep while reported remaining time < threshold, with hysteresis, on top of LRU.
  - *Does not contain:* a hazard rate, an index, or size.
  - — [v1 HTML](https://arxiv.org/html/2609.18849v1)
- **(d) Price.** *Does not contain* a memory price. The ledgers serve trust, not scarcity — [v1 HTML](https://arxiv.org/html/2609.18849v1)

### Inferences
- **It targets (a) most directly (inference).** Its 3Sigma-style baseline conditions history on elapsed time, as our engine conditions F on idle time. At 90% of a call, that baseline erred far more than progress did.
- **But its metrics are point-based (inference).** Accuracy is median point-ETA error, and timeliness is a point trigger. Our fetch uses a probability threshold. A wide conditional distribution late in a long call gives a small F(read time | t), so our rule declines to prefetch and degrades to an on-demand read. A point-error metric does not capture that behaviour.
- **What not to claim (inference).** The paper should not claim that pre-call or history forecasts predict long, environment-bound calls.
- **What to claim instead (inference).**
  - A calibrated conditional distribution makes the fetch decision risk-aware and size- and price-aware.
  - Progress events, when available, can be one more conditioning input.
  - Their Table 3 concedes that pre-call sorting works for the short-dominated bulk of calls, where per-tool distributions are informative.
- **Their trigger as a special case (inference).** "first reports a return within that budget" is the point-estimate special case of our rule: fetch when predicted remaining ≤ read time. It has no size- or price-dependent threshold, and its lead time is unspecified.
- **Shared authorship (inference).** Huanchen Zhang co-authored both this paper and Continuum, so reviewers may read this paper's critique of Continuum as a deliberate follow-on.

### Gaps
- **Controller settings.** The values of the keep threshold, the hysteresis band and the lead time are not reported, and no public code was found to check them.
- **The courtesy setting.** It is unclear whether the matched-environment "courtesy" applies to the 90% numbers as well as the midpoint ones.

---

## 4. Intent-Aware Caching (arXiv 2611.09715): what did the experiment with an age-conditioned arrival probability test, and what did it find?

### Takeaway
**The experiment.** Appendix B.4's "Full-Horizon" variant replaced IAC's binary recency test 1[age < M_{i,t}] with a continuous score as the primary key. The score is "the probability that the next turn arrives within the next H seconds given the current age of the entry, computed from the learned log-normal delay model".

**The result.** On ShareChat, Full-Horizon had a lower cache hit rate at all four capacities:

| Capacity (tokens) | Full-Horizon | IAC |
|---|---|---|
| 80K | 21.5% | 23.7% |
| 110K | 27.4% | 29.8% |
| 140K | 31.0% | 34.6% |
| 170K | 35.3% | 38.5% |

**The authors' two explanations:**
1. A near-unique continuous score dominates the lexicographic order, so the reuse-probability and size keys "rarely take effect".
2. Per-(intent, turn) full distributions are "noisy", while the median "is robust".

**What the score lacked.** It had no reload cost and no price. Size remained only a later tie-breaker, and H is not stated.

**Distributions and fit.** They are log-normal per (intent, turn), fitted as the mean and variance of log-delays over a recent-log window. No goodness-of-fit statistics are reported; log-normality is assumed by citing ServeGen (NSDI '26).

**Released code.** The released code ranks by a *continuous* 40-second-horizon score with the continuation probability folded in. That is not the paper's Eq. (1).

### Cited Findings

#### Record (verified)
- **Title and authors.** "Intent-Aware Caching for Efficient LLM Serving". The 6 authors are Kazi Hasan Ibn Arif, JinYi Yoon, Dimitrios S. Nikolopoulos, Hans Vandierendonck, Deepu John and Bo Ji — [arXiv abs](https://arxiv.org/abs/2611.09715)
- **Version and venue.** v1 (Fri 6 Nov 2026) is the only version ("[cs.AI]"). The Comments field reads "NeurIPS 2026 | Code: this https URL" — [arXiv abs](https://arxiv.org/abs/2611.09715), [v1 HTML](https://arxiv.org/html/2611.09715v1)
- **Headline result (abstract).** "IAC reaches a 34.6% cache hit rate. This substantially outperforms both the previous learning-based approach (18.8%) and traditional methods such as least recently used (LRU, 12.2%), while reducing average serving latency by up to 46.1%." — [v1 HTML](https://arxiv.org/html/2611.09715v1)

#### Main policy, verbatim (§4.2–4.3, Eq. (1))
- **Statistics (§4.2).** "P(t+1|i,t)=C(i,t+1)/C(i,t), where C(i,t) denotes the number of intent-i conversations among the N most recent conversations that reach turn-t." — [v1 HTML](https://arxiv.org/html/2611.09715v1)
- **Delays (§4.2).** "Inter-turn delays are typically long-tailed and follow a log-normal distribution [39]. Therefore, we learn `\log D_{i,t}\sim\mathcal{N}(\mu_{i,t},\sigma_{i,t}^{2})`, where μ_{i,t} and σ²_{i,t} are the mean and variance of the log-delay, respectively." — [v1 HTML](https://arxiv.org/html/2611.09715v1)
- **The oracle it mimics (§4.3).** The policy imitates "TB+S [14], which achieves the best oracle performance on our conversational workloads". The oracle's priority is `\left(\mathbf{1}[\Delta\text{next\_arrival}<\text{Threshold}],\text{size}\right)`, "where ... size is the number of tokens, and Δnext_arrival is the time until the next reuse (set to infinity when no future reuse occurs)." IAC's §2 cites [14] for LRB — [v1 HTML](https://arxiv.org/html/2611.09715v1)
- **The median window (§4.3).** "we summarize its central reuse timescale by the median delay obtained from the log-normal model as `M_{i,t}=\exp(\mu_{i,t})`. We use this median as the estimated reuse window: if a cached item is younger than M_{i,t}, it is considered to be still within its likely reuse period". — [v1 HTML](https://arxiv.org/html/2611.09715v1)
- **Eq. (1).**
  - `\mathrm{priority}(v)=\bigl(\underbrace{\mathbf{1}[a_{v}<M_{i_{v},t_{v}}]}_{\text{Recency signal}},\;\underbrace{P(t_{v}+1\mid i_{v},t_{v})}_{\text{Reuse signal}},\;s_{v}\bigr).`
  - "Let a_v and s_v be the age (the time since it was cached) and the size (the number of cached tokens) of node v".
  - "Nodes are compared lexicographically (i.e., recency first, followed by reuse, and finally size) ... and the node with the lowest priority is evicted first".
  - — [v1 HTML](https://arxiv.org/html/2611.09715v1)
- **Why the median (§5.4, Fig. 9).** "We compare three estimators of the reuse window M_{i,t} derived from the learned log-normal delay model: the mean e^{μ+σ²/2}, the median e^{μ}, and the mode e^{μ−σ²}. As shown in Fig. 9, the median provides the best performance." — [v1 HTML](https://arxiv.org/html/2611.09715v1)

#### The Full-Horizon experiment (Appendix B.4, Table 5): setup, metric, numbers, explanation
- **Setup, verbatim.** "We test a variant, called Full-Horizon, that replaces the binary recency test `\mathbf{1}[\mathrm{age}<M_{i,t}]` with a continuous score, i.e., the probability that the next turn arrives within the next H seconds given the current age of the entry, computed from the learned log-normal delay model. Full-Horizon uses this probability as the primary signal." — [v1 HTML](https://arxiv.org/html/2611.09715v1)
- **Metric and numbers.** Table 5 reports "Cache hit rate (%) on ShareChat with the binary median-based recency test (IAC) and with the full delay distribution (Full-Horizon)" (Full-Horizon vs IAC):
  - 80K tokens: 21.5 vs 23.7
  - 110K: 27.4 vs 29.8
  - 140K: 31.0 vs 34.6
  - 170K: 35.3 vs 38.5
  - — [v1 HTML](https://arxiv.org/html/2611.09715v1)
- **Explanation, verbatim.** "Full-Horizon stays below IAC at every capacity on ShareChat for two reasons. First, IAC keeps a binary recency threshold like the TB+S oracle it mimics, so recency, reuse probability, and size act in a clean order. A continuous recency score is almost unique per entry, so it dominates the ranking, and the reuse-probability and size signals rarely take effect. Second, Full-Horizon relies on the full delay distribution, including its spread and tail, which is noisy per (i,t) state, whereas the median is robust. This also matches Fig. 9, where the median outperforms the mean and the mode." — [v1 HTML](https://arxiv.org/html/2611.09715v1)
- **Consistency with the main results.** Table 5's IAC column matches the main ShareChat results: 34.6% at 140K, and "a 23.7% cache hit rate with only 24.4 GB of KV cache (80K tokens)" with Llama-3-70B (§5.1). B.4 itself names only "ShareChat" and does not state H — [v1 HTML](https://arxiv.org/html/2611.09715v1)
- **Main replay setting (A.4).**
  - "the first 40,000 conversations are used as the history/training split, and the next 10,000 conversations are used as the replay trace".
  - "we use the 40,000-conversation training split as the recent-log window for IAC".
  - Real inter-turn delays are kept, and new conversations arrive "using a Poisson process at the target arrival rate".
  - — [v1 HTML](https://arxiv.org/html/2611.09715v1)

#### Did the score account for size, reload cost or a memory price?
- **Size.** By the B.4 text, size stays a later key in the lexicographic order and "rarely take[s] effect". The text implies the continuation probability also stays a separate later key, rather than being folded into the score — [v1 HTML](https://arxiv.org/html/2611.09715v1)
- **Reload cost and price.** No reload cost or price appears anywhere in IAC's policy; the objective is cache hit rate. Prefill cost appears only in the background ("time complexity O(nL_t²d)"). The main body contains no "price", "prefetch", "hazard", "SSD" or "Lagrang" (term scan) — [v1 HTML](https://arxiv.org/html/2611.09715v1)
- **Single tier.** IAC runs on SGLang v0.5.3's RadixTree. "During eviction, SGLang considers leaf nodes as eviction candidates", and "node.age = system.clock − node.timestamp" (A.2). Capacities are in tokens of GPU KV ("140K token capacity (43 GB for Llama-3-70B)") — [v1 HTML](https://arxiv.org/html/2611.09715v1)

#### Distributions and fit
- **Form.** Log-normal per state (intent i, turn t), from the mean and variance of log-delays (§4.2, above). Intents come from a 30-category classifier. The chosen embedder, E5-Large-Inst, reaches "75%" accuracy at "20 ms" (Table 1) — [v1 HTML](https://arxiv.org/html/2611.09715v1)
- **Fit quality is not reported.**
  - No goodness-of-fit statistic appears.
  - Log-normality is asserted by citing [39], "ServeGen: Workload characterization and generation of large language model serving in production" (NSDI '26).
  - Support is qualitative: Fig. 2(b), "The delay CDF, clipped at 600 s, motivates using (intent, turn) as the workload category", and the B.1 PDFs (Fig. 10).
  - The authors' own B.4 explanation calls the per-state spread and tail "noisy".
  - — [v1 HTML](https://arxiv.org/html/2611.09715v1)
- **Right-censoring.** "If a request does not receive a follow-up within 30 minutes, we treat the corresponding state as completed." (A.2) — [v1 HTML](https://arxiv.org/html/2611.09715v1)
- **Defaults.** "If a state (i,t) is unobserved in the recent-log window, or if C(i,t)=0, ... we set P(t+1|i,t)=0.5 and use a default median reuse window of 150 s." (A.2) — [v1 HTML](https://arxiv.org/html/2611.09715v1)
- **Online use (B.3).**
  - A rolling evaluation on WildChat with N = 10K: "IAC keeps a clear advantage over LRU in every fold (+49% to +83%)".
  - "Each update is a single pass that counts continuations and fits the log-normal delay parameters for every (i,t) state, with no model training. It takes about 1.4 s on CPU".
  - — [v1 HTML](https://arxiv.org/html/2611.09715v1)
- **Synthetic workloads (§5.1).** For ShareGPT and LMSys-Chat, "inter-turn delays are sampled from a fixed long-tailed distribution with mean 4.6 and variance 1, following prior workload studies [39]" — [v1 HTML](https://arxiv.org/html/2611.09715v1)

#### Released code (default branch HEAD, commit 085e226; files dated 13 Nov 2026, after arXiv v1)
- **The README describes Eq. (1).** It gives the priority as "(paper Eq. 1)": "IAC Priority(i, t, age) = ( 1[age < M_{i,t}],  P(t+1 | i, t),  size )" — [README](https://github.com/hasanar1f/intent-aware-caching/blob/085e226ef8b52244ed607f86849e6ecfa10648e9/README.md)
- **The code ranks by something else.** In `iac/policy.py`:
  - `DEFAULT_HORIZON_S = 40.0`.
  - `reuse_within(p_next, mu, sigma, age, horizon)` returns `p_next * (s_now - s_later) / ((1.0 - p_next) + p_next * s_now)`, where `s_now` and `s_later` are the log-normal survival at `age` and at `age + horizon`.
  - `IACPolicy.priority` returns `(reuse_within(p_next, mu, sigma, age, self.horizon), size)`.
  - — [policy.py](https://github.com/hasanar1f/intent-aware-caching/blob/085e226ef8b52244ed607f86849e6ecfa10648e9/iac/policy.py)
- **That code is what serving calls.** `iac/serving.py` calls `self.iac.priority(tag.intent, tag.turn, self.now - node.iac_time, len(node.key))`. The median `exp(mean log delay)` appears only in a diagnostic snapshot (`"M": median`). I found no `1[age < M]` test in the Python code — [serving.py](https://github.com/hasanar1f/intent-aware-caching/blob/085e226ef8b52244ed607f86849e6ecfa10648e9/iac/serving.py)
- **Statistics code.** In `iac/stats.py`:
  - `DEFAULT_P_NEXT = 0.5`, `DEFAULT_REUSE_WINDOW_S = 150.0`, `DEFAULT_SIGMA = 1.3`.
  - μ and σ are the mean and population standard deviation of log-delays.
  - Shrinkage toward turn-level statistics uses p = (n_next + k·p_b)/(n + k), with `DEFAULT_SHRINKAGE = 20.0`.
  - `DEFAULT_WINDOW = 50_000` and `DEFAULT_UPDATE_INTERVAL_S = 60.0` are set in `policy.py`; `DEFAULT_SESSION_TIMEOUT_S = 1800.0` in `data.py`.
  - — [stats.py](https://github.com/hasanar1f/intent-aware-caching/blob/085e226ef8b52244ed607f86849e6ecfa10648e9/iac/stats.py), [policy.py](https://github.com/hasanar1f/intent-aware-caching/blob/085e226ef8b52244ed607f86849e6ecfa10648e9/iac/policy.py)

#### Corrections to the earlier notes
- **H, not X (Note B line 55).** Note B writes "within the next X seconds". The paper writes "H seconds" and does not give H's value; the code's default is 40 s — [v1 HTML](https://arxiv.org/html/2611.09715v1), [policy.py](https://github.com/hasanar1f/intent-aware-caching/blob/085e226ef8b52244ed607f86849e6ecfa10648e9/iac/policy.py)
- **Paper versus code (Note B line 54).** Note B's tuple is the paper's Eq. (1), correctly quoted. But the released code's serving path ranks by a different, continuous score — [serving.py](https://github.com/hasanar1f/intent-aware-caching/blob/085e226ef8b52244ed607f86849e6ecfa10648e9/iac/serving.py)
- **Scope of the negative result (Note B lines 11 and 58).** They say a "full conditional-probability score did worse than a median threshold", which is correct. The scope must be added:
  - The data were ShareChat chat traces.
  - The metric was hit rate.
  - The score was a single fixed H, placed lexicographically ahead of the continuation probability and size.
  - The score included no size, reload-cost or price weighting.
  - — [v1 HTML](https://arxiv.org/html/2611.09715v1)

#### What IAC does and does not contain, relative to (a)–(d)
- **(a) Forecasts.**
  - *Contains:* per-category (intent, turn) log-normal gap distributions, learned from recent logs by the engine-side system. Age enters as a median threshold in the paper, or as a within-H conditional probability in B.4 and the code.
  - *Does not contain:* tool keys, framework-sent distributions, or agentic tool gaps (the traces are chat).
  - — [v1 HTML](https://arxiv.org/html/2611.09715v1)
- **(b) Fetch.** *Does not contain* any fetch, prefetch or lower tier — [v1 HTML](https://arxiv.org/html/2611.09715v1)
- **(c) Retain.**
  - *Contains:* lexicographic eviction mimicking a threshold-plus-size oracle.
  - *Does not contain:* hazard × read time, an index normalized by holding time, or any size-price factor beyond tie-breaking.
  - — [v1 HTML](https://arxiv.org/html/2611.09715v1)
- **(d) Price.** *Does not contain* a price — [v1 HTML](https://arxiv.org/html/2611.09715v1)

### Inferences
- **The released score is our F(H|t) (inference, an algebraic identity).** The code's `reuse_within` is exactly P(arrival in (a, a+H] | no arrival by a) for a "defective" log-normal, one that returns with probability p and otherwise never. That is our F(H|t), with H fixed at 40 s, no max over H, no normalization by expected holding time, no size-price factor, and size as a tie-breaker.
- **B.4 does not test our index (inference).** Its score omits the size-price weighting and the holding-time normalization in (1+λL)·max_H F(H|t)/τ(H), uses one fixed H, and is placed ahead of the other keys lexicographically. Its objective is hit rate on chat traces with minutes-scale gaps, in a single GPU tier.
  - The authors' reason 1 is an artifact of lexicographic composition, which a scalar index avoids.
  - Their reason 2, noisy tails per key, is a real risk for us. Our pooled fallbacks, and possibly shrinkage like the IAC code's, are the answer. Our paper should show calibration of F(x|t) and a comparison against a median-threshold rule.
- **Reviewer risk (inference).** A reviewer may cite B.4 against conditional-probability scores. The precise reply is that B.4's score lacks size, read-cost and price weighting and holding-time normalization. The authors' own released code uses a continuous conditional score, though I cannot tell whether that code produced the paper's tables.

### Gaps
- **Which run produced Table 5.** I cannot tell which code version produced Table 5 or the main results, or which H was used in B.4.
- **Model and hardware.** These are not stated for Table 5, though its numbers match the main ShareChat results.
- **Code history.** I did not determine whether an earlier commit of the repository implemented Eq. (1).

---

## 5. Claim by claim: what the four papers do and do not contain, and how to scope the novelty statement

### Takeaway
**(a) Forecasts.** None of the four can be cited for *framework-sent* per-tool gap distributions, but most of (a)'s other ingredients are taken:
- per-tool empirical distributions learned by the engine (Continuum, CacheWise);
- the first word of the bash command as the key (Continuum);
- argument-level refinement (CacheWise);
- conditioning on elapsed time (CacheWise; IAC's B.4 variant and code);
- per-category log-normal gaps (IAC).

**(b) Fetch.** None times an SSD read, and none uses a probability threshold that depends on size and price to start any read. The only prefetch is Ask the Tool's point-estimate "one lead time before the predicted return", from DRAM to GPU.

**(c) Retain.** None uses a hazard rate or a Gittins-type index:
- CacheWise approximates Belady by the conditional mean of remaining time.
- IAC mimics a threshold-plus-size oracle.
- Continuum's TTL rule solves p(τ)·B = 1, not the hazard condition (inference).

**(d) Price.** None learns a price, uses a capacity multiplier, or gives a bound. Continuum's Eq. (1) contains a fixed latency charge for memory-time that cancels in the rule it solves.

**Counter-evidence to answer directly:**
- Ask the Tool's end-of-call evidence against history-based, elapsed-conditioned estimates.
- IAC's B.4 result against an age-conditioned arrival probability.

### Cited Findings

| Element of our claims | Continuum v7 | CacheWise v1 | Ask the Tool v1 | IAC v1 |
|---|---|---|---|---|
| Per-key duration or gap distribution | Yes: per-tool empirical CDF, online, engine-side (§4.2, §5.1) | Yes: per-tool and per-argument-cluster empirical, offline (§5.2, §6.1) | No in its method; compared against history predictors (§4.1) | Yes: log-normal per (intent, turn), recent-log window (§4.2) |
| Key = tool + first command word | Yes, bash on SWE-Bench (§5.1); code `words[0]` | No: TF-IDF + KMeans argument clusters (§5.2) | n/a | n/a |
| Distributions sent by the agent framework | No: client sends program_id only (§5) | No: engine parses tool_name/tool_args (§5.3) | Harness sends progress events, not distributions (§3) | No: engine-side intent classifier (§4.1) |
| Conditioned on elapsed idle time | No: TTL fixed at call start (Alg. 1, §5.2) | Yes: E[τ_i(t) given τ_i(t) − T_i > t − T_i], rebuilt every 3 iterations (§5.2–5.3) | Only its 3Sigma-style baseline (§4.1) | Median-age test (Eq. 1); within-H probability (B.4, code) |
| Read from lower tier before arrival | No (§5.2) | No: on-demand swap-in (§6.4) | Yes: refresh one lead time before predicted return, point estimate (§5.1, App. D) | No |
| SSD | Evaluation only, via LMCache (§6.3) | No | No | No |
| Fetch or keep threshold depends on size and price | No | No | No | No |
| Retention or eviction rule | TTL = argmax P(τ,f)·B − τ (Eq. 2) | evict largest predicted remaining time, Belady approximation (§5.2) | keep while remaining < threshold, hysteresis, LRU underneath (§5.1) | lexicographic (recency test, continuation probability, size) (Eq. 1) |
| Hazard rate or Gittins-type index | No | No | No | No |
| Memory priced | Eq. (1): fixed MemUsage/M latency charge; cancels in Eq. (2) | No: hard capacity M (§4) | No: trust ledgers only (§5.3) | No |
| Price learned online, dual ascent, lower bound | No | No | No | No |

Sources for every cell: [Continuum v7 HTML](https://arxiv.org/html/2511.02230v7); [CacheWise v1 HTML](https://arxiv.org/html/2606.16824v1); [Ask the Tool v1 HTML](https://arxiv.org/html/2609.18849v1); [IAC v1 HTML](https://arxiv.org/html/2611.09715v1). The code checks are cited in Sections 1, 2 and 4. Each cell is backed by the verbatim quotes in Sections 1–4.

- **Term scan of the four main bodies.**
  - None contains "newsvendor", "critical ratio", "hazard", "Gittins", "Lagrang" or "shadow" price.
  - "Belady" appears only in CacheWise and IAC.
  - "prefetch" appears only in Ask the Tool.
  - "SSD" appears in Continuum (evaluation only) and CacheWise (background).
  - Hits for "dual" are substrings of "individual" or of "greedy dual size-frequency".
  - — [Continuum v7](https://arxiv.org/html/2511.02230v7), [CacheWise v1](https://arxiv.org/html/2606.16824v1), [Ask the Tool v1](https://arxiv.org/html/2609.18849v1), [IAC v1](https://arxiv.org/html/2611.09715v1)
- **Dates relevant to priority.**
  - Continuum's per-tool empirical CDF and first-word key: since v1, 4 Nov 2025 (with a confidence-bound TTL at that point).
  - Continuum's cost-benefit TTL: by v4, 4 May 2026.
  - CacheWise: 15 Jun 2026.
  - Ask the Tool: 16 Sep 2026.
  - IAC: 6 Nov 2026 (NeurIPS 2026 per its arXiv comment).
  - — [Continuum abs](https://arxiv.org/abs/2511.02230), [CacheWise abs](https://arxiv.org/abs/2606.16824), [Ask the Tool abs](https://arxiv.org/abs/2609.18849), [IAC abs](https://arxiv.org/abs/2611.09715)

### Inferences
- **(a), wording to use (inference).** "The agent framework, which knows the tool and its command, supplies per-tool gap distributions. The engine conditions them on elapsed idle time and uses the full conditional distribution, not a point summary, for fetch, retention and eviction across GPU, host and SSD."
  - *Credit:* Continuum (per-tool empirical CDF, first-word key, pooled fallback), CacheWise (elapsed-time-conditioned per-tool and per-argument-cluster distributions), and IAC (per-category log-normal gaps).
  - *Do not claim as new:* "per-tool distributional forecasts", "first-word keys" or "conditioning on elapsed time".
- **(b), wording to use (inference).** "Starting a lower-tier read when the conditional probability of arrival before the read completes reaches a newsvendor critical ratio q = λL/(1+λL)."
  - No counterpart exists in the four papers.
  - Nearest are Ask the Tool's point-estimate lead time (DRAM to GPU, value unspecified) and Continuum's expected-benefit-minus-holding-cost argmax, which sets a retention horizon, not a read start.
  - The size and price dependence of q is the part with no precedent.
- **(c), wording to use (inference).** "Retention by hazard × read time against q, and a Gittins-type eviction index that reduces to Belady and LRU."
  - Belady as the target is not new: CacheWise says so explicitly, and IAC mimics TB+S. The derivation and the index are new.
  - The paper should state the contrast to Continuum: charging the expected occupancy turns its p(τ)·B = 1 into h(τ)·B = 1.
  - It must answer IAC B.4 by noting that B.4's score had no size, read-cost or price weighting and no holding-time normalization, and was used lexicographically.
- **(d), wording to use (inference).** "A capacity-coupled shadow price on host memory, learned online by multiplicative dual ascent, with a weak-duality lower bound."
  - None of the four has this.
  - Continuum's fixed MemUsage/M conversion is the nearest thing in these four papers, and it cancels out of their decision rule.
- **Ask the Tool's evidence (inference).** Do not claim that history-based forecasts predict long, environment-bound calls. Show instead three things:
  - the forecast-driven decisions degrade safely: fetches decline when F(read time | t) is low, and eviction falls back toward LRU;
  - the per-tool distributions are calibrated on the short-dominated bulk of calls;
  - optionally, progress can enter as one more conditioning variable.
- **Baselines (inference).** The public code of two papers differs from their text: Continuum runs a fixed 2 s TTL gated on the per-tool mean, and IAC runs a continuous 40 s-horizon score. If either serves as a baseline, state which variant (paper or code) was run.

### Gaps
- **Not re-checked here.** Venues beyond the arXiv records were not re-checked, including Continuum's ICLR 2026 workshop listing cited by Notes A and B.
- **Out of scope.** Papers beyond these four were not examined. Notes A and B cover TokenCake, Adaptive KV Retention, KVLearn, the Alibaba trace study and others.
- **Code used for experiments.** For Continuum and IAC, I could not establish which code produced the published experiments. For CacheWise, the vLLM integration was not in the linked repository.
