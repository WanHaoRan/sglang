# Paper intro draft

Started 2026-11-06 from a discussion of how to open the paper. This is a draft to edit, not settled text. The problem is stated
without SGLang: SGLang's HiCache appears only as evidence and as the engine the system is built on. The last two sentences of
paragraph 5 and contribution 3 describe the expected-cost policy, which is not built (`PAPER_PLAN.md`, "Design point"). Paragraphs 4 and 5 need
the prior-work changes listed under "Closest prior work" before they can be used.

## Draft opening

Brackets hold the scope of a number, or something still to fill in.

An LLM agent works in a loop. The model replies with a tool call, the harness runs the tool, and the next request resends the
whole session plus the tool's output. While a test suite runs or a package installs, the serving engine holds that session's
history, gigabytes of KV state, and must decide where it waits: GPU memory, host memory or SSD. The right choice depends on one
fact the engine never sees: when the session will come back.

Multi-tier KV caches treat that history as a cache. Recency decides what stays in fast memory, and a history moved to SSD is
read back when its request arrives. Agents break both rules. Idle time says little about return: a session that has waited
minutes on a test run is certain to come back, while one that just finished its task looks recent and never will. And reading
at arrival puts the slowest tier on the critical path. In our setup a returning turn reads about 1.6 GB [x70], so an SSD that
sustains about 2 GB/s for reads and writes together brings back a little over one turn per second, however fast the GPU is.

Figure 1 shows a production three-tier cache, SGLang's HiCache, under sustained agent load. For over an hour the SSD runs at its
limit and time to first token holds near 20 s. Then, within minutes, TTFT climbs past 120 s and stays there for 40 minutes,
while SSD reads fall to about a third. Reads waiting in the queue have filled the host memory set aside for them, new lookups
are refused, histories that already arrived are evicted before their turn runs, and every lost history is recomputed on the
GPU. Recomputation is the KV cache's escape hatch, and under load it turns a shortage of SSD bandwidth into a collapse of GPU
throughput.

The obvious remedy is to tell the engine when each session will return, and the agent can: the tool call in the reply picks
out the long gaps well [AUC 0.84 for gaps ≥ 300 s, x70]. But knowing roughly when doesn't tell the engine what to do. Once
landed reads are protected, exact return times cut mean TTFT a further 2.9x [1.47 → 0.50 s, 44 users, 70 GB], yet the same
routine fed a realistic forecast gains nothing [1.51 s]. The forecast is wide and often has two modes: for `pip` commands the
median gap is 26 times the 10th percentile. A read timed to a point estimate lands early and holds host memory other sessions
need: 35 % of such reads were revoked for room, against 18 % with exact times.

So the problem is not predicting the next request. It is acting on a prediction that is usually off by seconds and sometimes by
minutes, when arriving early costs memory every other session needs and arriving late costs a stall or a recomputation.
Classic informed prefetching assumes the reverse: hints name the blocks to be read, and the reader's progress sets the time
[TIP; Cao et al.]. Here what will be read is certain, the session's own history, and only when is uncertain. The two halves of
the decision also sit on opposite sides of the API: the agent knows what it is waiting for, and only the engine sees free
memory, the read queue and the GPU's backlog. [System] splits the decision along that line. The agent sends a distribution over
the next gap, keyed by the tool call. The engine puts a price on host memory and decides by expected cost when to read each
history, what to protect and what to evict.

## Contributions

Draft text for the end of the introduction, then the evidence and positioning behind each point. The thesis in one line: the
engine knows what an agent will read next, the session's own history, but only the agent knows roughly when.

This paper makes three contributions.

- **How agents break a three-tier KV cache.** Between tool calls a session's history leaves fast memory, so a returning turn
  reads gigabytes from SSD, and an SSD that sustains 2 GB/s serves only about one such turn per second [x70]. We show that a
  production three-tier cache does not slow down gracefully at that limit. It tips into a recompute storm: reads waiting for the
  SSD hold host memory, histories that have landed are evicted before their turn runs, and recomputing them moves the bottleneck
  from the SSD to the GPU. Time to first token rises from about 20 s to 126-150 s while SSD reads fall to a third [x70], and a
  second setting reaches the same state through a different trigger [x10].
- **[System], which moves KV by the agent's clock.** The agent sends, with each request, how long each kind of tool call
  usually takes; once the reply names a tool call, the engine knows roughly when the session will return. [System] ranks every
  session in GPU and host memory by its forecast next use, starts each SSD read so it lands just before the turn arrives, and
  holds what it read until the turn runs. With exact forecasts it lowers mean time to first token 1.4-10x against SGLang's
  HiCache across our settings [5.17 → 0.50 s at 44 users, 70 GB; 53.12 → 12.06-12.46 s at 128 users, 160 GB]. Timing each read
  to the forecast, rather than issuing reads in deadline order, reads each history once and cuts SSD traffic 29 % [128 users].
- **Acting on forecasts that are wrong.** A forecast's errors are paid in shared host memory. Where memory is short, a
  realistic forecast acted on as exact gains nothing over no forecast, while exact timing is worth 2.9x [44 users, 70 GB];
  where memory is plentiful, the same error costs 0.07 s [128 users, 160 GB]. Only the engine sees which case it is in, so in
  [System] the agent sends a distribution and the engine decides by expected cost, with a price on host memory that it learns
  from how often reads are revoked for room.

### Behind each contribution

**1. How agents break a three-tier KV cache.**
- **Numbers.** Campaign 14 (x70, original HiCache): the SSD ran at its 2.1 GB/s limit, reads plus writes, for over an hour with
  TTFT at 15-21 s. Then TTFT rose to 126-150 s for 40 minutes, recomputation went from under 400 to 9,000-15,000 tokens per
  turn, turns served fell about 70 %, and SSD reads dropped to about a third. At x10 (campaign 19, analysed 2026-11-05) the
  trigger was spent retry budgets, with the SSD far from saturated (reads 0.53, writes 0.64 GB/s). At 128 users the baseline
  recomputes 3,784 tokens per turn while its SSD is 39-48 % busy (`CAMPAIGN23_FINDINGS.md`).
- **Why it stands out.** It is counterintuitive and measured in a production engine, and none of the six papers studies how a
  three-tier cache fails under overload. CachedAttention and Bidaw protect queued requests' KV, which breaks one link of the
  loop; the baseline adopts that protection (read pins) and the paper shows what is left.

**2. [System], which moves KV by the agent's clock.**
- **Numbers.** The x70 ablation (campaign 16, 20,217 turns, `ORACLE_FINDINGS.md`): baseline 18.5 s, read pins alone 13.2 s,
  plus deadline-ordered reads 11.6 s, plus eviction by next use 5.7 s, with host hits rising from 14 % to 41 % of returning
  turns. At 128 users, reads timed to land just before the turn used 29 % fewer SSD pages than deadline-ordered reads (1.07M
  against 1.52M), one read per reading turn, and left the SSD 47-78 % busy instead of 97-98 %. Realistic forecasts keep nearly
  all of the 128-user gain (12.14-12.56 s) but add nothing beyond read pins at 44 users, which is contribution 3.
- **Why it stands out.** No prior system reads from SSD before the request arrives: CachedAttention and Bidaw read once the
  request is queued, KVFlow and Tokencake move KV ahead of time only from host memory to GPU, and Continuum and InferCept reload
  on demand. Eviction by next use exists in other forms (KVFlow counts workflow steps, Bidaw bytes of other traffic, Continuum
  has none); this one ranks every session by forecast time across GPU and host memory and needs no workflow graph.

**3. Acting on forecasts that are wrong.**
- **The policy** (planned, `PAPER_PLAN.md`, "Design point"): land each read at quantile q = λ·tokens / (λ·tokens + 1); take
  room back from reads that landed early for sessions that have not returned (soft pins); evict by the chance of return within a
  horizon given the idle time so far; learn λ from revocations. With exact hints it reduces to the just-in-time routine, and
  with no information to the engine without hints.
- **Numbers.** The forecast error is the same at both settings (actual minus predicted arrival: p10 -5.5 s, median -0.3 s, p90
  +4.9 s), but it costs 1.01 s at 44 users and 0.07 s at 128 users. Forecasts often have two modes (for `pip` the median is 26
  times the 10th percentile), and 35 % of reads were revoked for room against 18 % with exact times.
- **Why it stands out.** It inverts classic informed prefetching, where hints name the blocks and the reader's progress sets the
  time: here what will be read is certain and only when is uncertain. Continuum prices memory against a per-tool distribution,
  but only to decide how long to keep KV in GPU; Bidaw's probabilistic eviction counts bytes, not time. Expected-cost decisions
  about when to read from a slow tier are new, and so is the measurement that reads timed to a single predicted time,
  Tokencake's rule, gain nothing when memory is short.
- **Status: offline only.** At equal stall the rule holds 14-49 % fewer token-seconds of host memory than any fixed quantile,
  and its eviction ranks returns with AUC 0.900 against 0.825 for point forecasts and 0.763 for LRU at x70. Until the
  read-timing rule is built and run against `jit_p50`, `jit_exact` and the hint-free arm, and its eviction is compared with
  Bidaw's, this contribution is a promise.

### How they fit

Contribution 1 is the problem, 2 is the system and how far it can go, and 3 is the twist that makes this research rather than
engineering.

### What reviewers will check first

- **The strongest baseline without hints:** read pins plus queue-aware fetch and eviction (CachedAttention-style, option I in
  `CAMPAIGN23_FINDINGS.md`), plus Bidaw's read order. At 44 users, pins alone already give 79 % of the gain.
- **Repeat runs:** original HiCache varies about 25 % from run to run near its knee, and most arms have one run.
- **Realistic forecasts before contribution 3 exists:** the claim rests on the 128-user result (12.14-12.56 s against 29.19 s
  without hints), which overstates the hint's value because that arm also lacks the queue-aware mechanisms.
- **The workload:** gaps scaled 10x or 70x, and no workload paced like a human user.

## What each paragraph does

1. The workload change: agents turn requests into a loop whose gaps are set by tools.
2. The assumptions it breaks ("recency predicts reuse", "fetch on demand"), and one piece of arithmetic.
3. One vivid failure from a real system (Figure 1), explained in general terms.
4. The obvious fix and why it is not enough. This is what makes the problem unsolved rather than merely unaddressed.
5. The problem as a question, the contrast with classic prefetching, and the approach.
6. The contributions: the problem (1), the system and how far it can go (2), and the twist that makes this research (3).

## Where the numbers come from

| Draft text | Source | Scope |
|---|---|---|
| A returning turn reads about 1.6 GB | Campaign 14 original HiCache, `results/20261008_204128_three_tier_wc_NAT160`: turns with an SSD read landing in minutes 50-125, mean 1.58 GB, median 1.54 GB (5,484 turns) | x70; engine and client predate the current code |
| About 2 GB/s for reads and writes together | Same run, iostat: reads plus writes 2.10-2.11 GB/s in every 10-minute bin from minute 50 to 129, matching the 2.11 GB/s shared cap measured on 2026-10-13 | This VM's cloud volume |
| Figure 1: over an hour at the SSD's limit with TTFT near 20 s, then past 120 s for 40 minutes, reads down to about a third | Same run, 10-minute bins: reads 1.90-1.96 GB/s and TTFT 15-21 s in minutes 50-129; TTFT 126-150 s in minutes 130-169; reads 0.61-0.80 GB/s after minute 130 | x70; the old client's backlog cap; "lookups are refused" is inferred from the code, since that log has no refusal event |
| AUC 0.84 for gaps ≥ 300 s | `DIST_HINT_DESIGN.md`: tool plus the first word of its command (0.66 for the tool alone); the online table scores 0.837 on campaign 18's send times | x70 |
| 1.47 → 0.50 s (2.9x), and 1.51 s | `PAPER_PLAN.md`, `CAMPAIGN23_FINDINGS.md`: hint-free arm with read pins, `jit_exact_c44`, `jit_p50_c44`, on 7,768 shared turns | 44 users, 70 GB, 2 h; one run each, near-repeats within 0.01-0.12 s in campaign 25 at 5x faster arrivals |
| `pip`: median 26 times the 10th percentile | `PAPER_PLAN.md`: p10 0.85 s, p50 22 s | Online table at x10 |
| 35 % vs 18 % of reads revoked for room | `PAPER_PLAN.md`: 1,743 of 5,001 read launches against 381 of 2,088 | 44 users, 70 GB, just-in-time routine |

## Before using it

- **Figure 1's run is old.** Campaign 14's engine predates the current code and its client still had the backlog cap. Remake
  Figure 1 from a current-code run of original HiCache, where refusals are logged.
- **"Idle time says little about return" needs a scope.** It holds at x70, where LRU ranks returns with AUC 0.763, but not at
  x10, where LRU reaches 0.922 (`PAPER_PLAN.md`, rule 3). Cite x70 or soften the sentence.
- **Paragraphs 4 and 5 must cite Continuum and Tokencake** (see below). Both already tell the engine when a paused agent
  returns, and Continuum already weighs a per-tool duration distribution against a price on memory.
- **The end of paragraph 5 is a promise.** Until the expected-cost policy is built and run, the closing paragraph has to
  describe the just-in-time routine plus pins, and the headline is the 2.9x headroom that realistic forecasts do not capture yet.
- **Quote replay numbers as ratios** (26x, AUC) or state the gap scale, since gap seconds come from a replay stretched 10x or 70x.
- **Optional sentence for paragraph 2**, against "buy a faster SSD": at our 96 KiB per token, the median 126K-token context of
  human-paced use [TraceLab] is about 12 GB of KV, and even a 7 GB/s NVMe drive needs almost 2 s to bring one back.

## Closest prior work (checked 2026-11-06)

Read from the papers themselves (arXiv HTML, and the InferCept and Bidaw PDFs).

| Work | Tiers | What it decides | Signal for "when" | Forecast error | Moves data before the request arrives |
|---|---|---|---|---|---|
| InferCept (ICML 2024) | GPU, CPU | Per paused request: keep in GPU, drop and recompute, or swap to CPU, by least wasted GPU memory | Interception time profiled per augmentation type, or the time since the call started | The elapsed-time estimate reaches 93 % of an exact oracle | No: swapped context returns when the call ends |
| CachedAttention (USENIX ATC '24) | GPU, DRAM, SSD | SSD-to-DRAM prefetch and eviction, layer-wise loading, asynchronous saving | Look-ahead over the job queue, so only requests that have already arrived | Not applicable | No: only once the request is queued |
| KVFlow (NeurIPS 2025) | GPU, CPU (built on SGLang 0.4.4, compared with HiCache) | Eviction of radix-tree nodes by steps-to-execution; CPU-to-GPU prefetch | A step graph supplied with the workflow: steps, not seconds | Conservative prefetch on branches; misprediction not studied | Yes, the fixed prompts of agents due next step |
| Continuum (arXiv 2511.02230; ICLR 2026 workshop poster) | GPU; LMCache CPU and SSD offload in the evaluation | How long to pin a paused agent's KV in GPU (a TTL); program-level FCFS. No global order by next use: under memory pressure it unpins the latest-arriving program | Empirical CDF of each tool's past durations; the engine parses the tool from its own output (first word of the bash command for SWE-bench) | TTL maximizes expected benefit minus memory share × time; set once when the call starts | No: offloaded KV reloads on demand |
| Tokencake (arXiv 2510.18586) | GPU, CPU | Offload at call start when worthwhile; upload before the predicted end; GPU memory reserved for critical agents | Moving average blending a developer hint with history; cold-start estimates and agent priorities come from a static DAG of the application, declared through its API | Point estimate; an early finish triggers an immediate upload; tool latencies in the evaluation are simulated | Yes, CPU to GPU near the predicted end |
| Bidaw (FAST '26) | DRAM and SSD; GPU holds KV only while computing | Requests wait in a preparing queue until their KV is in DRAM, with SSD reads ordered by waiting time over KV size; DRAM eviction by lowest hit potential; queued requests' KV never evicted; which tensor to cache | Length of the previous answer (a lower bound on the next reuse distance), plus each user's history of reuse distances | A per-user distribution over reuse distance (bytes, not seconds), pruned by that lower bound and weighted by the hit rate of a Belady ghost cache | No: SSD reads start when the request arrives |
| This work | GPU, DRAM, SSD (about 2.1 GB/s shared by reads and writes) | Eviction by forecast next use; when to start each SSD read; protection of landed reads; planned: a price on host memory | Client-sent table per tool category, resolved from the engine's own reply | Today the p50 is planned on as if exact, with an early start of p50 - p10; planned: a distribution and a learned price, re-decided every tick | Yes, SSD to host memory, timed to the forecast arrival |

**No longer new, so cite it:**
- Eviction by forecast next use: KVFlow (steps-to-execution), CachedAttention (queue look-ahead), Bidaw (predicted access).
- Tool-keyed duration distributions, the first-word-of-command key, the engine reading the tool from its own output, and an
  expected-value rule with a memory price: Continuum, for GPU retention.
- Moving KV back ahead of a predicted return: Tokencake, host memory to GPU, from a point estimate.
- Protecting the KV of queued requests from eviction, which is what read pins do in the hint-free arm: CachedAttention and
  Bidaw.
- Predicting the next access from the reply: Bidaw, from the answer's length.
- Probabilistic next-use eviction: Bidaw's hit potential. Planned rule 3 has to be compared with it, not only with LRU and
  point forecasts.

**Still new:**
- Timing SSD reads before arrival on a saturated device whose bandwidth reads and writes share. The others bring data back on
  demand (InferCept, Continuum), only for requests already queued (CachedAttention, Bidaw), or only from host memory (KVFlow,
  Tokencake).
- What overload does in a production three-tier engine: host memory held by queued reads, landed reads lost before admission,
  and recomputation moving the bottleneck to the GPU. Bidaw orders SSD reads under contention, but none of the six studies
  this collapse. Its fix for the losses, protecting queued requests' KV, is prior art, so read pins belong in the baseline as
  adopted practice, not among the contributions.
- The measurement that point-timed staging, Tokencake's class of rule, gains nothing where host memory is short (1.51 against
  1.47 s at 44 users, 70 GB), while exact timing gains 2.9x.
- Planned: Continuum's expected-value rule extended from GPU retention to read timing and host-memory holds, re-decided every
  tick as the session stays idle, with the price learned from revocations.

**Open issues this raises:**
- **The case for the client sending the distribution.** Continuum learns per-tool distributions inside the engine from request
  gaps, with only a program id from the client, and Bidaw learns from its own answers. The paper needs a reason the engine cannot do the same: replicas behind a
  router each see only part of an agent system's sessions; the client times the tool itself, without queueing or network; the
  table outlives engine restarts; the client knows waits the engine never sees (human turns, harness timeouts). None is measured.
- **The hint-free rival has a citation.** CachedAttention's scheduler-aware fetching and eviction is close to the hint-free arm
  with queue-state mechanisms (option I in `CAMPAIGN23_FINDINGS.md`). Build that arm and present it as the CachedAttention-style
  baseline.
- **Paragraph 2 needs a scope.** "Multi-tier KV caches ... recency decides" holds for production caches such as HiCache, but
  CachedAttention and Bidaw already evict by more than recency. Name the systems that use LRU, after checking each.
- **Paragraph 3's mechanism is partly prior art.** Landed reads evicted before admission is what CachedAttention's and Bidaw's
  protection of queued requests prevents. Keep Figure 1 as the failure of a production engine, but make the problem the
  headroom left once that protection is in place.
- **Recheck `lit.md` section 8**, which is not on this machine, against these six.

**Suggested wording for paragraph 4**, after its first sentence: "Recent systems do this for GPU memory. Continuum keeps a paused
agent's KV for a time-to-live chosen from the tool's past durations [Continuum], and Tokencake moves it back from host memory
shortly before the call's predicted end [Tokencake]. Neither has to time a read from a tier that takes seconds per history and is
already saturated, or account for the host memory that early reads occupy." Then continue with "With an SSD tier, knowing
roughly when doesn't tell the engine what to do."

**Suggested wording for the end of paragraph 5:** "[System] splits the decision along that line. The agent sends a distribution
over the next gap, keyed by the tool call. Like Continuum, the engine weighs that distribution against a price on memory; unlike
it, the decisions it prices are when to start each SSD read before the turn returns and what to hold in host memory meanwhile.
It re-makes them every tick as the session stays idle, and learns the price from how often reads are revoked for room."

**Links:** [InferCept](https://arxiv.org/abs/2402.01869) ·
[CachedAttention](https://www.usenix.org/conference/atc24/presentation/gao-bin-cost) ·
[KVFlow](https://arxiv.org/abs/2507.07400) · [Continuum](https://arxiv.org/abs/2511.02230) ·
[Tokencake](https://arxiv.org/abs/2510.18586) · [Bidaw](https://www.usenix.org/conference/fast26/presentation/hu-shipeng)
