# Agent cache design: discussion of 2026-11-23

This note records the design discussion of 2026-11-23. It is the basis for updating the engine and for rewriting the
paper's design section (`sections/design.tex` in `icml_agent_cache_2027.zip`). The math will be written later. This
note fixes the structure, the mechanisms and the evidence behind each.

Every number comes from the campaign records in `agent_cache/results/` (campaigns 23, 26, 28, 32 and 34) or from the
Nov 20 trace study. Each number names its campaign.

## Decisions

1. **Time to next GPU use.** The design is organized around the time until a session's prefix cache is next needed
   on the GPU. That time has two parts: the remaining gap (agent side) and the queue wait (engine side).
2. **Queue wait is not fixed.** The queue wait and GPU time depend on the current workload, the queue and the
   recompute that the cache policy itself causes.
3. **Architecture, not a model.** The design section becomes a set of components, each with its policy and its
   evidence. The derivations move to the appendix and are written later.

## 1. Time to next GPU use

For session i, all decisions key on

    T_i = G_i + W_i

| | Part 1: remaining gap G_i | Part 2: queue wait W_i |
|---|---|---|
| Who knows it | the agent (tool runtime, human turns) | the engine (queue, GPU work, admission rate) |
| Before the request arrives | a distribution from the hints, conditioned on idle time | predicted from current load (section 2) |
| After it arrives | 0 | nearly known; shrinks as the request moves up |
| How it goes wrong | human turns cannot be forecast | it depends on the policy itself (section 2) |

Each mechanism is defined against T:
- **Fetch:** a read should land just before T. It starts at T − r_i, where r_i is the read time under the current
  storage queue.
- **Hold:** a landed read stays pinned until admission.
- **Evict:** evict the prefix whose T is largest, by class (section 3).

### Where the gain sits

TTFT_i ≈ W_i + S_i + prefill of the new tokens + recompute of U_i lost tokens. A policy acts on three terms:

| Term | What the policy does to it |
|---|---|
| S_i, exposed read | starts the read too late for T |
| U_i, recompute | the prefix is not resident at T: evicted early, or its read was refused for lack of host memory |
| W_i, queue wait | the sum of everyone's recompute and prefill, so it is shared by every request behind |

Campaign 32 on the same engine (FA3, FreeInference, 128 users, 160 GB), over the 9,158 returning requests all six
arms sent before minute 110:

| Arm | Mean TTFT | After admission | Queue, not its own read | Read, then GPU | Read only |
|---|---|---|---|---|---|
| Original HiCache | 33.18 s | 0.55 | 23.32 | 8.82 | 0.49 |
| Four rules | 5.18 s | 0.18 | 4.17 | 0.77 | 0.06 |

The exposed read alone is at most 0.5 s in both arms. The 28 s gap goes through recompute into the queue: HiCache
recomputed 36 % of prompt tokens, the rules arm 12 %. A model explains the gain only if W responds to U.

### Regimes

The read time r against the two parts gives three regimes:

| Regime | What matters | Evidence |
|---|---|---|
| W ≥ r (saturated) | The queue hides the read. Land just before T and pin until T; forecasts do not matter. | Triton (campaign 28): reads waited 40–68 s for the SSD and still landed 1.3–1.5 s before admission. The hinted arms were within 3 s of each other. |
| W < r (lighter load) | Part 1 must hide the rest of the read. A p50 point estimate lands reads early and holds host memory. | FA3 campaign 32, minutes 30–70, about 6 s queues: 57 % of JIT p50's reads started before arrival, and it entered the refused-read loop first. Campaign 34 (fixed engine): its early reads still held 136K host tokens at the median (p90 280K), and it ran 0.6–0.7 s behind the other arms. |
| Recorded pace | Part 1 is mostly unforecastable; part 2 carries the gain. | Human turns hold 76–95 % of resume read time; per-tool forecasts hide 0–0.6 % of it (Nov 20 trace study). |

## 2. Queue wait depends on the load

### The mechanism

W is set by the GPU work ahead of a request, and the cache policy creates part of that work through two channels:

1. **Recompute.** When a prefix is evicted too early, or its read is refused, the request recomputes it.
   - On FA3 that costs 0.042–0.044 ms per token under load, about 1.1 s for a 25K-token history.
   - The GPU serves requests in arrival order, so every request queued behind waits roughly that much longer.
2. **Memory.** A policy that lands reads early holds host memory, other sessions' reads are refused, they
   recompute, and the queue grows. A longer queue means landed reads are held longer, which holds more memory. This
   was campaign 32's loop.

Same users and arrivals in campaign 32:

| | Original HiCache | Four rules |
|---|---|---|
| Share of prompt tokens recomputed | 36 % | 12 % |
| Queue wait (mean, wait split above) | 23.32 s | 4.17 s |
| Requests served per minute | 87.8 | 125.0 |
| Users with a request before minute 110 | 433 | 557 |

### The estimator

1. **Queue wait as work ahead, not position × rate.** For a request in the queue:

       W_i ≈ Σ over requests j ahead of i of s_j
       s_j = ρ[(L_j + Δ_j)² − (L_j − U_j)²] + decode interference

   - L_j and Δ_j are known.
   - U_j is the part of j's prefix that will not be in GPU or host memory at admission: evicted, refused, or landing
     too late. So the tier state feeds the GPU queue estimate directly.
2. **Before arrival.** W also includes the work that joins the queue first: the current queue, plus the requests
   other sessions are expected to send in the meantime (their gap forecasts), minus what the GPU serves by then. The
   hints then predict load as well as single arrivals.
3. **Storage has its own queue.** The read time r_i is not bL/β:
   - one read thread, 83–96 % busy on FreeInference, serves reads, while writes share the same disk (about 2.11 GB/s
     for reads and writes together);
   - r_i therefore depends on the pages queued ahead and on the write backlog.

   The design then has two live queues, GPU and storage, with the same treatment.

### The marginal cost of GPU time

When the GPU is saturated, one extra second of prefill delays every request behind it by about one second:

    cost of recompute ≈ ρU(2L − U) × (1 + requests queued behind it)

From minute 90 of campaign 32, 76–85 requests were queued, so one recompute cost about 80 times what a per-session
model charges. Decisions that change with this cost:

- **Read or recompute.** With an empty queue on FA3, reading 20K tokens takes 0.8–1.1 s against 0.86 s to recompute
  them, so recomputing a short prefix can be right. With 80 requests queued, reading always wins.
- **Eviction protection** tightens with load: losing a prefix costs more when the queue is long.
- **Early reads.** When GPU time is expensive, a refused read, which leads to recompute, costs far more than holding
  memory a little longer.

### Consequence for the paper's model

The draft takes queueing delays as given and folds them into the constant C of the Lagrangian, which is what makes
the problem separate by session. With W depending on the policy, either:
- add a second price, on GPU time, next to the host-memory price, so sessions separate again, each charged for the GPU
  time it causes; or
- drop the separation and present the design as protecting GPU time first.

Undecided (open question 1 below).

## 3. Eviction

### Claims to drop

- **"Recency is a weak proxy."** Not supported at recorded pace: human gaps dominate, and idle age gives nearly the
  right order.
- **A gain from the retention index.** The index alone moved mean TTFT by +0.003 s (campaign 26). With exact hints
  in campaign 32, return-chance eviction left 5.0 % of prefixes missing at admission against 3.2 % for next-use
  eviction (minutes 30–65), and was 0.16 s slower.

### What worked

Protecting prefixes whose T is small and certain: read pins until admission, and evicting queued requests' prefixes
last. Default SGLang reads at arrival with no pin. At 44 users and 70 GB, about half of its landed reads were evicted
before use (974 of 2,065 and 1,209 of 2,139 in two runs).

### Eviction order by T, by class

1. **Ended sessions (T = ∞).** This needs the agent's end-of-session signal.
2. **Host copies of KV the GPU still holds.** Free to drop under exclusive tiering. Under write-through they are a
   fixed cost: 0.65M of the 1.63M host tokens in campaign 34's census.
3. **Idle sessions, largest expected T first.** The forecast only breaks ties within this class; at recorded pace the
   order is close to idle age.
4. **Staged reads of sessions not yet back.** Least likely to return within one read time first (the draft's Preempt
   rule).
5. **Queued requests,** farthest from admission first.
6. **Never:** landed reads of requests near the head of the queue.

On the motivation bullet "evict the data of arrived requests for the more urgent need": if it means taking room from
queued requests far from admission for those near the head, that is class 5. Evicting arrived requests before idle
ones would go against the evidence above.

### Two limits on eviction

- **The write path.** A page cannot leave host memory until it is on the SSD, and pending writes hold host memory.
  - Campaign 32's first refused reads came while pending writes held up to 166K tokens.
  - In its loop, 74–93 % of writes rewrote pages the SSD already held.
  - With rewrites skipped and writes merged, pending writes stayed under 90K tokens in campaign 34.
  - So eviction cannot outrun the write path, and the write path is part of the cache design.
- **The SSD tier evicts too.** The cleaner deletes the oldest page groups once the volume is 90 % full, down to 80 %.
  In campaign 34, recompute rose from 3–4K to up to 290K tokens a minute after its first pass (minute 87–88). The same
  T order should apply there, or at least the cleaner should spare open sessions.

## 4. Architecture

```
 agent harness -- hints (tool key, gap distribution, end of session) --+
                                                                       v
   +------------------- time-to-use estimator --------------------+
   |  gap model (part 1)  +  queue model (part 2: work ahead,     |
   |                         expected arrivals, recompute)         |
   +-----------------------------+---------------------------------+
                                 | T_i, read time r_i, GPU-time cost
   +-----------------------------v--+        +------------------------------+
   | tier planner                   |------->| I/O scheduler                |--> SSD
   | when to read, hold, evict      |        | reads by deadline, not-before|
   +---------------+----------------+        | writes deduplicated, merged  |
                   | pins, budget            | reads held on write backlog  |
   +---------------v----------------+        +------------------------------+
   | host memory manager            |  ready?  +---------------+
   | budget, pins, preemption,      |<-------->| GPU scheduler |
   | census of what holds memory    |          +---------------+
   +--------------------------------+
```

| Component | Policy | Exists in the engine today | To change |
|---|---|---|---|
| Time-to-use estimator | T = G + W; W from work ahead including recompute; r from the storage queue | `mem_cache/next_turn_estimators.py`: `residual_wait(position, rate)` (line 101) with the admission rate over the last 60 s (`RATE_WINDOW_S`, line 24); read time from `_refresh_load_model` and `_load_seconds` in `mem_cache/next_turn_prefetch.py` (lines 1483, 1612) | work-ahead W with expected U per queued request; W before arrival from the queue plus expected arrivals; a GPU-time cost from the queue behind |
| Tier planner | read starts at T − r; hold until admission; evict by T class | `next_turn_prefetch.py`: `read_not_before` (line 1448: predicted admission − load − 2 s, `NTP_HOLD_MARGIN_S` line 128; positions 0–1 read at once); eviction strategies in `mem_cache/next_use_eviction.py` (`NextUseStrategy` line 139, `ReturnChanceStrategy` line 175) | class-ordered eviction keyed on T; protection that tightens with the GPU-time cost; the read-or-recompute choice for short prefixes on an idle GPU |
| Host memory manager | a budget for pins; preempt staged reads before refusing an arrived request; know what holds memory | refusal path `_launch_storage_hit` (`mem_cache/unified_radix_cache.py` line 2988); `MAX_REFETCHES = 2` (`next_turn_prefetch.py` line 110); room retry (`SGLANG_HICACHE_NEXT_TURN_ROOM_RETRY`); census `mem_cache/host_census.py` (`HoldReason`) | charge refusals at the GPU-time cost; keep the census on by default |
| I/O scheduler | deadline-ordered reads with a not-before time; no rewrite of stored pages; merged writes; reads wait while pending writes hold host memory | `write_backup_storage` (`unified_radix_cache.py` line 2027); merging (`managers/cache_controller.py` line 333); skip stored writes (`mem_cache/storage/nixl/nixl_utils.py`); read hold on pending writes (`unified_radix_cache.py` line 631); switches in `environ.py` lines 775–784, all off by default | decide which switches become defaults; feed the write backlog into r |
| SSD tier | evict by T, or at least spare open sessions | `HiCacheL3Cleaner` in `mem_cache/storage/nixl/nixl_cleaner.py`: oldest group mtime first between the high and low watermarks | a session-aware order |
| Admission coupling | admit when the prefix has landed; a late read only makes admission pass over its own request | wait_complete admission | none planned |

**Information levels.** No hints, end signal, distribution and exact hints become what the system falls back on as
hints get poorer, not separate arms.

**Still open in the engine (`agent_cache/OPEN_ITEMS.md` section 1):** a lookup that finds nothing still counts
toward the per-request lookup cap. Campaign 34 still saw 0–17 lookups per window refused at the cap.

## 5. Checks and runs

### Offline check of the queue model (no GPU)

- **Goal:** show that the queue wait depends on recompute, and that a work-ahead estimate predicts it.
- **Method:** for each returning request in campaign 32's six runs, predict its queue wait at arrival two ways:
  position × admission rate, and work ahead including expected recompute. Compare both with the measured
  `queue_duration` in each server log.
  - Data: `agent_cache/results/compare_c32_designs/runs/<arm>/server.log` and the client logs beside them.
- **Expected result:** the work-ahead estimate tracks the 23.3 s against 4.2 s queue gap between HiCache and the
  rules arm. The rate-based one lags it, because it only sees the rate after the queue has grown.
- **Expected lesson:** whether a work-ahead W is worth building into the estimator, and how much decode interference
  it must model.

### Runs pending (need the GPU)

- **Goal:** put original HiCache and the oracle (ec_exact) on the fixed engine, so the campaign 34 figures compare
  policies on one engine. They currently mix campaign 32's runs (no fixes) with campaign 34's.
- **Queue lines,** with campaign 34's `settings.env` (the five fix switches on):

      hicache three_tier_wc none - engine_fa3fix_20261122 --attention-backend fa3 --max-total-tokens 668096 --host 127.0.0.1
      ec_exact three_tier_wc_oracle_v2_nu_owned oracle - engine_fa3fix_20261122 --attention-backend fa3 --max-total-tokens 668096 --host 127.0.0.1 --hicache-storage-next-turn-read-timing expected_cost --hicache-storage-next-turn-memory-price 5e-5 --hicache-storage-next-turn-price-learning harm --hicache-storage-next-turn-soft-pins on --radix-eviction-policy return_chance --radix-eviction-policy-config {"horizon_s":30}

- **Expected result:** ec_exact close to ec_full and jit_exact (campaign 34's jit_exact ties ec_full: −0.06 s,
  interval −0.19 to +0.06). HiCache gains too, since two fixes act on the write path it uses.
- **Notes:**
  - Campaigns 32 and 34 ran on GPU unit `GPU-0f61782e`. Record the unit used.
  - Afterwards, set the campaign of `hicache` and `ec_exact` in `ARMS` in `agent_cache/scripts/c34_pricing.py`.
    Then rerun it and `c34_figures.py`.

## 6. Open questions

1. **The paper's model:** a second price on GPU time, or no per-session separation?
2. **The host-memory price:** keep it as one fixed knob in the memory manager, or drop it from the main design?
3. **The SSD write-path fixes:** present them as design components? The design needs them to hold under overload,
   but they have upstream precedent: Mooncake already skips rewrites.
4. **Default configuration:** exclusive tiering or write-through? This decides whether eviction class 2 is free.
5. **The motivation's eviction bullet:** which meaning was intended (section 3)?
