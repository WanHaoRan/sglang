## position
I agree that every decision should key on T = G + W, and that pre-arrival and post-arrival read timing should be one decision rather than two mechanisms. I would change the weighting, though. The records and my checks say the gain is the post-arrival schedule: pins, deadline order by admission position, and the hold. Pre-arrival reads paid in one setting only (campaign 26, 44 users / 70 GB: 0.14-0.21 s, results/campaign26/DECISIONS.md:632), and they never fire once W > r.

The reason is a cost the draft leaves out. Admission passes over a turn whose read is still under way (python/sglang/srt/managers/scheduler.py:4010-4012). So a late read only reorders admissions while other queued requests are ready, and its system cost depends on whether the GPU has other ready work. In campaign 34 it did: whenever requests were queued, the KV pool stayed at about 97 % use in every arm and window, so lateness cost no GPU capacity. When the queue is short, a late read is the turn's full stall.

So I propose one expected-cost rule: earliness priced at the memory price, lateness priced at the probability that the GPU starves. At light load it gives the paper's critical ratio on T = G + W. Under load it gives no early reads and a pull rule that lands each read at, not before, admission. In a toy two-queue model that rule tied the engine-like hold when memory was ample (27.2-27.4 s, while pinned memory fell 41K → 5K tokens) and beat it by 16 s when memory was tight.

Three things are missing:
- r must come from the storage queue. The pre-arrival r was 4-6x short once the read thread saturated.
- W should count only the requests ahead that are ready. Near admission this cuts the p90 error from 9-15 s to 2-5 s.
- The closed-loop law R = N/X − Z should be the paper's account of why W depends on the policy (through throughput X), in place of a work-ahead estimator.

On eviction, I agree to treat Preempt and Retain as one decision. I would rank by slack (time until the history is needed minus time to restore it), with the restore cost set by where the other copy is: on SSD it costs a read; if the host copy is the only one it costs a recompute. Writes are jobs whose deadline is host room. Under GPU saturation they should outrank timed reads whenever pending writes threaten the room the next reads need. Fix 3 approximates this.

## fetch_rule
STATEMENT. Keep one read job for each history that a turn will need.
- Its due time T is when admission would take the turn if its history were resident: T = A + W.
- Before arrival, A = now + G, with G the remaining gap from the hint, conditioned on idle time. After arrival, A = now.
- Release the job at the time s that minimises E[c_E (T − C(s))+ + c_T (C(s) − T)+]. Release means park it for earliest-deadline dispatch; host memory is taken at dispatch. C(s) is the landing time.
- There is recourse: a turn that arrives before s is released at arrival.
- c_E = memory price × room tokens.
- c_T = GPU capacity lost per second of lateness. That is the probability that no other queued request is ready when this one reaches the front, because admission passes over a turn whose read is under way (python/sglang/srt/managers/scheduler.py:4010-4012).
- If the hazard of T increases with idle time, this reduces to "release when P(T <= now + r) >= q, q = c_E/(c_E + c_T)".
- The trace's gap entries have a decreasing hazard (sigma_high 1.79 and 1.86; fetch fact-finder's CPU check). For them a threshold fires right after the turn ends and then pins memory through the tail. So keep the argmin with recourse that read_start already computes (python/sglang/srt/mem_cache/next_turn_policy.py:165-201), and add c_T to it.

HOW G AND W ENTER.
- Before arrival, the spread of T is G's spread plus W's error. Under load W dominates for tool gaps (gap p50 1 s against W of 10-30 s). At light load G dominates.
- W = (ready requests ahead + 0.5) / admission rate, counting only queued turns that admission will not pass over. The tick already counts these for room retry (python/sglang/srt/mem_cache/next_turn_prefetch.py:661-667).
- Offline on campaign 34, minutes 30-110, counting only ready requests changes the p90 absolute error of the remaining wait as follows:

| Time before admission | ec_full | jit_exact | jit_p50 |
|---|---|---|---|
| 2 s | 8.9 → 1.8 s | 10.4 → 1.8 s | 15.2 → 1.8 s |
| 5 s | 9.9 → 4.4 s | 11.2 → 4.3 s | 16.3 → 4.8 s |

- Turns that had their own SSD read waited a median 6.2, 11.8 and 20.6 s longer than their queue position predicted (ec_full, jit_exact, jit_p50).
- Weighting the queue by tokens (a KV-pool model) does not help. At arrival, ec_full: |err| p50 2.5 against 2.6 s, p90 14.9 against 14.3 s.
- A prefill work-ahead sum is the wrong model here. Admission is bound by the KV pool at 19-23 running requests, and decode takes 61-75 % of GPU time (calibration fact-finder; /home/wanhr/sglang/agent_cache/results/campaign34/checks/decode_scale.out).
- No projection of W to the arrival time is needed for tool gaps. W moves a median 0.9 s over 1 s and 3.2 s over 17 s, less than the estimator's own error at 10-20 s (p90 6.9 s). Human gaps are too uncertain to read early for.
- Scripts: /tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/designer-scheduling/w_estimators.py and /tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/designer-scheduling/w_estimators2.py.

HOW r AND THE STORAGE QUEUE ENTER.
- C(s) = s + lags + p × (pages in flight + parked pages due before T + expected 'now' reads) + p × own pages + the expected remainder of a write hold.
- p is the read thread's EWMA in ms per page, which already absorbs interference from writes.
- Under earliest-deadline order a read cannot land before the work due ahead of it. In overload, its release time therefore does not matter, provided it is parked before the dispatch frontier reaches T.
- Backward scheduling (latest start times over the deadline-ordered list) matters only when the read thread is underloaded. There read_not_before already sums the pages ahead, pessimistically (python/sglang/srt/mem_cache/next_turn_prefetch.py:1448-1481). It lands reads near admission: campaign 34 ec_full, last landing to admission p50 0.8 s, p90 4.1 s (/home/wanhr/sglang/agent_cache/results/campaign34/checks/read_waits_ec_full.out).
- The real gap is before arrival. The pre-arrival planner's r counts only pages already dispatched (python/sglang/srt/mem_cache/unified_radix_cache.py:2334-2337).
- Campaign 34 jit_p50's staged reads:

| Minutes | Planned r, median | Actual stage → land, median | Landing vs the turn's arrival, median |
|---|---|---|---|
| 30-60 | 5.2 s | 6.2 s | 6.2 s before |
| 60-90 | 8.3 s | 30.3 s | 15.9 s after |
| 90+ | 8.3 s | 52.5 s | 56.1 s after |

- Script: /tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/designer-scheduling/staged_reads.py.

REGIMES.
1. Light GPU load, host memory binding (c_T ≈ 1, c_E > 0). This is the paper's critical ratio on T = G + W with W ≈ 0. Early reads pay only with sharp hints.
2. GPU saturated with ready work queued (c_T ≈ 0).
   - No read before arrival.
   - After arrival, release by ready position (a pull rule): start a turn's read once at most K ready requests are ahead of it.
   - K is small (2-8) or equals the admissions during its own read time, never sized from the storage backlog.
   - Reads then land at or just after admission, and pins hold almost nothing.
3. Host memory ample (c_E ≈ 0). Release as soon as the read is known, at background priority.
4. Storage saturated.
   - Earliest deadline by admission order; early release gains nothing.
   - While the GPU is saturated, keep reading rather than recompute.
   - Recompute only when the GPU would otherwise starve and the recompute would finish before the read lands. On FA3, recomputing 20K tokens takes about 0.62 s on an idle GPU against 0.8-1.1 s to read them (results/campaign32/checks/mechanisms_findings.txt:174).

NEW AGAINST THE ENGINE.
- c_T, taken from the ready count. Today read_start charges every second of stall at 1, and the JIT and hold paths use fixed margins.
- Release by ready position after arrival, instead of the time hold with its 2 s margin (NTP_HOLD_MARGIN_S, next_turn_prefetch.py:128).
- W counted over ready requests only.
- Before arrival, r taken from the deadline-ordered queue plus the write hold.
- One function replaces _check_pending (before arrival) and read_not_before / read_deadline (after arrival). The 1 s JIT margin, 2 s hold margin, early_s and 120 s lead horizon become quantiles.
- Early release is gated on c_T > 0. Expected-cost timing's GPU-bound test (next_turn_prefetch.py:698-712) is a binary version of this gate.

EVIDENCE.
- Late reads cost no GPU capacity in campaign 34. From minute 30, whenever requests were queued, the KV pool was at least 90 % used on 98.7-99.6 % of batch log lines and on average 2.6-3.1 % unused, the same in all three arms and every window. Campaign 32 hicache: 3.7-5.0 % unused. Script: /tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/designer-scheduling/pool_idle.awk.
- From the records:
  - Early reads paid only at 44 users / 70 GB (campaign 26).
  - They cost 0.6-0.9 s on p50 hints (campaign 26; results/campaign34/DECISIONS.md:145-146).
  - They did not fire in saturation: campaign 28 chose arrival in 2,354 of 2,354 decisions, and campaign 34 ec_full staged 1 read.
  - The large gains came from avoiding earliness after arrival: the hold in campaign 29, −77.7 s (results/campaign29/DECISIONS.md:388); pins in campaign 23, 79 % of the gain.
- Toy two-queue model (/tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/designer-scheduling/toy_two_queues.py). Closed loop with FreeInference-like gaps, 21 GPU slots with pass-over, LRU host with pins ≤ 0.3 H, one earliest-deadline read server at 18.3K tokens/s. At 128 users with 1.5M host it lands near campaign 34's late phase (2.33 against 2.36 req/s, TTFT about 27 s).

| Regime | Rule | Mean TTFT | Notes |
|---|---|---|---|
| 128 users, 1.5M host, 8 seeds | engine-like hold | 27.32 s | pins 40.7K; se 0.6-0.8 s for all four rules |
| | land 3 s late | 27.17 s | pins 15.2K |
| | pull K = 2 | 27.42 s | pins 4.6K |
| | pull K = 8 | 27.30 s | pins 12.3K |
| | all four | | GPU never starved |
| 128 users, 0.7M host | engine-like hold | 46.3 s | |
| | pull K = 2 | 30.0 s | |
| | pull K = 8 | 30.3 s | |
| | read at arrival, pinned | 105.8 s | |
| | hold with W over ready requests | 87.8 s | more accurate W, released earlier |
| 40 users, 300K host | hold, no early reads | 0.26 s | GPU idle with a waiting read 22 % of the time |
| | exact early reads | 0.12 s | idle 4 % |
| | p10-timed early reads | 1.65 s | |
| | expected cost on the trace's entries | 0.26 s | never staged |

- Toy outputs: /tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/designer-scheduling/toy_two_queues.out, /tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/designer-scheduling/toy_two_queues_set3.out and /tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/designer-scheduling/toy_seeds_c34like.out.

## eviction_and_writes
PREEMPT AND RETAIN ARE ONE DECISION.
- A staged read, an idle history, and a queued turn's history beyond the ready buffer are all histories held in host memory ahead of their T. Giving one up turns it into a later read, or a recompute.
- Rank them together by slack: a low quantile of T minus the time to restore. Evict the largest slack first.
- The restore cost depends on where the other copy is:
  - Clean on SSD: a read, which costs storage time and is free while the read thread is idle.
  - Only copy (failed write, deleted by the cleaner, refused GPU→host copy): a recompute. That costs GPU time, which under the user cap delays every request behind it.
  - Dirty (pending write): not evictable until the write acks.
- Order in practice:
  1. Ended sessions (T = ∞).
  2. Host copies of KV the GPU still holds. They are free to drop under exclusive tiering, and are 0.65M of 1.63M tokens in campaign 34 (results/campaign34/DECISIONS.md:98-102).
  3. Idle clean histories, by slack.
  4. Staged pins.
  5. Queued histories beyond the ready buffer (slack W − r), only while the read thread has spare capacity.
  6. Never the ready buffer or reads in flight.
- Under the pull rule, staged pins mostly disappear under load, so Preempt matters at light load only. It never acted on FA3: 0 ec_bump events in campaigns 32 and 34 (evict fact-finder).
- Today the engine does the reverse. A read short of room evicts other arrived turns' queued-tier leaves first, and releases staged pins only afterwards; the pins sit outside the eviction heap (python/sglang/srt/mem_cache/unified_radix_cache.py:3036-3077).
- Wherever eviction order alone was isolated, it moved TTFT by 0.1 s at most: campaign 26, and campaign 32 at −0.07 s [−0.24, +0.10] (results/campaign32/DECISIONS.md:59-64). Present the ranking as protection of histories whose T is small and certain, not as a source of gain.
- "Clean" must mean acked complete. _drain_backup releases the write's host lock at the ack without checking completed_tokens (unified_radix_cache.py:3511-3515). A failed write therefore leaves a host copy that eviction treats as safe.

WRITES ON THE SHARED DISK.
- Reads and writes share one device of about 2.1 GB/s (iostat in campaigns 32 and 34), on separate threads with no I/O priority.
- Campaign 34 (calibration fact-finder): write-through demand was 4.0-4.8K tokens/s. While reads kept the disk 74-100 % busy, the write thread managed 5.5-7.4K tokens/s (8.7-11.6 ms/page), so write utilisation was about 0.55-0.85, a thin margin.
- In campaign 32, rewrites of recomputed history (74-93 % of writes) pushed it over, and pending writes reached 515-675K tokens (results/campaign32/DECISIONS.md:171-183).
- In scheduling terms, a write is a job whose deadline is host room: a pending write host-locks its pages (unified_radix_cache.py:2055-2060).
- A late read is free while the GPU has ready work. So under GPU saturation, writes should outrank timed reads whenever pending writes exceed the room the next pulled reads need. At light load (c_T ≈ 1), reads go first.
- Disk priority classes, highest first:
  1. Reads for the ready buffer ('now' reads).
  2. Writes, when pending writes threaten that room.
  3. Other timed reads.
  4. Background writes.
- Fix 3 is a fixed-threshold version of this: timed reads stop at 50K pending tokens and resume at 25K (python/sglang/srt/mem_cache/storage_read_scheduler.py:147-150, 170-181). Its threshold should instead come from the host room the read planner needs, which the census already reports.
- Write order: FIFO is close to eviction order under LRU-like eviction. Reorder only if write utilisation nears 1.
- Cut demand first:
  - skip stored pages (fix 1);
  - merge writes (fix 2);
  - skip pages that will never be read: ended sessions' last turn, and, in this block-trace replay, generated outputs. The replay rebuilds prompts from recorded reply blocks, so outputs are never re-sent (inferred from /home/wanhr/sglang/agent_cache/scripts/replay_programs.py:7-9). That is about 20 % of the new KV per request (inferred: 280 output against 1.1K new tokens, campaign 34 workload).

SSD TIER.
- Deleting an SSD page turns a future read into a recompute, the most expensive eviction.
- The cleaner deletes least recently written page groups between 90 % and 80 % (python/sglang/srt/mem_cache/storage/nixl/nixl_cleaner.py:168-212).
- Keys are chained, so a deleted head cuts off the whole chain (python/sglang/srt/mem_cache/storage/nixl/hicache_nixl.py:658-664).
- Use the same order by T: ended sessions first, then the largest T, tails before heads, and refresh mtime on read.
- Campaign 34 (evidence fact-finder): after the first pass, 116 returning requests lost at least 1K tokens of history in 22 minutes, against 16 before. That is 1.1-1.3M tokens per arm (1.5-1.7 % of history), in bursts of up to 292K tokens a minute. Small in a 2 h run; it grows with run length or a smaller volume.

## paper_structure
Organize the design section as an architecture around one deadline (T = G + W) and two queues (GPU admission and SSD I/O), coupled by host memory.

1. Overview, with a figure: time-to-use estimator → read planner → I/O scheduler, and host memory manager ↔ GPU admission.
2. Time to next use.
   - G comes from the hints, conditioned on idle time.
   - W = ready requests ahead / admission rate, with its measured error.
   - Why W depends on the policy: under the user cap, R = N/X − Z. A policy moves mean TTFT only through GPU throughput X: recompute, and an idle KV pool. This replaces taking W as the constant C.
3. Read release, the main mechanism.
   - Earliest deadline by admission order, memory taken at dispatch, pins until admission.
   - Under load: release by ready position (pull).
   - Before arrival: release only when the GPU queue is short.
   - One cost statement: earliness at the memory price against lateness at the probability the GPU starves. The four regimes are its cases.
   - Read versus recompute is the same rule.
4. Host memory: one ranking of staged, idle and queued histories by slack. Ended sessions first, clean before dirty, never the ready buffer.
5. I/O scheduler: reads and writes on one device; a write share tied to host room; skip and merge writes, crediting upstream precedent (Mooncake skips rewrites).
6. SSD tier: deletion turns a read into a recompute; the same order by T, tails first.

Hint levels (none, end signal, distribution, exact) become what each component falls back on, not separate arms.

Move to the appendix:
- the earliness/tardiness derivation: the critical ratio for an increasing hazard, and the argmin with recourse otherwise;
- the closed-loop derivation;
- the Lagrangian and its weak-duality floor;
- the retention index and price learning, if kept at all. The engine uses harm learning, and λ never moved on FA3: it stayed at 5e-5 in campaigns 32 and 34 for ec_full and sat at the 1e-7 floor for campaign 32 ec_exact.

Drop or restate these claims:
- "F(x|t) rises as the session stays idle": false for this trace's entries (decreasing hazard).
- "A read started at arrival completes before the head, so nothing is read early": the engine holds reads after arrival, and that schedule is the gain.
- r_i = bL_i/β: r is mostly the storage queue (4-6x the planned value in saturated windows), and β is shared with writes.
- "Recency is a weak proxy", and any TTFT gain from the retention index: at most 0.1 s wherever isolated.
- The multiplicative price update as described: not what the engine runs.
- "Storage holds every prefix cache": the cleaner deletes pages.
- "Beyond position b/(2ρβ) reading is cheaper" as a constant: those are Triton constants; on FA3 the crossover is about 33-37K tokens (inferred), and the choice depends on load.
- The motivation bullet on evicting arrived requests: keep it only as "queued histories beyond the ready buffer".

## experiments
- **W at the decision horizon (offline, partly done)**: Q: Which estimate of the queue wait is accurate enough to time reads, and how large is its error at the horizons the rule uses?
  - method: Offline over the server logs of campaign 34 (three arms), campaign 32 (six) and campaign 28. Per request, arrival and admission come from ReqTimeStats, and reading intervals from s2h_io. Predictors: position × 60 s admission rate (the engine's), ready requests ahead × rate, token-weighted queue, recent waits, and the closed-loop R = N/X − Z. Error at arrival is binned by the predicted wait; error is also measured at 2, 5, 10 and 20 s before admission, per window. /tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/designer-scheduling/w_estimators2.py does the campaign 34 part.
  - expected: As found on campaign 34: counting only ready requests cuts the p90 absolute error 2 s before admission from 8.9-15.2 s to 1.8 s; tokens add nothing; the error grows to about 40 % of the horizon at 10-20 s. Campaign 32's loop windows should show a larger lag.
  - decides: Whether the engine and the paper use W = ready requests ahead / rate, with no KV-pool or prefill work-ahead model; which error quantile replaces the 2 s hold margin; and confirms that tool gaps need no projection of W.
- **Cost of a late read (offline, partly done)**: Q: When a read lands after its turn would have been admitted, does the GPU lose capacity, or does another ready request take the slot?
  - method: Offline: from Decode and Prefill batch lines (token usage, #queue-req) together with per-request reading intervals, measure the KV pool left unused, and the time spent with free capacity while every queued request waits on a read. Per window and arm for campaigns 28, 32 and 34, and campaign 26 at 44 users. /tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/designer-scheduling/pool_idle.awk does the first part.
  - expected: Campaign 34: 2.6-3.1 % unused in every window (the decode reserve), so c_T ≈ 0 from minute 30. Campaign 32 hicache: 3.7-5.0 %. Campaign 26 at 44 users: short queues, with c_T near 1 for reading turns.
  - decides: Whether the lateness weight must depend on the ready count, i.e. whether the paper's stall term is weighted by the probability the GPU starves.
- **Read time from the storage queue (offline, partly done)**: Q: How far off is the pre-arrival planner's r, and does an earliest-deadline-aware r (in flight + parked and due earlier + write hold) predict stage-to-landing time?
  - method: Offline replay of srs_park, srs_dispatch, s2h_io, srs_write_hold and ntp_due over campaign 34 jit_p50 and jit_exact, and campaign 32 for the loop. At each staging and each hold decision, compute the earliest-deadline-aware r and compare it with the logged load_ms and the actual landing. /tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/designer-scheduling/staged_reads.py does the first comparison.
  - expected: Planned r p50 8.3 s against an actual 30-52 s in jit_p50 from minute 60; the earliest-deadline-aware r within about 25 %. In those windows early reads add nothing, since they landed 16-56 s after the turn arrived.
  - decides: Whether the pre-arrival r must include the parked queue and the write hold, or whether gating early reads on c_T alone is enough.
- **Two-queue DES: read release rules**: Q: Across regimes, which release rule gives the lowest mean TTFT: the engine's time hold with a 2 s margin, a hold with a late margin, a pull rule with fixed K or K sized by the own read time, or early reads gated on c_T with exact, p50 and distribution hints?
  - method: A CPU discrete-event simulation grown from /tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/designer-scheduling/toy_two_queues.py, calibrated on campaign 34: closed-loop client of the trace's 729 users (the load_model.py route); KV-pool-bound admission with pass-over; FA3 prefill and decode costs; page-level histories with growth and shared prefixes; one read thread with the 512-page window; pins within 0.3 of the pool; refusals; the 3-lookup cap. Regimes: campaign 34 (1.63M host), 0.7-1.0M host, 40-60 users, and campaign 32's write locks. Five or more seeds. Validate against campaign 34 (arms within 0.7 s, no refusals) and campaign 26 at 44 users.
  - expected: As in the toy. With ample memory all rules tie (27.2-27.4 s, se 0.6-0.8) while pins fall up to 9x. With tight memory, pull K = 2 gives 30 s against 46 s for the hold. At light load, exact early reads give 0.12 s against 0.26 s, p10 timing gives 1.65 s, and the expected-cost rule on the trace's entries never stages.
  - decides: Whether to replace the time hold with a pull rule, and which K; whether to keep early reads only behind the c_T gate; whether the paper needs a pre-arrival section at all.
- **Disk sharing between reads and writes (DES)**: Q: Under GPU saturation, should writes outrank timed reads when pending writes hold host room, and how much margin does the write path have?
  - method: Add to the DES one write thread on the shared 2.1 GB/s device, with write ms/page rising from 3-4 to 9-12 as reads load the disk (as in campaign 34). Host pages stay locked until the write acks. Model rewrites of recomputed history (fix 1 on and off) and merging (fix 2). Compare the fix 3 read hold with a priority rule: buffer reads > writes when pending writes exceed the room the next pulled reads need > other reads > background writes. Also skip writes of ended sessions. Sweep write demand from 0.5 to 1.2 of the write capacity under read load.
  - expected: Campaign 32's loop reappears with rewrites once demand nears capacity. Any writes-first rule removes it at no TTFT cost while c_T ≈ 0. Campaign 34's margin is a write utilisation of about 0.55-0.85.
  - decides: Whether fix 3 becomes a design component (a write share tied to host room) and its default; whether writes need ordering by eviction position (expected: not below a utilisation of about 0.9).
- **Closed-loop law (offline)**: Q: Once the 128-user cap binds, is mean TTFT set by throughput (R = N/X − Z, minus decode), so that a policy acts only through X?
  - method: Per minute, from client.jsonl and the server logs of campaigns 32 and 34: users in session (N), completed requests (X), mean gap of the requests sent (Z), and decode time. Compare predicted and measured mean TTFT per arm in minutes 85-110, and attribute arm differences to recompute GPU time and an idle KV pool.
  - expected: Within about 10 % per arm. Campaign 32's arm gaps are explained by X (87.8 against 125 requests a minute); campaign 34's arms tie because X ties (141.5-141.9 requests a minute).
  - decides: Whether the paper explains W's dependence on the policy through throughput rather than a work-ahead estimator, and whether GPU time is priced at about N per GPU-second under the cap.
- **SSD cleaner order (offline)**: Q: How much of the recompute after cleaner passes would an order by T avoid (refresh on read, tails first, ended sessions first)?
  - method: From campaign 34's h2s and s2h events (rid to session, page counts, times), rebuild per-chain write and read times. Replay the passes at 90/80 % with alternative orders, and count later lookups that would find their head deleted.
  - expected: Refresh on read, or deleting tails first, removes most of the 65 zero-hit and 18 partial-hit losses after the first pass (1.1-1.3M tokens per arm).
  - decides: Whether the SSD tier needs a session-aware order or only LRU (refresh mtime on read).
- **Ceiling of early reads on this trace (offline)**: Q: Given G, W and r as recorded, how much read time can a gap hide?
  - method: For returning requests that read in campaigns 34 and 32 (and campaign 26), compute min(G, r − W)+ per request from recorded gaps, waits and read times, with exact hints as the bound, by regime window.
  - expected: Near 0 on FA3 from minute 30 (W > r); a few tenths of a second per reading request at light load.
  - decides: How much space the paper gives to early reads, and which setting must exercise them (light GPU load, small host pool, sharp hints).

## risks
- The toy model is stylized: no GPU tier, no shared prefixes, all-or-nothing histories, a refusal recomputes the whole history, no write path, rough constants. Its regime results are directional until the calibrated simulation reproduces campaigns 34 and 26.
- Late reads are free only while other queued requests are ready. A pull rule with too small a K starves the GPU once more turns need reads (more host pressure, a slower SSD). Sizing K from the storage backlog fed back on itself in the toy (89.5 s against 30.0 s for K = 2 with tight memory), so K needs a cap, or must come from the turn's own read time.
- Fairness and tails: a pull rule moves delay onto turns that read. In campaign 34 they already waited a median 6-21 s more than their queue position predicted. Means can tie while p90/p99 and per-user TTFT get worse, so report tails and the reading/non-reading split.
- A more accurate W on its own makes things worse. Counting only ready requests releases reads earlier (toy: 87.8 s against 46.3 s for the hold with tight memory). The estimator and the target quantile must change together.
- The closed-loop law relies on the replay client's user cap. Under open-loop overload W grows without bound, so the paper should cover both regimes and not generalize from the capped phase.
- The FA3/FreeInference setting may never exercise the pre-arrival rule. In campaign 34 the queue was empty most of minutes 0-30, when few reads happen, and non-empty from minute 30 on, when early reads stop paying. The paper's evidence for early reads would rest on campaign 26 (Triton) unless a new setting is run.
- There is one run per arm on FA3 and no measured spread in the saturated phase. The toy's seed spread (sd about 2 s) suggests sub-second differences between rules need repeats.
- Replay artifact (inferred from scripts/replay_programs.py:7-9): block traces rebuild prompts from recorded reply blocks, so generated outputs (about 20 % of new KV per request) are written to host and SSD but never reused. Write-path conclusions may overstate write demand compared with a real agent loop.
- Writes-first can starve reads when write demand exceeds write capacity under load. Then only cutting demand helps (skip stored pages, skip never-read pages, merging), and the bottleneck may just move.
- Engine coupling the toy does not model: late reads interact with the 3-lookup cap (defect 1), room retry, plan expiry and pin release after 600 s. A pull rule must not raise capped lookups or revocations.