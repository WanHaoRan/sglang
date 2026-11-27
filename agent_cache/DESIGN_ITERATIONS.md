# Design iterations of the agent KV cache work

Written 2026-11-26. One entry per design iteration, in order of time: what it is, its main points, the arm label it ran under and the record to read. The table at the end lists them one line each. A campaign is a set of GPU runs on one workload, and an arm is one server configuration in it. Paths are relative to `agent_cache/`. Every number names its campaign; one run per arm unless said.

**What a new reader needs first**

- **The system.** SGLang's hierarchical cache (HiCache) keeps each agent session's prefix cache (the KV of its context) in GPU memory, in host memory or on one SSD. Between two turns the agent runs a tool call, the session is idle and its KV moves down the tiers; the next turn needs it back on the GPU, or recomputes it. A request waits in a queue until the GPU admits it (admission). The metric is TTFT, the time to a turn's first token.
- **Hints.** A gap is the time from a reply to the session's next request, and a hint is what the agent framework tells the engine about it. Exact hints give the true gap (an oracle, so an upper bound). Table hints give the p10, p50 and p90 of past gaps after the same kind of tool call, which a real agent can keep; "p50 hints" use the table's median as one number.
- **The box.** One H200, Qwen3-30B-A3B with bf16 KV, a GPU pool of about 668K tokens, a host pool of 160 GB (1.63M tokens) unless a size is given, and one SSD of about 2 GB/s.
- **Host room.** HiCache copies every page the GPU holds to host memory (write-through), so the room left for idle sessions is the host pool minus the GPU pool: about 44K tokens at 70 GB and 959K at 160 GB.
- **Workloads.** Campaigns 14 to 26 replay the LMCache agentic trace (768 sessions) with the recorded gaps stretched x70 or x10; their numbers are mean TTFT on the turns every compared run completed (shared turns). Campaigns 28 on replay public traces (FreeInference, TraceLab) at the recorded pace with at most 128 users in session; their numbers are mean TTFT over each arm's requests sent in the first 6,600 s.
- **Attention backend.** Campaigns 14 to 29 use Triton attention; campaigns 30 and 32 to 35 use FlashAttention 3 (FA3), whose prefill takes 0.042 to 0.047 ms a token against Triton's 0.11 ms. The same workload loads the box far less on FA3: TraceLab saturates it from minute 6 on Triton and never on FA3 (campaigns 29 and 33).

## 1. Original HiCache, the baseline (upstream code; reference run campaign 14, 2026-10-08)

Stock HiCache with the SSD tier: every prefilled page is copied to host memory and on to the SSD (write-through), eviction is by recency (LRU), a prefix is read from the SSD only once its request has arrived, and admission waits for that read (`wait_complete`).

- **A read that has landed is not protected.** Once it has finished it is an ordinary host entry and can be evicted before its request is admitted. In campaign 14 (x70 gaps) 665 turns recomputed a history the SSD had already delivered, 59 % of all recomputed history, and the run fell into a "recompute storm" (mean TTFT of 100 to 180 s for 30 to 50 min).
- **Empty lookups spend the lookup budget.** A request may ask the SSD 8 times (paced retries, 8 scheduler passes apart, and the re-query at admission of upstream #39283). Polls for text the SSD never held use the budget up, and history lost while the request queues is then recomputed. At x10 gaps that storm never ended (95.8 s, campaign 19); on FA3 94 % of its lookups find nothing (campaign 34).
- **Where it stands.** 18.5 s at x70 (campaign 14); 5.17 s at 44 users and 70 GB (campaign 21; a repeat ran 1.43 s slower, campaign 25); 115.7 s on FreeInference (campaign 28); 33.5 s on FA3 (campaign 32) and 31.88 s with the write fixes (campaign 34). The other read policy, `timeout` (wait up to a deadline, then recompute), did worse (53.3 against 22.2 s over campaign 14's measurement window) and was dropped.
- **Arms and records.** `three_tier_wc` (label `hicache` from campaign 28), `three_tier_to`. Read `ORACLE_FINDINGS.md` sections 2 to 4 and `results/campaign14/DECISIONS.md`.

## 2. First oracle (v1): reads anchored on admission (plan 2026-10-01; campaign 15, 2026-10-09)

The client sends with each request the exact time until the session's next request. The engine predicts when that request will be admitted (arrival plus queue wait) and starts its SSD read one load time earlier: before the request arrives (staging), or at once for a queued request that lost history (refetch).

- **Result (campaign 15: x70 gaps, 768 sessions, 5 h).** Under `wait_complete` 20.8 against 22.2 s for the baseline; under `timeout` 41.1 against 53.3 s (means over the measurement window).
- **Why so little.** All reads share one first-in-first-out thread, so a staged read waited about 30 s behind others. Host pages were taken before the read waited, so queued reads held 39 % of the host pool and the share of returning turns served from host memory fell from 16 % to 10 %. Landed reads were still unprotected (509 turns lost one), and the hints were not used for eviction.
- **Superseded by** oracle v2, which answers each cause with one lever.
- **Arms and records.** `three_tier_wc_oracle`, `three_tier_to_oracle`. Read `ORACLE_PLAN.md`, `ORACLE_FINDINGS.md` section 5 and `results/campaign15/README.md`.

## 3. Oracle v2 and its levers (plan 2026-10-10; campaigns 16 to 19, 2026-10-11 to 10-18)

Three levers added one at a time on top of HiCache, each behind a flag. On campaign 16's 20,217 shared turns (x70 gaps) the baseline is at 18.5 s and oracle v1 at 17.3 s.

| Arm | What it adds | Mean TTFT | What changed |
|---|---|---|---|
| `_dl` | Read pins, no hint needed: a read the SSD delivered stays locked (pinned) in host memory until its request is admitted, within 30 % of the host pool | 13.2 s | no turn lost a landed read and the storm did not happen (p99 TTFT from 169 to 46 s) |
| `_oracle_v2` | Exact hints and reads in deadline order: a read waits parked without host memory and is dispatched earliest deadline first into a 512-page window, up to 120 s before the predicted admission; the baseline's paced retries are off | 11.6 s | host memory held by waiting reads fell from 28 % to 2 % |
| `_oracle_v2_nu` | Next-use eviction ("_nu") on GPU and host: ended sessions first, then sessions over 60 s overdue, then sessions due back after 300 s, then entries with no hint, last sessions due back within 300 s | 5.7 s | returning turns served from host memory rose from 14 % to 41 % and SSD reads fell 19 % |

- **The same arm in other settings.** 15.8 to 11.2 s at x10 (campaign 17); on the rebuilt replay client 1.09 to 0.58 s at x70 (campaign 18, within the run-to-run spread) and 95.8 to 20.3 s at x10 (campaign 19). How much of those gains needed a hint was not separated then (entry 6).
- **Why superseded.** The hints were exact. Reads staged far ahead sit pinned (a median 9.8 s before admission at 44 users and 70 GB), so where host room is short the arm adds nothing over pins alone (1.52 against 1.51 s, campaign 23's analysis). Next-use eviction also did not register staged reads, so unpinned ones were evicted and read twice (27 % of the arm's SSD reads in campaign 18). It stays in later campaigns as "the old oracle".
- **Arms and records.** `three_tier_wc_dl`, `three_tier_wc_oracle_v2`, `three_tier_wc_oracle_v2_nu` (label `oracle_v2` from campaign 28). Read `ORACLE_V2_PLAN.md` (design), `ORACLE_V2_HANDOFF.md` (how campaign 16 was run) and `ORACLE_FINDINGS.md` sections 6 to 9.

## 4. Gap-table hints, the first realistic hint (2026-10-31; campaigns 20 and 21)

A hint a real agent can send. The agent framework keeps, for each kind of tool call (the tool plus the command's first word, such as `execute_bash/pytest`), the p10, p50 and p90 of the gaps seen so far, and sends the table and the end of a session with its requests. The engine reads the tool call from its own reply, looks it up and, in this first step, runs oracle v2 on the p50 as if it were exact.

- **With a long queue the p50 is enough.** x10 gaps, 128 users (campaign 20): 20.37 s against 20.27 s with exact hints and 95.83 s for the baseline. A queue wait of about 26 s hides the forecast's errors.
- **With no queue and little host room it costs.** 44 users, 70 GB (campaign 21): 1.95 s against 1.48 s with exact hints and 5.17 s for the baseline, so the p50 keeps 87 % of the gain.
- **Weak points.** One number is acted on as exact, and every gap of 300 s or more was missed and its session marked ended (all 159 in campaign 20). This led to later read timing (entry 5) and to using the whole distribution (entry 7).
- **Arms and records.** Client `HINTS=dist` on `three_tier_wc_oracle_v2_nu` ("old p50"); `three_tier_wc_oracle_v2_nu_table` (the engine reads the table itself). Read `REALISTIC_HINTS.md` (the discussion of 27 and 28 October), `DIST_HINT_DESIGN.md`, `results/campaign20/DECISIONS.md` and `results/campaign21/DECISIONS.md`.

## 5. Just-in-time (JIT) reads and the "owned" arm (2026-11-01 to 11-05; campaigns 24, 23 and 25)

Oracle v2 with reads timed to land just before use, and with the next-turn prefetcher as the one owner of a turn's SSD reads.

- **JIT timing.** A read before arrival starts at the predicted admission minus the read time minus 1 s (on table hints also p50 minus p10 earlier, so it lands near the gap's p10). Alone, on exact hints (`jit_70`), it took oracle v2 from 1.52 to 0.67 s (44 users, 70 GB, 7,410 shared turns).
- **"Owned" adds the gate and the hold.** Gate: a planned turn whose history is in memory makes no SSD lookup at arrival, otherwise at most 3. Hold: a queued turn's read waits until the predicted admission minus the read time minus 2 s. Lookups at arrival that find nothing fell 90 to 93 %. TTFT gained 0.17 s more at 44 users (noise level); at 128 users the GPU queue sets TTFT (12.46 against 12.06 s) and the gain is 29 % fewer SSD pages.
- **Result (campaign 23: 44 users, 70 GB, x10 gaps, 7,768 shared turns).** Baseline 5.17 s, oracle v2 1.48 s, `jit_exact` 0.50 s, `jit_p50` 1.51 s. With a three-minute queue the hold decides: on TraceLab `jit_exact` is about 78 s ahead of oracle v2 (campaign 29; the size is not clean, entry 8).
- **Why superseded.** On table hints it does no better than no hints at all (1.51 against 1.47 s): reads land early and stay pinned (30.9K tokens pinned against 6.2K with exact hints, on about 44K of free host room), and 1,743 of 5,001 read launches are refused room. A fixed later quantile helps only in part: `fq90`, which lands at the p90, reaches 0.94 s against 1.35 to 1.60 s (campaign 26).
- **Arms and records.** `three_tier_wc_oracle_v2_nu_jit` (`jit_70`); `three_tier_wc_oracle_v2_nu_owned` (`jit_exact`, `jit_p50`, `fq90`), the server arm every later arm builds on. Read `CAMPAIGN23_FINDINGS.md`, `results/campaign23/DECISIONS.md` and `results/campaign24/DECISIONS.md`.

## 6. Hint-free and queue-state arms (2026-11-01 to 11-14; campaigns 21, 25 and 26)

Arms that use no gap forecast, built to measure how much of the gain needs a hint at all.

- **Pins only (`three_tier_wc_dl_poll0`, "the hint-free arm").** HiCache plus read pins, with paced retries off. At 44 users and 70 GB it gives 79 % of the gain: 1.51 s against 5.33 s for HiCache and 0.49 s for `jit_exact` (7,410 shared turns). The pins carry it, not the retry setting (campaign 25). At 128 users it keeps about 58 % (29.19 s against 53.12 s for HiCache and 12.06 s for oracle v2, campaign 25), because queued turns lose their history.
- **Queue-state mechanisms (`nofc_noend`, "no hints").** They need only what the engine sees: evict a queued turn's history last, time its read from its queue position (the gate and the hold), and re-read what it lost. Reads start only at arrival, and of the hint the arm uses only which request continues which session. `nofc` adds the end-of-session signal (ended sessions go first).
- **Result (campaign 26: 44 users, 70 GB, 6,715 shared turns, two runs each).** `nofc_noend` 0.71 and 0.72 s, `nofc` 0.69 and 0.80 s, against HiCache at 5.47 and 6.62 s and `jit_exact` at 0.51 and 0.54 s: 96 % of the way with no hint. At 128 users the seven arms that protect queued turns span 20.57 to 20.98 s (all turns of each run's first 6,900 s).
- **Standing.** Not superseded: it is the yardstick for any hint. Exact hints beat it by 0.14 to 0.21 s (shown, two runs a side); table hints under the four rules do not (not shown). With 100 GB of host memory the no-hint arm, `jit_p50` and `jit_exact` tie at 0.41 to 0.42 s, because under 1 % of turns read from the SSD: 30 GB more memory bought 0.52 s, more than any hint.
- **Arms and records.** `three_tier_wc_dl_poll0` (labels `hintfree_*`), `three_tier_wc_poll0`; `nofc` and `nofc_noend` (the `_owned` arm with `--hicache-storage-next-turn-read-timing arrival` and `return_chance` eviction scored `lru` or `queue`). Read `CAMPAIGN23_FINDINGS.md`, `results/campaign25/DECISIONS.md` and `results/campaign26/DECISIONS.md`.

## 7. The four rules, "Pricing the room" (deck 2026-11-06; built 2026-11-11; campaigns 26, 28, 29, 32 to 34)

Use the whole gap distribution, conditioned on how long the session has been idle, and charge host memory a price λ (seconds of TTFT per token-second held) so that each session can decide alone.

- **The rules** (the paper's design section calls them the actions Fetch, Preempt, Retain and Price), with L the tokens to read back:
  1. Fetch: start the read at the time with the lowest expected stall plus λ × L × expected seconds pinned. For a gap with one mode that is once the chance that the turn comes within one read time reaches λL / (1 + λL); λ was fitted at 5e-5.
  2. Soft pins: an arrived turn's read takes room from the staged read least likely to be needed within its own read time.
  3. Retain: evict the session least likely to return within 30 s.
  4. Price: move λ with refused and idle room (as built: with the harm staged reads do to arrived turns).
- **Result (campaign 26: 44 users, 70 GB, 6,715 shared turns).** All four (`ec_full`) 0.66 s; rules 1 and 3 0.66 and 0.78 s; rule 1 alone 0.67 s; rule 3 alone 1.48 s, so the gain is rule 1's. Beside them: `jit_p50` 1.35 to 1.60 s, no hints 0.71 and 0.72 s, `jit_exact` 0.51 and 0.54 s. The rules repair `jit_p50` but are not shown to beat no hints. The learned price and soft pins added nothing on TTFT: soft pins took back 462 of 1,569 staged reads, and 452 of those were read again.
- **At saturation the rules decide nothing.** FreeInference on Triton (campaign 28): the four hinted arms at 47.7 to 49.8 s against HiCache at 115.7 s. FA3 with the write fixes (campaign 34): `ec_exact` 10.66 s, `jit_exact` 10.92 s, `ec_full` 10.96 s, `jit_p50` 11.47 s, HiCache 31.88 s; there `ec_full` started 1 read before arrival and its price never moved (review of 23 November).
- **Why superseded.** The model takes the queue as given, prices host memory only and leaves out the SSD write path, while the measured gain sits after arrival (entry 6) and in the write path (entry 9).
- **Arms and records.** `ec_r1`, `ec_r3`, `ec_r13`, `ec_r134`, `ec_full` (all four rules on table hints) and `ec_exact` (on exact hints, from 2026-11-22): the `_owned` arm with `--hicache-storage-next-turn-read-timing expected_cost`, soft pins, `--radix-eviction-policy return_chance` and price learning. Read `PAPER_PLAN.md` ("Formulation"), `results/design_unified_1126/inputs/deck1_pricing_the_room_1106.txt`, `results/campaign26/DECISIONS.md` and `results/campaign29/DECISIONS.md` ("Both campaigns side by side").

## 8. Two engine defects found on the public traces (2026-11-19 and 11-20; one fixed 2026-11-22)

Defects of the engine, not of a policy, that inflated the differences between the hinted arms in campaigns 28 and 29.

- **Defect 1, the lookup cap (not fixed).** A queued request may ask the SSD three times, and a lookup that finds nothing counts. In the replay most returning requests spend all three on text no tier holds; when their real history later leaves memory, it is recomputed. On TraceLab 175 to 333 requests an arm were refused a read this way and carry about 99 % of the history those arms recomputed (campaign 29). Two fixes are proposed in the record; neither is written.
- **Defect 2, the overdue rank (fixed).** Next-use eviction ranked a turn that had waited over 60 s in the queue as "not coming" and evicted its history right after ended sessions. The fix (commit `94c3dd0c85`, engine copy `engine_queuedfix_20261122`) ranks a queued turn last. Run check on TraceLab's first 30 minutes: requests whose lookups ran out 55 to 70 s after arrival fell from 88 to 6, and those of them refused a read from 57 to 1.
- **What they change.** The 15.6 s lead of the four rules over `jit_p50` on TraceLab (campaign 29) was defect 2 acting through defect 1, not the rules. That every hinted design beats HiCache at saturation stands.
- **Arms and records.** None of their own; the check run is `jit_exact_qfix` in `results/qfix_check/`. Read `OPEN_ITEMS.md` sections 0 to 3.

## 9. FA3 write-path fixes (found in campaign 32, 2026-11-22; verified in campaign 34, 2026-11-23)

On FA3 the arms were separated by the SSD write path, not by the policy. Four engine fixes, each behind an environment switch that is off by default, plus a log of what holds host memory (the census).

- **The loop (campaign 32: FreeInference, 128 users, 160 GB).** A read finds no host memory and is refused; its history is recomputed; write-through writes the recomputed pages to the SSD again (74 to 93 % of writes were pages the SSD already held); the one write thread falls behind and its pending pages hold host memory (515 to 675K tokens); more reads are refused. Every hinted arm ended in it; the policy only set when it started, from minute 42 (`jit_p50`; minute 67 by a stricter test) to minute 84 (`ec_full`). The hinted arms ended at 14.6 to 18.0 s, HiCache at 33.5 s.
- **The fixes.** (1) Do not rewrite a page the SSD holds. (2) Merge queued writes into transfers of up to 128 pages. (3) Reads with a deadline wait while pending writes hold 50,000 host tokens. (4) Retry a read refused host room without spending one of the turn's lookups.
- **Result (campaign 34, same users).** No read was refused host room while the client was sending, pending writes stayed under 90K tokens, and the hinted arms were 5.8 to 9.5 s faster (HiCache 3.06 s: its loss is its lookup budget, entry 1). They are now within about 1 s of each other (10.66 to 11.47 s). On TraceLab the box does not saturate on FA3 and five arms sit at 0.63 to 0.66 s (campaign 33).
- **Limits.** All four fixes ran together, so the run does not say which one is needed (fix 4 never acted). From minute 88 the SSD's cleaner deletes the oldest pages, those of live sessions included, and they are recomputed.
- **Arms and records.** Campaign 34's five arms on engine copy `engine_fa3fix_20261122` with the switches of `results/campaign34/settings.env`. Read `OPEN_ITEMS.md` section 7, `results/campaign32/DECISIONS.md` and `results/campaign34/DECISIONS.md`.

## 10. Revised design of 23 November: T = G + W and five components (2026-11-23; not built)

The design reorganized around one quantity, the time T until a session's prefix cache is next needed on the GPU: T = G + W, with G the remaining gap (the agent knows it, as a distribution) and W the queue wait (the engine knows it).

- **Why.** The records put the gain after arrival and in the I/O path: read pins 3.82 s; reads timed to the queue with queued turns' history kept, 1.02 s more; the write fixes 5.8 to 9.5 s; 30 GB more host memory 0.52 s. Reads before arrival added at most 0.14 to 0.21 s with exact hints and cost 0.6 to 0.9 s on p50 hints; eviction order moved 0.1 s at most.
- **Five components.** (1) A time-to-use estimator: T, and each read's landing time from the queue of reads at the SSD. (2) Read release: one rule for when a read starts, before or after arrival, and a pin until admission. (3) A host memory manager: what holds the room, and one ranking of what may leave. (4) An I/O scheduler: reads by deadline, writes deduplicated and merged. (5) The SSD tier: evict by the same order and spare live sessions.
- **What the review of the same day found (CPU only).** The note's "work ahead" model of W is not needed: queue position over admission rate is unbiased for turns that do not read, and a turn that reads is admitted at the later of its GPU turn and its read's landing. The paper's threshold rule for Fetch misfires here, because the trace's gaps have a falling return rate (1.81 s of cost against 1.10 s for reading at arrival). Room is the largest host lever: under write-through 0.65M of the 1.63M host tokens only mirror the GPU. At saturation a GPU-second of recompute cost about 94 TTFT-seconds (campaigns 32 and 34, minutes 90 to 109).
- **Why superseded.** It is an architecture with no statement of optimality. On 2026-11-26 the author asked to keep the paper's four actions and have them cover the queue, the hint and the write, provably optimal under a stated model.
- **Arms and records.** No arm. Read `agent_cache_design_discussion_2026-11-23.md`, `results/design_review_1123/critique.md` (its `README.md` lists the other files) and `results/design_unified_1126/inputs/deck2_revised_1123.txt`.

## 11. Unified four-action design, "Four actions, three prices" (2026-11-26; measured 2026-11-27)

Every movement of KV between tiers is a job with a due time, T = A + W: the session's next arrival A, known as a distribution (the hint), plus the queue wait W. GPU time, host memory and the SSD each have a price in seconds of TTFT (μ, λ, κ), a second of one turn's stall weighs c, and each of the paper's four actions is a comparison at those prices, on every tier.

- **The actions.** Fetch: when a copy moves up (after arrival a read lands at the q-quantile of admission, q = λL / (c + λL); before arrival it starts early only on a sharp hint; one tier up, the prefill pass no longer waits on the host-to-GPU copy, the "load hold"). Retain: which copies each tier keeps, by one index (the Gittins index of the gap), with the SSD write as retention one tier down and the SSD cleaner as eviction by the same index. Preempt: who yields now (a turn whose read or copy has not landed is passed over; room is taken in index order). Price: read off the queues, not tuned.
- **What is proved.** 21 theorems and two propositions (`results/design_unified_1126/PROOFS.md`, part 0): derived, re-derived by a referee, checked numerically. The queue enters through one identity: with the GPU pool full, mean TTFT = (N − B) g − Z′, so a cache policy moves the mean only through GPU-seconds per request. Not proved: index order for many sessions under a hard capacity (1 % from the priced bound at 512 sessions in simulation).
- **What is built** (every rule behind a switch, off by default): the load hold; Fetch and Price by measured prices; Retain on the SSD (pages in use kept young, ended sessions' own pages first, their last turn not written, sessions ended only by their hint); queued turns evicted latest arrival first; the admission pass-over. Reviewed twice by independent agents; every defect found was fixed with a test.
- **Measured** (same users and requests, paired with `ec_full`; two runs of `ec_full` differ by 0.01 s): everything on is faster in all four settings: −1.79 s (16 %) at campaign 34's setting, −2.49 s (22 %) with the SSD cleaner from minute 50, −3.39 s (26 %) with 112 GB of host memory (tail cut by a third or more, median up from 0.26 to 2.99 s), −0.14 s (21 %) with 64 users. The load hold alone is −1.61 s (4.8 % more decode steps a second); the cache rules alone −0.38 s and −0.99 s, all from the SSD's Retain rules once the cleaner deletes (the simulator: −0.28 and −0.93 s). The measured price of host room is too low at saturation and stages reads early, yet the fixed price ties (+0.04 s); the admission pass-over ties on the mean (−0.02 s) and trades a lower median for a heavier tail.
- **Records.** `results/campaign35/DECISIONS.md` (campaigns 35 to 45, findings 1 to 10); `results/compare_unified/` (paired tables, mechanisms, figures, `summary.png`); `results/design_unified_1126/DESIGN.md`; the "Four Actions, Three Prices" deck.

## Also on record

- **Replay client.** Rebuilt on 2026-10-17 (`REPLAY_REDESIGN.md`): runs from campaign 18 on do not pair with campaigns 13 to 17.
- **Write-back.** An arm that copies a page to host memory only when the GPU evicts it (`three_tier_wc_wb`) was held back in campaign 16: in its smoke the SSD was written but never read (0 hits in 1,281 lookups), because the engine looks up the SSD only under a match that host memory backs (`ORACLE_FINDINGS.md` section 6). It matters for any plan to drop the host copy of pages the GPU still holds (entry 10).

## Summary

| # | Iteration | Date (2026) | Arm label(s) | Headline result | Record |
|---|---|---|---|---|---|
| 1 | Original HiCache | 10-08, reference run | `three_tier_wc` (`hicache`) | 18.5 s at x70 (campaign 14); 31.88 s on FA3 (campaign 34) | `ORACLE_FINDINGS.md` |
| 2 | First oracle | 10-09 | `three_tier_wc_oracle` | 20.8 against 22.2 s (campaign 15) | `ORACLE_PLAN.md` |
| 3 | Oracle v2 | 10-11 | `three_tier_wc_dl`, `_oracle_v2`, `_oracle_v2_nu` (`oracle_v2`) | 13.2, 11.6 and 5.7 s against 18.5 s (campaign 16) | `ORACLE_V2_PLAN.md`, `ORACLE_FINDINGS.md` |
| 4 | Gap-table hints | 10-31 | `HINTS=dist` on `_oracle_v2_nu` | p50 1.95 s against exact 1.48 s (campaign 21) | `DIST_HINT_DESIGN.md` |
| 5 | JIT reads, "owned" arm | 11-01 | `_oracle_v2_nu_owned`: `jit_exact`, `jit_p50`, `fq90` | 0.50 s exact, 1.51 s p50, baseline 5.17 s (campaign 23) | `CAMPAIGN23_FINDINGS.md` |
| 6 | Hint-free, queue-state arms | 11-01 to 11-14 | `three_tier_wc_dl_poll0`, `nofc`, `nofc_noend` | 0.71 s with no hints: 96 % of the way to exact hints (campaign 26) | `results/campaign26/DECISIONS.md` |
| 7 | Four rules | 11-06 deck, 11-11 built | `ec_r1` to `ec_full`, `ec_exact` | `ec_full` 0.66 s (campaign 26); 10.96 s on FA3 (campaign 34) | `PAPER_PLAN.md`, `results/campaign26/DECISIONS.md` |
| 8 | Two engine defects | 11-20 found, 11-22 one fixed | check run `jit_exact_qfix` | requests it cost a read: 57 before the fix, 1 after (run check) | `OPEN_ITEMS.md` |
| 9 | FA3 write-path fixes | 11-22 to 11-23 | campaign 34's arms, engine `engine_fa3fix_20261122` | 5.8 to 9.5 s faster on the same users (campaign 34 against 32) | `results/campaign34/DECISIONS.md` |
| 10 | T = G + W, five components | 11-23 | none | not run; reviewed on CPU | `agent_cache_design_discussion_2026-11-23.md`, `results/design_review_1123/critique.md` |
| 11 | Four actions, three prices | 11-26 design, 11-27 measured | campaigns 35 to 45: `ec_full`, `ec_full_load`, `unified`, `unified_load`, `unified_load_po`, `unified_load_fixedprice` | −1.79 s (16 %) at campaign 34's setting, −2.49 s with an early cleaner, −3.39 s with 112 GB, −0.14 s with 64 users; the load hold alone −1.61 s | `results/design_unified_1126/DESIGN.md`, `PROOFS.md` |
