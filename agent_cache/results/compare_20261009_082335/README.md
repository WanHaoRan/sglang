# Oracle smoke C — next-turn prefetch case 2 on a dense queue, observe vs on (2026-10-09)

Step-7 smoke of `../../ORACLE_PLAN.md` (run C of `frozen/oracle_scripts_staging/oracle_smokes.sh`). Case 2: when a
hinted turn waits in the queue and part of its context has left GPU and host memory, re-issue its prefetch early enough
that the pages are back in host memory by admission. Engine at `294c82b7fc`, scripts as applied 07:00Z. Both arms use
the same client with oracle hints on (`ORACLE_HINTS=1`: each turn carries a `sglang.next_turn` kv_hint with the
session's next rid and the recorded gap), so they differ only in the server's `--hicache-storage-next-turn-prefetch`
mode.

| arm | server |
|---|---|
| `three_tier_wc_ntpobs` | `three_tier_wc` (HiCache L1 + write-through L2 + nixl L3 on the SSD, wiped before boot; `wait_complete` prefetch; #39283 admission re-query and paced retries, 8 x 8) with next-turn prefetch in `observe` mode: it plans every hinted turn, runs the estimators and logs each `ntp_*` event, including when a read would be due, but never reads |
| `three_tier_wc_oracle` | the same in `on` mode: stages a read before the turn arrives (case 1), and asks the scheduler to re-issue a queued turn's prefetch once its predicted admission is no further away than the estimated read time (case 2) |

`ntpobs` is the oracle arm without the actions ("ntp" = next-turn prefetch, "obs" = observe). It is the control for
`three_tier_wc_oracle` and the arm that validates the estimators (plan experiment E1). Run A
(`../compare_20261009_070112/`) showed that observe does not change serving: TTFT, tiers and per-turn prefetches match a
mode-off arm.

Workload: 32 conversations of the 262K agentic trace (offset 32), 12 turns each, all 32 live from the start (closed
loop, concurrency 32), recorded gaps x1 capped at 1,200 s, 900 s client cap. Qwen3-30B-A3B-Instruct-2507, bf16 KV, TP 1
on one H200 (the cookbook's H200 config), context 131,072. The pools are shrunk so that 32 sessions evict: L1 65,536
tokens (`--max-total-tokens`), host 12 GB = 122,112 tokens, L3 cleaner at 70%/60%.

Files: `compare.png`, `compare.csv`, `timeline_<arm>.png`, `turns_<arm>.csv`, `events_<arm>.csv`, `oracle_eval.csv`,
`oracle_rids_<arm>.csv`. Regenerate from `agent_cache/`: `python3 scripts/timeline.py --manifest
results/compare_20261009_082335/manifest.txt` and `python3 scripts/oracle_eval.py --manifest
results/compare_20261009_082335/manifest.txt --out results/compare_20261009_082335`. The breakdowns below come from
`frozen/oracle_scripts_staging/analyze_c*.py` (git-ignored): `python3 frozen/oracle_scripts_staging/analyze_c.py
compare_20261009_082335`, or `../frozen/oracle_scripts_staging/c_pairs/<c2|c3|c4>` for a rerun paired with this observe
arm.

## Result

Whole run, returning turns (352 per arm).

| | three_tier_wc_ntpobs | three_tier_wc_oracle |
|---|---|---|
| turns / errors | 384 / 0 | 384 / 0 |
| 95% of turns done | 268.7 s | **215.1 s** |
| TTFT mean / p50 / p90 / p99 (s) | 16.76 / 20.67 / 33.88 / 40.62 | **13.13 / 16.85 / 25.91 / 31.24** |
| server queue wait, mean (s) | 16.13 | 12.77 |
| returns: device / host / SSD / recompute | 109 / 76 / 64 / 103 | 80 / 71 / 156 / 45 |
| uncached tokens per returning turn | 4,465 | 2,322 |
| tokens s2h / h2d / d2h (= h2s) | 0.89M / 0.71M / 1.67M | 1.54M / 1.47M / 0.93M |
| tokens evicted from host | 2.45M | 3.20M |

Same (conversation, turn), oracle − observe: **−3.63 s [95% CI −4.13, −3.13]**, p50 −1.11 s; 196 turns faster by more
than 0.5 s, 20 slower, 136 within 0.5 s. Queue wait, paired: −3.36 s. That CI covers per-turn noise only: two runs of
identical code differed by 0.98 s on the same turns, and across four oracle runs oracle − observe ranges from −2.66 to
−3.63 s (Reruns, below).

## What it shows

1. **Case 2 works: queued turns get their evicted context back from the SSD instead of recomputing it.** Recomputes fall
   from 103 to 45 and SSD restores rise from 64 to 156 (63 of the observe arm's recomputes became restores), so the
   uncached tokens per returning turn halve. Of the 156 restores, 127 came from the oracle's re-issued prefetches, 27
   from #39283's admission re-query and 2 from case-1 staging; in the observe arm 61 of 64 came from the re-query. The
   re-query has less left to catch: 80 → 44 requests saw their device prefix shrink before admission, and 139 → 104 had
   spent the re-query's 8 attempts. 238 re-fetches were issued and 202 of them hit the SSD. Reads land p50 1.0 s and p90
   3.3 s before admission; 2 of the 162 turns the oracle acted on were late (admitted no more than 50 ms after their
   last read ended).
2. **The TTFT gain is shorter queueing.** The paired queue wait falls 3.36 s of the 3.63 s: less prefill (uncached
   tokens −48%) drains the queue faster for every turn. Turns that recompute in both arms still gain 6.4 s (n 40), and
   turns that moved from a device hit to an SSD restore gain 4.0 s (n 33). Turns restored from the SSD in both arms gain
   5.3 s (n 55), more than the queue average: in the observe arm 61 of 64 restores came from the admission re-query,
   which holds the turn at the queue head while its read runs.
3. **The gain grows with the context.** By turn index: 0.0 to −0.2 s for turns 1-4, whose contexts still fit (73 of the
   76 host hits are turns 2-4), then −0.9 s at turn 5, −3.0 at 6, −6.7 at 7, −6.3 at 8, −6.1 at 9 and −8.5 / −8.3 at
   turns 10 / 11. The observe arm's recomputes start at turn 5 (one at turn 4). Per-turn-index means vary a lot between
   runs (turn 9: −6.1, −5.2, −2.5 and −2.9 s in the four oracle runs); the rise from turn 5 to turns 10-11 holds in all
   four.
4. **The oracle reads and loads more, and writes less.** s2h +73% and h2d +105%, while d2h and h2s fall 45%: a turn
   restored from the SSD writes back only its new tail, whereas a recomputed turn rebuilds its whole context, and
   write-through backs every new page up to host and SSD again. Host evictions rise 31%, and 33 turns that were device
   hits in the observe arm were SSD restores here, consistent with the re-fetched contexts pushing other sessions' pages
   out. Those turns still came out faster (item 2).
5. **Most re-evictions were a stale probe, now fixed.** The arm logged 144 `ntp_reevict` lines (40 at the 2-read cap).
   143 came a median 0.09 s after the rid's last read ended (p10 0.00 s, p90 1.75 s), and 130 of 144 reported exactly
   the missing count of the read before them. `_wants_refetch` judged a re-eviction by the residency probe taken on the
   0.5 s cadence, which could predate the landing of the read it was judging, and so re-fetched pages that were already
   back. Of the 104 uncapped re-evictions, 18 re-issues found under 256 tokens left to read and were declined, and 33
   looked up less than half the span: only the tail beyond what had just landed. 8 were declined at issue and 45 asked
   for at least half the span: the real re-evictions. Each of the 33 issued false re-fetches counted toward the 2-read
   cap, and 16 of the 45 remaining recomputes had hit the cap. 30 of the 33 tail-only lookups missed the SSD (a turn's
   new tail is fresh tool output, never on the SSD), which accounts for most of the 34 issued re-fetches that missed and
   never hit later. Fix: `_wants_refetch` probes again right before it decides
   (`python/sglang/srt/mem_cache/next_turn_prefetch.py`; regression test
   `test_a_landed_read_is_not_taken_for_a_re_eviction`). Reruns below.
6. **Real re-evictions remain, in a small host-only region.** Under write-through each device-resident node keeps a host
   copy that host eviction cannot take, so only host − L1 = 56,576 tokens can hold host-only pages. A re-fetched context
   waiting in the queue is an unlocked leaf there, next to every other session's evicted context. At this run's eviction
   rate (3.2M tokens in 448 s, ~7.1K tokens/s) that region turns over every ~8 s, and another turn's read can take a
   context milliseconds before its admission (Reruns, item 4). This is a small-pool effect, but it is the eviction
   cascade the plan's section 7 warns about.
7. **The estimators err early, and early is the safe side here.** `L̂` overestimates the read by 0.72 s (p50 of read −
   estimate). The queue-length wait estimate, which places the case-2 trigger, under-predicts the wait: MAE 4.4 s, bias
   −2.2 s (observe arm: 5.1 s, −3.2 s). Both errors move the trigger earlier, so reads land a second early and only 2
   were late. The cost of early is time exposed to re-eviction (item 6). The wait EWMA beats the queue-length estimate
   in the observe arm (MAE 4.4 s, bias −3.1 s) and is much worse with the oracle on (7.5 s, −7.1 s). Arrival prediction
   is tight: |err| p50 115 ms (observe arm) and 67 ms, p90 269 / 234 ms.
8. **Case 1 hardly runs at x1 gaps.** The next turn arrives seconds after the previous one finishes, leaving no window
   to stage before arrival: 2 case-1 dues and 2 staged reads. Case 1 is covered by B3 (`../compare_20261009_092437/`).

## Reruns: the re-eviction fix, and run-to-run spread

Three more oracle arms on the same knobs and client, each on a cold L3 and paired with this directory's observe arm. C2
and C3 run the fix (working tree). C4 runs the pre-fix code from a frozen copy (`frozen/engine_294c82b7fc_stale_probe`:
`294c82b7fc`'s `next_turn_prefetch.py`, otherwise the same tree). Compare dirs: `../compare_20261009_150239/` (C2),
`../compare_20261009_151214/` (C3), `../compare_20261009_152319/` (C4); drivers
`frozen/oracle_scripts_staging/oracle_smoke_c{2,3,4}.sh`.

| | observe | pre-fix C (above) | pre-fix C4 | fixed C2 | fixed C3 |
|---|---|---|---|---|---|
| TTFT mean / p50 / p90 / p99 (s) | 16.76 / 20.67 / 33.88 / 40.62 | 13.13 / 16.85 / 25.91 / 31.24 | 14.10 / 16.67 / 28.23 / 33.92 | 13.73 / 15.76 / 28.65 / 35.48 | 13.64 / 16.59 / 27.99 / 35.09 |
| paired vs observe (s) | | −3.63 [−4.13, −3.13] | −2.66 [−3.11, −2.22] | −3.03 [−3.51, −2.54] | −3.12 [−3.64, −2.58] |
| returns: device / host / SSD / recompute | 109 / 76 / 64 / 103 | 80 / 71 / 156 / 45 | 87 / 71 / 148 / 46 | 91 / 72 / 144 / 45 | 86 / 73 / 147 / 46 |
| re-fetches issued / missed the SSD | | 238 / 36 | 214 / 35 | 191 / 0 | 207 / 3 |
| `ntp_reevict` (capped) | | 144 (40) | 118 (34) | 71 (19) | 82 (21) |
| read end to re-eviction, p50 | | 0.09 s | 0.16 s | 0.97 s | 0.78 s |
| uncapped re-evictions: span / tail only / under 256 left / declined at issue | | 45 / 33 / 18 / 8 | 34 / 29 / 20 / 1 | 42 / 0 / 0 / 10 | 55 / 0 / 0 / 6 |
| recomputes: capped / a re-fetch hit the SSD | | 16 / 4 | 16 / 3 | 12 / 13 | 7 / 10 |
| tokens s2h / evicted from host | 0.89M / 2.45M | 1.54M / 3.20M | 1.58M / 3.29M | 1.67M / 3.59M | 1.53M / 3.39M |

1. **The fix removes the stale-probe artefacts.** No tail-only lookups or too-short declines are left, issued re-fetches
   almost never miss the SSD, and re-evictions come 0.8-1.0 s after the read instead of 0.1-0.2 s. Every uncapped
   re-eviction left either looked up the span or was declined at issue. The declines cluster in passes that re-fetch
   several turns at once: `tick()` checks the read limiter before the scheduler issues any of them.
2. **TTFT does not change.** Fixed minus pre-fix, two runs each, per turn: +0.07 s [−0.17, +0.32]. C's oracle run (13.13
   s) was the best of the four; the same code ran 14.10 s in C4.
3. **Run-to-run spread is about 1 s, more than the per-turn CIs suggest.** C4 − C, identical code: +0.98 s [+0.61,
   +1.32] on the same turns (C3 − C2: −0.10 s [−0.55, +0.37]). In a closed loop a small timing difference reroutes
   evictions, so a bootstrap over turns covers only within-run noise. The oracle beats observe in all four runs; a
   difference under ~1 s between single runs is not evidence here. The observe arm ran once.
4. **What the fix exposes: re-fetched pages are evicted again just before admission.** With the cap no longer spent on
   false re-evictions, fewer turns end capped and more recompute although a re-fetch hit the SSD (13 / 10 vs 4 / 3).
   Example, C2's `c2-t8` at queue position 1: its re-fetch (11,200 tokens) is in host memory at 15:06:12.589 ("HiCache
   prefetch success ... loaded=11200"). Reads for two other queued turns then evict host pages: `c25-t10`'s arrival
   lookup (26,048 tokens at 12.909) and `c1-t9`'s re-fetch (11,200 at 12.982). `c2-t8` is admitted at 13.011 with only
   the shared 7,808-token prefix and recomputes 11,335 tokens. Neither safety net can catch this. #39283 compares the
   admission match with the match at the request's last `_prefetch_kvcache`, and the re-fetch itself set that to the
   shrunken length. The prefetcher's 0.5 s re-probe cannot see an eviction 30 ms before admission. Holding a host lock
   on re-fetched (and staged) pages until admission, the plan's "unprotected after insert" risk (section 7), is the
   change to try. Why the eviction took 0.3 s-old pages was not established; with 56,576 tokens of host-only room, one
   26K-token eviction frees nearly half of it.

## Notes and deviations

- A smoke, not a measurement: 32 sessions, 352 returning turns, one observe run. Not a deployment configuration: the
  pools are shrunk (65,536-token L1, 12 GB host) to force eviction at 32 sessions; the rest follows the cookbook's TP-1
  bf16 H200 setup for this model.
- An SSD restore here is a turn with storage credit and at least half its prompt cached. One more oracle turn (157) had
  storage credit with less than half the prompt cached, and counts as a recompute.
- "Replacing unresolved storage-hit accounting" goes from 0 to 107 lines with the oracle, only on requests read from the
  SSD twice (a re-fetch after the request's own read). 84 re-credit the identical span; the rest move the credit to the
  later read, net 5,952 of 1.2M tokens. Reporting only.
- Re-eviction classes come from the scheduler pass after each `ntp_reevict`: "span" and "tail only" are issued lookups
  for at least or under half the evicted count, "under 256 left" is a `prefetch_start` below the prefetch threshold, and
  "declined at issue" is a larger `prefetch_start` that started no read (limiter or a pending lookup).
