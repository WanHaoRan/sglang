# Campaign 20: the p50 of the online table at x10 (first step of DIST_HINT_DESIGN.md)

Requested 2026-10-31: "do the smoke test and evaluation with only the x10 gap with 5h timeout -- the same as the previous
campaign 19 workload. Leave out the x70 case."

**The question:** how much of the oracle's gain at x10 survives when the exact gap is replaced by the p50 of the client's
online table?

**The arm.** `three_tier_wc_oracle_v2_nu` on the frozen engine `frozen/engine_5df71c4bf0_v2/python`, as in campaign 19. The
only change is the client: `HINTS=dist ORACLE_HINTS=0` (`HINT_SOURCE=online`, the default, tool + first word keys, explicit
ends). The engine reads the entry's p50 as the version 1 `next_gap_s` and ignores the version 2 table. No new baseline or oracle
arm: campaign 19's (`../campaign19/`) are the comparison, same users, seed and arrival times.

**The workload.** Campaign 19's: `USERS=768 TURNS=50 ARRIVAL=0.1 GAP=10 C=128 SEED=0 CLIENT_TIMEOUT=18000` on
`traces/lmcache_agentic_trace_262k.json`.

**Cookbook.** `docs/cookbook/autoregressive/Qwen/Qwen3.mdx` read before running. This run stays pinned to campaign 19's server
for comparison, which differs from the page in the usual ways: no `--tool-call-parser` or `--reasoning-parser` (the replay sends
no tools and thinks off), `--context-length 131072`, one GPU, and the HiCache flags the page does not cover.

**Launcher.** `frozen/oracle_scripts_staging/c20_dist/c20_dist_x10.sh smoke|full` (log `launcher.log`). Smoke acceptance is
`c20_dist/check_smoke.py`: 0 errors and tracebacks, every hint with a point has a plan with that gap, no plan without one, and
the list grows during the run.

## Smoke (2026-10-31 20:22-20:44Z, `manifest_smoke.txt`)

Passed. 32 users x up to 12 turns at x10, through `run_compare.sh`: 363 turns, 0 client errors, 0 aborted, 0 tracebacks, every
boot check matching. 329 turns carried a point and the server made exactly those 329 plans, each with the client's p50 as its
gap; the 34 turns without a point (29 with no table entry yet, 5 session ends) made none. The table grew from 1 to 12 entries,
and 48.3 % of true gaps fell at or below the sent p50.

## Full run (2026-10-31 20:45Z to 2026-11-01 01:30Z, `manifest.txt`)

**The answer: at x10 the whole gain survives, but x10 is too forgiving to say the p50 is good enough in general.**

Accepted (`all/acceptance.txt`): every user finished, 24,879 turns, 0 errors, 0 aborted, 0 tracebacks, pools as profiled. All
24,082 plans carried the client's p50. Every plan was admitted (23,706) or expired (376), and all 376 of those turns still
came. Campaign 19's rule "every plan arrives" cannot hold under a forecast, so it became this one. The turns, gaps and replay
prompts are the same as campaign 19's on all 24,879 turns.

TTFT rows are over the 15,033 turns all three runs completed; the rest are whole-run numbers.

| | Baseline (c19) | Oracle (c19) | p50 (c20) |
|---|---:|---:|---:|
| Mean TTFT | 95.83 s | 20.27 s | 20.37 s |
| TTFT p50 / p90 / p99 | 125.9 / 172.3 / 198.8 s | 24.2 / 36.0 / 43.0 s | 23.5 / 36.4 / 43.4 s |
| Paired difference from the baseline, resampling users | | -75.55 [-80.33, -70.60] s | -75.46 [-80.35, -70.54] s |
| Same, resampling 10-min blocks | | [-105.41, -49.84] s | [-105.27, -49.81] s |
| Users finished / never started | 413 / 232 | 768 / 0 | 768 / 0 |
| Last user finished | 18,196 s | 16,907 s | 16,905 s |
| Mean wait to start | 5,748 s | 732 s | 738 s |
| Turns/s, 2,100-9,000 s | 0.72 | 2.27 | 2.27 |
| History from GPU / host / SSD / recomputed | 37 / 13 / 23 / 27 % | 35 / 37 / 28 / 0 % | 34 / 38 / 27 / 0 % |
| Turns recomputing more than a page | 36.5 % | 0.7 % | 0.8 % |
| SSD read / written | 8.28 / 10.39 TB | 16.64 / 2.16 TB | 16.58 / 2.16 TB |
| Planned turns' admission wait p50 / p99 / max | | 26.0 / 50.0 / 71.8 s | 26.1 / 52.1 / 66.4 s |

The p50 against the oracle, paired: +0.09 s on those 15,033 turns (users, [+0.03, +0.16]); +0.045 s on all 24,879 (10-min
blocks, [-0.116, +0.196]). Neither interval holds the spread between runs, which has never been measured at x10, and single
turns differ by a median 2.5 s between the two runs. So the small difference means nothing either way. All of the -75 s is the
baseline's storm: turns the baseline sent before it started (2,280 s) contribute +0.04 s.

**How wrong the forecast was.** The sent p50 is at or above the true gap on 45.1 % of turns. |true - p50|: median 0.22 s, mean
9.24 s, within 30 s on 96.3 %. In the congested part (2,280-10,800 s), arrival error of live plans: p50 0.50 s, p90 8.72 s, p99
40.44 s; the oracle's 0.06, 0.24, 1.02 s.

**TTFT by true gap** (same turns, 24,111 returning; `all/score.txt`):

| True gap | Turns | Oracle | p50 |
|---|---:|---:|---:|
| 0-5 s | 7,369 | 23.53 s | 23.19 s |
| 5-15 s | 13,611 | 23.64 s | 23.86 s |
| 15-60 s | 2,568 | 24.56 s | 25.05 s |
| 60-120 s | 324 | 24.86 s | 24.15 s |
| 120-300 s | 80 | 23.53 s | 23.70 s |
| 300 s or more | 159 | 24.42 s | 22.91 s |

**What the wrong forecasts cost.**
- 376 plans (1.6 %) expired and ended their sessions in the engine, while the agents were still working: all 159 gaps of 300 s or
  more, and 376 of the 563 gaps of 60 s or more (true gaps 66-1,200 s, median 205 s; sent p50 median 7.6 s). The expected loss on
  long gaps did not appear: those turns' TTFT is 22.79 s against the oracle's 23.49 s.
- 38 of those turns recomputed more than a page, against 10 (mean recompute 111 against 101 tokens per turn).
- 246 of them had a read staged and thrown away: 4.77M tokens, 0.47 TB at 96 KiB per token. Total SSD reads did not rise.
- Reads landed closer to admission, a median 6.6 s before it against 12.0 s, and 85 of 8,594 were late against 16 of 9,075. The
  queue wait covered this.

**Why nothing was lost (`all/why.txt`, `all/details.txt`).**
- The queue wait is much longer than the error: 29.6 s mean in the congested part, and TTFT is mostly that wait.
- Reads start at the same moment. A read comes due once its history is gone and the predicted admission is within 120 s. For gaps
  under 60 s (97.7 %) that is about 2.1 s after the previous turn in both arms, when next-use eviction takes the just-finished
  session (`ORACLE_FINDINGS.md` section 9.11). Only long gaps differ: for 300 s or more the oracle's read comes due at 595 s and
  the p50's at 2.2 s.
- The gaps are easy at x10: p10 / p25 / p50 / p75 / p90 of 0.6 / 1.0 / 7.1 / 7.5 / 17.7 s. By calculation, not a run, one fixed
  7.1 s for every turn is within 30 s on 95.0 % of turns against the table's 96.3 %. Its median error is 4.86 s against 0.22 s.

**How the engine uses the gap, and why it rarely matters here.**
- TTFT is the queue wait: in the congested part TTFT minus admission wait is 0.44 s. Both the GPU and the SSD are at their limits there:
  2.27 turns/s, and the SSD 98 % busy at 1.86 GB/s in both runs (the 53 % often quoted is the whole-run mean). So a forecast could hurt by
  adding GPU work or SSD reads, and it added neither: recompute stayed at 43 tokens per turn (oracle 45, baseline 6,545) and total reads at
  16.58 TB against 16.64 TB.
- When to read: a read is issued once the predicted admission is within 120 s (`NTP_LEAD_HORIZON_S`), and the read scheduler launches
  earliest deadline first whenever it has room. With a 30 s wait, any forecast under about 90 s means "read now". One sent p50 was over
  90 s (largest 92.8 s); 349 of 24,111 true gaps were (1.4 %).
- What to evict: among sessions due within 300 s, the one due last goes first, and due time is gap + queue wait. The session that just
  finished is due last with either number. By the rule, an error of about 5 s (p10 / p90 of true gap - p50: -5.29 / +5.19 s) only swaps
  it with sessions that finished around the same time; the staged-turn sets differ a lot while the totals match (see "Logged").
- When to give up: expiry ends the session and evicts it first, but an exact hint also evicts a long-gap session early (the "far" tier).
  Both runs read those turns back from the SSD: 19.0K against 19.1K tokens per turn for gaps of 300 s or more.

**Tested by the server's queue when the turn was sent (`all/why_cheap.txt`, same turns, queue in that range in both runs).**

| Queue | Turns | Oracle TTFT | p50 TTFT | SSD tokens per turn, oracle / p50 |
|---|---:|---:|---:|---:|
| 0 | 197 | 0.51 s | 0.51 s | 427 / 427 |
| 1-9 | 3,021 | 0.47 s | 0.48 s | 63 / 64 |
| 10-39 | 1,788 | 11.59 s | 11.69 s | 3,123 / 2,586 |
| 40 or more | 18,171 | 29.85 s | 29.91 s | 8,933 / 8,729 |

- With 40 or more queued, the 274 turns of expired plans lost nothing (30.64 s against 29.74 s). 268 read after arriving: a median
  2.96 s, p90 4.36 s, against a median TTFT of 29.09 s, and all 268 finished before the first token.
- With no queue they pay: 15 turns, all gaps of 300 s or more, mean TTFT 0.45 s against 0.66 s, p90 0.74 s against 1.30 s, recompute 53
  against 99 tokens per turn. A small sample, but it is the cost with nothing covering it. The other 182 no-queue turns show no change,
  but they read nothing from the SSD, so they do not test the forecast. Queue 1-9 is mostly before eviction began: no test either.

So the p50 is cheap here mainly because of the congestion. By the same reasoning one fixed number should do about as well; not measured.

**What this run does not show.**
- **x70.** Its queue is nearly empty, so nothing covers a late read, and gaps of 300 s or more hold 35 % of idle time. x10 also
  cannot rank methods better than the p50, since a better forecast has nothing left to gain here.
- **Whether hints produce the gain at all.** This server also turns off paced retries and pins reads, which need no hints, and the
  gain is one absent storm. The hint-free arm (`ORACLE_FINDINGS.md` sections 8.7 and 11, "Separate the retry change at x10") has
  never been run. With accuracy making no difference here, it is the run that would show what the hints add at x10.
- **The real engine path.** The engine still takes the p50 as a version 1 action; reading the table and the reply is not built.

**Logged, not chased.**
- The 60 s wait rule is not met, as in campaign 19: 51 turns, longest 66.4 s, at 9,851-10,103 s. The second SSD cleanup was at
  10,062 s, so 24 of them came before it; campaign 19's 67 all came after its cleanup. Both arms peak in the same bin, 165-170
  min (5-min mean TTFT 47.7 s and 50.6 s). This run's waits were not traced as section 9.11 traced campaign 19's.
- Only 5,844 turns were staged in both arms, with 2,677 in the oracle's alone and 2,556 in the p50's alone, so eviction differs a
  lot turn by turn and not in aggregate. No repeat exists to say how much of that is run-to-run.
- Server-counted `prompt_tokens` differ on 4,779 turns (mean -0.64 tokens); `replay_prompt_tokens` are identical on all turns.
- The launcher's `tar --exclude` came after the paths and did nothing, so `scripts_at_launch.tgz` also holds `__pycache__`.
- Over all 24,111 returning turns the p50 run's p99 is 52.63 s against 50.44 s, with 391 turns over 50 s against 251, mostly in 145-170
  min; p99.9 goes the other way (63.28 s against 68.07 s). No repeat exists to say whether this is the p50 or run-to-run.

Scripts in `frozen/oracle_scripts_staging/c20_dist/` (`accept.py`, `score.py`, `why.py`, `details.py`, `compare_c19.py`, `why_cheap.py`,
`figure_notes.py`), outputs and `c20_overview.png` in `all/`, `c20_summary.txt` from `scripts/c16_summary.py`, per-arm files in
`three_tier_wc_oracle_v2_nu/`.

## Which workload would expose the forecast's cost (analysis, no run; `all/x70_where.txt`)

Campaign 18's x70 workload, unchanged, with `HINTS=dist`: one 5 h run, because its baseline and oracle arms exist. The calculation checks
out where it can be checked: it gives 1.6 % expired at x10 (measured 1.56 %) and 1.51 turns/s at x70 (measured 1.34-1.71).

| Gap scale | GPU load | Plans expired | Gaps over 120 s | Sent p50 over 120 s | Users finishing in 5 h |
|---:|---:|---:|---:|---:|---:|
| x10 | 218 % | 1.6 % | 1.0 % | 0.0 % | 768 |
| x40 | 98 % | 5.3 % | 5.8 % | 1.4 % | 768 |
| x70 | 66 % | 6.9 % | 12.0 % | 9.5 % | 768 |
| x100 | 50 % | 7.3 % | 20.6 % | 14.0 % | 639 |
| x200 | 29 % | 7.6 % | 69.2 % | 75.4 % | 366 |

By calculation for 128 users: GPU load against 2.3 turns/s, with an 11 s reply; x70 measured 747 users finished.

Read it on calm turns, not on the overall mean. 81 % of x70's -0.52 s comes from turns sent with 5 or more requests queued, in the baseline's
spikes, and identical smoke runs have differed by more than the gain (`ORACLE_FINDINGS.md` section 9.7). On turns sent with under 5 queued in
both arms, gaps under 60 s gain nothing (-0.02 s), and gaps of 60 s or more show a sharp signal (4,189 turns, 17.5 %):

| True gap | Turns | Baseline median | Oracle median | Baseline mean | Oracle mean |
|---|---:|---:|---:|---:|---:|
| 60-120 s | 1,888 | 0.88 s | 0.33 s | 1.12 s | 0.70 s |
| 120-300 s | 1,518 | 1.07 s | 0.33 s | 1.30 s | 0.74 s |
| 300 s or more | 783 | 1.07 s | 0.40 s | 1.33 s | 0.91 s |

Expected, not measured: about 1,660 turns (6.9 %) expire and read on arrival, which alone is about 0.04 s on the overall mean; losing the
whole band would be about 0.08 s of the 0.52 s gain. Worse eviction order is on top and not estimated here (the model in the next section
puts the total near 0.24 s): with no queue wait to dominate it, the forecast sets the order. Watch the GPU share of history, 58 % in the
oracle arm against 46 % in the baseline.

x10 with fewer users alone is no alternative: about 40 users gives the same GPU load, but about 1M tokens of live context fits in 2.3M of GPU
and host memory, so nothing would need reading (next section). x200 adds a new path, since the p50 passes the 120 s read horizon on 75 % of turns and so sets
read timing, but it needs new baseline and oracle arms (15 h in all), and only about 366 users finish.

For a stronger contrast on the same workload, no code is needed: `HINT_KEY=none` gives every turn one table-wide p50.

## Lower cap, smaller memory or more load? (CPU model, no run; `all/qps_calibration.txt`, `qps_sim.txt`, `qps_sim_window.txt`)

Asked 2026-11-01: "How about increasing the QPS?", then "should we lower the max user cap from 128 to somewhere lower? I think stretch the gap
is not realistic." The model is `frozen/oracle_scripts_staging/c20_dist/qps_sim.py`, a closed-loop extension of `scripts/next_use_replay.py`.
Its one fitted setting is 7,000 tokens per context shared between sessions. It then gives:

| Real run | Measure | Real | Model |
|---|---|---:|---:|
| x70 oracle | returning turns from GPU / host / SSD | 42 / 31 / 21 % (6 % cold) | 44 / 34 / 23 % |
| x70 baseline | mean TTFT | 1.09 s | 1.45 s |
| x10 oracle | SSD read rate; SSD share of returns; mean TTFT | 1.86 GB/s; 35 %; 23.7 s | 1.92 GB/s; 38 %; 26.8 s |
| x10 p50 | mean TTFT against the oracle; wasted reads | equal; 0.47 TB | equal; 0.34 TB |

It reads less than x70's real 0.95 GB/s (0.63 GB/s), and it has no storms, so its no-hint column means little under congestion (real x10 baseline
95.8 s). Predictions, mean TTFT in seconds; GPU load is the exact-gap rate against 2.4 turns/s:

| Gaps | Users | Host memory | GPU load | Exact gap | p50 | One fixed number | No hints |
|---|---:|---:|---:|---:|---:|---:|---:|
| x10 | 128 | 160 GB | 100 % | 26.75 | 26.82 | 26.75 | 26.73 |
| x10 | 48 | 160 GB | 80 % | 0.57 | 0.60 | 0.60 | 0.57 |
| x1 | 20 | 80 GB | 67 % | 0.28 | 0.28 | 0.28 | 0.28 |
| x10 | 40 | 70 GB | 69 % | 0.35 | 0.36 | 0.41 | 0.55 |
| x10 | 44 | 70 GB | 73 % | 0.70 | 0.98 | 1.43 | 2.10 |
| x10 | 48 | 70 GB | 80 % | 2.87 | 4.74 | 8.15 | 6.61 |
| x10 | 52 | 70 GB | 61 % | 10.37 | 11.96 | 18.07 | 13.60 |
| x70 | 128 | 160 GB | 66 % | 0.35 | 0.59 | 2.11 | 1.45 |
| x70 | 160 | 160 GB | 82 % | 0.67 | 3.56 | 26.13 | 18.15 |

- **A lower cap alone removes the signal.** At x10 with 40-56 users and 160 GB under 1 % of returning turns need the SSD. At x1 and x3 over 99 % find
  their history on the GPU, at 160 GB and at 80 GB.
- **A lower cap with less host memory keeps it.** x10, about 44 users, `HOST_GB=70`: the only x10 setting tried with exact < p50 < no hints and no
  queue. The window is narrow, so it needs a probe (about 1 h each at 40, 44 and 48 users on the oracle server, looking for near-zero queue and
  0.4-1 GB/s of SSD reads), and then baseline, oracle and p50 arms: about 18 GPU-hours. 70 GB is just above the 66 GB that mirrors the GPU pool; this
  model has not been run that low.
- **A risk there:** at 48 users and 80 GB no hints beats exact gaps (1.15 s against 2.88 s), because the routine evicts the just-finished session and
  reads it back. If that is real, the comparison stops being about forecast accuracy.
- **More load:** `ARRIVAL` alone changes nothing once the cap binds (x70 users finish at 0.047/s against 0.1/s arriving). Raising the cap at x70 exposes
  the forecast strongly: the p50 reads 40-60 % more from the SSD than exact gaps, which reaches the SSD's 1.9 GB/s limit first. At x10 more load only hides it.
- **Rule of thumb:** aim for the oracle arm reading around 1 GB/s with the GPU under about 85 %, at whatever gap scale.

## Figures (`../compare_20261031_204525/`)

The usual comparison figures with campaign 19's two arms beside this one, from `manifest_with_c19.txt` (the run's own `manifest.txt` is
untouched). From `scripts/`, in the container, where matplotlib is:

    python3 timeline.py --manifest ../results/compare_20261031_204525/manifest_with_c19.txt --tier-rule private --steady-state \
      --ttft-over-time --paired-cdf c19_three_tier_wc_oracle_v2_nu --paired-max 2000

Campaign 19's `compare.csv` rows and its `turns_` and `events_` files regenerate exactly. `compare_steady.csv` differs slightly for those
arms because the cut moved from 952.8 s to 953.4 s, this run's first host eviction. The regenerated `turns_`, `events_` and `timeline_`
files of campaign 19's arms were removed from this folder; the originals are in `../compare_20261018_103537/`.

In the paired figures, "faster on 69 %" of the slowest 15 % is selection, not a gain: the slowest 15 % of the p50 run makes the oracle
look faster by the same margin (-3.36 s against -3.02 s, 69 % in both; `all/figure_notes.txt`).
