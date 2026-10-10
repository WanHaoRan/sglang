# Campaign 15 — the oracle next-turn prefetch on campaign 14's workload, `wait_complete` and `timeout` (2026-10-09/10)

The oracle of `../../ORACLE_PLAN.md` (steps 1-9, `--hicache-storage-next-turn-prefetch on`) on campaign 14's x70
steady-state workload, one arm per prefetch policy. The baselines are campaign 14's runs: the user dropped the planned
reruns to save 10 h (`DECISIONS.md`). Engine: the frozen copy `agent_cache/frozen/engine_294c82b7fc_ntpfix`
(`294c82b7fc` plus the fresh re-eviction probe); campaign 14 ran `engine_2abe005bb2`, whose difference is inactive with
the flag off. Client: `run_campaign14.sh`'s knobs unchanged, plus `ORACLE_HINTS=1` (campaign 14 sent no hints).

| arm | server | run |
|---|---|---|
| `three_tier_wc` | L1 natural (668,160 tokens) + L2 160 GB write-through + nixl L3 on `/mnt/ssd`, `wait_complete` | campaign 14 |
| `three_tier_wc_oracle` | the same + the oracle on (retry flags pinned at the defaults, 8 / 8) | here, 2026-10-09 16:41-21:43Z |
| `three_tier_to` | as `three_tier_wc` with the `timeout` prefetch policy | campaign 14 |
| `three_tier_to_oracle` | the same + the oracle on (new arm, added for this campaign) | here, 21:43Z-02:47Z |

Workload as campaign 14: 768 fresh conversations of the 262K agentic trace, Poisson arrivals at 0.1/s, the user cap
following the server backlog (target 32), stationary start, x70 gaps capped at 1,200 s, 5 h per arm. Qwen3-30B-A3B bf16
KV, TP 1 on one H200, context 131,072.

Files, one directory per oracle arm, each against its campaign-14 baseline (`manifest_vs_c14.txt`): `steady.csv`,
`steady_paired.csv`, `steady_bins.csv`, `steady.png` (window 2,700-16,800 s); `oracle_eval.csv` and
`oracle_rids_<arm>.csv`; `diagnosis.txt` (`frozen/oracle_scripts_staging/c15_diagnosis.py`: SSD, host pool, reads by
case, Little's law); `oracle_timeline_7200_120.png` (two minutes of the run, `scripts/oracle_timeline.py`). Regenerate
from `agent_cache/`: `python3 scripts/steady_state.py --manifest results/campaign15/<arm>/manifest_vs_c14.txt --results
results --window-start 2700 --window-end 16800 --iostat <campaign 14's and 15's iostat_vdc.log, concatenated> --out
<dir>`; `python3 scripts/oracle_eval.py --run results/<run dir> --out <dir>`; `python3 scripts/oracle_timeline.py --run
results/<run dir> --start 7200 --seconds 120` (in the container: matplotlib).

## Result

Window 2,700-16,800 s after the client started (campaign 14's), returning turns. "Congested" = 5-min bins with a TTFT
median above 10 s (`steady_bins.csv`).

| | `three_tier_wc` (c14) | `three_tier_wc_oracle` | `three_tier_to` (c14) | `three_tier_to_oracle` |
|---|---|---|---|---|
| TTFT mean / p50 / p90 / p99 (s) | 22.2 / 10.4 / 40.4 / 173.9 | 20.8 / 16.8 / 36.1 / 154.9 | 53.3 / 10.7 / 155.4 / 188.7 | **41.1 / 8.7 / 112.9 / 138.6** |
| queue wait, mean (s) | 21.6 | 20.2 | 51.9 | 39.9 |
| turns/s; users finished per hour | 1.217; 169 | 1.218; 169 | 0.988; 124 | **1.114; 139** |
| users finished by the 5-h stop | 765 | 766 | 646 | **708** |
| turns/s while congested (share of bins) | 1.34 (55 %) | **1.57** (57 %) | 0.64 (72 %) | **0.78** (66 %) |
| turns/s while light (live users) | 1.06 (94) | 0.74 (73) | 1.91 (141) | 1.77 (143) |
| user cap, mean; live users | 171; 133 | 178; 130 | 145; 149 | 152; 153 |
| returns: device / host / SSD / recompute | 21 / 16 / 50 / 13 % | 21 / 10 / 58 / 11 % | 11 / 26 / 28 / 35 % | 8 / 21 / 36 / 35 % |
| uncached tokens per returning turn | 1,846 | 1,389 | 5,795 | 5,079 |
| SSD read / written | 15.6 / 3.4 TB | 18.0 / 2.7 TB | 11.0 / 7.9 TB | 18.0 / 7.8 TB |
| SSD busy, mean; ≥ 90 % in (10-s samples) | 61 %; 49 % | 70 %; 68 % | 54 %; 22 % | 86 %; 77 % |
| host pool held by in-flight SSD reads, p50 | 382K tokens (23 %) | 642K (39 %) | 50K (3 %) | 183K (11 %) |

Same (conversation, turn), oracle − baseline: `wait_complete` **−1.41 s [−2.09, −0.73]**, p50 +0.01 s (16,594 pairs);
`timeout` **−7.28 s [−8.38, −6.16]**, p50 −0.09 s (13,421 pairs). The CIs cover within-run noise only (see Notes).

## What it shows

1. **Under `timeout` the oracle clearly helps.** Mean TTFT falls 23 % (53.3 → 41.1 s) and p90 / p99 by 27 %; turns/s
   rise 13 % and 62 more users finish in the 5 h. The baseline left the SSD idle (54 % busy, ≥ 90 % in only 22 % of
   samples): the timeout cut reads and 35 % of returns recomputed. The oracle starts reads earlier, into that idle
   bandwidth (86 % busy), so more pages land before admission: uncached tokens per turn −12 %. Throughput rose while
   congested (0.64 → 0.78 turns/s, +22 %), and the run spent 34 % of its bins light against 28 %.
2. **But wastefully: the timeout cuts half of the oracle's reads.** 5,018 of 9,663 oracle reads were cut (73.7M tokens
   short of their lookups), against 11 % of the requests' own reads; 4,473 of the arm's 15,810 reads (28 %) were cut
   while still queued and then skipped by the read thread (campaign 14: 504 of 8,464). 116M tokens were re-fetched and
   43M of them (37 %) credited from the SSD; 711 staged reads went unused; 2,982 re-evictions (1,085 at the cap). This
   is plan decision 3's gap: a read's timeout budget runs from the read's start, not from its turn's admission.
3. **Under `wait_complete` the averages barely move, but throughput while congested does.** Mean TTFT −1.4 s, p90 / p99
   −11 %, the median worse (10.4 → 16.8 s), turns/s equal. Split by phase, the oracle arm completed 17 % more turns/s
   while congested (1.34 → 1.57; 13-31 % for median thresholds of 3-30 s). Its light bins had fewer live users (73
   against 94) and fewer turns (0.74 against 1.06/s): light phases are demand-limited, so the mean hides the gain. The
   SSD was the bottleneck there: reads queued ~30 s behind a median 9,820 pages, 73 % ended within 50 ms of admission
   (late), and in-flight reads held a median 39 % of the host pool, which cut host hits from 16 % to 10 % of returns
   (`ORACLE_PLAN.md` section 5).
4. **`wait_complete` stays the better policy.** With the oracle, `timeout` reaches 41.1 s mean TTFT and 1.11 turns/s;
   plain `wait_complete` had 22.2 s and 1.22. The oracle narrows the gap between the policies without closing it.
   `three_tier_wc_oracle` has the best mean and p90 of the four (20.8 s, 36.1 s), `three_tier_to_oracle` the lowest p99
   (138.6 s).
5. **How well the oracle predicted depends on the read queue.** Under `timeout` the queue at a trigger was short (p50
   1,203 pages for case 1, 2,941 for case 2, against 9,820 under `wait_complete`), the load estimate was close (read −
   L̂ p50 −0.3 s, against +6.5 s), reads ended a median 0.85 s before admission and 34 % were late (73 %). Arrival error
   p50 45 ms (`wait_complete`) and 123 ms (`timeout`); all 19,454 planned turns arrived and were admitted, and 19,006 of
   19,012 (the other 6 were still pending at the stop). The case mix follows the queue: 9,499 staged and 1,587 re-fetch
   triggers under `wait_complete`, 3,628 and 5,585 under `timeout`.

## Notes and deviations

- Single runs against campaign 14's single runs, a day apart, on different engine copies (the difference inactive for a
  baseline) and with or without hints (ignored with the flag off). In smoke C two runs of identical code differed by
  0.98 s mean TTFT at 32 sessions (`../compare_20261009_082335/README.md`); the spread at this scale is not measured.
  Both campaigns moved through congested and light phases at different times (`steady.png`), so read the phase split,
  not only the means. The `timeout` gains are larger and consistent across phases; the `wait_complete` mean is not
  evidence on its own.
- The backlog cap fixes the mean queue wait at backlog / throughput (Little's law holds within 0.4 s for all four), so
  latency gains show up as throughput.
- Not a deployment configuration: pools as the variable, triton attention, context 131,072
  (`../campaign14/DECISIONS.md`).
- `scripts_at_launch.tgz` holds `agent_cache/scripts/` as launched; the cut of the baselines is in `cut_baselines.log`.
