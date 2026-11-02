# Campaign 24: a healthy setting for x10 at 44 users

Why: campaign 21's oracle arm churned at 70 GB (`../campaign21/DECISIONS.md`), so its forecast comparison would mostly measure that. Started 2026-11-01 05:02Z on my judgment under "do what you think should".

**Probes.** Exact hints, x10, 44 users, 75 min each (`CLIENT_TIMEOUT=4500`), past the 55-70 min where that arm was worst:

| Label | Host memory | Routine | Engine | What it tests |
|---|---|---|---|---|
| `jit_70` | 70 GB | just-in-time reads | `frozen/engine_table_20261101` | whether read timing alone removes the churn |
| `horizon_90` | 90 GB | as campaign 21 | `frozen/engine_5df71c4bf0_v2` | whether more host-only room removes it (about 250K tokens against 44K) |

**A setting passes** if, from 30 min to the end: recompute stays under about 200 tokens per returning turn, throughput holds near 1.9 turns/s, the queue stays near empty, and SSD reads stay around 0.4-1 GB/s, so forecasts still have something to affect. Fixed before seeing the results.

**Then.** Baseline, oracle and online p50 at the setting that passes, 2 h each, then the hint-free arm there. If both pass, prefer 70 GB with just-in-time reads, the harder test and the more interesting system result, and consider 90 GB as the control. If neither passes, go up in memory. Campaigns 22 and 23 stay waiting until those arms finish.

**Added 2026-11-01 05:15Z, with the first probe 12 minutes in, before the period the criteria read.** Campaign 21's GPU pool was filling from 30 min on (`../campaign21/DECISIONS.md`), and neither probe changes that pool. So read the pool alongside the pass criteria. If both probes fail with the pool at 0.95 or above for much of the run, the next step is fewer users rather than more host memory. The criteria themselves are unchanged.

**Superseded 2026-11-01 05:30Z.** Campaign 21's setting was judged acceptable (`../campaign21/DECISIONS.md`, "The stop was a wrong call"), so there is no gate: the pass criteria and the fallback above are not used, and the three cases run at 44 users and 70 GB. `jit_70` finishes and is read against campaign 21's two attempts over the same 75 min. `horizon_90` is cancelled: the `horizon_90 cancelled -` line in `manifest.txt` makes the launcher skip it.

**The runs it was compared with are deleted.** Campaign 21's first attempt and restart were deleted on 2026-11-01 (`../campaign21/DECISIONS.md`). The numbers below against them can no longer be recomputed, and until they are updated `jit_table.py` and `plot_jit.py` fail and `../compare_c24_jit_vs_c21/` still draws them.

**`jit_70` result.** 75 min, 7,671 turns, no errors or tracebacks. Figures in `../compare_c24_jit_vs_c21/` (start with `jit_overview.png`), table from `frozen/oracle_scripts_staging/c24_probe/jit_table.py`. Mean TTFT on shared returning turns against campaign 21's restart, by the restart's send time:

| Sent at | Oracle, restart | Just-in-time | Turns |
|---|---:|---:|---:|
| 0-30 min | 0.36 s | 0.34 s | 3,082 |
| 30-55 min | 1.14 s | 0.41 s | 2,281 |
| 55 min to the end | 4.68 s (median 2.99) | 2.16 s (median 0.41) | 1,204 |
| Whole run | 1.42 s | 0.70 s | 6,567 |

The two oracle attempts differ by 0.21 s over 30-55 min, the only measure of noise there is. Recompute stays at about 30-100 tokens per returning turn over 30-50 min, against 650-1,700, and returning turns over 75 min rise 13 % (7,412 against 6,537). Live sessions stay at 44 until the client's timeout, so the late calm is not the client winding down.

Bursts remain, at 54 min (in the same minute as both oracle attempts) and 62-64 min (mean TTFT up to 15.9 s in a minute, worse than the restart's worst minute, 13.5 s). But it returns within about 3 min to 0.33 s TTFT, about 40 recomputed tokens and a GPU pool at 0.6, and stays there to the end. The restart was still bad when stopped at 78 min; whether it would have recovered is unknown, and the oracle rerun answers that.

**Pins.** p90 of pinned tokens at dispatch over 30-50 min: restart 60-70K, `jit_70` 10-37K, against 44K of room. `jit_70` passes the room only in its bursts (61K at 50-55 min, 88K at 60-65 min). So recompute appears when pin peaks pass the room, under both routines, which restores the explanation set aside in `../campaign21/DECISIONS.md` on medians alone. The restart's sustained GPU pool fill looks like an effect: same workload and pool, and `jit_70` falls back to 0.6.

**Read with campaign 21's p50 arm.** The horizon routine also comes out of its episode by itself, at 85 min, and has another at 105-110 min (`../campaign21/DECISIONS.md`). So the calm at the end of this run says little alone, and 75 min does not reach the later episode. What stands is the paired gain on the same turns.

**Read with campaign 21's oracle rerun.** The rerun leaves its episode at 80 min, so the restart would have recovered too. Every arm slows down and recovers at the same place in the workload (`../campaign21/DECISIONS.md`, "Episodes follow the workload"), so just-in-time reads did not end the episode early by themselves: this run reaches that place sooner, 18 min from turn 5,500 to 7,000 against the oracle's 25. On the same turns over that stretch TTFT is 1.91 s against 5.06 s and 23.62 s for the baseline, and over 3,000-5,500 it is 0.40 s against 0.99 s and 0.72 s. Its 75 min end at 7,671 turns, before the second heavy stretch at about 8,750.

**Limits.** One run, 75 min, exact hints. Exact hints are this routine's easiest case: reads start later, so a wrong forecast has less slack; `jit_p50_c44` in campaign 23 is the test. The engine is `engine_table_20261101`, not the v2 freeze; with exact hints only read timing should differ, which was not checked further.

**What it leaves as it was.** `read_timing` times only the staged read. The arrival fetch, the queued-turn refetch and the admission re-query are the oracle arm's, so this run is not the one owner of reads the user asked for on 2026-10-31 (`../../DIST_HINT_DESIGN.md`, later steps; items 7 and 8 of the long version's section 4), and it does not test dropping the arrival fetch. On the 7,642 turns it shares with campaign 21's arms: 7,443 arrival calls, 3,935 lookups that found nothing, 203 reads and 19.0 % of pages started at arrival, 1.163 reads per turn that reads, 8.9 % of pages from later reads, against 31.7 %, 1.329 and 18.4 % for the oracle (`frozen/dist_hint_design/arrival_fetch.py`). Staged reads land a median 1.8 s before admission (p90 2.8 s), against 9.6 s (p90 44.7 s). (Corrected 2026-11-01: a counting bug had given 17.6 %, 1.30 and 19.4 % against 25.5 %, 1.71 and 39.8 %, and 1.7 s against 6.8 s; `../campaign23/DECISIONS.md`.) The start time applies only before arrival, so when the queue wait exceeds the read time plus 1 s the turn arrives first: 94 of this run's 119 turns that waited 5 s or more had their read come due only after arrival. In campaign 19's oracle arm at 128 users that condition held for 8,501 of 8,521 staged reads, so campaign 23's 128-user arms would mostly measure the arrival fetch (inferred, not run).

**Where the gain comes from.** On the 7,410 planned turns in the baseline, the oracle and this run, mean TTFT is 1.52 s for the oracle and 0.67 s here (`frozen/oracle_scripts_staging/c24_probe/shallow_queue.py`, from `results/`). Under 1 s of queue in both arms (6,196 turns): 0.50 and 0.36 s, about 14 % of the net gain. Oracle queued 1 s or more and this run not (900 turns): 7.19 and 0.53 s, recomputed tokens 3,858 and 226, about 95 %. This run queued 1 s or more (314 turns): 5.39 and 7.15 s, about -9 %. Those rows are chosen on outcome, so the last two each flatter one arm and only the net is unbiased. Where both queued 5 s or more (42 turns) this run has no edge, 12.32 s against 13.60 s for it.

The oracle does not skip the pre-prefetch. On the shallow turns it stages more, 669 reads against 273, and 20.5 % of them are read again against 5.5 %. Pinned tokens at read dispatch over 30-50 min: median 48K and 0K, p90 71K and 35K, against 44K of room. My reading, not tested: early reads hold host memory, that causes evictions and recompute, and those form the queue. One run each, on different engine freezes.
