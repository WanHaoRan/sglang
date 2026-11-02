# Campaign 21: does an inaccurate forecast cost anything once the queue is gone? x10 gaps, 44 users, 70 GB of host memory

Requested 2026-11-01: "Let's run the x10, 44 user cap, 70GB host memory case as you suggested. Run these cases 1) the baseline (original sglang),
2) oracle with exact gap hint and 3) the online p50 estimation."

**The two stopped oracle attempts are deleted.** On 2026-11-01, by request ("delete these incomplete data"): their run directories and driver logs, `../compare_c21_stopped/`, and the copies of their tables in `../compare_c21_arms/` and `../compare_c24_jit_vs_c21/`. Anything below that quotes the first attempt or the restart was measured before that and can no longer be recomputed: the noise estimates and the comparisons against the restart. So no arm at this setting has a repeat on disk. `../compare_c21_arms/` is redrawn without them, its paired figures now against the finished oracle run, and the load tables below are recomputed from that run alone, which moves only the oracle's numbers (most at 1,100-1,200K, 7.07 to 7.88 s). `../compare_c24_jit_vs_c21/` still draws the attempts, and `c24_probe/jit_table.py` and `c24_probe/plot_jit.py` still read them and fail.

**The question.** Campaign 20 found the p50 as good as exact gaps at x10 with 128 users, but its queue hid forecast errors. With no queue and history
still leaving memory, what does the p50 cost against exact gaps, and how much of the gain over no hints does it keep?

**The arms**, in this order, each on a fresh boot with the SSD wiped, on the frozen engine `frozen/engine_5df71c4bf0_v2/python`:

| Label in `manifest.txt` | Server | Client hints |
|---|---|---|
| `three_tier_wc_oracle_v2_nu` | oracle v2 + next-use | exact gaps (`ORACLE_HINTS=1`) |
| `three_tier_wc_oracle_v2_nu_p50` | the same | p50 of the online table (`HINTS=dist`, `HINT_SOURCE=online`) |
| `three_tier_wc` | original HiCache, as in campaigns 18 and 19 | sent as in the oracle arm, and ignored |

The oracle runs first so that a bad setting shows before the other two arms run. "Original sglang" is read as the `three_tier_wc` arm on the same frozen
engine, the baseline of every earlier campaign, so that only the server configuration differs.

**The workload.** Campaign 19's except for the cap and host memory: `USERS=768 TURNS=50 ARRIVAL=0.1 GAP=10 C=44 SEED=0 CLIENT_TIMEOUT=7200` on
`traces/lmcache_agentic_trace_262k.json`, `LEVEL=NAT160 HOST_GB=70` (run dirs are named `custom_L1nat_H70`), device pool as profiled (668,160 tokens).

**Cookbook.** `docs/cookbook/autoregressive/Qwen/Qwen3.mdx`, as in campaign 20: no tool or reasoning parser (the replay sends no tools, thinking off),
`--context-length 131072`, one GPU, and the HiCache flags the page does not cover.

**Launcher.** `frozen/oracle_scripts_staging/c21_x10_c44/c21.sh` (copy here as `launcher.sh`, log `launcher.log`); health snapshots from `health.py`.

**Predictions from the CPU model** (`../campaign20/DECISIONS.md`), mean TTFT: exact 0.70 s, p50 0.98 s, no hints 2.10 s, 1.76 turns/s, SSD reads 0.43, 0.65 and
0.88 GB/s. The window is narrow: 40 users 0.35 s, 48 users 2.87 s, 52 users 10.4 s.

**What the model leaves out, and this setting may hit.** Host-only room is about 44K tokens (712K of host pool, 668K mirroring the device). A read that cannot get
host memory is shortened or dropped (`unified_radix_cache.py:2962`), a write-through backup that cannot get it is skipped (`:1723`), and both end in recompute. The pin
budget (30 % of the pool, about 214K tokens) is larger than that room.

**Restart.** 2026-11-01 03:43Z, by request: "Stop the ongoing runs, set the timeout to be 2 hours for all of them, then restart them." So every arm here and in campaign 22 has `CLIENT_TIMEOUT=7200`, down from 18,000.
The stopped oracle arm was `20261101_024420_three_tier_wc_oracle_v2_nu_custom_L1nat_H70` (59 min, 5,645 turns, no summary line, not in `manifest.txt`, since deleted); it counts for nothing but the note below.

**What the stopped arm showed.** Memory filled at about 30 min. Recompute then rose from 30-130 to 1,000-1,200 tokens per returning turn by 45 min, with 12-19 % of turns recomputing over a page. 5M tokens of reads went unfulfilled for `host_capacity`. From 30 min on,
5 % of returning turns made 86 % of the recompute, each losing 5K-36K tokens after staged and re-fetched reads were cut short. TTFT stayed low: median about 0.4 s, mean about 1.1 s, almost no queue. So any p50 cost measured here may partly be this shortage of host memory.

**Stop rules.** Any traceback or client error: read it, stop if real. The first plan was to stop after the oracle arm above about 1,000 tokens of recompute per turn. Having seen how it arises, that is no longer the rule: both hinted arms face the same memory limit, so the
p50 against the oracle stays a fair comparison, and the request is all three arms. Stop only on a real collapse: TTFT in the tens of seconds with the GPU no longer keeping up, or recompute in the thousands per turn as in campaign 19's baseline storm. Almost no SSD reads would also mean the test says nothing, but the stopped arm rules that out.

**Stopped 2026-11-01 05:02Z after about 78 min of the restarted oracle arm** (`20261101_034414_three_tier_wc_oracle_v2_nu_custom_L1nat_H70`, 6,804 turns, no summary line, not in `manifest.txt`, since deleted), under the collapse rule above and "do what you think should". Memory filled at 30 min, as in the first attempt. From 55 min recompute held at 2,900-4,000 tokens per returning turn with 49-59 % of turns recomputing over a page, against 60-180 and about 2 % before; throughput fell from 1.9 to 0.8-1.2 turns/s and median TTFT rose from 0.3 s to 2.2-4.4 s. 25.4M tokens of reads went unfulfilled for `host_capacity`.
The cause is this memory size with this routine. Host-only room is about 44K tokens, pinned reads held 56-87K, so reads forced device evictions (about 16K tokens/s), re-fetched spans were evicted again (115-195 re-evictions per 5 min) and the rest was recomputed. p50 against exact hints here would mostly measure that churn. Campaign 24 looks for a healthy setting; the three cases then run there. The stopped run was kept as evidence of the failure, until the deletion noted at the top.

**The two attempts are the first same-setting repeat at x10.** On shared returning turns: before 30 min TTFT 0.367 s and 0.358 s (paired -0.009 s), from 30 min on 0.971 s and 1.170 s (paired +0.199 s, median absolute difference 0.235 s). So once churn starts, differences of about 0.2 s between single runs mean little.

**A correction on the cause.** Recompute starts rising at 30 min, when the median of pinned tokens (16-30K) is still under the 44K of room, so the median does not explain the onset; peaks were not checked. The GPU pool fills over the same period: in the restart it sits at 0.95 or above on 5 % of scheduler lines at 30-40 min, 34 % at 50-60 min and 73 % at 70-80 min, with running requests going from 18 to 25. Which way that runs is open: slower turns keep more requests in flight, and a fuller pool evicts more. The worst stretch starts at 55 min, with pins at 56-87K.

**Peaks now checked.** The paragraph above relied on medians. p90 of pinned tokens at dispatch was 60-70K over 30-50 min, above the 44K of room, and recompute appears when those peaks pass the room under both routines (`../campaign24/DECISIONS.md`). So the pin explanation stands after all, and the sustained pool fill looks like an effect.

**The stop was a wrong call.** On 2026-11-01, after seeing the figures: "Why do you stop it? It looks OK to me actually." That reading is right. Mean TTFT peaked at 3-6 s with a queue of 2-4, it levelled off from 55 min with no errors, and SSD reads stayed at 0.7-1.0 GB/s against the disk's 1.9; campaign 19's baseline storm, which the rule named, was 95.8 s. I stopped on the rule's "recompute in the thousands" clause. "p50 would mostly measure churn" was never measured, and the fairness argument under Stop rules still held. What is real: time per turn rose from about 9.5 s to 26-32 s, which accounts for nearly all of the fall in turns/s (gaps and reply lengths did not change), and prefilled new tokens rose from about 1,500 to about 4,000 per second. That is a result of this setting, not a reason to stop it.

**The onset repeats.** In both attempts the GPU pool dips to 0.60 at 52 min and jumps at 54 min (0.90-0.97 and 0.86-0.90 over 54-56 min, queue 1.0-4.4), with TTFT 2.33 s and 3.05 s on the 115 shared turns the first attempt sent from 54 min. Contexts stay ordinary (largest in flight 50-60K), so no single huge session explains it. Only the restart shows what follows.

**Rerun.** All three arms at this setting, same engine, 2 h each, no pass gate, in the order p50, baseline, oracle: the oracle goes last because two attempts already cover its first 78 min, so the p50 can be read against them as soon as it ends. `c21_wait.sh` starts `c21.sh` when campaign 24 ends; `horizon_90` is cancelled there. `scripts/start_server.sh` now carries the new arms from campaign 24; a dry-run showed the existing arms unchanged (`../campaign24/scripts_at_launch.tgz`). **Stop only for errors: a traceback, client errors or a dead server.** Slowdown is the result. This replaces the stop rules above.

**p50 arm (ended 2026-11-01 08:22Z).** `../20261101_062014_three_tier_wc_oracle_v2_nu_custom_L1nat_H70/`: 2 h, 10,021 turns, no errors or tracebacks; the client's timeout ended it. Figures in `../compare_c21_arms/` (start with `arms_overview.png`), table from `frozen/oracle_scripts_staging/c21_x10_c44/compare_arms.py`, which takes the other arms as they land. Mean TTFT on shared returning turns against the stopped restart, by the restart's send time:

| Sent at | Oracle, restart | p50 | Paired | Turns |
|---|---:|---:|---:|---:|
| 0-30 min | 0.36 s | 0.39 s | +0.04 s | 3,082 |
| 30-55 min | 1.14 s | 1.39 s | +0.25 s | 2,281 |
| 55 min to the restart's end | 4.68 s (median 2.99) | 6.93 s (median 5.68) | +2.25 s | 1,204 |
| Whole restart, 78 min | 1.42 s | 1.94 s | +0.52 s | 6,567 |

The two attempts differ by -0.01 s before 30 min and +0.21 s over 30-55 min. Against the first attempt the p50 is +0.03 s and +0.52 s. Nothing measures the noise from 55 min, where only one attempt ran, so that row waits for the oracle rerun.

**It recovers by itself.** The p50 arm's mean TTFT per 5 min is under 1 s to 40 min, then 2.76 and 1.89 s, 0.58 s at 50-55 min, 4.9-11.0 s over 55-85 min, 0.48-0.79 s over 85-100 min, 9.83 s at 105-110 min, 0.70 s at 110-115 min and 2.56 s at 115-120 min, when the run ended. Calm stretches run at 1.4-1.7 turns/s, episodes at 0.7-1.0. So this is a run of episodes, not a state it stays in, and "levelled off" above was wrong: the same routine under the p50 came out of it 7 min after the point where I stopped the oracle restart. Episodes start in the same minutes across runs (54 and 70 min), so the workload sets them off.

**What the forecast costs.** With no congestion (0-30 min) the cost is small and lands where it should. Against the restart, the 45 turns whose plan expired go from 0.82 to 1.31 s, with SSD history 3,994 to 12,503 tokens, and the other 3,022 turns are +0.03 s. Under congestion (30-78 min) the extra is not tied to the turn's own error: +0.76 s with an expired plan, +0.77 s without, and +0.63 to +1.07 s in every band of true gap over sent p50 but one (2-4, -0.03 s on 158 turns). So there it is shared through the queue, or it is the variation between runs; this run cannot separate the two. 171 of 9,703 plans expired (1.8 %), all on gaps of 60 s or more. The sent p50 was at or above the true gap on 44.7 % of turns, with |true - p50| median 0.20 s, mean 7.01 s.

**TTFT follows pins.** Over the 50 5-min bins of these three runs: median pins under 20K, TTFT 0.41 s and 147 recomputed tokens; 20-44K, 0.86 s and 698; above the 44K of room, 4.44 s and 3,006. This is correlation. A queue also keeps reads pinned for longer, so it may run both ways, and what ends an episode is not established.

**Baseline (ended 2026-11-01 10:24Z).** `../20261101_082229_three_tier_wc_custom_L1nat_H70/`: 2 h, 8,037 turns against the p50's 10,021, no errors or tracebacks. It is level with the hinted arms to 50 min, then storms from 52 to 110 min: mean TTFT 14-47 s per 5 min, 0.3-0.5 turns/s, 10-19K tokens recomputed per returning turn, GPU pool 0.97, queue 8-15. Unlike campaign 19's storm this one ends by itself, 1.93 s at 110-115 min. Mean TTFT on shared returning turns, by the baseline's send time, oracle being the stopped restart (no second oracle run reaches past 59 min):

| Sent at | Baseline | Oracle | p50 | Just-in-time, exact |
|---|---:|---:|---:|---:|
| 0-30 min | 0.37 s | 0.36 s | 0.40 s | 0.34 s |
| 30-50 min | 0.77 s | 1.07 s | 1.34 s | 0.40 s |
| 55-85 min | 26.5 s | 4.73 s | 6.30 s | 1.55 s |
| 0-78 min | 2.96 s | 1.16 s | 1.47 s | 0.50 s |
| Whole baseline, 2 h | | | 5.17 -> 1.95 s (7,768 turns) | |

**Before the storm the hinted routine loses to no hints.** Over 30-50 min recompute per returning turn is 250 for the baseline, 849 and 952 for the two oracle attempts, 1,264 for the p50 and 57 for just-in-time reads. The first attempt's +0.11 s is within the attempts' 0.21 s spread, but all three horizon runs lie on the same side and the p50's +0.57 s is beyond it. So the gain is all in preventing or shortening the storm, as at x70 (campaign 18), and the horizon routine pays for it in calm periods. Just-in-time reads are the only arm ahead of the baseline in calm, -0.37 s over 30-50 min, with exact hints.

**It looks like two stable states (withdrawn below, under "Episodes follow the workload").** With 44 users, gaps of about 12 s and a closed loop, rate = 44 / (gap + latency) and in flight = rate x latency. Calm: latency 12 s, 1.8 turns/s, 22 in flight. p50 episode: 37 s, 0.9 turns/s, 33 in flight. Baseline storm: 90 s, 0.43 turns/s, 39 in flight. All three match the run. The 668K GPU pool holds about 26 requests of 26K tokens, so calm sits about 15 % under the pool, and once in flight passes it, evictions and recompute keep latency up. Every arm is pushed over at about 54 min (client send times, in-flight tokens going from 550-700K to about 1M), and the arms differ in how much recompute holds them there. This is my reading, not a tested claim: nothing here shows what ends an episode.

**Oracle (ended 2026-11-01 12:26Z).** `../20261101_102434_three_tier_wc_oracle_v2_nu_custom_L1nat_H70/`: 2 h, 10,319 turns against the p50's 10,021 and the baseline's 8,037 (+28 % and +25 %), no errors or tracebacks. On the 7,768 returning turns all three ran, mean TTFT is 5.17 s for the baseline, 1.48 s for the oracle and 1.95 s for the p50, so the p50 keeps 87 % of the oracle's gain, in TTFT and in turns. One run each. The baseline stops at 8,037 turns and never reaches the second heavy stretch below, so whole-run means that are not paired compare different work.

**Episodes follow the workload.** With each arm's turns in its own send order instead of on the clock, every arm slows down and recovers at the same turns (`../compare_c21_arms/by_position.png`, tables from `frozen/oracle_scripts_staging/c21_x10_c44/load_curve.py`): about 5,500-7,000 turns, and from about 8,750 in the two arms that get that far. Getting from 5,500 to 7,000 takes the baseline 54 min, the p50 29, the oracle 25 and just-in-time reads 18. So the arms differ in how hard they are hit and how long they take to get through, not in when an episode comes or goes. The two-state reading above is withdrawn. What ends an episode is that the load falls. Metastable failure is not the framing for this data.

The load is the context of live sessions not parked in a gap of 60 s or more. It averages 740K tokens over the first 3,000 turns, 970K to 5,500, 1,095K to 7,000, 830K to 8,750 and 1,070K after. Reply lengths and median gaps do not change across those stretches, new tokens per turn are no higher in the heavy ones, and after the ramp the mean prompt only moves between 25K and 27K, so what moves the load is which sessions are live and which are parked. Mean TTFT per minute against it, from minute 20:

| Load | Baseline | Oracle | p50 | Just-in-time, exact |
|---|---:|---:|---:|---:|
| under 800K | 0.35 s | 0.43 s | 0.48 s | 0.34 s |
| 800-900K | 0.39 s | 0.50 s | 0.51 s | 0.36 s |
| 900-1,000K | 0.87 s | 1.10 s | 0.79 s | 0.51 s |
| 1,000-1,050K | 1.61 s | 1.68 s | 2.07 s | 0.70 s |
| 1,050-1,100K | 10.15 s | 2.69 s | 4.68 s | 1.39 s |
| 1,100-1,200K | 25.81 s | 7.88 s | 7.21 s | 5.59 s |
| 1,200-1,400K | 40.40 s | 14.47 s (1 min) | 14.23 s | none |

The GPU pool averages about 0.85 at 1,050-1,100K, and the queue starts there (0.7-3.4 requests, against 0.2-1.0 just below). Starting from an episode, all of the next 5 min stay under 1.5 s in 8 of 8 cases under 900K, 6 of 10 at 900-1,000K and 5 of 127 above. Starting from calm, a minute at 5 s or more comes within 5 min in 15 of 113 cases under 1,000K, 3 of 14 at 1,000-1,050K and 3 of 7 above. The count of parked sessions alone shows no such order, so its context has to be counted.

**Limits of the load reading.** It is partly an effect: a slower arm enters long gaps less often and parks fewer sessions, so at turns 6,000-6,500 the baseline reads 1,155-1,255K where the oracle and the p50 read 1,084-1,132K. That places the baseline's slow minutes further right and flatters its curve. All minutes come from one workload, so they are not independent, and the top bands rest on few minutes. Some carry-over is not ruled out, since few minutes share a load with different histories: the baseline at 1,050-1,100K has 15.9 s after an episode (4 min) and 2.3 s from calm (1 min); the p50 at 1,100-1,200K has 8.3 s (21 min) and 3.7 s (3 min).

**The same turns, stretch by stretch.** Mean TTFT and recomputed tokens per returning turn, positions in the oracle's order, so nothing depends on when an arm got there:

| Turns, mean load | Returning turns | Baseline | Oracle | p50 | Just-in-time, exact |
|---|---:|---:|---:|---:|---:|
| 0-3,000, 740K | 2,883 | 0.37 s, 37 | 0.35 s, 55 | 0.39 s, 93 | 0.34 s, 35 |
| 3,000-5,500, 970K | 2,421 | 0.72 s, 219 | 0.99 s, 859 | 1.24 s, 1,149 | 0.40 s, 56 |
| 5,500-7,000, 1,095K | 1,453 | 23.62 s, 13,509 | 5.06 s, 3,404 | 6.92 s, 4,564 | 1.91 s, 984 |
| 7,000-8,750, 830K | 1,005 | 3.03 s, 2,018 | 0.74 s, 403 | 0.92 s, 634 | see below |
| 8,750 to the end, 1,070K | 1,228 | not reached | 4.06 s, 2,521 | 3.12 s, 1,879 | not reached |

Just-in-time reads over 7,000-8,750 run only on 653 turns: baseline 3.67 s, oracle 0.74 s, just-in-time 0.33 s. Its engine is `engine_table_20261101`, and its 75 min end at 7,671 turns.

**This run repeats the stopped restart.** On the same turns, restart then rerun: 0.36 and 0.35 s, 0.99 and 0.99 s, 4.68 and 4.73 s (1,263 turns to the restart's end). The first attempt against the restart is 0.85 and 0.99 s over 3,000-5,500. So noise is at most about 0.14 s below the knee and 0.05 s on the one repeat in the heavy stretch, though the baseline itself has never been repeated. Against that, the p50's cost per stretch is +0.04 s, +0.25 s, +1.86 s and +0.16 s (on all 1,704 turns the two share there, 0.61 and 0.77 s). Over 8,750 to the end the p50 is ahead, 4.06 and 3.12 s, with no repeat there, so I read that stretch as even. On all 9,689 turns the two arms share the p50 costs +0.26 s.

**The earlier answers.** The rerun leaves the first episode at 80 min of client time, 4 min past the restart's last send at 76 min (78 min from launch), so the restart would have recovered too. Before the knee all three oracle runs stay above the baseline, 0.85-0.99 s against 0.72 s. Against the oracle rerun the forecast's cost is as read before: with no congestion the 45 turns with an expired plan go from 0.56 to 1.31 s and the other 3,022 are +0.03 s; over 30-78 min the extra is +0.18 s with an expired plan and +0.77 s without, so it is not tied to the turn's own error.

**The gain is a small shift of a steep curve.** Interpolating the band means, the load each arm carries at a given mean TTFT:

| Mean TTFT | Baseline | Oracle | p50 | Just-in-time, exact |
|---:|---:|---:|---:|---:|
| 1 s | 975K | 938K | 963K | 1,042K |
| 3 s | 1,041K | 1,081K | 1,048K | 1,103K |
| 5 s | 1,055K | 1,112K | 1,085K | 1,126K |

So the oracle's 3.5-fold fall in mean TTFT corresponds to about 5 % more load at 5 s, the p50's 2.7-fold fall to about 3 %, and at a 1 s target the horizon routine carries no more than the baseline; just-in-time reads carry 6-7 % more at all three. This setting sits on the knee, which is why mean TTFT moves so much. The reading is rough for the reasons above, and it probably understates the gain because the baseline's curve is flattered. It comes from swings in one workload; running each routine at several user counts is the direct test.

**A fourth arm, added 2026-11-01 by request ("we are good to go").** `three_tier_wc_dl_poll0`: paced retries off, pins and deadline order, no hints used, same setting and 2 h. It runs from campaign 22's launcher once this launcher ends, and its line lands in `manifest.txt` here.
Against the oracle arm it shows what hints, staging and next-use eviction add together with no queue to hide timing.

**The fourth arm was cut short.** The host went down at 14:25Z and came back at 14:28Z, with no cause found in the journal, 117 min into the arm: `../20261101_122705_three_tier_wc_dl_poll0_custom_L1nat_H70/`, client `client_122833_cmp_three_tier_wc_dl_poll0`, 10,355 turns, no errors or tracebacks, no summary line, no line in `manifest.txt`. The container and the launchers of campaigns 22 and 23 went with it, so neither runs until restarted.

**After this campaign.** Campaign 22 (`../campaign22/`) is scheduled to start when this one ends. If this one is stopped under the rules above, `touch ../campaign22/GO` lets it start.
