# Campaign 23: the real table path and just-in-time reads

**Restarted 2026-11-01 by request, with the gate and the hold ("build the gate and the hold, then restart campaign 23").** This section replaces the engine and arm rows further down. The host reboot at 14:25Z ended the earlier launcher while it was still waiting, so no arm of this campaign had run. It starts with `GO`, without campaign 22, which is not running.

**What the arms now test.** Three changes against campaign 21's and campaign 19's oracle arms, all behind flags in `three_tier_wc_oracle_v2_nu_owned`:
- **Just-in-time staging** (`--hicache-storage-next-turn-read-timing just_in_time`): a read before arrival starts at predicted admission minus read time minus 1 s. This is what `jit_70` tested.
- **The gate** (`--hicache-storage-next-turn-arrival-gate on`): the user's 2026-10-31 instruction, in the form agreed that day. The prefix match at arrival stays, because it costs no I/O and admission's re-query needs it. A planned turn whose match covers its committed key to within the 256-token threshold gets no SSD lookup. Otherwise it looks up, at most 3 times per turn across all triggers. A turn with no plan looks up as before.
- **The hold** (`--hicache-storage-next-turn-arrival-hold on`): a parked read of a queued turn waits until predicted admission minus read time minus 2 s, and goes at once at position 0 or 1 or with no measured rate.

**Engine.** `frozen/engine_owned_20261101/python`: `engine_table_20261101` plus those two switches, off by default. Seven files differ. Unit tests: 1,712 pass; 2 fail in `test_pp_prefetch_ticket.py`, and they fail the same way on `engine_table_20261101`. `c23_engine/mutation_check.py` broke each rule of the gate and hold in turn and a test caught it (25 cases).

**The first launch crashed, 15:49Z.** The scheduler died 12 s into the client at the first declined lookup, `KeyError: 'declined_gate'`: the real counters are a plain dict with fixed keys, and the test fixtures use a `defaultdict`, so every unit test passed. `owned_smoke` caught it as designed; no 2 h arm was spent. Fixed, with a regression test on the real cache that failed before the fix, and the freeze was cut again under the same name. The crashed run never reached `manifest.txt` and has since been deleted (below). The checks missed it as a traceback, because a scheduler exception prints as `Scheduler hit an exception: Traceback`, not at the start of a line; they caught it from client errors. They now match it anywhere, and none of campaigns 19-21 or 24's runs contains that form.

**Host shutdown, 16:05Z.** The second launch, at 16:00Z, was 4 min into its client when the host went down; it came back at 16:16Z. That run had 241 turns, no errors or tracebacks, and the gate had declined 128 lookups, so the fix held on the GPU; the hold had not acted yet. By request ("clean and then resume the previous unfinished the work"), both aborted `owned_smoke` runs, their compare folders and driver logs, and the iostat log that covered only them were deleted, and the campaign relaunched from its first arm at 16:19Z. Nothing had finished, so nothing was lost. A reboot ends the launcher, which runs inside the container, and the arm in progress; finished arms stay in `manifest.txt` and a relaunch skips them.

**Preempted again, 16:27Z.** The host is a preemptible VM (the user, 2026-11-01). The third launch was 4 min into its client, 189 turns and no errors; its run dir `../20261101_161951_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70/` is kept and is not in `manifest.txt`. Relaunched from the first arm at 19:12Z, as the user asked, without weighing whether 2 h arms fit between preemptions.

**`owned_smoke` passed, 19:49Z** (`../20261101_191247_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70/`, 30 min, 128 users, 70 GB, 3,329 turns, no errors or tracebacks). The first time the gate and the hold ran under a real queue. These numbers show the mechanism only; no other arm has this setting.
- **The hold does what it was built for.** On the 370 held reads with a full record, whose turns waited a median 15.9 s in the queue, the read stayed parked a median 10.0 s (p90 22.2 s) and was dispatched a median 4.3 s before admission (p10 1.6 s, p90 13.8 s). None was dispatched after its admission and none was left unreleased. So a stuck read, the risk this smoke was for, did not appear.
- **The gate removes the blind lookups.** 1,331 lookups were declined as covered. Of the 3,223 arrival calls, 2,404 now look nothing up and 151 looked up and found nothing, against 3,900 or so in `jit_70` on about twice as many turns.
- **It is still not one read per turn here.** 1.21 reads per turn that reads, 13.7 % of pages from later reads, and 13 lookups refused at the cap, on a setting chosen to be harsh. (First recorded as 1.43 and 27.7 %; see the counting bug under `jit_exact_c44` below.) Whether this falls at the real settings is one of the questions for the 2 h arms.
- **Worth watching.** A staged read that lands before its turn arrives is not held, and stays pinned through the queue wait: p90 39.1 s on 148 such reads, 958 read-seconds. It happens when the queue grows after the read started.

**`table_smoke` passed, 20:21Z** (`../20261101_194908_three_tier_wc_oracle_v2_nu_table_custom_L1nat_H70/`, 30 min, 44 users, 70 GB, 3,144 turns, no errors or tracebacks). The engine read its own gaps from the table and matched the client exactly (`c23_engine/check_table_smoke.py`): the key on all 3,144 turns, the gap to within 1 ms on all 3,034 plans, and no plan on the 110 turns with no p50 (81 session ends, 29 with no entry). So the real table path gives the same numbers as the version 1 point campaign 21's p50 arm ran on, and `jit_p50_c44` can be read against that arm. One difference remains by design: under just-in-time reads a table plan starts earlier by p50 - p10, which a version 1 point cannot express.

**`jit_exact_c44` done, 22:24Z** (`../20261101_202142_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70/`, 2 h, 44 users, 70 GB, exact gaps, no errors or tracebacks). Tables: `compare_c44.out`, `load_curve_c44.out`, `arrival_same_turns_jit_exact_c44.out`, `arrival_fetch_jit_exact_c44.out`, `read_start_timing_jit_exact_c44.out`, `shallow_queue_jit_exact_c44.out` here; figures `../compare_c23_c44/`; scripts `c23_engine/compare_c44.py`, `plot_c44.py`, `load_curve_c44.py`. Campaign 21's tables are unchanged.

| Same returning turns | Baseline | Oracle | p50 | This arm |
|---|---|---|---|---|
| All 7,768, mean TTFT, s | 5.17 | 1.48 | 1.95 | 0.50 |
| All 7,768, recomputed tokens per turn | 2,870 | 977 | 1,329 | 114 |
| Positions 0-3,000 (2,883), s | 0.37 | 0.35 | 0.39 | 0.32 |
| Positions 3,000-5,500 (2,421), s | 0.72 | 0.99 | 1.24 | 0.37 |
| Positions 5,500-7,000 (1,453), s | 23.62 | 5.06 | 6.92 | 1.12 |
| Positions 7,000-8,750 (1,005), s | 3.03 | 0.74 | 0.92 | 0.41 |
| Turns sent in 2 h, each arm's own | 8,037 | 10,319 | 10,021 | 12,201 |

Against the oracle on all 9,980 returning turns they share: paired -1.05 s, faster on 6,952, slower on 3,028.

- **The hardware did not move, so the GPU unit caveat below is answered as far as it can be.** On the first 3,000 turns, in all five arms at this setting: decode 45.0-45.4 ms per token (this arm 45.4), prefill 0.100-0.102 ms per uncached token on first turns (0.101), first-turn TTFT median 0.22-0.24 s (0.22). Over each whole run, read time per page in reads of 500 pages or more, p10 1.83-1.96 ms (1.83). Mean TTFT over that stretch is 0.32 s, 0.02 s below the 0.34-0.39 s expected; with the speeds equal I read that as this arm's own effect.
- **Staging gives most of the gain; the gate and hold are not separated from noise.** On the 7,412 turns shared with `jit_70`: oracle 1.52 s, `jit_70` 0.67 s, this arm 0.49 s. So just-in-time staging alone gives 0.85 of the 1.03 s; the gate and hold add 0.18 s, most of it in one episode (position 6,500: 5.82 and 1.92 s). One run each, no repeat. Turns queued 5 s or more, of the 7,430 planned turns shared: oracle 476, `jit_70` 119, this arm 51.
- **The gate does what the 2026-10-31 instruction asked.** Over the run it declined 5,928 lookups as covered, let through 1,486 first lookups and 682 later ones, and refused 25 at the cap. On the 7,642 turns all four arms share:

| | Baseline | Oracle | `jit_70` | This arm |
|---|---|---|---|---|
| Arrival lookups that found nothing | 3,300 | 2,926 | 3,935 | 227 |
| SSD reads | 2,121 | 2,004 | 883 | 793 |
| SSD pages | 574,019 | 287,700 | 217,169 | 206,693 |
| Share of pages started at or after arrival, not staged or re-fetched while queued | 100 % | 34.1 % | 19.8 % | 16.2 % |
| Turns that read | 2,005 | 1,508 | 759 | 694 |
| Reads per turn that reads | 1.058 | 1.329 | 1.163 | 1.143 |
| Pages from a turn's later reads | 7.3 % | 18.4 % | 8.9 % | 8.1 % |
| Staged reads, landing to admission, median (p90), s | | 9.6 (44.7) | 1.8 (2.8) | 1.7 (2.5) |

- **One fetch: each page once, but not always one I/O.** Over the run, 1,239 of 1,463 reading turns read once (1.167 per reading turn). In all 224 others the first read was cut short and the next fetched the rest: only one turn lost a landed span and read it again, 268 of 434,495 pages. Cut reads are 344 of 1,707 (20 %; the oracle's 1,317 of 2,988, 44 %) and load a median 48 % of what the lookup found. The cause is host room, by the code rather than a log line: `_launch_storage_hit` falls back to a shorter prefix when eviction cannot free the whole read (`unified_radix_cache.py:2978-2987`), and that is its only trim. The dispatcher's budget is 356,064 tokens against about 44K of room, so it does not stop a read launching into no room. Another 72 turns made one cut read and recomputed the rest; 69 of them were returning turns, mean TTFT 6.64 s, 10,296 tokens recomputed. Holding a cut read until room appears might give one I/O per turn. Not built or tested.
- **The hold acted 173 times, all sound.** On the 172 with a full record, queue wait median 8.8 s, parked median 3.7 s (p90 10.9 s), dispatched a median 3.2 s before admission (p10 1.0 s, p90 11.8 s). None was dispatched after admission and none left unreleased. At 44 users there is little queue for it to act on, so the 128-user arms are its test.
- **Pins stay under the room for longer.** Pinned tokens at read dispatch, median and p90: 30-50 min 0K and 37K against the oracle's 48K and 71K; 50-75 min 20K and 53K against 65K and 105K; room 44K. Re-fetched spans evicted again before admission: 197 against 1,336, each over 120 min.
- **Load carried at a latency target moves further than any arm before.** Load at a mean TTFT of 1, 3, 5 s: baseline 975K, 1,041K, 1,055K; oracle 938K, 1,081K, 1,112K; p50 963K, 1,048K, 1,085K; `jit_70` 1,042K, 1,103K, 1,126K; this arm 1,082K, 1,138K, 1,176K. That is about 9-12 % over the baseline, read from swings in one run and partly endogenous, so it still needs the user-count sweep in `../../PAPER_PLAN.md`.
- **What it does not fix: the spike near position 8,750.** Oracle 10.28 s, p50 7.32 s, this arm 8.35 s per 250 turns, at the same place in the workload, minutes 85-90 here. The GPU pool is 0.92-0.98 full and the queue 2-10 for those minutes in all three arms, so this looks like the GPU being short of room for what is running, which no read routine can give back. This arm recovers from it faster, 0.28 s by position 9,250 where the oracle stays at 2-3 s. About half of all TTFT seconds now sit on the 565 planned turns queued 1 s or more.

**A counting bug in `arrival_fetch.py`, fixed 2026-11-01.** The script gave a read revoked before any I/O the next read's `s2h_io`, so those pages counted twice. It now stops at the next lookup, and its totals equal the `s2h_io` lines exactly on all five runs checked. This changes campaign 21, 24 and the owned smoke, where the deadline scheduler revokes reads while they are parked. Earlier figures on 7,642 turns were too high: `jit_70` 1.30 reads per reading turn and 19.4 % later pages, now 1.163 and 8.9 %; the oracle 1.71 and 39.8 %, now 1.329 and 18.4 %; the baseline 1.11, now 1.058; the oracle's staged reads landing to admission 6.8 s, now 9.6 s. Campaigns 16, 18 and 19 change by at most 3 reads, so the design doc's table stands. `shallow_queue.py` counts `s2h_io` directly and was not affected.

**`jit_p50_c44` done, 2026-11-02 00:26Z** (`../20261101_222451_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70/`, 2 h, 44 users, 70 GB, online p50, no errors or tracebacks). The first full arm on the real table path: the engine made its own pick on all 10,419 turns and 10,089 plans. Same GPU unit as `jit_exact_c44`, straight after it. Tables here: `compare_c44.out`, `load_curve_c44.out`, `arrival_same_turns_jit_p50_c44.out`, `arrival_fetch_jit_p50_c44.out`, `read_start_timing_jit_p50_c44.out`, `shallow_queue_jit_p50_c44.out`, `forecast_cost_c44.out`, `read_denied_c44.out`, `recompute_by_gate_c44.out` (one-off, no script kept). Scripts: `c23_engine/forecast_cost.py`, `read_denied.py`; `hw_control.py`, `split_reads.py`, `split2.py`, `hold_stats.py`, `direct_reads.py`, `spike.py` and `reread_samples.py` moved there from the scratchpad because the host is preemptible.

| Same returning turns | Baseline | Oracle | p50 | `jit_exact_c44` | This arm |
|---|---|---|---|---|---|
| All 7,768, mean TTFT, s | 5.17 | 1.48 | 1.95 | 0.50 | 1.51 |
| All 7,768, recomputed tokens per turn | 2,870 | 977 | 1,329 | 114 | 940 |
| Positions 0-3,000 (2,883), s | 0.37 | 0.35 | 0.39 | 0.32 | 0.42 |
| Positions 3,000-5,500 (2,421), s | 0.72 | 0.99 | 1.24 | 0.37 | 0.94 |
| Positions 5,500-7,000 (1,453), s | 23.62 | 5.06 | 6.92 | 1.12 | 5.24 |
| Positions 7,000-8,750 (1,005), s | 3.03 | 0.74 | 0.92 | 0.41 | 0.60 |
| Turns sent in 2 h, each arm's own | 8,037 | 10,319 | 10,021 | 12,201 | 10,419 |

Against the oracle on all 9,977 returning turns they share: paired -0.11 s, faster on 5,610, slower on 4,367. One run each, no repeats.

- **The question this arm was for: a forecast does cost more once the routine uses the timing.** Exact gaps over the p50 are worth 1.01 s under the new routine and 0.47 s under the old one. The p50 keeps 78 % of exact gaps' gain over the baseline, against 87 % before. The new routine gains 0.44 s with the p50 and 0.98 s with exact gaps, so with the p50 it only reaches the old oracle. "p50 is enough" was at least partly a fact about the old routine.
- **Hardware.** Same unit as `jit_exact_c44`. Decode per token matches over the first 15 min (median 25.0, 39.5, 44.8 ms there, 24.2, 38.9, 44.7 ms here). First-turn TTFT median 0.24 s, prefill 0.099 ms per uncached token, big-read p10 1.94 ms per page, all with the other arms. Decode then runs slower here only (minute 40: 66.5 against 54.3 ms), which puts the first 3,000 turns at 45.8 ms median and 47.1 mean against 45.0-45.4 and 44.3-45.5 elsewhere. Since the speeds match until eviction starts, I read that as this arm's own load.
- **The cost is shared, not paid by the turns the forecast got wrong.** On the 10,076 returning turns both just-in-time arms share (exact 0.71 s, p50 1.65 s), 63 % of the gap sits in stretch 5,500-7,000 and 62 % of it is queue wait (mean 0.88 s against 0.30 s). Forecast error, actual minus predicted arrival: p10 -5.5 s, median -0.3 s, p90 +4.9 s (exact: within 0.12 s). Turns forecast to within 1 s are two-thirds of the turns and hold 54 % of the gap; they lose 0.76 s each, others 0.86-1.83 s. 1,670 turns read the SSD only under the p50 and 280 only under exact gaps; those 1,670 hold 63 % of the gap and their own forecasts were typical (median error -0.3 s).
- **Recomputation now lives where a read finds its data but gets no host room.** By the code (`_launch_storage_hit`) rather than a log line, a revoked launch means eviction could not free any pages and a cut read means it freed some.

| Whole run | Oracle | p50 | `jit_exact_c44` | This arm |
|---|---|---|---|---|
| Read launches revoked, of all dispatches | 2,329 of 5,317 | 2,554 of 5,933 | 381 of 2,088 | 1,743 of 5,001 |
| Pinned tokens at dispatch, median: revoked, launched | 62K, 52K | 59K, 47K | 42K, 7K | 53K, 40K |
| Recomputed tokens, M | 11.38 | 12.76 | 2.36 | 10.72 |
| Share on turns whose read was cut or revoked | 86 % | 84 % | 78 % | 88 % |

  Those are 694 of this arm's 10,076 returning turns, mean TTFT about 6 s. They account for 91 % of the extra 8.36 M tokens against `jit_exact_c44`. The cap refused a lookup on 214 of the returning turns, 2.15 M tokens, overlapping that group.
- **Why there is less room: reads that land early stay pinned.** A landed read holds its pages until its turn is admitted, with 600 s the only limit (`PREFETCH_PIN_MAX_AGE_S`). Mean pinned tokens over the run: 30.9K against 6.7K, on about 44K of room. Pins of 10 s or more: 446 against 22, and 72 % of this arm's pinned token-seconds. About half of that is gaps above the key's p90 (median true gap 58 s, p50 12 s). The other half is gaps within range, where the read started early on purpose by p50 - p10 (median 10.8 s). Staged reads land a median 4.7 s before admission (p90 24.6 s); exact gaps 1.7 s (2.5 s).
- **Not separated: pins against eviction.** Eviction also runs on the forecast, and device hits fall to 18,784 tokens per turn from 21,571. These two runs cannot split that from the pins. An arm with exact gaps for eviction and the p50 for reads would. Whether a time limit on pins or holding a read until room appears recovers the gap is untested; neither is built.
- **The gate still works under a forecast; one fetch gets worse.** On the 8,008 turns all five arms share:

| | Baseline | Oracle | p50 | `jit_exact_c44` | This arm |
|---|---|---|---|---|---|
| Arrival lookups that found nothing | 3,461 | 3,051 | 3,092 | 236 | 294 |
| Turns that read | 2,067 | 1,543 | 1,996 | 725 | 1,768 |
| SSD reads | 2,184 | 2,053 | 2,612 | 830 | 2,256 |
| SSD pages | 592,964 | 293,759 | 375,501 | 215,881 | 430,992 |
| Reads per turn that reads | 1.057 | 1.331 | 1.309 | 1.145 | 1.276 |
| Pages from a turn's later reads | 7.1 % | 18.5 % | 19.1 % | 8.3 % | 17.5 % |

  Over the run the gate declined 3,757 lookups as covered, let through 2,721 first and 2,549 later ones, and refused 215 at the cap (exact gaps: 5,928, 1,486, 682, 25). 1,865 of 2,537 reading turns read once. Of the other 672, 646 had the first read cut short. 73 turns read pages twice, about 14,700 of 610,945 (2.4 %), and 31 lost a landed span; exact gaps had 1 turn and 268 pages. Cut reads are 1,208 of 3,258 (37 %; exact gaps 20 %). This arm reads 15 % more pages than the old p50 routine on the same turns, because its reads are larger: 244 pages per reading turn against 188, on fewer reading turns.
- **The hold stays sound.** 470 held, 449 with a full record: queue wait median 7.3 s, parked median 1.7 s (p90 7.0 s), dispatched a median 4.8 s before admission (p10 1.2 s, p90 18.1 s). None after admission, none unreleased.
- **Load carried at a latency target.** At a mean TTFT of 1, 3, 5 s: baseline 975K, 1,041K, 1,055K; p50 963K, 1,048K, 1,085K; this arm 950K, 1,061K, 1,117K; `jit_exact_c44` 1,082K, 1,138K, 1,176K. So -3 %, +2 %, +6 % over the baseline, against +11 %, +9 %, +11 % with exact gaps. Rough, as above.
- **Below the knee it is still slower than the baseline** (0.42 against 0.37 s, 0.94 against 0.72 s in the first two stretches), as the oracle and p50 arms are. Exact gaps are not (0.32, 0.37 s).

**`jit_exact_c128` done, 2026-11-02 02:30Z** (`../20261102_002657_three_tier_wc_oracle_v2_nu_owned_NAT160/`, 2 h, 128 users, 160 GB, exact gaps, no client errors). Tables here: `compare_c128.out`, `why_c128.out`, `late_reads_c128.out`, and `arrival_fetch_`, `arrival_same_turns_`, `read_start_timing_`, `hold_stats_`, `read_denied_`, `split_reads_`, `direct_reads_jit_exact_c128.out`. Scripts: `c23_engine/compare_c128.py`, `why_c128.py`, `late_reads_c128.py`. Campaigns 19 and 20 ran 5 h, so every number here keeps each arm's first 2 h. They ran on `engine_5df71c4bf0_v2`, and all three changes arrive together, so nothing separates them at this setting.

| Same returning turns, first 2 h | Baseline | Oracle | p50 | This arm |
|---|---|---|---|---|
| All 8,175, mean TTFT, s | 53.12 | 12.06 | 12.14 | 12.46 |
| All 8,175, recomputed tokens per turn | 3,784 | 35 | 34 | 33 |
| Positions 0-2,000 (1,908), s | 0.33 | 0.34 | 0.34 | 0.32 |
| Positions 2,000-4,000 (1,916), s | 2.15 | 2.12 | 2.19 | 2.25 |
| Positions 4,000-6,000 (1,922), s | 46.70 | 15.45 | 15.78 | 16.02 |
| Positions 6,000-8,000 (1,918), s | 136.25 | 26.02 | 25.99 | 26.81 |
| Past position 8,000 (511), s | 153.40 | 27.86 | 27.77 | 28.77 |
| Turns sent in 2 h, each arm's own | 8,511 | 16,228 | 16,210 | 16,101 |

Against the oracle alone, on the 15,529 returning turns they share: 20.88 s and 21.28 s, paired +0.40 s, faster on 7,485, slower on 8,044. One run each, no repeats.

- **No TTFT gain at this setting: the queue at the GPU decides it.** On those turns queue wait is 20.46 s against 20.86 s, and everything after admission is 0.42 s in both. Every hinted arm recomputes about 33 tokens per turn. From minute 45 the GPU pool stays 0.98 full with about 30 running and 59-73 queued, in all three hinted arms. So hints already removed what memory was costing, and no read routine can shorten what is left. At 44 users and 70 GB the same routine took 1.48 s to 0.50 s; the two settings differ in both users and host memory, so these runs do not say which of the two matters.
- **The +0.40 s looks mostly like speed, not the routine.** On the first 3,000 turns, while the SSD reads only about 7 MB/s, decode takes 50.22 ms per token against the oracle's 49.21 (p50 49.75), while prefill is faster, 0.102 against 0.113 ms per uncached token. The arm generated 0.8 % fewer tokens in 2 h. This is a closed loop with a cycle near 58 s (128 sessions at 2.2 turns/s), so 0.8 % of throughput equals about 0.45 s of wait, though there the two are one number seen twice. Turns that read in neither arm lose 0.32 s. Campaign 19's GPU unit is unrecorded, so this cannot be settled without a repeat on one unit.
- **A just-in-time read sometimes makes its own turn wait, a small part of the gap.** A turn without a read is never passed by a later arrival by 0.1 s or more, in any arm, so being passed measures waiting on a read. Reading turns queued 5 s or more: passed 16.7 % (oracle 4.5 %, p50 25.9 %), by a mean 3.66 s (4.06, 6.66), which is 0.61 s per reading turn (0.18, 1.73). Spread over the shared turns that is about 0.07 s of the 0.40 s. So about one read in six lands after its turn could have gone. The 2 s margin, the read estimate or the wait forecast could each be why; not separated. Reads land a median 2.23 s before admission (p10 0.26 s).
- **Exactly one fetch per turn, as the 2026-10-31 instruction asked, at this setting.** 3,504 reads on 3,504 reading turns; no turn read twice, no read cut. On the 16,167 turns all three share:

| | Oracle | p50 | This arm |
|---|---|---|---|
| Arrival lookups that found nothing | 6,372 | 6,503 | 473 |
| Turns that read | 5,319 | 5,245 | 3,504 |
| SSD reads | 5,362 | 5,417 | 3,504 |
| SSD pages | 1,524,899 | 1,529,525 | 1,079,202 |
| Reads per turn that reads | 1.008 | 1.033 | 1.000 |
| Turns read twice | 36 | 170 | 0 |
| Landing to admission, median (p90), s | 10.45 (20.41) | 5.11 (18.73) | 2.23 (6.31) |

  The gate declined 6,470 lookups as covered, let through 3,464 first and 3 later ones, and refused none. Of the 473 empty lookups, 451 are first turns, which have no plan, and 22 are planned. At 44 users and 70 GB this routine reads 1.145 times per reading turn. The difference is host room: at 160 GB no read is cut or revoked in any arm, which agrees with what the 44-user arms showed splits a read.
- **The gain here is I/O.** 29 % fewer SSD pages on the same turns. Per returning turn, host hits are 10,716 tokens against 8,872 and SSD 4,360 against 6,165, with device about equal, so more stays in host memory; why is not separated. Mean pinned tokens over 2 h: 32.9K against 148.8K (pins median 2.2 s against 10.5 s, p90 6.3 s against 20.7 s). The SSD goes from 97-98 % busy at about 1,800 MB/s to 47-78 % busy at 900-1,550 MB/s over minutes 45-120, though reads rise through the run. What that room buys is unmeasured, because the GPU is the limit here.
- **The hold's real test passed.** 3,203 reads held, none left unreleased, none dispatched after admission. On the 3,151 with a full record: queue wait median 31.0 s, parked median 26.3 s (p90 34.9 s), dispatched a median 3.8 s before admission (p10 1.9 s, p90 8.2 s). The queue forecast it relies on holds up: median actual over predicted wait at arrival is 1.00.
- **One traceback, after the client ended, no effect on results.** At shutdown a read still in flight for a turn that would never come hit the freed host pool (`AttributeError` on `kv_buffer` in `mha.py:696`, thread `prefetch_io_aux_func`), seconds after the client had finished. A bug only at exit; not fixed, since the engine is frozen during the campaign. The watcher treats any traceback as an alert, so `jit_p50_c128` may wake it the same way at its end.
- **`read_start_timing.py`'s "case 2" row misleads under the hold.** It reads `trig_lead_ms`, the planned trigger rather than the dispatch, and shows p10 -2.49 s. Measured from the log, dispatch leads admission by p1 0.87 s and median 3.75 s, and the read first comes due a median 30.9 s before admission.

**`jit_p50_c128` done, 2026-11-02 04:35Z** (`../20261102_023031_three_tier_wc_oracle_v2_nu_owned_NAT160/`, 2 h, 128 users, 160 GB, online p50, no client errors, no tracebacks). The last arm: the launcher printed `C23 ALL DONE`, and the container is up with no server running. Same GPU unit as `jit_exact_c128`, straight after it (`GPU-b59d6a34`). Tables here: `compare_c128.out`, `why_c128.out`, `late_reads_c128.out`, `forecast_cost_c128.out`, `staged_reads.out`, `late_why_c128.out`, `late_deadline_c128.out`, `late_by_service_c128.out` and `late_split_c128.out` (one-offs, no script kept), and `arrival_fetch_`, `arrival_same_turns_`, `read_start_timing_`, `hold_stats_`, `read_denied_`, `split_reads_`, `split2_`, `reread_samples_`, `direct_reads_jit_p50_c128.out`. Scripts: `c23_engine/forecast_cost_c128.py`, `staged_reads.py`, `late_why_c128.py`, `late_deadline_c128.py`, `plot_c128.py`. Figures: `../compare_c23_c128/`.

| Same returning turns, first 2 h | Baseline | Oracle | p50 | Exact, this routine | p50, this routine |
|---|---|---|---|---|---|
| All 8,175, mean TTFT, s | 53.12 | 12.06 | 12.14 | 12.46 | 12.56 |
| All 8,175, recomputed tokens per turn | 3,784 | 35 | 34 | 33 | 34 |
| Positions 0-2,000 (1,908), s | 0.33 | 0.34 | 0.34 | 0.32 | 0.33 |
| Positions 2,000-4,000 (1,916), s | 2.15 | 2.12 | 2.19 | 2.25 | 2.30 |
| Positions 4,000-6,000 (1,922), s | 46.70 | 15.45 | 15.78 | 16.02 | 16.17 |
| Positions 6,000-8,000 (1,918), s | 136.25 | 26.02 | 25.99 | 26.81 | 27.08 |
| Past position 8,000 (511), s | 153.40 | 27.86 | 27.77 | 28.77 | 28.62 |
| Turns sent in 2 h, each arm's own | 8,511 | 16,228 | 16,210 | 16,101 | 16,082 |

- **A forecast costs almost no TTFT here.** On the 15,503 returning turns the four hinted arms share: exact 21.27 s, this arm 21.34 s, paired +0.07 s, median +0.005 s, faster on 7,551, slower on 7,952. Under the old routine, on the same turns: oracle 20.86 s, p50 20.96 s, +0.10 s. At 44 users this routine's forecast cost 1.01 s, with the same forecast error (p10 -5.4, median -0.3, p90 +4.9 s, against -5.5, -0.3, +4.9 s). One run each, no repeats, so neither difference here is known to be above noise.
- **Both new arms lag their rivals by the same amount.** This arm against the old-routine p50, on 15,511 turns: 20.96 s and 21.34 s, paired +0.38 s; exact against the oracle was +0.40 s. Queue wait is 20.58 s against 20.93 s, and everything after admission is 0.42 s in both. Tokens generated are down 0.8 % in both pairs (3,214,504 against 3,241,525). Decode on the first 3,000 turns is 49.90 ms per token against the old p50's 49.75, a smaller gap than exact's 50.22 against 49.21, so this arm weakens the earlier reading that the lag is mostly GPU speed. Routine and unit still arrive together: this needs a repeat on one unit.
- **About half of the I/O gain survives.** On the 16,137 turns all four share, except the last two rows, which cover each arm's whole first 2 h:

| | Oracle | p50 | Exact, this routine | p50, this routine |
|---|---|---|---|---|
| Arrival lookups that found nothing | 6,364 | 6,493 | 472 | 512 |
| Turns that read | 5,301 | 5,228 | 3,477 | 4,147 |
| SSD reads | 5,344 | 5,400 | 3,477 | 4,276 |
| SSD pages | 1,518,868 | 1,523,453 | 1,071,057 | 1,290,748 |
| Reads per turn that reads | 1.008 | 1.033 | 1.000 | 1.031 |
| Turns read twice | 36 | 170 | 0 | 129 |
| Mean pinned tokens | 148.8K | 127.0K | 32.9K | 66.4K |
| SSD busy, per 15 min, minutes 45-120 | 97-98 % | 97-98 % | 47-78 % | 60-92 % |

  Pages fall 15 % against the old-routine p50, where exact fell 29 % against the oracle: 52 % of the saving. From minute 60 the SSD reads 1,610-1,770 MB/s, near its ceiling of about 1,850 MB/s, against exact's 1,300-1,555 MB/s. The gate holds: 6,340 covered, 4,083 first reads, 6 re-reads, none refused, and empty lookups stay near exact's.
- **Why more turns read than with exact: looks like eviction on the wrong time, not separated.** 2,192 turns read in both arms, 1,842 only in this one (531,455 pages), 1,179 only in exact (349,857). By this arm's forecast error on the turn: 5 s or more before the forecast, reads go 280 to 549; within 1 s, 1,532 to 2,097; 1 s or more after, 1,257 to 1,014. A turn forecast too late looks far off and goes to the SSD first; one forecast too soon stays. The rise on turns forecast within 1 s has no direct evidence behind it.
- **Exactly one fetch does not survive, and the failure has one cause.** 129 turns read twice (exact 0), 37,070 pages, 2.9 %. All 129 go the same way: the first read lands before the turn arrives, the plan times out 60 s after the forecast arrival, which releases the pin, the span is gone from host memory by the time the turn comes, and the second read comes a median 124 s after the timeout. 121 of them came due after the forecast arrival had already passed. In all, 515 reads came due before arrival (exact 17), 463 of them with the forecast arrival already missed, by a median 12.5 s and with the turn still away: the routine still assumes the turn arrived on time and is queueing. Those 463 turns' 565 pins hold 37.5K of the 66.4K mean pinned tokens, and reads due at or after arrival hold 22.6K (all of exact's 32.9K). In `read_start_timing_jit_p50_c128.out`, 514 of the 550 reads due before arrival had a wait estimate above load + 1 s, so they could have waited for the turn; counting only plans made in the first 2 h, 508 of 515 (at 44 users 287 of 2,094, with exact hints 0 of 1,198). Mean pinned tokens on one definition, each arm's first 2 h, at 44 users: oracle 39.3K, p50 32.3K, `jit_exact_c44` 6.2K, `jit_p50_c44` 30.9K (the 6.7K above also counts pins after the client stopped). Summary and options for next steps: `../../CAMPAIGN23_FINDINGS.md`.
- **A rule that follows, not built: a read goes out before its turn arrives only if the queue wait could not cover it.** It should bring this setting close to one fetch and take most of that pin load away, and at 44 users, where the wait is short, it should change little. It also would not fix 44 users' cost: there 87 % of pinned token-seconds come from reads due before the forecast, and 6 % from those past it (`staged_reads.out`).
- **The same long pins that cost at 44 users are harmless here.** Pins of 10 s or more: 483 against exact's 142, 72 % of the token-seconds; 57 % of those come from true gaps above their key's p90 (median p50 12.1 s, true 88.6 s), the rest from a short p50 and the early start (median p50 17.6 s, true 42.8 s, early start 11.4 s). Peak pinned in the 2 h is 400,768 tokens (exact 239,232), on 959,488 of room and a budget of 488,294. All 8 reads revoked at launch came at 7,234-7,237 s, after the client stopped sending, and 3 of 4,298 reads were cut. Recompute stays at 34 tokens per turn, against 940 at 44 users. This fits the 44-user finding that pins cost through room, though the two settings differ in users as well as memory.
- **The hold is sound but does not put reads on time under p50.** 2,826 reads held, none unreleased, none dispatched after admission, dispatch a median 3.4 s before admission (p10 1.8 s, p90 6.8 s). Reading turns queued 5 s or more are passed 32.1 % of the time (exact 16.7 %, old p50 25.9 %, oracle 4.5 %), by a mean 5.06 s, or 1.62 s per reading turn (0.61, 1.73, 0.18). Reads land a median 1.43 s before admission (p10 0.16 s, p90 7.90 s). Turns without a read are again never passed.
- **Why reads are late: mostly the SSD's ceiling.** It is not the turn's own forecast: passed share is 31-34 % in every forecast-error band up to 30 s, including turns forecast within 1 s, and 5.6 % beyond (`late_deadline_c128.out`). It follows the deadline's error only weakly: 33.6 % within 3 s, 37.6 % at 3-10 s too late. It follows the SSD: with the read rate under 1.6 GB/s around the admission, the two arms match (this arm and exact: 4.0 and 6.1 %, 8.8 and 10.0 %, 17.3 and 16.7 %); at 1.8 GB/s or more this arm's passed share is 48.1 % against 19.5 %, and 47 % of its reads fall there against 8 %. 926 reads needed no hold because the backlog already used up the wait (exact 223), and 66 % of those are passed (48 %). Two things stay open. On held reads the queue estimate runs later than real admission by a mean 1.57 s against 0.46 s, the read estimate does not differ (-1.01 s, -1.06 s), and turns passed ahead explain only 0.09 s of that gap. The ceiling also cannot explain the old routine, where the oracle and p50 read at the same rate and p50 is passed five to six times as often.

**2026-11-04: where the 44-user gain comes from.** Asked by the user. Six analyses (`gain44/w1_*.json`) and six independent checks of them (`gain44/w2_*.json`), with every cited output in `gain44/` and scripts in `../../frozen/oracle_scripts_staging/c23_gain44/`; the summary is in `../../CAMPAIGN23_FINDINGS.md`. The hint-free arm (`../20261101_122705_three_tier_wc_dl_poll0_custom_L1nat_H70/`, campaign 21's setting, 117 of 120 min), not analysed before, ties the oracle on the same turns (1.47 against 1.48 s on 7,768 turns) and carries 79 % of the gain over original HiCache. On the 7,410 turns all seven arms share: original HiCache 5.33 s, hint-free 1.51, oracle 1.52, `jit_70` 0.67, `jit_exact_c44` 0.49 s. Checked: every arm started with an empty SSD (`L3 wiped`, and no SSD hit before the first host eviction), replayed the same workload and ran at the same speed. Corrections to this file:
- The gate and hold add 0.17 s, not 0.18 s; 0.18 is a difference of rounded means.
- "Re-fetched spans evicted again before admission: 197 against 1,336" counts `ntp_reevict`, which is logged after any re-fetch that left history missing. Only 48 of 197 and 230 of 1,336 follow a re-read that landed; 78 and 638 follow one revoked at launch.
- "62 % of it is queue wait" (`jit_p50_c44`): 64.9 % once the 201 unplanned turns' waits from ReqTimeStats are counted; `forecast_cost.py` gives them 0.
- "The p50 keeps 78 % of exact gaps' gain over the baseline, against 87 % before": the hint-free arm alone keeps 79 %, so neither p50 routine adds to it on these turns.
- The spike near position 8,750 is not beyond any read routine: new exact recovers by position 9,250 and averages 1.93 s past 8,750 against the oracle's 3.77 s, on 1,519 shared turns.
- Staged landing to admission with p50 hints, 4.7 s (p90 24.6 s), is on shared turns; over all staged reads it is 5.2 s (p90 38.3 s).
- The hint-free arm's deadline order and read window are inert without hints, because every park is due at once. Descriptions of it in `../campaign21/DECISIONS.md` and `../campaign22/DECISIONS.md` should read "read pins and paced retries off".

**The GPU may be a different unit now.** Its UUID changed across this shutdown, `GPU-661d1d6b` before and `GPU-a385db19` after, and again at 16:27Z to `GPU-b59d6a34`. The instance is the same type: H200, Xeon 8468, 16 vCPU, 196 GB (`hardware_after_shutdown_20261101.txt`), and the SSD volume kept its data. No earlier record says which unit campaigns 19-21 and `jit_70` ran on. The speeds match (above). A paper should still run its repeats on one recorded unit.

**Two changes from the agreed design, both found while building it:**
- **The text past the key can be on the SSD.** The design said what is left after the committed key is new tool output and cannot be stored. Replaying the rule over the logs (`c23_engine/gate_rule_check.py`) shows it removes 3,729 of 3,810 empty lookups in `jit_70`, but also refuses 28 reads that found data, 32,896 tokens or about 0.2 % of what that run read, and 133 reads and about 0.5 % in campaign 21's oracle. They sit mostly at turns 1-3 and come from other sessions with the same opening. Kept as agreed, since they become small recomputes; the largest is 14,144 tokens.
- **The hold's read estimate counts the pages ahead of the read, not only those in flight.** Campaigns 19-20 ran the SSD 98 % busy in the congested stretch. With only in-flight pages counted, reads due together would all start too late. Counting the reads ahead makes them start earlier exactly when the SSD is short of time.

The cap of 3 lookups would have refused 12 lookups in `jit_70`, 11 of which found data, and 455 (347) in campaign 21's oracle, whose early reads are lost more often. It should bind rarely under the hold; `ntp_gate decision=capped` counts it.

**Arms, in order** (`manifest.txt` labels):

| Label | Setting | Client | Compare with |
|---|---|---|---|
| `owned_smoke` (30 min) | 128 users, 70 GB | exact gaps | technical checks only (`c23_engine/owned_smoke_check.py`): acts, no stall, no errors |
| `table_smoke` (30 min) | 44 users, 70 GB, read horizon, no gate or hold | only the v2 table | every plan's gap against the client's `dist_hint.p50` |
| `jit_exact_c44` | 44 users, 70 GB, 2 h | exact gaps | campaign 21 oracle and baseline, same turns |
| `jit_p50_c44` | same | online p50 | campaign 21 p50 and baseline, same turns |
| `jit_exact_c128` | 128 users, 160 GB, 2 h | exact gaps | campaign 19 oracle and baseline (`compare_20261018_103537`), same turns in the first 2 h |
| `jit_p50_c128` | same | online p50 | campaign 20 p50 (`compare_20261031_204525`) and campaign 19 baseline, same turns in the first 2 h |

`owned_smoke` runs first because a stuck read would block its turn, and 128 users on 160 GB would not show it: campaign 19 parked 7 reads in its first 30 min. Its numbers compare with nothing.

**How far the comparisons go.** Same trace, seed, users, arrival rate and gaps. The older arms ran on `engine_5df71c4bf0_v2`. Campaigns 19 and 20 ran 5 h (`CLIENT_TIMEOUT=18000`), so their whole-run numbers (95.8, 20.3, 20.4 s) cannot be set beside a 2 h arm: only the same turns can. `jit_70` shows the staging change alone, for its 75 min, so it helps separate that from the gate and the hold at 44 users. At 128 users nothing separates them. Every arm is one run, and no repeat exists at either setting.

**Stop only for errors**: a traceback, client errors, a dead server, or a client that stops sending. Slowdown is a result.

---

Earlier plan, kept for the record:

Started on my own judgment under "speed up and get this project published" and "do what you think should" (2026-11-01). It runs after campaign 22. To stop it first: `touch CANCEL` here.

**Engine.** `frozen/engine_table_20261101/python`: HEAD's `python/` (identical to `engine_5df71c4bf0_v2`) plus the uncommitted table intake (`reply_key.py`, `--hicache-storage-next-turn-reply-key-source`) and `--hicache-storage-next-turn-read-timing`. Unit tests pass, and the reply reader matches the client on all 25,617 recorded replies.

**Arms** (x10, `CLIENT_TIMEOUT=7200` except the smoke; `manifest.txt` labels):

| Label | Setting | Client | Compare with |
|---|---|---|---|
| `table_smoke` (30 min) | 44 users, 70 GB, read horizon | only the v2 table (`HINT_POINT=none`) | every plan's gap against the client's `dist_hint.p50` |
| `jit_exact_c44` | 44 users, 70 GB, just in time | exact gaps | campaign 21 oracle |
| `jit_p50_c44` | same | online p50 | campaign 21 p50 |
| `jit_exact_c128` | 128 users, 160 GB, just in time | exact gaps | campaign 19 oracle, first 2 h |
| `jit_p50_c128` | same | online p50 | campaign 20, first 2 h |

**Why just-in-time reads.** The routine reads as soon as the history is gone and admission is within 120 s, so exact timing is barely used and pins are held far too long: in campaign 21's first arm reads came due a median 10 s (p90 55 s) before admission but needed 1.75 s (p90 3.5 s), on a host pool with 44K tokens of room. A read now starts at admission minus the load estimate (backlog included) minus 1 s, and table plans also minus p50 - p10. Eviction and the dispatch deadline keep the p50.

**Questions.** Does it cut SSD reads, pins and cut reads? Does a forecast now cost more against exact hints, since the routine finally uses the timing? That second answer is what shows whether "p50 is enough" was a fact about forecasts or about the old routine.
