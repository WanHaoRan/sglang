# Campaign 12, x10 — campaign 11 with Poisson arrivals and 30-50-turn conversations (2026-10-02)

Campaign 11 (`../compare_20261001_013443/`, campaign 8's x10 workload on the same merged engine) rerun with users who
come and go. The same 128 conversations of the 262K agentic trace (offset 32) arrive as a Poisson process at 0.2/s
(seed 0; the last arrives at 12.3 min), and each replays min(recorded turns, U[30, 50]) turns (seed 120): 4,671 turns
and 112.4M prompt tokens, against campaign 11's 4,676 and 112.2M. Gaps are the recorded ones x10, capped at 1,200 s; wall
cap 9,736 s per arm. Qwen3-30B-A3B-Instruct-2507, bf16 KV, one H200, merged engine (upstream #39283: admission-time SSD
re-query, paced retries 8/8).

This is the x10 scale of campaign 12. The x70 and x1 scales are in `../compare_20261001_211629/` and
`../compare_20261002_200123/`; the comparison with the all-at-once baselines is `../campaign12/README.md`, and the
design log is `../campaign12/DECISIONS.md`. **This directory holds all four x10 arms**: a host outage stopped the first
driver run after `hbm_host`, and the other three arms ran from `../compare_20261002_143343/` (see the notes).

| arm | server |
|---|---|
| `hbm_host` | HiCache host tier, write-through: L1 668,160 tokens + L2 160 GB (1,627,648 tokens) |
| `three_tier_to` | + nixl L3 on the attached SSD (wiped before boot), `timeout` prefetch |
| `three_tier_wc` | same, `wait_complete` prefetch |
| `hbm_lru` | no HiCache, radix LRU, L1 only |

Files: `compare.png`, `compare.csv`, `timeline_<arm>.png`, `turns_<arm>.csv`, `events_<arm>.csv` (timeline.py);
`live_timeline.png` and `live_phases.csv` (live conversations and TTFT over time, statistics per phase);
`gap_distribution.png`; `summary.csv` and `paired.csv` (the tables below); `iostat_vdc.log` (SSD samples every 10 s over
both driver runs). Regenerate from `scripts/`:

```
python3 timeline.py --manifest ../results/compare_20261002_060200/manifest.txt
python3 live_timeline.py --manifest ../results/compare_20261002_060200/manifest.txt --label "x10, campaign 12" --ref ../results/compare_20261001_013443/manifest.txt --ref-label c11 --out ../results/compare_20261002_060200
python3 campaign_summary.py --manifest ../results/compare_20261002_060200/manifest.txt --iostat ../results/compare_20261002_060200/iostat_vdc.log
python3 gap_distribution.py --set x1=../results/compare_20261002_200123/manifest.txt --set x10=../results/compare_20261002_060200/manifest.txt --set x70=../results/compare_20261001_211629/manifest.txt --this x10 --iostat ../results/compare_20261002_060200/iostat_vdc.log --title "campaign 12 (swebench trace, OFFSET=32, Poisson arrivals, 30-50 turns)" --out ../results/compare_20261002_060200/gap_distribution.png
```

## Result

Whole run, returning turns. A return counts as recompute when more than half of its prompt beyond the 7,808 shared
tokens was uncached; `compare.csv` and `compare.png` use timeline.py's older whole-prompt rule, so their tier counts
differ.

| | hbm_host | three_tier_to | three_tier_wc | hbm_lru |
|---|---|---|---|---|
| turns / errors / conversations cut by the wall cap | 4,671 / 0 / 0 | 4,671 / 0 / 0 | 4,671 / 0 / 0 | 4,662 / 0 / 2 |
| client wall (s) | 6,491 | 4,674 | 4,625 | 9,736 |
| 90 % of conversations ended (min) | 81.3 | 50.5 | 49.5 | 150.2 |
| returning-turn TTFT mean / p50 / p90 / p99 (s) | 48.2 / 11.0 / 155.4 / 171.8 | 23.1 / 10.2 / 80.3 / 98.6 | 22.1 / 10.0 / 74.6 / 95.5 | 100.3 / 115.8 / 186.2 / 197.8 |
| worst 10-min window, median TTFT (s) | 166.8 (from min 50) | 82.3 (from min 30) | 81.0 (from min 30) | 190.5 (from min 90) |
| turns done at 95 % (s) | 4,711 | 2,841 | 2,784 | 8,306 |
| server queue wait, mean (s) | 46.9 | 22.4 | 21.5 | 97.7 |
| returning turns: device / host / SSD / recompute | 1,003 / 2,113 / 0 / 1,427 | 1,006 / 2,430 / 548 / 559 | 1,012 / 2,440 / 563 / 528 | 832 / 0 / 0 / 3,702 |
| uncached tokens per returning turn (mean) | 6,166 | 2,473 | 2,342 | 14,485 |
| SSD restore TTFT p50 / p90 (s), tokens per restore (median) | – | 75.4 / 93.1, 17.6K | 67.8 / 91.8, 17.7K | – |
| tokens: d2h / h2s / h2d / s2h | 29.1M / – / 34.4M / – | 12.3M / 12.3M / 50.9M / 10.4M | 11.7M / 11.7M / 51.6M / 10.7M | – / – / – / – |
| tokens evicted from device / host | 62.8M / 27.5M | 62.6M / 21.1M | 62.6M / 20.8M | 66.1M / – |
| peak scheduler queue | 68 | 63 | 63 | 78 |
| SSD read / written (TB); %util mean / p90 | – | 1.00 / 1.18; 18 % / 50 % | 1.02 / 1.13; 17 % / 50 % | – |
| SSD reads while active, mean / peak (MiB/s); samples at ≥ 95 % util | – | 615 / 1,806; 1 % | 658 / 1,934; 6 % | – |

Same (conversation, turn), mean difference with 95 % bootstrap CI (`paired.csv`; 4,543 pairs, 4,534 with `hbm_lru`):
- `three_tier_to` − `hbm_host`: **−25.1 s [−26.2, −23.9]**, p50 −0.52 s.
- `three_tier_wc` − `three_tier_to`: −0.97 s [−1.17, −0.78], p50 −0.02 s.
- `hbm_lru` − `hbm_host`: +52.0 s [+50.5, +53.5], p50 +40.4 s.

By phase (`live_phases.csv`): arrival lasts until the last conversation arrives; peak lasts until the live count first
falls below 90 % of the arm's own peak (117 live in the HiCache arms, 125 in `hbm_lru`).

| TTFT mean / p90 (s), span (min) | arrival | peak | departure |
|---|---|---|---|
| `hbm_host` | 3.89 / 11.2, 0-12.3 | 65.7 / 141.2, 12.3-27.0 | 89.9 / 164.8, 27.0-108.2 |
| `three_tier_to` | 3.90 / 11.4, 0-12.3 | 36.7 / 86.8, 12.3-21.6 | 38.1 / 87.9, 21.6-77.9 |
| `three_tier_wc` | 3.94 / 11.6, 0-12.3 | 33.3 / 85.0, 12.3-20.4 | 37.1 / 85.9, 20.4-77.1 |
| `hbm_lru` | 7.65 / 28.6, 0-12.3 | 109.0 / 146.4, 12.3-50.5 | 154.5 / 192.7, 50.5-162.3 |

## What it shows

1. **The SSD arms halve `hbm_host`'s TTFT.** They hold mean TTFT at 22.1-23.1 s against 48.2 s, and save 25.1 s per
   turn. They recompute 528-559 returns (12 %), where `hbm_host` recomputes 1,427 (31 %), and they end 90 % of the
   conversations in 50 minutes against 81. `hbm_lru` averages 100.3 s, 52.0 s per turn above `hbm_host`.
2. **The SSD is reached through re-issued lookups.** 545 of the 583 (`three_tier_to`) and 546 of the 584
   (`three_tier_wc`) requests that read from the SSD did so after a re-issued lookup, i.e. the #39283 admission
   re-query or a paced retry, a median 64-72 s after the request arrived (`scripts/requery_share.py`). A restored turn's
   TTFT is therefore mostly queueing: p50 68-75 s, p90 92-93 s.
3. **The GPU queue binds, not the SSD.** Each SSD arm reads 1.0 TB at a mean %util of 17-18 % (p90 50 %), and only
   1-6 % of the samples in which it reads reach 95 %. The mean server queue wait is 21.5-22.4 s, and the scheduler
   queue peaks at 63.
4. **The arrivals spread the load.** 1,952-1,960 returning turns (43 %) fall in the 12.3-minute arrival phase, at 3.9 s
   mean on every HiCache arm. Host evictions start at 11.4 min, near the last arrival.
5. **The prefetch policy moves the mean by 1 s** (`wait_complete` − `timeout`: −0.97 s [−1.17, −0.78]).

Against campaign 11, which started all 128 conversations at once on the same engine, `hbm_host` falls from 115.4 to
48.2 s, the SSD arms from 69 to 22-23 s and `hbm_lru` from 139.7 to 100.3 s. The paired comparison is in
`../campaign12/README.md`.

## Notes and deviations

- **Two driver runs, one manifest.** The first run (this directory, launched 06:02Z) finished `hbm_host`. The host went
  down at ~08:33Z, about 40 minutes into `three_tier_to`; that run (`../20261002_075232_three_tier_to_NAT160`, 3,560 of
  4,671 turns) is not used. The relaunch (`../compare_20261002_143343/`, 14:33Z) ran `three_tier_to`, `three_tier_wc`
  and `hbm_lru`. Its three manifest lines were appended to `manifest.txt` here on 2026-10-05, as campaign 11 did for its
  fifth arm, so `manifest.txt` now equals `../campaign12/manifest_x10.txt`. `iostat_vdc.log` covers both runs and has no
  samples from 08:33 to 14:33Z.
- **Engines.** `hbm_host` ran the working tree with oracle step 2 (`2abe005bb2`); its
  `--hicache-storage-next-turn-prefetch` flag was off and the client sends no hints. The other three arms ran the
  frozen `2abe005bb2` engine (`agent_cache/frozen/engine_2abe005bb2`, recorded in each `run.txt`).
- **Wall cap.** `hbm_lru` hit 9,736 s with 2 conversations unfinished (9 turns).
- **One run per arm** and one draw of arrivals (seed 0) and lengths (seed 120).
- **Worst 10-min window**: the highest median TTFT over fixed 10-minute bins of send time holding at least 20
  returning turns.
- **SSD restores** are the returns counted as SSD in the table; tokens per restore is their median storage-hit count.
- **SSD traffic** is integrated from `iostat_vdc.log` over each arm's client window. "While active" counts the samples
  reading more than 1 MiB/s.
- `gap_distribution.png` shows the gaps of all three campaign-12 scales; its dashed lines mark the SSD read time of
  this campaign's median restore at the device ceiling and at the loaded read rate.
