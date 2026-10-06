# Campaign 12, x1 — campaign 9 with Poisson arrivals and 30-50-turn conversations (2026-10-02/03)

Campaign 9 (`../compare_20260923_031116/`, the recorded gaps unscaled) rerun with users who come and go. The same 128
conversations of the 262K agentic trace (offset 32) arrive as a Poisson process at 0.2/s (seed 0; the last arrives at
12.3 min), and each replays min(recorded turns, U[30, 50]) turns (seed 120): 4,671 turns and 112.4M prompt tokens,
against campaign 9's 4,676 and 112.2M. Gaps are the recorded ones x1, capped at 1,200 s; wall cap 9,736 s per arm.
Qwen3-30B-A3B-Instruct-2507, bf16 KV, one H200, merged engine (upstream #39283: admission-time SSD re-query, paced
retries 8/8).

This is the x1 scale of campaign 12. The x70 and x10 scales are in `../compare_20261001_211629/` and
`../compare_20261002_060200/`; the comparison with the all-at-once baselines is `../campaign12/README.md`, and the
design log is `../campaign12/DECISIONS.md`.

| arm | server |
|---|---|
| `hbm_host` | HiCache host tier, write-through: L1 668,160 tokens + L2 160 GB (1,627,648 tokens) |
| `three_tier_to` | + nixl L3 on the attached SSD (wiped before boot), `timeout` prefetch |
| `three_tier_wc` | same, `wait_complete` prefetch |
| `hbm_lru` | no HiCache, radix LRU, L1 only |

Files: `compare.png`, `compare.csv`, `timeline_<arm>.png`, `turns_<arm>.csv`, `events_<arm>.csv` (timeline.py);
`live_timeline.png` and `live_phases.csv` (live conversations and TTFT over time, statistics per phase);
`gap_distribution.png`; `summary.csv` and `paired.csv` (the tables below); `iostat_vdc.log` (SSD samples every 10 s over
the whole campaign). Regenerate from `scripts/`:

```
python3 timeline.py --manifest ../results/compare_20261002_200123/manifest.txt
python3 live_timeline.py --manifest ../results/compare_20261002_200123/manifest.txt --label "x1, campaign 12" --ref ../results/compare_20260923_031116/manifest.txt --ref-label c9 --out ../results/compare_20261002_200123
python3 campaign_summary.py --manifest ../results/compare_20261002_200123/manifest.txt --iostat ../results/compare_20261002_200123/iostat_vdc.log
python3 gap_distribution.py --set x1=../results/compare_20261002_200123/manifest.txt --set x10=../results/compare_20261002_060200/manifest.txt --set x70=../results/compare_20261001_211629/manifest.txt --this x1 --iostat ../results/compare_20261002_200123/iostat_vdc.log --title "campaign 12 (swebench trace, OFFSET=32, Poisson arrivals, 30-50 turns)" --out ../results/compare_20261002_200123/gap_distribution.png
```

## Result

Whole run, returning turns. A return counts as recompute when more than half of its prompt beyond the 7,808 shared
tokens was uncached; `compare.csv` and `compare.png` use timeline.py's older whole-prompt rule, so their tier counts
differ.

| | hbm_host | three_tier_to | three_tier_wc | hbm_lru |
|---|---|---|---|---|
| turns / errors / conversations cut by the wall cap | 4,671 / 0 / 0 | 4,671 / 0 / 0 | 4,671 / 0 / 0 | 4,671 / 0 / 0 |
| client wall (s) | 5,449 | 3,886 | 3,908 | 9,293 |
| 90 % of conversations ended (min) | 80.3 | 54.3 | 54.6 | 145.5 |
| returning-turn TTFT mean / p50 / p90 / p99 (s) | 52.6 / 15.6 / 164.8 / 182.7 | 32.3 / 15.0 / 101.8 / 119.7 | 32.7 / 14.8 / 103.2 / 124.0 | 99.1 / 115.5 / 187.5 / 203.4 |
| worst 10-min window, median TTFT (s) | 169.7 (from min 40) | 102.5 (from min 20) | 103.9 (from min 30) | 195.1 (from min 70) |
| turns done at 95 % (s) | 4,725 | 3,160 | 3,177 | 8,095 |
| server queue wait, mean (s) | 51.3 | 31.5 | 31.9 | 96.6 |
| returning turns: device / host / SSD / recompute | 968 / 2,160 / 0 / 1,415 | 966 / 2,275 / 617 / 685 | 972 / 2,266 / 610 / 695 | 944 / 0 / 0 / 3,599 |
| uncached tokens per returning turn (mean) | 6,188 | 3,112 | 3,145 | 14,018 |
| SSD restore TTFT p50 / p90 (s), tokens per restore (median) | – | 95.1 / 113.8, 18.2K | 96.8 / 115.1, 18.3K | – |
| tokens: d2h / h2s / h2d / s2h | 29.2M / – / 34.8M / – | 15.2M / 15.2M / 48.8M / 11.8M | 15.4M / 15.4M / 48.7M / 11.8M | – / – / – / – |
| tokens evicted from device / host | 63.3M / 27.6M | 63.4M / 25.4M | 63.4M / 25.5M | 64.1M / – |
| peak scheduler queue | 71 | 70 | 70 | 81 |
| SSD read / written (TB); %util mean / p90 | – | 1.14 / 1.46; 24 % / 55 % | 1.13 / 1.48; 25 % / 57 % | – |
| SSD reads while active, mean / peak (MiB/s); samples at ≥ 95 % util | – | 602 / 1,859; 0 % | 609 / 1,901; 3 % | – |

Same (conversation, turn), mean difference with 95 % bootstrap CI (`paired.csv`, 4,543 pairs each):
- `three_tier_to` − `hbm_host`: **−20.3 s [−21.2, −19.4]**, p50 −0.41 s.
- `three_tier_wc` − `three_tier_to`: +0.40 s [+0.23, +0.58], p50 0.00 s.
- `hbm_lru` − `hbm_host`: +46.6 s [+45.1, +48.0], p50 +31.3 s.

By phase (`live_phases.csv`): arrival lasts until the last conversation arrives; peak lasts until the live count first
falls below 90 % of the arm's own peak (108 live in the HiCache arms, 121 in `hbm_lru`).

| TTFT mean / p90 (s), span (min) | arrival | peak | departure |
|---|---|---|---|
| `hbm_host` | 6.35 / 16.1, 0-12.3 | 73.6 / 154.2, 12.3-28.2 | 101.0 / 174.0, 28.2-90.8 |
| `three_tier_to` | 6.42 / 16.2, 0-12.3 | 41.5 / 92.4, 12.3-21.8 | 60.9 / 113.0, 21.8-64.8 |
| `three_tier_wc` | 6.38 / 16.0, 0-12.3 | 43.3 / 90.0, 12.3-22.2 | 61.1 / 113.9, 22.2-65.1 |
| `hbm_lru` | 8.61 / 34.5, 0-12.3 | 106.2 / 127.6, 12.3-41.9 | 154.7 / 196.4, 41.9-154.9 |

## What it shows

1. **The SSD arms cut `hbm_host`'s TTFT by 38 %.** They hold mean TTFT at 32.3-32.7 s against 52.6 s, and save 20.3 s
   per turn. They recompute 685-695 returns (15 %), where `hbm_host` recomputes 1,415 (31 %), and they end 90 % of the
   conversations in 54 minutes against 80. `hbm_lru` averages 99.1 s, 46.6 s per turn above `hbm_host`.
2. **The SSD is reached through re-issued lookups.** 620 of the 630 (`three_tier_to`) and 606 of the 616
   (`three_tier_wc`) requests that read from the SSD did so after a re-issued lookup, i.e. the #39283 admission
   re-query or a paced retry, a median 91-92 s after the request arrived (`scripts/requery_share.py`). A restored turn's
   TTFT is therefore mostly queueing: p50 95-97 s, p90 114-115 s.
3. **The GPU queue binds, not the SSD.** Each SSD arm reads 1.1 TB at a mean %util of 24-25 % (p90 55-57 %), and only
   0-3 % of the samples in which it reads reach 95 %. The mean server queue wait is 31.5-31.9 s, and the scheduler
   queue peaks at 70.
4. **The arrivals spread the load.** 2,058-2,060 returning turns (45 %) fall in the 12.3-minute arrival phase, at 6.4 s
   mean on every HiCache arm. Host evictions start at 11.0-11.1 min, just before the last arrival.
5. **The prefetch policy moves the mean by 0.4 s** (`wait_complete` − `timeout`: +0.40 s [+0.23, +0.58]).

Against campaign 9, which started all 128 conversations at once on the pre-merge engine, `hbm_host` falls from 130.5 to
52.6 s and `hbm_lru` from 150.0 to 99.1 s. The SSD arms fall from 131 to 32-33 s, a change that also includes the
#39283 re-query, which campaign 9 did not have. The paired comparison is in `../campaign12/README.md`.

## Notes and deviations

- **Engines.** All four arms ran the frozen `2abe005bb2` engine (`agent_cache/frozen/engine_2abe005bb2`, recorded in each
  `run.txt`); its `--hicache-storage-next-turn-prefetch` flag was off and the client sends no hints.
- **No wall cap was hit.** `hbm_lru` finished every conversation in 9,293 s.
- **One run per arm** and one draw of arrivals (seed 0) and lengths (seed 120).
- **Worst 10-min window**: the highest median TTFT over fixed 10-minute bins of send time holding at least 20
  returning turns.
- **SSD restores** are the returns counted as SSD in the table; tokens per restore is their median storage-hit count.
- **SSD traffic** is integrated from `iostat_vdc.log` over each arm's client window. "While active" counts the samples
  reading more than 1 MiB/s.
- `gap_distribution.png` shows the gaps of all three campaign-12 scales; its dashed lines mark the SSD read time of
  this campaign's median restore at the device ceiling and at the loaded read rate.
