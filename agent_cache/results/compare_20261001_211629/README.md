# Campaign 12, x70 — campaign 7 with Poisson arrivals and 30-50-turn conversations (2026-10-01/02)

Campaign 7 (`../compare_20260922_023714/`) rerun with users who come and go. The same 128 conversations of the 262K
agentic trace (offset 32) arrive as a Poisson process at 0.2/s (seed 0; the last arrives at 12.3 min), and each replays
min(recorded turns, U[30, 50]) turns (seed 120): 4,671 turns and 112.4M prompt tokens, against campaign 7's 4,676 and
112.2M. Gaps are the recorded ones x70, capped at 1,200 s; wall cap 12,600 s per arm. Qwen3-30B-A3B-Instruct-2507, bf16
KV, one H200, merged engine (upstream #39283: admission-time SSD re-query, paced retries 8/8).

This is the x70 scale of campaign 12. The x10 and x1 scales are in `../compare_20261002_060200/` and
`../compare_20261002_200123/`; the comparison with the all-at-once baselines is `../campaign12/README.md`, and the
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
python3 timeline.py --manifest ../results/compare_20261001_211629/manifest.txt
python3 live_timeline.py --manifest ../results/compare_20261001_211629/manifest.txt --label "x70, campaign 12" --ref ../results/compare_20260922_023714/manifest.txt --ref-label c7 --out ../results/compare_20261001_211629
python3 campaign_summary.py --manifest ../results/compare_20261001_211629/manifest.txt --iostat ../results/compare_20261001_211629/iostat_vdc.log
python3 gap_distribution.py --set x1=../results/compare_20261002_200123/manifest.txt --set x10=../results/compare_20261002_060200/manifest.txt --set x70=../results/compare_20261001_211629/manifest.txt --this x70 --iostat ../results/compare_20261001_211629/iostat_vdc.log --title "campaign 12 (swebench trace, OFFSET=32, Poisson arrivals, 30-50 turns)" --out ../results/compare_20261001_211629/gap_distribution.png
```

## Result

Whole run, returning turns. A return counts as recompute when more than half of its prompt beyond the 7,808 shared
tokens was uncached; `compare.csv` and `compare.png` use timeline.py's older whole-prompt rule, so their tier counts
differ.

| | hbm_host | three_tier_to | three_tier_wc | hbm_lru |
|---|---|---|---|---|
| turns / errors / conversations cut by the wall cap | 4,671 / 0 / 0 | 4,671 / 0 / 0 | 4,671 / 0 / 0 | 4,671 / 0 / 0 |
| client wall (s) | 8,734 | 5,328 | 5,263 | 11,386 |
| 90 % of conversations ended (min) | 118.5 | 62.3 | 60.7 | 160.2 |
| returning-turn TTFT mean / p50 / p90 / p99 (s) | 57.3 / 0.59 / 180.9 / 192.7 | 2.41 / 0.29 / 7.13 / 34.1 | 2.21 / 0.28 / 9.11 / 23.2 | 84.6 / 84.5 / 183.5 / 197.1 |
| worst 10-min window, median TTFT (s) | 183.1 (from min 50) | 3.09 (from min 30) | 0.51 (from min 30) | 189.4 (from min 100) |
| turns done at 95 % (s) | 6,512 | 3,137 | 3,043 | 8,805 |
| server queue wait, mean (s) | 55.7 | 2.00 | 1.94 | 82.0 |
| returning turns: device / host / SSD / recompute | 1,228 / 1,503 / 0 / 1,812 | 1,648 / 1,863 / 825 / 207 | 1,800 / 1,782 / 867 / 94 | 773 / 0 / 0 / 3,770 |
| uncached tokens per returning turn (mean) | 8,362 | 1,004 | 560 | 15,016 |
| SSD restore TTFT p50 / p90 (s), tokens per restore (median) | – | 3.81 / 22.9, 19.3K | 10.5 / 21.8, 19.1K | – |
| tokens: d2h / h2s / h2d / s2h | 39.1M / – / 22.2M / – | 5.63M / 5.63M / 46.9M / 16.6M | 3.61M / 3.61M / 45.6M / 17.1M | – / – / – / – |
| tokens evicted from device / host | 60.6M / 37.4M | 51.8M / 20.6M | 48.5M / 19.1M | 68.6M / – |
| peak scheduler queue | 74 | 39 | 26 | 75 |
| SSD read / written (TB); %util mean / p90 | – | 1.65 / 0.54; 18 % / 99 % | 1.64 / 0.35; 17 % / 99 % | – |
| SSD reads while active, mean / peak (MiB/s); samples at ≥ 95 % util | – | 954 / 2,081; 32 % | 1,055 / 2,010; 36 % | – |

Same (conversation, turn), mean difference with 95 % bootstrap CI (`paired.csv`, 4,543 pairs each):
- `three_tier_to` − `hbm_host`: **−54.9 s [−57.1, −52.7]**, p50 −0.14 s.
- `three_tier_wc` − `three_tier_to`: −0.20 s [−0.35, −0.05], p50 0.00 s.
- `hbm_lru` − `hbm_host`: +27.3 s [+26.0, +28.5], p50 +12.0 s.

By phase (`live_phases.csv`): arrival lasts until the last conversation arrives; peak lasts until the live count first
falls below 90 % of the arm's own peak (128 live in every arm).

| TTFT mean / p90 (s), span (min) | arrival | peak | departure |
|---|---|---|---|
| `hbm_host` | 0.23 / 0.49, 0-12.3 | 56.5 / 180.0, 12.3-53.6 | 105.9 / 185.5, 53.6-145.6 |
| `three_tier_to` | 0.24 / 0.48, 0-12.3 | 2.72 / 6.70, 12.3-29.7 | 3.61 / 14.2, 29.7-88.8 |
| `three_tier_wc` | 0.24 / 0.51, 0-12.3 | 1.93 / 4.95, 12.3-28.5 | 4.07 / 17.9, 28.5-87.7 |
| `hbm_lru` | 1.76 / 6.85, 0-12.3 | 89.7 / 165.7, 12.3-72.2 | 132.2 / 191.3, 72.2-189.8 |

## What it shows

1. **The SSD tier carries the load.** Both SSD arms hold mean TTFT at 2.2-2.4 s, against 57.3 s for `hbm_host` and
   84.6 s for `hbm_lru`, and save 54.9 s per turn over `hbm_host`. They restore 825-867 returns (18-19 %) from the SSD
   and recompute 94-207 (2-5 %), where `hbm_host` recomputes 1,812 (40 %). They end 90 % of the conversations in about
   an hour; `hbm_host` takes two hours and `hbm_lru` 2.7.
2. **The arms split when the live set outgrows L2.** Through the 12.3-minute arrival phase every HiCache arm answers at
   0.23-0.24 s mean. Host evictions start at 15.3 min. From there `hbm_host` recomputes and queues (mean server queue
   wait 55.7 s, queue peak 74), while the SSD arms' queue peaks at 26-39.
3. **The SSD sets the restore time.** 91-92 % of the SSD loads fall between minutes 20 and 40. Over the run the SSD's
   %util p90 is 99 %, and a third of the samples in which it reads are at ≥ 95 %; it reads about 1 GiB/s on average,
   with 10-s peaks near 2 GiB/s. A restored turn waits behind the other reads on the single SSD read thread
   (`../campaign12/README.md`, point 2; `../campaign12/x70/read_backlog*.png`): its TTFT p50 is 3.8 s under `timeout`
   and 10.5 s under `wait_complete`, p90 22-23 s.
4. **The prefetch policy moves the mean by 0.2 s and splits the tails.** `wait_complete` has the higher p90 (9.1 vs
   7.1 s) and the lower p99 (23.2 vs 34.1 s). `timeout` recomputes more returns (207 vs 94) and restores faster (p50 3.8
   vs 10.5 s), as in campaign 7.
5. **`hbm_lru` recomputes 83 % of its returns** and adds 27.3 s per turn to `hbm_host`.

Against campaign 7, which started all 128 conversations at once, every arm lands within 10 s per turn, and the SSD arms'
tails grow (`three_tier_to` p99 17.5 → 34.1 s). The paired comparison is in `../campaign12/README.md`.

## Notes and deviations

- **Engines.** `hbm_host` and `three_tier_to` ran engine `47ceeb6d26` (= `ace8ac79a7`, campaign 11's). `three_tier_wc`
  and `hbm_lru` ran the working tree with oracle step 2 (`2abe005bb2`); its `--hicache-storage-next-turn-prefetch` flag
  was off and the client sends no hints, so no behaviour difference is expected (`../campaign12/DECISIONS.md`).
- **One run per arm** and one draw of arrivals (seed 0) and lengths (seed 120).
- **Worst 10-min window**: the highest median TTFT over fixed 10-minute bins of send time holding at least 20
  returning turns.
- **SSD restores** are the returns counted as SSD in the table; tokens per restore is their median storage-hit count.
- **SSD traffic** is integrated from `iostat_vdc.log` over each arm's client window. "While active" counts the samples
  reading more than 1 MiB/s.
- `gap_distribution.png` shows the gaps of all three campaign-12 scales; its dashed lines mark the SSD read time of
  this campaign's median restore at the device ceiling and at the loaded read rate.
