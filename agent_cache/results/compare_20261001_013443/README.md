# Campaign 11 — campaign 8's dense x10 workload on the merged engine with upstream #39283 (2026-10-01)

Campaign 8 (`../compare_20260922_165655/`) rerun on HEAD ace8ac79a7, which contains upstream SGLang #39283: at
admission, if a queued request's GPU + host match shrank since arrival, the SSD lookup is re-issued and the request waits
for it. #39283 also made paced miss-retries the default (poll every 8 scheduling passes, at most 8 attempts; campaigns
7-10 ran with retries off). Setup changes, smoke test and per-arm notes in `DECISIONS.md`.

| arm | server |
|---|---|
| `hbm_host` | HiCache host tier, write-through: L1 668,160 tokens + L2 160 GB (1,627,648 tokens) |
| `three_tier_to` | + nixl L3 on the attached SSD (wiped before boot), `timeout` prefetch |
| `three_tier_wc` | same, `wait_complete` prefetch |
| `hbm_lru` | no HiCache, radix LRU, L1 only |
| `three_tier_wc_norq` | `three_tier_wc` with `--hicache-storage-prefetch-retry-max-attempts 0`: no admission re-query, no paced retries (campaign 8's behaviour on this engine) |

Workload as campaign 8: 128 conversations of the 262K agentic trace (offset 32), 40 turns, all 128 live from the start,
no replacement; recorded gaps x10, capped at 1,200 s; 9,000 s cap per arm. Qwen3-30B-A3B-Instruct-2507, bf16 KV, one
H200. Client without the old 100-connection cap.

Files: `compare.png`, `compare.csv`, `timeline_<arm>.png`, `turns_<arm>.csv`, `events_<arm>.csv`, `gapclass.csv` and
`gapclass_paired.csv` (whole run), `delay_components.png` and `turn_components_*.csv` (campaign 8 vs 11),
`iostat_vdc.log`. Regenerate: `python3 scripts/timeline.py --manifest manifest.txt`,
`python3 scripts/gapclass.py --manifest manifest.txt --window 0,9100`, `python3 scripts/delay_components.py --out . "x10
campaign 8 (before #39283)=../compare_20260922_165655/manifest.txt" "x10 campaign 11 (#39283 merged)=manifest.txt"`.

## Result

Whole run, returning turns. The workload is a fixed batch of 128 conversations, so arms that finish early drain their
load; compare whole runs and same-turn pairs, not a fixed time window.

| | hbm_host | three_tier_to | three_tier_wc | hbm_lru | three_tier_wc_norq |
|---|---|---|---|---|---|
| turns / errors | 4,667 / 0 | 4,676 / 0 | 4,676 / 0 | 4,543 / 0 | 4,666 / 0 |
| 90% of conversations done | 126.8 min | **82.2 min** | **81.0 min** | 150.5 min (cap) | 127.5 min |
| TTFT mean / p50 / p90 / p99 (s) | 115.4 / 134.3 / 252.1 / 266.7 | **69.2 / 79.3 / 145.1 / 169.8** | **68.4 / 74.7 / 144.1 / 174.5** | 139.7 / 146.2 / 252.8 / 266.3 | 116.3 / 135.3 / 251.6 / 265.8 |
| server queue wait, mean (s) | 113.3 | 67.9 | 67.1 | 137.0 | 114.2 |
| returns: device / host / SSD / recompute | 550 / 1,461 / 0 / 2,528 | 603 / 1,434 / 1,150 / 1,361 | 594 / 1,457 / 1,158 / 1,339 | 440 / 0 / 0 / 3,975 | 528 / 1,435 / 9 / 2,566 |
| SSD read / written (iostat) | — | 2.3 / 2.8 TB | 2.3 / 2.8 TB | — | 0.05 / 5.2 TB |

Same (conversation, turn), mean difference with 95% bootstrap CI:
- `three_tier_wc` − `three_tier_wc_norq`: **−47.8 s [−49.3, −46.5]**, p50 −49.9 s (4,538 pairs).
- `three_tier_to` − `hbm_host`: −46.1 s [−47.5, −44.7]; `three_tier_wc` − `three_tier_to`: −0.8 s [−1.1, −0.6].
- `three_tier_wc_norq` − `hbm_host`: +0.9 s [+0.6, +1.2]; `three_tier_wc_norq` − campaign 8 `three_tier_wc`: +0.2 s
  [−0.0, +0.4].

## What it shows

1. **#39283 is the whole change.** With the re-query off, the SSD arm matches `hbm_host` and campaign 8 to within 1 s.
   With it on, both SSD arms cut mean TTFT by 47 s (−41%) and finish 90% of the conversations 45 min sooner. The non-SSD
   arms reproduce campaign 8 (hbm_host 115.4 vs 114.8 s, hbm_lru 139.7 vs 138.7 s): neither the ~945 merged upstream
   commits nor removing the client's connection cap changed them.
2. **Mechanism: queue-time losses come back from the SSD instead of being recomputed.** 1,141-1,145 of the ~1,150 SSD
   restores per arm came from the admission re-query. Each costs ~4 s at the queue head: 2.1 s from re-query to data in
   host memory (0.8 s of it the read itself), then ~2.0 s to admission. Recomputes fall from 56% to 30% of returns, so
   prefill work drops (uncached tokens per turn 11.5K → 6.2K), admission speeds up and the mean queue wait falls from
   113 to 67 s. Most of the TTFT gain is shorter queueing, not faster restores.
3. **Most remaining recomputes are a retry-budget artefact.** 1,263 of `three_tier_to`'s 1,361 recomputes (93%; 1,263 of
   1,339 in `three_tier_wc`) had already spent all 8 attempts while queued. The paced miss-polls re-query the new tail
   of the prompt, which is fresh tool output and can never be on the SSD, every 8 passes, and a ~118 s queue wait
   exhausts the budget. At admission the context had been evicted, but the shared counter blocks the re-query
   (`scheduler.py` `_prefetch_after_device_hit_loss`), so the turn recomputes. Re-query alone (poll interval 0, max
   attempts 8) should convert most of these into restores; untested.
4. **The prefetch policy does not matter** (`timeout` vs `wait_complete`: −0.8 s mean, 0.0 s median).
5. **SSD traffic:** each re-query arm read ~2.3 TB (s2h 23.8-23.9M tokens) and wrote ~2.8 TB, mean utilization 25%,
   p90 58%, peaks 99%. With the re-query off the arm wrote 5.2 TB, nearly all rewrites: the store ended at 330 GB, so
   the L3 cleaner (70%/60% watermarks) never fired.

## Notes and deviations

- Single run per arm. Gap classes are uninformative at x10 (99% of gaps are under 1 min; `gapclass.csv`).
- `delay_components.py` now times an L3 read from the last lookup before it. The first-lookup version attributed the whole
  queue wait of a re-queried turn (~116 s) to the SSD read.
- `three_tier_wc_norq` ran from its own driver (`../compare_20261001_104416/`, `../chain_c11_norq.sh`); its manifest line
  was appended here. hbm_host, hbm_lru and the norq arm hit the 9,000 s cap with 2, 27 and 2 conversations unfinished; the
  re-query arms finished every conversation.
