# Campaign 10 — realistic agentic cadence: 80 live sessions × 50 turns (2026-09-24)

Four arms, identical pools, identical client workload, one fresh boot each (`scripts/run_compare.sh`; settings in
`config.txt`, rationale and go/no-go in `DECISIONS.md`).

| arm | server | pools |
|---|---|---|
| `hbm_lru` | no HiCache, radix LRU | L1 668,160 tokens (natural pool, bf16 KV) |
| `hbm_host` | HiCache, host tier, write-through | L1 + L2 160 GB (1,627,648 tokens) |
| `three_tier_to` | + nixl L3 on the attached SSD, `timeout` prefetch | L1 + L2 + L3 (wiped before boot) |
| `three_tier_wc` | same, `wait_complete` prefetch | same |

Workload (chosen from a verified literature survey, `../analysis_gapscale/workload_survey/`):
- **80 live sessions**, closed loop with replacement, entering at 0.1 sessions/s (MORI runs 20-80 concurrent agent
  programs per replica on this GPU and model; AgentX uses 32-100 live session trees).
- **50 turns** per session: the 283 SWE-bench conversations of the 262K trace with ≥ 45 turns; context grows from
  ~10K to ~44K tokens.
- **Gaps calibrated to TraceLab** (4,265 real Claude Code / Codex sessions), baked into the trace
  (`traces/lmcache_agentic_trace_262k_ge45_tl_s1.json`): tool time p50 0.1 s / p90 10 s (cap 3.2 min), plus a human think
  pause after 1 in 8 calls, p50 1.4 min / p90 20.6 min (cap 30 min). Mean 49 s, median 0.19 s; 2.2% of gaps ≥ 10 min.
  A synthetic cadence, not a replay of recorded timing.
- 7,200 s per arm; analysis window 30-120 min of client time. Qwen3-30B-A3B-Instruct-2507 on one H200.

Files: `compare.png`, `compare.csv`, `timeline_<arm>.png`, `turns_<arm>.csv`, `events_<arm>.csv`, `gapclass.csv`,
`gapclass_paired.csv`, `iostat_vdc.log`. Regenerate: `python3 scripts/timeline.py --manifest manifest.txt` and
`python3 scripts/gapclass.py --manifest manifest.txt`.

## Result

Whole arm, returning turns (`compare.csv`):

| | hbm_lru | hbm_host | three_tier_to | three_tier_wc |
|---|---|---|---|---|
| turns / errors | 3,832 / 0 | 9,358 / 0 | 9,523 / 0 | 9,522 / 0 |
| turns/s, minutes 30-120 | 0.33 | 1.26 | 1.29 | 1.29 |
| TTFT mean / p50 / p90 / p99 (s) | 53.7 / 14.6 / 141.7 / 170.7 | 0.45 / 0.27 / 0.62 / 4.90 | 0.35 / 0.28 / 0.57 / 1.37 | 0.36 / 0.29 / 0.57 / 1.35 |
| returns: device / host / SSD / recompute | 1,939 / 0 / 0 / 1,780 | 7,957 / 1,014 / 0 / 159 | 8,276 / 829 / 185 / 0 | 8,283 / 821 / 185 / 0 |
| peak scheduler queue | 51 | 6 | 3 | 3 |

By the gap that preceded the turn, minutes 30-120, TTFT mean (95% bootstrap CI), and the same (conversation, turn)
paired against `hbm_host` (`gapclass.csv`, `gapclass_paired.csv`):

| gap | share | hbm_host: tier, TTFT | three_tier_to: tier, TTFT | paired difference |
|---|---|---|---|---|
| < 1 min | 89% | GPU 93%; 0.43 s | GPU 96%; 0.35 s | −0.08 s [−0.09, −0.06] |
| 1-5 min | 7% | host 93%; 0.57 s | host 92%; 0.39 s | −0.18 s [−0.28, −0.10] |
| 5-10 min | 1.2% | host 84%, recompute 15%; 1.14 s | host 81%, SSD 20%; 0.47 s | −0.72 s [−1.12, −0.35] |
| ≥ 10 min | 2.4% | recompute 100%; 3.21 s | SSD 99%; 1.02 s | **−2.20 s [−2.55, −1.85]** |
| all | | 0.52 s | 0.38 s | −0.14 s [−0.17, −0.12] |

`three_tier_wc` matches `three_tier_to` in every row (≥ 10 min: −2.14 s [−2.50, −1.82]).

## What it shows

1. **The tier a turn hits is set by the idle gap before it.** Tool-call gaps (under 1 min) are served from the GPU,
   pauses of 1-5 min from host memory, and returns after ≥ 10 min fall out of host memory: recomputed without an SSD,
   restored from it with one. The SSD's whole benefit sits in the 2-4% of turns that follow a human think pause.
2. **The SSD turns those returns from a 3.2 s recompute into a 1.0 s restore** (paired −2.2 s) and cuts the p99 of all
   returning turns from 4.90 to 1.37 s. Short-gap turns also get faster (−0.08 to −0.18 s mean): the host-only arm's
   long recomputes slow the batches they share. At the median the SSD arms are ~0.02 s slower (paired p50 +0.02 s),
   a small fixed cost of the storage tier whose cause is not established.
3. **The prefetch policy does not matter at this load.** The SSD was lightly used (≤ 34% utilization; 678 GB read and
   1,421 GB written over the campaign, i.e. per SSD arm ~347 GB restored and ~725 GB written through), so `timeout` and
   `wait_complete` are each other's replicate: they differ by ≤ 0.06 s, against the 2.2 s SSD effect.
4. **Restores are exposed but small.** 180 per SSD arm in the window, ~19.2K tokens (~1.9 GB) each, all after pauses of
   7.7-33 min (median 22 min). The SSD fetch takes 0.62 s on average, 61% of a restored turn's 1.02 s TTFT; hiding it
   would bring these turns close to a host hit (~0.3-0.4 s). This is the realistic-cadence headroom for starting the
   fetch before the turn arrives, and the pause that would have to be predicted is a human's, not a tool's.
5. **GPU memory alone collapses.** `hbm_lru` holds ~40% of the working set: every return after more than about a
   minute recomputes, the queue grows to 51, and TTFT reaches 111 s mean in the window (server queue wait 107 s). Its
   admission-time tier shares show the queue-time invalidation of campaigns 8-9: 100% recompute in the window.

## Notes and deviations

- Single run per arm, one gap seed (seed 1 = the median of seeds 1-7 on mean gap and on the ≥ 10 min share).
- `three_tier_to` took 408 s to boot (vs 3-4 min before), most likely the SSD processing the discard of the 331 GB
  deleted just before; no effect on the measurement.
- The host was shut down at 08:23Z, after `ALL DONE` and `timeline.py`; the interrupted `gapclass.py` step was re-run
  after the host came back (17:01Z). No run data was affected.
- The last SSD arm's L3 store (~676 GB) is still on /mnt/ssd; the next three-tier arm wipes it.
