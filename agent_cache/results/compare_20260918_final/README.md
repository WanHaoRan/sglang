# HiCache tiers under a 32-way agentic replay (2026-09-18)

Three arms, identical pools and identical client workload, one fresh boot each (`scripts/run_compare.sh`):

| arm | server | pools |
|---|---|---|
| `hbm_lru` | no HiCache, radix LRU | L1 65,536 tokens |
| `hbm_host` | HiCache, host tier, write-through | L1 65,536 + L2 12 GB (91,520 tokens) |
| `three_tier` | HiCache, host + nixl L3 on the local NVMe, `timeout` prefetch | L1 65,536 + L2 12 GB + L3 (wiped before boot) |

Client: LMCache swebench trace, 32 conversations all live (`NCONV=32 C=32`), 12 turns each, gaps x10
(`hbm_lru` uncapped and ended after 380/381 turns because conv 27's last gap was 3,006 s; the other two arms
with `GAP_CAP=600`, which changes only that one gap). Qwen3-32B-FP8, fp8 KV (128 KiB/token), A100 80 GB.
Files: `timeline_<arm>.png` (per-arm timeline), `compare.png`, `compare.csv`, `turns_<arm>.csv`, `events_<arm>.csv`.
Regenerate: `python3 scripts/timeline.py --manifest results/compare_20260918_final/manifest.txt`.

## Result: the three arms are indistinguishable

| | hbm_lru | hbm_host | three_tier |
|---|---|---|---|
| returning-turn TTFT p50 / p90 (s) | 141 / 247 | 142 / 248 | 143 / 246 |
| turns done at 95 % (s) | 2163.4 | 2112.5 | 2103.3 |
| returning turns: device / host / storage / recompute | 196 / 0 / 0 / 152 | 156 / 43 / 0 / 150 | 146 / 43 / 11 / 149 |
| uncached tokens per returning turn (mean) | 7,427 | 7,052 | 6,961 |
| tokens: d2h / h2s / h2d / s2h | – | 2.59M / – / 106K / – | 2.55M / 2.55M / 150K / 191K |
| tokens evicted from device / host | 2.67M / – | 2.63M / 2.50M | 2.64M / 2.67M |
| peak scheduler queue | 28 | 28 | 28 |

43-44 % of returning turns recomputed their whole private context in every arm; the tiers restored only in the
first ~100 s (host) plus 11 storage hits. TTFT is dominated by admission queueing (queue ~25 for 35 minutes:
32 live conversations against a 64K-token pool that admits 3-5 requests of 12-20K tokens at a time).

## Why the tiers did not help here (from the event log)

1. **The host pool is smaller than the churn.** L1+L2 = 157K tokens against a live set of ~500K. The host LRU
   evicted 2.5-2.7M tokens, i.e. a conversation's tail left the host before the conversation returned; host hits
   happened only while the total cached KV was still below the pool (first ~100 s).
2. **Prefetch admission is budgeted at half the host pool** (`cache_controller.py:581`: 45,760 tokens), and each
   prefetch reserves its requested length, so about three 15K prefetches fit at once: 145 of 354 prefetch requests
   never reached the L3 lookup (rate-limited).
3. **The `timeout` policy gives up before the SSD can deliver.** Budget ~1 s + 0.25 s per Ki-token (~4.7 s for
   15K tokens); the L3 read ran at <= 0.5 GiB/s because the SSD was busy writing: `h2s_io` occupied 835 s of the
   2,780 s run at the 0.37 GiB/s write ceiling. 167 of 209 lookups ended with zero usable tokens.
4. **Write amplification.** Every recomputed turn re-inserts its KV, which write-through backs up again: 2.55M tokens
   (312 GiB) were written to L3 for ~480K tokens of distinct content (the NVMe holds 50 GB afterwards). The re-writes
   are what starved the reads in (3). Only 194K of the 465K tokens that prefetches asked for were found on L3 at
   lookup time.

## What this says about the design

With pools this far below the live set, the closed-loop 32-way workload is capacity-bound at admission, and the
eviction/restore machinery churns without effect: write-through costs bandwidth on every recompute and the
restores are throttled by the host budget and the prefetch timeout. The tiers can only show a benefit when
(a) L1 admits the live in-flight set (`f * c * context <= L1 / 1.2`, RUNBOOK 5.3), (b) L2 is large enough that a
conversation's tail survives its gap, and (c) prefetch capacity and the timeout budget cover the restore of a full
context at the SSD's contended read rate. Candidate next cells: the same client at `C=8-12` with the runbook's
PH/PL pools (L1 131K, host 48/18 GB), `wait_complete` instead of `timeout`, and a prefetch budget above 0.5 x host.
