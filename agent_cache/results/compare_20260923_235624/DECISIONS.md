# Campaign 10 — realistic agentic cadence: 80 live sessions x 50 turns, TraceLab tool + think gaps

Started 2026-09-23T23:56:24Z, driver log `../compare_driver_c10.log`. Same model, pools and four arms as campaigns 7-9
(Qwen3-30B-A3B-Instruct-2507, bf16 KV, L1 668,160 tok, L2 160 GB write-through, L3 nixl on /mnt/ssd; hbm_host,
three_tier_to, three_tier_wc, hbm_lru). What changes is the workload, chosen from a literature survey
(`../analysis_gapscale/workload_survey/`) to be less extreme than 128 sessions at x10/x1 and more realistic than a
uniform x70 stretch of the recorded gaps.

## Setting

- Sessions: closed loop with replacement, at most 80 live (C=80). A session keeps its slot through its gaps; when it
  ends the next conversation starts. Conversations enter as a seeded Poisson process at 0.1/s (ARRIVAL=0.1, SEED=0),
  so the 80 slots fill in ~13 min. C=80 = MORI's top concurrency per replica (H200, Qwen3-30B-A3B); AgentX (AIPerf)
  uses 32-100 live session trees; the SGLang HiCache blog example uses 80 clients.
- Conversations: the 283 of the 262K trace with >= 45 turns, truncated at 50 turns (13,989 turns; mean 49.4). Prompt
  p50 ~10K at turn 1 to ~44K at turn 50. Trace: `traces/lmcache_agentic_trace_262k_ge45_tl_s1.json`, made by
  `scripts/make_tl_trace.py --seed 1` (seed 1 = median of seeds 1-7 on mean gap and on the >=10 min share).
- Gaps (baked into pre_gap, replayed with GAP=1 GAP_CAP=0, identical on every arm): tool = min(192 s, lognormal p50
  0.1 s / p90 10.0 s) = TraceLab (arXiv 2606.30560) Table 7 per-step tool execution (p99 3.2 min as the cap); plus,
  with probability 1/8 ("around 8 steps" per user request), a human think pause = min(30 min, lognormal p50 1.4 min /
  p90 20.6 min) = TraceLab's human thinking events. Result: mean 49.4 s, p50 0.19 s, p90 73 s, p99 27 min; 10.8 % of
  gaps > 1 min, 3.4 % > 5 min, 2.2 % > 10 min; 12.5 % of gaps contain a pause. This is a synthetic cadence calibrated to
  TraceLab, not a replay of recorded timing: the trace's own gaps come from an unattended SWE-bench harness (p50
  0.71 s, no human pauses), under which the host and SSD tiers are never reached at any load this box sustains.
- Wall cap 7,200 s per arm (CLIENT_TIMEOUT, enforced in the client); analysis window 1,800-7,200 s of client time.
- Client fixes since campaign 9: no client-side connection cap (TCPConnector(limit=0)); SGLANG_TIMEOUT_KEEP_ALIVE=3600
  in the server environment; per-request ReqTimeStats logged on every arm.

## Expected (queue-aware model calibrated on campaign 7; a wrong number should be investigated, not believed)

- ~1.35-1.5 turns/s, ~12-15 requests in flight, queue ~0 on the three host-pool arms. Live working set ~1.7M tokens
  (~1.04x L2), peaks ~1.9M.
- Returns by gap class: < 1 min (~90 %) from L1; 1-5 min from L2; 5-10 min partly SSD; >= 10 min (~2.2 % of returns)
  essentially all SSD in the three-tier arms, all recompute in hbm_host. So ~2-3.5 % of returns are SSD restores
  (~200-300 per arm), each ~20K tokens (~1 s at the SSD ceiling). timeout and wait_complete should be identical here
  (the SSD is ~5 % busy): they serve as each other's replicate / noise floor.
- hbm_lru collapses by construction (L1 holds ~40 % of the working set; every return after > ~1 min recomputes).
- Whole-window means will differ little between hbm_host and the SSD arms: the result must be read by gap class and
  with (conv, turn) pairing (`scripts/gapclass.py --manifest manifest.txt`).

## Go / no-go (hbm_host, minutes 30-45)

turns/s 1.3-1.6, running ~12-16, queue <= 5, device token usage < 0.7. If the queue stays > 5 for > 5 min, stop and rerun
all arms at C=72.

## Smoke test (23:42-23:56Z, not part of the campaign)

three_tier_wc with small pools (L1 131,072 tok, L2 20 GB), 24 live sessions, 600 s: 497 turns, 0 errors, 0 retries,
gap_slept equal to the baked gap on every turn, 41 SSD restores (gaps 1 s-10 min), cap acted cleanly, ReqTimeStats on
every request, gapclass.py ran end to end. Its directories (compare_20260923_234157, 20260923_234158_three_tier_wc_*)
were deleted after this note.

## 00:44Z go / no-go (hbm_host, client minutes 30-45): GO

1.24 turns/s (a little under the expected 1.3-1.6 because the mean turn latency is 10.8 s, not 9-10 s), running 13 on
average, queue mean 0.00 / max 2, device token usage 0.53, TTFT mean 0.38 s / p50 0.27 s, 0 errors. The stop criterion
(queue > 5 for > 5 min) is far from triggered; the campaign continues unchanged at C=80.

## 08:15Z: campaign complete (ALL DONE 08:15:42Z); artefacts in README.md

All four arms ran to the 7,200 s cap with 0 errors. Returning-turn TTFT mean: hbm_host 0.45 s, three_tier_to 0.35 s,
three_tier_wc 0.36 s, hbm_lru 53.7 s. Returns after >= 10 min pauses: recomputed in hbm_host (3.21 s mean), restored
from SSD in the three-tier arms (1.02 / 1.08 s), paired difference -2.20 / -2.14 s. The host was shut down at 08:23Z
(after ALL DONE and timeline.py) and came back at 17:01Z; gapclass.py was re-run then. timeline.py was changed after
the run to label the first-turn panel with the real concurrency/arrival rate and to use a log x-axis when arms span
more than ~2 decades; compare.png was regenerated.
