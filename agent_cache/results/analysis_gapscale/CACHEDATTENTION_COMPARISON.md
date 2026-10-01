# Admission-anchored staging vs CachedAttention

CachedAttention = Gao et al., "Cost-Efficient Large Language Model Serving for Multi-turn Conversations with
CachedAttention", USENIX ATC '24 (arXiv 2403.19708; v1-v2 titled AttentionStore). Our design = admission-anchored staging
(AAS), `TAKEAWAYS.md`. Statements about the paper were checked against its text (104 checks: 92 accurate, 11 partly
accurate and reworded, 1 unverifiable and dropped); statements marked *(inference)* are ours, not paper facts. Evidence in
`cachedattention_comparison/`. Engine facts refer to the fork that ran campaigns 7-10 (commit 81124a061, storage retry
poll interval 0, so nothing re-queried the SSD for a queued request). Upstream #39283 (admission-time re-query; it also
made paced miss-retries the default, 8 passes / 8 attempts) entered the fork in merge ace8ac79a7 on 2026-09-24, after
campaign 10, and was pulled into this checkout on 2026-10-01.

## Bottom line

AAS extends CachedAttention's scheduler-aware idea into the time domain, for agent sessions on a write-through,
radix-tree engine. It is not a new paradigm. For queued requests under FCFS, the predicted admission time
Â = now + position / admission rate orders sessions exactly as CachedAttention's queue position does, and sessions still
between turns rank after all queued ones, which is CachedAttention's implied order. Most of the dense-load (x10/x1) loss
we measured comes from mechanisms our fork lacked and CachedAttention has (queue-driven fetching and eviction), not from a
gap only AAS fills; whether those mechanisms remove it is untested.

## What CachedAttention does

- **AttentionStore:** every request's KV is saved to host DRAM + SSD and fetched back when the session resumes, subject to
  eviction out of the store when the disks are full and a per-session TTL. HBM holds the running job's KV plus read/write
  buffers, not a cache that persists across turns.
- **Layer-wise pre-loading** (host → HBM) overlapped with the new tokens' prefill; an HBM read buffer lets loading start
  while the previous job runs, sized Sbuf = B(Tload·Lhist − Tpref·Lnew).
- **Asynchronous saving** (HBM → host), layer by layer during the job, with an HBM write buffer.
- **Scheduler-aware fetching:** disk → host prefetch for the first Lpw = Cmem/Skv waiting jobs from the queue head.
- **Scheduler-aware eviction:** host → disk demotion gives priority to items nearer the tail of a look-ahead window (the
  Fig. 9 walkthrough first looks for DRAM-resident KV with no queued job); items inside a (Cmem+Cdisk)/Skv-job window
  are never dropped from the store, though they can still be demoted to disk.
- **KV decoupled from positional encoding,** so saved KV can be truncated when a context overflows.
- **Evaluation:** ShareGPT with Poisson session arrivals; the paper describes no model of time between a session's turns.
  86% hit rate vs 58% for LRU and 48% for FIFO at 128 GB DRAM / 10 TB SSD; an arrival-rate sweep (0.5-2.0 sessions/s)
  lowers the hit rate from 82% to 77%.

## Where AAS stands out

| | CachedAttention | AAS | Strength |
|---|---|---|---|
| Cross-regime measurement | arrival-rate sweep only; no gap-scale variation, queue depth or lead time reported | agent traces at x70 / x10 / x1 + realistic cadence: backlogged SSD reads at x70, queue-time eviction and recompute at x10/x1, neither at realistic cadence | strong (measured) |
| Sessions between turns | only trigger: a queued job inside the head window | stage a session's SSD context before its next turn arrives | moderate: 82-92% of live sessions had no queued job in the sparse/realistic runs |
| Lead time | job counts sized by capacity | seconds: own read + earliest-deadline-first read backlog + margin | moderate: at x70, 58-80% of restore TTFT is waiting behind other reads; own-read lead hides 16-24%, 10 s lead 81-91% (oracle simulation) |
| Misprediction | no prediction | low-quantile gap, fall back to fetch-at-arrival + re-check at admission | moderate: over-estimates lose only the gain; a −25% gap error with a 2 s margin keeps 85-88% |

Weaker *(inference)*: CachedAttention's design partly avoids two capacity costs AAS carries. With no cross-turn GPU cache
it has no idle L1 mirror (28.7-30.6% of our L2 at x70, 3-7% at x10/x1; the whole mirror is ~40%), though it does hold
running jobs' KV in DRAM. Whether it avoids SSD rewrites (~94% of our x10/x1 writes) is unknown: the paper does not say
whether a disk copy survives a fetch. Page-level radix residency with a shared prefix is a real difference
(whole-session items would re-read ~1.4x the bytes), but it comes from SGLang, not from AAS.

## Where CachedAttention already does it

Keeping session KV in DRAM + SSD; placement driven by the job queue rather than LRU/FIFO history; prefetching
disk-resident KV for queued jobs; demotion priority for items near the window's tail; exempting near-future work from
being dropped from the store, within a budget; asynchronous saving; layer-wise pre-loading. Its queue mechanisms target
our dense-load failure; whether they fix it is untested (no CachedAttention baseline run; at x1 the queued private context,
1.7-1.8M tokens, exceeds the host-only part of L2, at most 0.96M, so some SSD reload remains under any order). In our SSD
arms the arrival-time match refreshed LRU stamps, so host eviction followed arrival order and hit the requests just
behind the queue head first; hbm_host and hbm_lru match only at admission.

**CachedAttention ideas AAS lacks:** a session-aware disk drop policy and per-session TTL (the fork's L3 cleaner drops
only the oldest-mtime files above a filesystem watermark, 70%/60% ≈ 1.75 TB in campaign 10, and never fired while each
campaign-10 SSD arm wrote ~725 GB); no idle HBM/host mirror across turns; an HBM read buffer filled before admission;
truncation via decoupled positional encoding (never exercised: no prompt overflowed); a dollar-cost model.

## Improvements, in priority order

1. **Queue-order eviction plus fair baselines:** upstream's re-query (#39283) cut x10 mean returning-turn TTFT from 116
   to 68 s in campaign 11 (paired vs the same engine with it off: −47.8 s); 93% of the remaining recomputes are blocked
   by a shared retry budget. Next: re-query without paced polls, and a faithful CachedAttention baseline. If they match
   AAS at x10/x1, drop the dense-regime novelty claim.
2. **Skip the arrival-time SSD lookup for fresh tool output** (cost measured in campaign-10 logs; fix untested): GPU-hit
   returns with a new tail ≥ 256 tokens wait p50 38 ms in the SSD arms vs 1.0 ms in hbm_host (same-arm tails < 256:
   ~1.1 ms); it is the SSD tier's only measured latency cost at realistic cadence (~6% of mean TTFT).
3. **Deduplicate SSD writes:** the nixl backend's `batch_set_v1` writes every page without an existence check, unlike
   the file backend; ~−90% writes at x10/x1.
4. **Measure the dense regime open-loop** (Poisson session arrivals) on throughput, and fill in x20-x50.
5. **Human returns from signals, not durations:** turn-end type plus a client warm-up hint; every campaign-10 restore
   followed a 7.7-33 min pause, and an uncontended restore needs only ~0.6 s of warning.
6. **Pre-arrival staging with a real predictor,** against a fixed 10-20 s lead (oracle first).
7. **Copy KV to host at request finish, not at end of prefill:** at x10/x1 only 3-7% of L2 is idle mirror, so ~33-36% of
   L2 holds copies of running contexts.
8. **Compute or load by backlog** instead of waiting then timing out (x70 timeout arm: 64 restores timed out after their
   reads queued 7.4 s on average, and those turns recomputed, 10.4 s mean TTFT).
9. **Working-set admission control keyed on Â:** the strongest potential differentiator (the CachedAttention paper
   describes no admission reordering), but large and needs a fairness bound.

Lower priority: session-aware L3 drop policy; restores ahead of write-through on the SSD (after a `fio` test); HBM read
buffer; invalidating chains orphaned by context compaction; Â-aware routing across instances.

## Risks to the claim

- AAS is not implemented: every gain above is an oracle or model estimate; no CachedAttention baseline has been run.
- At realistic cadence the timing lever is worth ~0.6 s on ~2.6% of turns.
- Cheaper levers (mirror reclaim, write dedup) may shrink the problem AAS targets.
- Results are single-seed; x10/x1 queues were shaped by the old 100-connection client cap.
