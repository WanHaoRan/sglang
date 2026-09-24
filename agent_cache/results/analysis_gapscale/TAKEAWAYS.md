# Prefetch timing in SGLang HiCache: two takeaways, checked

Evidence: code under `python/sglang/srt` (the live cache class is `UnifiedRadixCache`, not `HiRadixCache`); campaigns 7/8/9
= gap scale x70/x10/x1, 128 live sessions, SWE-bench replay, up to 40 turns, mean 36.5 (`../compare_20260922_023714`,
`../compare_20260922_165655`, `../compare_20260923_031116`); campaign 10 = 80 live sessions × 50 turns at a
TraceLab-calibrated cadence (`../compare_20260923_235624`). Raw analysis, scripts and citation checks:
`takeaway_assessment/`. "to" / "wc" = the `timeout` / `wait_complete` prefetch-policy arms.

## Takeaway 1 — sparse load: start the SSD fetch before the next turn arrives

**Claim.** Prefetch is issued when the request enters the queue; with a near-empty queue the fetch is exposed. Agent gaps
are predictable and the context is reused, so start the fetch earlier from a predicted tool-return time plus queue delay.

**Verdict.** Partly right. Worth building for the sparse regime after the corrections below. Not new as an idea.

Confirmed:
- The SSD lookup starts at arrival (`_add_request_to_queue` → `_prefetch_kvcache`, scheduler.py:3148-3155); `prefetch_start`
  is logged within 1-2 ms of the request's queue entry.
- At x70, 19.7% (to) / 18.6% (wc) of returning turns are restored from SSD, at TTFT p50 2.76 / 3.34 s vs 0.25 / 0.24 s
  for host hits.
- 97.3% of replies are followed by another turn; every SSD-hit gap was ≥ 46.9 s.

Corrections:
1. **The main cost is waiting behind other SSD restores, not one unhidden read.** Restores run on a single read thread in
   FIFO order and arrive in bursts: that wait is 58% (to) / 80% (wc) of storage-hit TTFT. Hits arriving to an empty queue
   expose only ~0.7 s. The read thread is idle ~75% of the read span, but 10-s demand peaks reach ~2.1x the SSD bandwidth.
2. **The lead must cover the read backlog.** Starting each fetch earlier by only its own read time hides 16-24%; the lead
   needs read time + L3 backlog (10-20 s here), with reads served earliest-deadline-first.
3. **Earlier is not better.** Fetched pages are inserted unprotected (unified_radix_cache.py:2024). Applying the run's
   observed L2 survival curve: fetching at the end of the reply would get ~77% re-evicted before use; 30 s / 60 s leads
   would lose 20-25% / 56-64% (model, not observed).
4. **Predictability is not shown.** The replay gaps are deterministic (trace x70). History-based predictors keep 54-71%
   of the gain; over-estimating the gap by ≥ 25% keeps essentially none.
5. **Under realistic cadence the target moves.** With measured tool times (TraceLab: p50 0.1 s, p90 10 s) contexts stay
   on the GPU across tool calls. Campaign 10 measured it: returns after < 1 min are 96% GPU hits, after 1-5 min 92% host
   hits, and all 180 SSD restores per arm followed pauses of 7.7-33 min (median 22 min). The predictor that matters is
   the human's return, not the tool's.

Headroom (oracle arrival times, EDF reads): a 10 s lead hides 81-91% of fetch time; a 20 s lead cuts mean returning-turn
TTFT 1.94 → 1.17 s (to, -40%) and 1.48 → 0.50 s (wc, -66%). At realistic cadence (campaign 10) the SSD is uncontended:
a restore costs 1.02 s TTFT, of which the fetch is 0.62 s (61%), so hiding it saves ~0.6 s on the ~2.6% of turns that
are restored.

Closest prior work (none times the fetch against a read backlog; all but SYMPHONY move KV only between DRAM and HBM):
TokenCake (arXiv 2510.18586, EuroSys '27: per-function-type duration estimate, gradual CPU→GPU upload before the predicted
return; released code is demand-triggered); Ask the Tool, Don't Guess (arXiv 2609.18849: reload one lead time before a
return predicted from the tool's live progress); SYMPHONY (arXiv 2412.16434: on an advisory, greedy promotion from
disk/host/remote to the fastest tier, no timing rule); KVFlow (NeurIPS '25: workflow-step-driven prefetch).

## Takeaway 2 — dense load: fetch late, just before admission

**Claim.** The engine writes KV to host and SSD right after a request finishes. With long queues the prefetch happens
too early, the prefetched content is evicted by newly finished requests, and the request recomputes. Host and GPU nodes
are coupled by lock status (waste). So start the prefetch later.

**Verdict.** The symptom is real and is the largest prize in the data; the mechanism is mostly different.

Confirmed:
- At x10 / x1, **93-95% of all prefill recomputes context that was in L1+L2 when the request arrived** and on the SSD
  the whole time (write-through copies every host page to SSD, 1:1).
- The loss grows with the server-side wait (x1): 0% under 10 s, 29-35% at 20-30 s, ≥ 94% beyond 60 s; median wait 111 s
  for these turns (102 s over all returns).
- 54% (x10) / 59% (x1) of returning turns are admitted with only the 7,808-token shared system prompt cached; 49% / 55%
  had their whole context in L1+L2 at arrival and lost it.

Corrections:
1. **Write timing.** 72-98% of write volume is triggered when the last prefill chunk completes: the host copy before the
   first token, the SSD copy chained after its host-write ack (p50 lag 0.02 s at x70, ~0.3 s at x10/x1). Intermediate
   chunks trigger no write of their own; they are written with the whole prompt after the last chunk. Only decode pages
   and the page straddling the prompt boundary are written at finish.
2. **It is mostly not prefetched content.** Prefetched-then-evicted pages are 0.4-1.9% of lost tokens. The dominant loss
   is the arrival-time L1/L2 match itself, evicted while the request waits; no SSD fetch was ever issued (an SSD query
   found ≥ 256 tokens for only 68/4,538 returns at x10 and 15/4,548 at x1). Three causes:
   - *No protection while queued.* The only host lock covering a queued request's match is the prefetch anchor: not
     taken when the unmatched tail is < 256 tokens (`prefetch_threshold`), dropped within milliseconds after an SSD miss,
     dropped when fetched pages are inserted.
   - *Worst-case eviction order.* The arrival-time match refreshes LRU stamps (unified_cache/unified_tree_core.py:836-839),
     so host eviction follows arrival order and evicts the requests just behind the head (next to be admitted) first;
     the blocked head itself is re-stamped on each admission attempt.
   - *Self-reinforcing pressure.* ~98% of host writes at x10/x1 come from other requests' prefill completions, mostly
     recomputes: 54-58M tokens written vs 3.6-5.3M in the x70 SSD arms.
3. **Admission never asks the SSD again.** It re-matches L1/L2 only. The paced retry is off by default
   (`--hicache-storage-prefetch-retry-poll-interval 0`); upstream's reactive admission-time re-query (sgl-project/sglang
   #39283, merged 2026-09-15) is absent from this fork.
4. **"Lock status" is inclusion.** A host copy is evictable only after its GPU copy is gone (`_is_host_leaf`,
   unified_cache/unified_tree_core.py:1889), so under write-through L2 ⊇ L1: ~40% of L2 (636-656K tokens) mirrors L1 in
   every run. `write_back` avoids the mirror but pays a synchronous GPU→host copy on eviction (~50x the per-token cost of
   a demote).
5. **"Start the prefetch later" cannot fix the main case, because there is no prefetch.** What is needed: re-resolve
   against the SSD shortly before predicted admission; protect what was matched or fetched until admission (a 10 s
   horizon ≈ 5% of L2 at x1; the whole queue ≈ all of L2); evict latest-predicted-admission first. Just-in-time reload at
   x1 needs ~0.6-0.7 GiB/s, about a third of the SSD.

Closest prior work: CachedAttention (USENIX ATC '24: disk→host prefetch for a look-ahead window of Cmem/Skv queued jobs;
a (Cmem+Cdisk)/Skv eviction window whose items are never dropped from disk; host→disk eviction takes the window's tail
first; windows counted in jobs); PCR (arXiv 2603.23049: SSD→DRAM prefetch for a waiting-queue window, 4 by default, plus
a look-ahead LRU); py-kvcache (arXiv 2609.11744: native vLLM's disk→CPU promotions are evicted before use; py-kvcache
preloads a bounded lookahead of waiting requests, one at a time, yielding to demand loads); SGLang #39283 (reactive
re-query at admission; also pins the device prefix under staged buffer-mode fetches).

## One design covering both: admission-anchored staging

- Predict each session's admission time Â: in a gap, a low quantile of its return time plus the predicted server wait;
  once queued, queue position / admission rate.
- Start its SSD read at Â − (own read + earliest-deadline-first backlog + margin), never earlier than a horizon H.
- Protect its matched and staged pages until admission, within an L2 budget; evict the latest Â first.
- At admission, re-match and re-query the SSD if needed, so a late prediction falls back to today's behaviour; an early
  prediction still costs L2 space and extra SSD reads.
- The same rule starts the read before arrival when the queue is empty (T1) and one or two positions before the head
  under backlog (T2). Optional: reclaim L2 mirrors of idle GPU-resident KV that the SSD already holds.

**Novelty.** No single piece is new (forestall: Kimbrel et al., OSDI '96; CachedAttention; TokenCake). Defensible: the
cross-regime measurement; one admission-anchored controller over in-gap and queued sessions with an SSD tier and a
backlog-aware lead; a misprediction-robustness study. Threats: #39283 alone may fix the dense regime; mirror reclaim alone
may cut x70 SSD/cold returns (LRU model, to arm: 21% → 2.7-4.5%).

## Measurement caveat

The replay client used aiohttp's default 100-connection pool with 128 sessions: at x1 requests waited p50 7.35 s / p95
50.6 s client-side (~20 s of the 131 s mean TTFT), x10 p50 2.1-2.4 s / p95 42 s; x70 was unaffected (p50 0.07 s).
Server-side queue times exclude this lag, but the cap held ≤ 100 requests at the server, so the x10/x1 server queues
were shaped by it; uncapped runs will queue more at the server. Fixed on 2026-09-23 (`TCPConnector(limit=0)`).

## Next experiments

1. x10: port #39283 or enable the paced retry; queue-tail-first eviction; `--schedule-policy lpm`.
2. Oracle headroom per regime.
3. Gap predictability: human think pauses (campaign 10: they cause every SSD restore); real Claude Code traces.
4. Mirror-reclaim ablation.
5. Replication: 2-3 seeds (all results so far are single runs).
