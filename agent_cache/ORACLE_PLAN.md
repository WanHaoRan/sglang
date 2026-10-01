# Oracle admission-anchored SSD staging — implementation plan

Status tracker at the bottom. One step at a time, each on the user's go. Line numbers are at HEAD ace8ac79a7 (2026-10-01).
Evidence: read-only code check + adversarial review (2026-10-01); data from campaigns 7, 10 and 11.

## Design (the user's rule)

The client knows every gap exactly, so with turn t it sends a hint `{session, next_rid, next_gap_s}`. The server:

1. At turn t's finish (`t_commit`) records the session's committed context and the hint.
2. Predicts arrival `P = t_commit + gap + δ̂` (δ̂ = HTTP + tokenization offset, learned online, ~0.1 s).
3. Predicts admission `A`: before arrival `A = P + Ŵ` (Ŵ = queue length / admission rate; an EWMA of measured waits
   logged alongside); after arrival `A = now + (queue position + 0.5) / admission rate` (FCFS keeps arrival order).
4. Estimates SSD load time `L̂ = fixed handoff + 2.5 ms/page × (own SSD-only pages + pages already queued on the read
   thread)` (2.5 ms per 64-token page ≈ 2.4 GB/s; the queued part is the read backlog of `read_backlog.png`).
5. Starts the prefetch on the first scheduler loop with `now ≥ A − L̂`:
   - **case 1, not yet arrived** (sparse load): stage under the next turn's own handle `(next_rid, 0)`; an arrival
     mid-read joins it (per-handle dedupe), a finished read is found in host memory and credited as storage;
   - **case 2, already queued** (dense load): refetch the context evicted while it waited, through the existing
     `_prefetch_kvcache(req)` (does not spend the 8-attempt retry budget; #39283 stays as the fallback).

Transport: `kv_hints` (existing, typed, versioned per-request envelope; reaches `Req.kv_hints`, nothing reads it). The chat
route has no `metadata` field (silently dropped) and does not accept `kv_hints` until step 2. Switch: server flag
`--hicache-storage-next-turn-prefetch off|observe|on` (off = no-op; observe = compute and log only).

## Steps

| # | Step | Files | Acceptance |
|---|---|---|---|
| 1 | Client hints: draw the next gap one turn early (only with `--oracle-hints`; deterministic without `--gap-sample`, which is refused), attach the `kv_hints` envelope `sglang.next_turn` v1 to every non-last turn, suppress it when `--max-seconds` certainly cuts the next turn, log `oracle_hint` per turn | `scripts/replay_agentic.py`, `scripts/start_client.sh` (`ORACLE_HINTS`) | dry run: hint gap of turn t = gap of turn t+1; gap lines identical with and without hints; body has no `kv_hints` without the flag; envelope validates with `decode_kv_hints_envelope` |
| 2 | Transport + switch: `kv_hints` on `ChatCompletionRequest`, forwarded by serving_chat; the server flag (default off) | `entrypoints/openai/protocol.py`, `serving_chat.py`, `arg_groups/fields/memory.py`, `field_order.py` (append at the end), one unit test | unit test; a malformed envelope is rejected; `server_args` shows the flag |
| 3 | Read-only residency probe `resident_prefix_len(key) -> (l1, l12)` (no split, no LRU stamp) + read-backlog helper counting allocated reads (`len(hash_value) × page` for ops with host indices) | `mem_cache/unified_cache/unified_tree_core.py`, its interface, `mem_cache/unified_radix_cache.py`, new CPU test | probe equals `match_prefix` lengths and leaves the tree unchanged |
| 4 | Estimators module (δ̂ dropping samples > 5 s, admission rate with a fallback below 5 admissions, Ŵ both variants with the wait EWMA excluding only requests credited with their own SSD read, L̂) | new `mem_cache/next_turn_estimators.py` + test | unit tests; optional offline replay on c11 logs |
| 5 | Prefetcher in observe mode: turn-end hook, one tick per loop, arrivals/admissions from queue snapshots, `HICACHE_EVT ntp_plan/ntp_arrive/ntp_due/ntp_admit` with prediction errors, no prefetch; refuses TP/PP > 1, DP attention, disaggregation, non-Python tree core | new `mem_cache/next_turn_prefetch.py`; one-line hooks in `base_prefix_cache.py`, `unified_radix_cache.py` (end of `cache_finished_req`), `scheduler.py` (init helper + tick delegate, large-class-style); fixture fix in `test_scheduler_hicache_events.py`; new test | unit tests; live smoke: one plan per hinted turn, arrival error p50 ~0.1 s, no extra prefetch |
| 6 | Case 1 staging: rate-limit pre-check (retry next tick on decline), cancel stale retry entries, expire only plans whose request never arrived (`finish(ABORT)` only when no read is running) | `next_turn_prefetch.py` + tests | small-pool smoke: staged reads land before arrival; storage-hit turns wait like GPU hits |
| 7 | Case 2 refetch: the scheduler delegate reports whether a read was issued; re-arm on re-eviction (or restore `last_match_len`) so #39283 still catches it; log `ntp_refetch issued=` and `ntp_reevict` | `next_turn_prefetch.py`, one line in the scheduler delegate + tests | dense smoke: refetch lands before admission; fewer re-queries and cap hits |
| 8 | Analysis: ignore staged-read events logged before a turn's send; new arms in colour maps; `oracle_eval.py` (estimator errors, trigger lead vs load time, late and wasted reads) | `scripts/delay_components.py`, `timeline.py`, `crossscale.py`, `read_backlog.py`, new `oracle_eval.py` | c11 outputs unchanged; eval matches a hand grep |
| 9 | Arms `three_tier_wc_ntpobs` (observe), `three_tier_wc_oracle` (on); every arm with hints on and retry flags pinned 8/8; `ORACLE_HINTS` through run_compare | `scripts/start_server.sh`, `scripts/run_compare.sh` | boot checks print the mode and caps |

Experiments: E0 smoke; E1 observe at sparse (c10 config: NCONV=283 C=80 ARRIVAL=0.1 TURNS=50 GAP=1) and dense (c11
config) to choose Ŵ and the margin; E2 case 1 at c10 cadence; E3 case 2 at x10 vs c11 `three_tier_wc` and `_norq`.

## Decisions

- Transport: `kv_hints` (taken with the go on step 1).
- Open: trigger margin (exactly `A − L̂`, or plus one inter-admission gap); `wait_complete` arms only at first (a staged
  read's `timeout` deadline would count from staging); case 2 for hinted requests only or every queued request.

## Risks to watch

Staged reads log `prefetch_start/s2h_*` under next_rid before that request exists (step 8 fixes the parsers); staged and
refetched pages are unprotected after insert (re-eviction, eviction cascade at x10); oracle reads share the single FIFO
read thread and the 0.5 × host-pool rate limit; Prometheus storage counters skew (client numbers do not); wall-clock
triggers are rank-local (TP=1 only).

## Status

| Step | State | Date |
|---|---|---|
| 1 | done: client hints (`replay_agentic.py` +46/−6, `start_client.sh` `ORACLE_HINTS`). Verified offline: dry-run hint gap of turn t = gap of turn t+1 and output otherwise identical; `--gap-sample` refused; a stub server received no `kv_hints` without the flag, valid envelopes (server's `decode_kv_hints_envelope`) with it, `gap_slept` = hinted gap, hints suppressed exactly for turns the wall cap cut; today's chat model drops the key. Live smoke deferred to step 2 | 2026-10-01 |
| 2-9 | not started | |
