# Oracle v2: admission-anchored SSD staging with a read scheduler, pinned staging and next-use eviction

Written 2026-10-10 from campaign 15 (`results/campaign15/`), its figures and an offline replay of its reads. It is
self-contained: every number is in this file, with its source, so nobody needs the conversation that produced it, the
campaign READMEs or `ORACLE_PLAN.md` (v1) to follow it. v1 stays the reference for the hint transport, the estimators and
the v1 trigger rule, which v2 keeps under a flag.

**Engine.** Working tree on top of `df900e0d15` (the `agent_cache` oracle v1 engine, which is the upstream merge
`ace8ac79a7` plus the prefetcher). Every `file:line` anchor below is at that tree; lines drift, function names do not.

**Box.** The H200 box of campaigns 13-15: one H200, Qwen3-30B-A3B-Instruct-2507 bf16 KV (6 MiB per 64-token page),
L1 668,160 tokens, L2 160 GB = 1,627,648 tokens, nixl L3 on `/mnt/ssd`. Anything that needs that box is in
`ORACLE_V2_HANDOFF.md`; everything else here runs on a CPU-only checkout.

---

## 0. The short version

1. §1-§2: what campaign 15 showed and why. Read once; nothing to run.
2. §4.1-§4.6: the engine changes, each behind a flag that defaults to today's behaviour, each with CPU tests
   (`python3 -m pytest test/registered/unit/mem_cache/test_storage_read_scheduler.py ...`, seconds, no GPU).
3. §4.7: the offline replay of campaign 14's accesses under LRU and next-use eviction (CPU, ~1 min).
4. §5: the GPU smokes and campaign 16 (H200, ~1 h of smokes + 4 arms x 5 h), in `ORACLE_V2_HANDOFF.md`.
5. §6: the go/no-go numbers for each lever.

---

## 1. Observations (campaign 15, 2026-10-09/10)

Campaign 15 ran the v1 oracle (`--hicache-storage-next-turn-prefetch on`) on campaign 14's workload: 768 fresh
conversations of the 262K agentic trace, Poisson arrivals at 0.1/s, the user cap steered to a server backlog of 32,
recorded gaps x70 capped at 1,200 s, 5 h per arm, one arm per prefetch policy (`wait_complete` = wc, `timeout` = to),
each against campaign 14's baseline without the oracle. All numbers are measured unless tagged; window = 2,700-16,800 s
after the client started, returning turns only (`results/campaign15/README.md`, `three_tier_*_oracle/diagnosis.txt`,
`oracle_eval.csv`; the 120-s figure numbers are from `three_tier_wc_oracle/oracle_timeline_7200_120.png` and the
server log it was drawn from).

### 1.1 Headline

|                                                        | wc baseline        | wc + oracle v1     | to baseline         | to + oracle v1     |
| ------------------------------------------------------ | ------------------ | ------------------ | ------------------- | ------------------ |
| TTFT mean / p50 / p90 (s)                              | 22.2 / 10.4 / 40.4 | 20.8 / 16.8 / 36.1 | 53.3 / 10.7 / 155.4 | 41.1 / 8.7 / 112.9 |
| turns/s; users finished in 5 h                         | 1.217; 765         | 1.218; 766         | 0.988; 646          | 1.114; 708         |
| turns/s while congested (bins with TTFT median > 10 s) | 1.34               | 1.57               | 0.64                | 0.78               |
| returns served by device / host / SSD / recompute (%)  | 21 / 16 / 50 / 13  | 21 / 10 / 58 / 11  | 11 / 26 / 28 / 35   | 8 / 21 / 36 / 35   |
| SSD read / written (TB)                                | 15.6 / 3.4         | 18.0 / 2.7         | 11.0 / 7.9          | 18.0 / 7.8         |
| SSD busy, mean; share of 10-s samples >= 90 %          | 61 %; 49 %         | 70 %; 68 %         | 54 %; 22 %          | 86 %; 77 %         |
| host pool held by in-flight SSD reads, p50 (tokens)    | 382K (23 %)        | 642K (39 %)        | 50K (3 %)           | 183K (11 %)        |

Paired on the same (conversation, turn): wc oracle minus baseline -1.41 s mean, +0.01 s median; to oracle minus
baseline -7.28 s mean. The user cap fixes the mean queue wait at backlog / throughput (Little's law holds within 0.4 s
in all four arms), so a better cache shows up as throughput, not as TTFT.

### 1.2 The wc oracle in a congested two minutes (7,200-7,320 s)

- 219 hinted turns admitted, 144 after an SSD read, 107 of those within 50 ms of the read's end ("late": admission
  waited on the read).
- Arrival prediction exact: hinted arrival error p50 45 ms, p90 298 ms (deterministic replay gaps).
- Admission predicted a median 8.0 s too early (`adm_err_abs_p50`). The queue-length estimator is unbiased (bias
  -0.06 s, MAE 8.7 s); the wait EWMA is biased -12.9 s because it excludes storage-credited turns, which were 58 % of
  returns.
- Staged reads (v1 case 1): trigger lead p50 32.9 s; trigger to read end p50 32.8 s; estimated load p50 26.0 s;
  the queue ahead of a staged read p50 9,820 pages (~60 GB), p90 12,933; 7,183 of 9,397 case-1 reads late (76 %).
- The read thread was busy 100 % of the window: 36,203 pages of I/O in 120 s at 3.32 ms per page against the load
  model's seed of 2.5 ms (OLS on campaign 11, where reads did not overlap write-through).
- The read queue was pinned at ~12.5K pages, which is the prefetch capacity limit, 0.5 x host pool = 12,716 pages;
  occupancy logged at read completion peaked at 834K tokens against the 814K cap.
- Every one of the 147 triggers in the window fired already past its ideal start (lead < estimated load), i.e. the
  oracle never chose a time; it fired the moment the backlog growth made the plan overdue.
- Triggers came in bursts: 18 in 1.4 s at 7,245 s (backlog seen 4,966 -> 8,937 pages during the burst), 38 in 2.5 s at
  7,260 s (6,017 -> 12,939), 42 in 9 s at 7,305 s (7,439 -> 13,036); the plans in a burst had admission leads spread
  over 14-34 s.

### 1.3 The to oracle

- The timeout cut 5,018 of 9,663 oracle reads (73.7M tokens short of their lookups) against 11 % of the requests' own
  reads; 4,473 of the arm's 15,810 reads (28 %) were cut while still queued and then skipped by the read thread
  (campaign 14 baseline: 504 of 8,464). 116M tokens were re-fetched and 43M (37 %) credited; 711 staged reads were
  never used; 2,982 re-evictions, 1,085 at the per-request cap.
- In the same two minutes only 1 of 58 reads was late, because admission no longer waits: a cut read admits the turn
  with whatever landed and the rest is recomputed (35 % of returns).
- The oracle's reads kept the SSD at 86 % busy (baseline 54 %), which is where the to arm's gain came from.

### 1.4 What the host pool held

- Under write-through a host copy is evictable only after its device copy is gone (`unified_tree_core.py`
  `_is_host_leaf`), so the host pool always mirrors the device pool: 668,160 of 1,627,648 tokens, 41 %.
- A read's host pages are allocated when its lookup hits, before it waits in the FIFO, so a read that queues 30 s
  holds its pages 30 s: in-flight reads held 39 % of the pool with the oracle, 23 % without.
- Room for host-only contexts at the median hold: 318K tokens with the oracle, 578K without.
- Host hits cost 0.30-0.34 s TTFT (p50); SSD restores 22-23 s. Each host hit avoids a ~1.5 GB read. If the host hits
  had been SSD reads: +36 % read volume on the wc baseline, +22 % on the wc oracle arm.

### 1.5 Offline replay of the run's reads (CPU, 2026-10-10)

The 11,606 SSD reads of `three_tier_wc_oracle` replayed through one FIFO read thread with their measured service times,
every read issued up to D seconds earlier but never before its plan existed, host pages held from issue until the
logged admission under the real 12,716-page cap (script: `scripts/read_shift_replay.py`, §4.8; it reproduces the real
completion times to 0.2 s at D = 0):

| D    | window reads finishing earlier | net issue shift | cap wait | SSD state at 7,200 s                          |
| ---- | ------------------------------ | --------------- | -------- | --------------------------------------------- |
| 10 s | 10.0 s                         | 10.0 s          | 0 s      | 9 s of work ahead of the real run             |
| 40 s | 37.8 s                         | 15.8 s          | 23.5 s   | 38 s ahead: 2,621 pages queued, 11,426 parked |

The real run at 7,200 s had 12,457 pages queued and 0 parked. The SSD was idle 2,626 s of the 7,200 s before the
window (36 %).

---

## 2. Analysis

**The problem in one inequality.** A returning turn's pages land in time only if

    trigger lead  >=  read-queue wait + own read time + insert lag.

v1 computes the lead as predicted admission minus an estimated load that includes the whole backlog, and the lead was
spent in the queue. Under wc the shortfall is the 8-s admission error and the 76 % late reads; under to it is cut
reads and recompute.

**Five mechanisms, in the order they bite.**

1. **The v1 trigger rule is unstable on a shared FIFO.** Each plan subtracts the whole backlog from its own trigger,
   so backlog growth pulls every pending trigger earlier at once, the stagings grow the backlog further, and the
   cascade stops only at the capacity cap. The 18/38/42-trigger bursts and the sawtooth at 12.5K pages are that loop.
   Any "shift earlier" built on the same rule inherits it.
2. **The cap is spent on waiting, not on work.** Host pages are allocated at the lookup hit. At 7,200 s the cap held
   12,457 queued pages and zero finished ones. The budget that should hold work done ahead of admission is consumed by
   work not yet started.
3. **Landed pages are unprotected.** A finished read is inserted as an ordinary host leaf and its anchor lock is
   dropped; nothing pins it until the request is reached in the admission loop. Under to this produced 2,982
   re-evictions and 116M re-fetched tokens.
4. **Write-through churn sets host residency.** Every prefill, every recompute included, writes through at about
   1,000 tokens/s, so the host-only room turns over in 30-60 s. Recompute causes writes, writes cause misses.
5. **Host hits are the cheapest SSD capacity, and the oracle spent them.** Staging displaced host hits from 16 % to
   10 % of returns and added 16 % read volume on a saturated SSD; that is why wc gained only in congested phases.

**What shifting earlier can and cannot do (from §1.5).** Issuing every read earlier does cover the gap, by more than
the 8 s needed, but only because the SSD did the work earlier, in idle time banked before the congested stretch, and
parked the finished pages in host memory. Two bounds: the idle bank (36 % here; zero in a stationary overload) and the
parking budget (the cap holds ~42 s of SSD work at 302 pages/s; v1 fills it with the queue). In a stationary overload
nothing can be banked and only fewer bytes (eviction) or a better order (deadlines) help.

**Lateness under wc is a tautology.** Admission is gated on the read, so a read that queued long ends at admission
by construction. The honest lateness is predicted-admission to admission, 8 s, which matches the load model's
under-prediction of 6.5 s plus noise.

**The hints are information the engine never uses.** They say which context is needed when; v1 uses that for neither
eviction nor read order.

---

## 3. Design: what v2 changes

Four levers, each a flag, each independently testable, in build order. "FIFO path" = today's engine, which stays
byte-for-byte the default.

| lever                                                    | flag                                                                                                         | mechanism                                                                                                                                                                                                                                                                                                                                                                                   | fixes                                                                                                  |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **c. Host pages at dispatch, landed pages pinned** | `--hicache-storage-read-order deadline` (allocation) + `--hicache-storage-prefetch-pin-fraction F` (pin) | A hit op parks in a scheduler-side queue holding no host memory; pages are allocated when the read is dispatched to the I/O thread, and the dispatched + pinned pages, not the queue, are charged against the prefetch cap. A landed span is host-locked until its request takes the host lock at admission, aborts, or its plan expires, within F x host pool.                             | §2 items 2 and 3                                                                                      |
| **b. Deadline-ordered dispatch**                   | same`deadline` order                                                                                       | Parked ops are dispatched by deadline: a request's own lookup or re-query and a queued request at the head get deadline "now"; a staged read gets its plan's predicted admission. The I/O thread holds at most`--hicache-storage-read-window-pages` of in-flight work, so completions follow deadlines, not trigger order. A timeout-policy clock starts at dispatch, never while parked. | §2 item 1 (no backlog term in any trigger), the FIFO lateness, to's cut-while-queued                  |
| **b'. Work-ahead**                                 | implied by b + c                                                                                             | Dispatch pulls the earliest deadline whenever the window has room and the cap has room; the oracle issues lookups as soon as a plan exists and the span is missing, within a lead horizon. The budget, not a per-request timer, limits how far the SSD runs ahead.                                                                                                                          | §1.5's banking, bounded by the cap                                                                    |
| **d. Online load model**                           | automatic with b                                                                                             | The I/O thread publishes an EWMA of ms per page; the estimator's seed is replaced once 8 reads are in. Deadlines are the only consumer.                                                                                                                                                                                                                                                     | the 6.5-s under-prediction                                                                             |
| **a. Next-use eviction**                           | `--radix-eviction-policy next_use`                                                                         | Device and host eviction order by the hinted next arrival: ended sessions first, then sessions whose next turn is beyond a horizon (farthest first), then un-hinted nodes by LRU, then sessions returning within the horizon (farthest first). The prefetcher registers each committed turn's tail node with its predicted next admission and updates it on arrival.                        | §2 items 4 and 5: fewer SSD reads is the only lever that adds capacity when the SSD is the bottleneck |
| **mirror reclaim / write-back**                    | existing`--hicache-write-policy write_back` arm only                                                       | Not built in v2. Measured as an ablation arm (§5) because it is one flag; the write-through mirror reclaim is a v3 item.                                                                                                                                                                                                                                                                   | §1.4                                                                                                  |

Everything runs on the scheduler thread at the existing lockstep points (the storage-event drain), so TP determinism
is untouched; v2 is validated for TP = PP = 1 only, like v1, because deadlines come from a rank-local clock.

---

## 4. Implementation steps

Each step says what it changes, how it is checked on CPU, and what a GPU run must confirm (§5). Run every test file in
its own process, as CI does.

### 4.1 Flags

**Status:** done 2026-10-10
**Goal:** three new server flags and one eviction policy, all defaulting to the FIFO path, validated at argument
resolution so a bad combination fails at boot, not mid-campaign.
**Runs on:** CPU checkout
**Touches:** `python/sglang/srt/arg_groups/fields/memory.py` (fields), `arg_groups/field_order.py`,
`arg_groups/choices.py` (`next_use`), `arg_groups/hicache_hook.py` and `kv_cache_hook.py` (validation)
**Takes:** none

- `--hicache-storage-read-order {fifo,deadline}` (default `fifo`).
- `--hicache-storage-read-window-pages N` (default 512): pages the I/O thread may hold in flight under `deadline`.
- `--hicache-storage-prefetch-pin-fraction F` (default 0.0 = no pin): share of the host pool that landed, unadmitted
  prefetch spans may hold.
- `--radix-eviction-policy next_use` with `--radix-eviction-policy-config '{"horizon_s": 300}'`.
- Validation: `deadline` needs `--hicache-host-memory-mode cache`, a storage backend and TP = PP = 1; `next_use` needs
  `--hicache-storage-next-turn-prefetch` not `off` and non-negative `horizon_s` / `overdue_grace_s`; F in [0, 0.5]
  (the prefetch cap is half the host pool, and pinned tokens count against it); the pin needs `deadline`.

```bash
# host: the flag definitions and their validation hook resolve and reject a bad combination (CPU, < 10 s)
cd /home/wanhr/Documents/codes/sglang && python3 -m pytest -q test/registered/unit/mem_cache/test_oracle_v2_flags.py
```

-> prints `passed`; a failure names the flag whose validation changed.

**Expected result:** the five tests pass (measured 2026-10-10: `5 passed`); `python3 -m sglang.launch_server --help | grep -c "storage-read-order"` prints 1.
**Expected lessons:** every lever is reachable from the command line and off by default, so campaign 14's baseline
command still boots the FIFO path unchanged.

### 4.2 The storage read scheduler (lever b, allocation half of c)

**Status:** done 2026-10-10 (CPU); GPU smoke pending (§5)
**Goal:** replace "allocate at the lookup hit, FIFO into the I/O thread" with "park without memory, dispatch by
deadline into a bounded window, allocate at dispatch", behind `--hicache-storage-read-order deadline`.
**Runs on:** CPU checkout
**Touches:** new `python/sglang/srt/mem_cache/storage_read_scheduler.py` (the policy, pure); `unified_radix_cache.py`
(`_drain_and_alloc_storage_hit` parks instead of launching; new `dispatch_storage_reads`, called from
`check_hicache_events` after the drain; `_can_terminate_prefetch` never cuts a parked op; occupancy charged at
dispatch, sized to the allocation, with the tracked key trimmed to it so the existing release paths subtract the same
amount; the operation's `start_time` reset at dispatch); `cache_controller.py` (the limiter counts pinned tokens
through `extra_occupancy_fn`)
**Takes:** none

The policy (`StorageReadScheduler(window_pages, budget_tokens, page_size)`): `park(handle, operation, pages, now)`,
`discard(handle)`, `is_parked(handle)`, `clear()`, and `dispatch(now, deadline_of, in_flight_pages, held_tokens) -> decisions`. Dispatch pops the parked read with the smallest deadline while the I/O thread holds fewer than
`window_pages` (every dispatched read counts, "now" reads included) and `held_tokens` + this read x page_size stays
within `budget_tokens` (`held_tokens` is what the cache already charged: the in-flight reads, charged at dispatch, plus
the pinned spans, so a read is charged once); a deadline that is `None` means
"now", sorts ahead of any timed deadline, and is dispatched even when the window or budget is full (today's path
allocated it immediately, and the request's admission is waiting on it). The deadline function is bound by the
prefetcher (`UnifiedRadixCache.read_deadline_fn`); without the prefetcher every deadline is "now" and the order is park
order, which is the drain order. The cache side: `_park_storage_hit` (the drain, instead of launching),
`_dispatch_storage_reads` (from `check_hicache_events` after the drain; drops parked handles that left
`ongoing_prefetch`, releases pins older than 600 s, launches the decisions through `_launch_storage_hit`, the former
drain closure), `_can_terminate_prefetch` returns False for a parked op, and `revoke_pending_prefetch` /
`release_aborted_request` discard the park.

```bash
# host: policy unit tests (deadline order, window, budget, the now bypass, stale handles) and the cache-side
# park/dispatch wiring with a fake controller (CPU, < 20 s)
cd /home/wanhr/Documents/codes/sglang && python3 -m pytest -q test/registered/unit/mem_cache/test_storage_read_scheduler.py \
  && python3 -m pytest -q test/registered/unit/mem_cache/test_storage_read_dispatch.py
```

-> prints `passed` twice (measured 2026-10-11: `6 passed`, `15 passed`; the second file also holds the pin and
controller cases of 4.3 and 4.4 and the admit-site pin release on a CPU tree).

**Expected result:** with the flag at `fifo` the drain code path is the pre-existing one (the lifecycle test
`test_storage_prefetch_lifecycle.py` passes unchanged, `8 passed`); with `deadline` a parked hit holds no host pages and
the fake controller's `prefetch_buffer` receives ops in deadline order with at most `window` pages outstanding.
**If it differs:** `test_storage_prefetch_lifecycle.py` fails: the FIFO path was touched; diff `_drain_and_alloc_storage_hit`
against `df900e0d15`.
**Expected lessons:** the read queue becomes a scheduler-side decision with no memory attached; the backlog the oracle
sees (`storage_read_backlog_pages`) shrinks to the window, which is what removes the feedback loop of §2 item 1.

### 4.3 Pinned landed pages (lever c, pin half)

**Status:** done 2026-10-10 (CPU); GPU smoke pending (§5)
**Goal:** a span the SSD delivered survives until its request takes it, within a budget.
**Runs on:** CPU checkout
**Touches:** `unified_radix_cache.py` (`_handle_prefetch_result` takes a host lock on `inserted_host_node` and records
`pinned_prefetch[handle]`; `_unpin_prefetch` at `init_load_back`, `cache_unfinished_req`, `cache_finished_req`,
`release_aborted_request`, `reset`; a pin older than 600 s is released by the dispatcher as a leak guard)
**Takes:** none

The pin is a host lock on the inserted tail node; `_is_host_leaf` needs childless and unlocked, so the whole landed
path is protected. The pin budget is `F x host pool` tokens; a span that does not fit is left unpinned (today's
behaviour) and counted (`srs_pin_skipped`).

```bash
# host: pin taken on insert, released on each exit path, budget respected, never double-released (CPU, < 20 s)
cd /home/wanhr/Documents/codes/sglang && python3 -m pytest -q test/registered/unit/mem_cache/test_storage_read_dispatch.py -k pin
```

-> prints `passed`.

**Expected result:** host lock counts return to zero after every path; `pinned_prefetch_tokens` never exceeds the
budget.
**Expected lessons:** the "protect what was fetched until admission" rule costs host memory only for spans that
actually landed, which is the correct unit; v1 paid for queued spans instead.

### 4.4 Prefetcher deadline mode (levers b', d)

**Status:** done 2026-10-10 (CPU); GPU smoke pending (§5)
**Goal:** with `deadline` order the oracle stops timing reads and starts labelling them: a plan's lookup is issued as
soon as its span is missing and its predicted admission is within the lead horizon; the dispatcher orders reads.
**Runs on:** CPU checkout
**Touches:** `python/sglang/srt/mem_cache/next_turn_prefetch.py` (`read_deadline(handle)`, the horizon, the parked cap,
deadline-mode branches in `_check_pending` and `_check_queued`; binds `tree_cache.read_deadline_fn`),
`cache_controller.py` (the I/O thread folds every completed read into `storage_read_ms_per_page`, an EWMA with
alpha 1/8 published after 8 reads and logged every 64); the prefetcher replaces the load model's two per-page seeds
with it on every tick (`_refresh_load_model`)
**Takes:** none

- `read_deadline(handle)`: for a staged handle before arrival, predicted arrival + the queue-length wait; after arrival,
  now + the residual wait at its queue position; `None` ("now") for anything the prefetcher does not know, for a queued
  request at position <= 1, and for a plan whose predicted admission has passed.
- Lead horizon `NTP_LEAD_HORIZON_S` = 120 s: a plan further out is rechecked at predicted admission minus the horizon.
- Parked cap `NTP_MAX_PARKED` = 256 plans with a lookup in flight or parked; beyond it new lookups wait.
- The v1 trigger rule stays the behaviour under `fifo` order, so campaign 15 remains reproducible.

```bash
# host: deadline mode issues at horizon, labels deadlines, keeps v1 under fifo; the online load model (CPU, < 20 s)
cd /home/wanhr/Documents/codes/sglang && python3 -m pytest -q test/registered/unit/mem_cache/test_next_turn_prefetch.py \
  && python3 -m pytest -q test/registered/unit/mem_cache/test_next_turn_estimators.py
```

-> prints `passed` twice (measured 2026-10-11: `32 passed` of which 9 are the deadline-mode and registry cases, and
`4 passed`); the 23 v1 cases in `test_next_turn_prefetch.py` are unchanged.

**Expected result:** under `deadline`, `ntp_due` events carry `lead_ms` that can be larger than `load_ms` (the plan
is not overdue when it fires), and the `ntp_stage` count equals the number of plans whose span was missing at the
horizon.
**Expected lessons:** the oracle's job splits in two: predict (arrival, admission, missing span) and label (deadline);
the SSD's job (when to read) moves to one place that sees all reads.

### 4.5 Next-use eviction (lever a)

**Status:** done 2026-10-10 (CPU); GPU smoke pending (§5)
**Goal:** host and device eviction prefer the context needed last, using the same hints.
**Runs on:** CPU checkout
**Touches:** new `python/sglang/srt/mem_cache/next_use_eviction.py` (`NextUseRegistry`, `NextUseStrategy`),
`evict_policy.py` is untouched, `mem_cache/utils.py` (factory entry), `next_turn_prefetch.py` (registers tail nodes
at commit, updates on arrival, prunes the previous tail), `unified_radix_cache.py` (passes the committed tail node id
to the prefetcher)
**Takes:** none

Priority tuple `(tier, key)`, smallest evicted first: tier 0 = registered with no next turn (ended session, LRU
within); tier 1 = next use overdue by more than `overdue_grace_s` (default 60 s, the prefetcher's minimum plan expiry;
oldest first: the turn is not coming); tier 2 = next use beyond `horizon_s` (farthest first); tier 3 = unregistered
(plain LRU); tier 4 = next use within the horizon (farthest first). An un-hinted request's tail is registered as ended;
with un-hinted traffic the policy degrades to "finished first, then LRU", which the flag's help text says. The
prefetcher ends a session's registration when its plan expires, when its queued turn is aborted and when a running
turn is aborted (review finding of 2026-10-10: a stale future next use would otherwise sit in the most protected tier
for the life of the node). The registry is bounded (65,536 entries, oldest registration dropped), since nothing
removes an entry when its node is evicted and un-hinted traffic adds one per finished request.

```bash
# host: strategy ordering, registry updates, and host eviction order on a CPU tree (CPU, < 20 s)
cd /home/wanhr/Documents/codes/sglang && python3 -m pytest -q test/registered/unit/mem_cache/test_next_use_eviction.py
```

-> prints `passed` (measured 2026-10-11: `6 passed`).

**Expected result:** on a CPU tree with four two-token sessions, eviction frees the ended one, then the far one, then
the un-hinted one, and keeps the one returning soonest (the test drives device eviction; host eviction uses the same
priority through `_is_host_leaf` and the strategy).
**Expected lessons:** eviction and prefetch now read the same fact; the host-only room is spent on the sessions that
will return before the SSD could have served them.

### 4.6 Observability

**Status:** done 2026-10-10
**Goal:** every v2 decision leaves a line the existing analysis scripts can parse (`HICACHE_EVT <kind> rid=<rid> k=v ...`).
**Runs on:** CPU checkout
**Touches:** the files above; new `scripts/srs_summary.py` (counts and distributions of these events for one boot
directory, optionally inside a time window); `scripts/oracle_eval.py` and `scripts/oracle_timeline.py` are unchanged
**Takes:** none

| event               | fields                                                    | meaning                                                                                                  |
| ------------------- | --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `srs_park`        | pages, deadline_ms                                        | a hit parked (deadline_ms =`now` for "now")                                                            |
| `srs_dispatch`    | pages, waited_ms, inflight_pages, pinned_tokens, launched | a parked read handed to the I/O thread (launched=0: the host allocation failed and the read was revoked) |
| `srs_pin`         | tokens, node, pinned                                      | pin taken on the inserted host tail                                                                      |
| `srs_pin_skipped` | tokens, pinned, budget                                    | a landed span left unpinned because the budget is full                                                   |
| `srs_unpin`       | tokens, reason, pinned                                    | pin released (admit, abort, replaced, leak)                                                              |
| `srs_rate`        | ms_per_page, samples                                      | the I/O thread's EWMA, logged every 64 reads                                                             |
| `nue_register`    | node, next_use_ms                                         | next-use registry update (ms from the commit or arrival to the next use,`ended` for none)              |

```bash
# host: counts and distributions of the v2 events of one boot (3 s on a 5 h log)
cd /home/wanhr/Documents/codes/sglang/agent_cache && python3 scripts/srs_summary.py --run results/<boot dir> --window 7200 7320
```

-> prints the counts table and `dispatch count equals s2h_io count: yes` on a `deadline` boot.

**Expected result:** `grep -c "HICACHE_EVT srs_dispatch"` on a §5 smoke log equals its `s2h_io` count under `deadline`.
**Expected lessons:** lateness, parking time and pin budget pressure are measurable per read, so §6's numbers do not
depend on a model.

### 4.7 Offline replay of campaign 14 under LRU and next-use (lever a, expected effect)

**Status:** done 2026-10-10
**Goal:** bound what next-use eviction can gain before spending GPU time: replay campaign 14's `three_tier_wc`
accesses through a two-tier model with LRU and with next-use eviction of the host-only room.
**Runs on:** host, CPU, `agent_cache/`
**Touches:** writes `results/analysis_oracle_v2/next_use_replay.csv`; reads campaign 14's `client.jsonl`
**Takes:** ~1 min, GPU idle

```bash
# host: replay the campaign 14 wc run's returns through L1 (LRU) + host-only room (LRU vs next_use) at three room sizes
cd /home/wanhr/Documents/codes/sglang/agent_cache && python3 scripts/next_use_replay.py \
  --client results/20261008_204128_three_tier_wc_NAT160/client_204548_cmp_three_tier_wc/client.jsonl \
  --l1-tokens 600000 --host-room-tokens 318000 578000 1250000 --horizon-s 300 \
  --out results/analysis_oracle_v2/next_use_replay.csv
```

-> prints one row per (room, policy): share of returning turns served by L1 / host / SSD and the SSD gigabytes;
the pass condition is that next_use serves a higher host share than LRU at every room size.

**Expected result:** measured 2026-10-10 (`results/analysis_oracle_v2/next_use_replay.csv`, 0.5 s on CPU), 768
sessions, 19,449 returning turns, L1 600K tokens:

| host-only room | policy   | device | host   | SSD    | SSD read |
| -------------- | -------- | ------ | ------ | ------ | -------- |
| 318K tokens    | LRU      | 32.4 % | 5.3 %  | 62.3 % | 27.9 TB  |
| 318K tokens    | next_use | 32.4 % | 27.5 % | 40.2 % | 18.9 TB  |
| 578K tokens    | LRU      | 32.4 % | 9.2 %  | 58.5 % | 26.2 TB  |
| 578K tokens    | next_use | 32.4 % | 41.2 % | 26.4 % | 12.4 TB  |
| 1,250K tokens  | LRU      | 32.4 % | 47.0 % | 20.7 % | 9.3 TB   |
| 1,250K tokens  | next_use | 32.4 % | 58.6 % | 9.0 %  | 3.8 TB   |

The real run's client-side tiers for the same turns were device 25.1 %, host 22.1 %, SSD 47.4 %, cold 5.3 %
(measured, campaign 14 `client.jsonl`). The model is cruder than the engine (whole contexts, no partial hits, no
prefetch holds, a fixed L1), so its absolute shares are not predictions; the ordering comparison at equal room is the
result: next-use eviction moves 22 to 32 points of returns from the SSD to the host at the two rooms the campaigns
actually had, and still 12 points at twice the room.
**Expected lessons:** at campaign 14's host room the SSD traffic is mostly reachable by eviction order alone, so the
`_nu` arm is worth its 5 h; the mirror lever (more room) adds less than the order does until the room doubles.

### 4.8 The read-shift replay used in §1.5

**Status:** done 2026-10-10
**Goal:** keep the simulation that produced §1.5 reproducible.
**Runs on:** host, CPU
**Touches:** reads `results/20261009_164101_three_tier_wc_oracle_NAT160/`; writes nothing
**Takes:** ~30 s

```bash
# host: replay the wc oracle arm's reads with earlier issue under the FIFO thread and the host cap
cd /home/wanhr/Documents/codes/sglang/agent_cache && python3 scripts/read_shift_replay.py \
  --run results/20261009_164101_three_tier_wc_oracle_NAT160 --window 7200 7320 --shifts 0 5 10 20 40
```

-> prints the §1.5 table; `shift 0` must reproduce the real completion times within 0.3 s mean.

**Expected lessons:** the two bounds on "shift earlier": the idle bank and the parking budget.

---

## 5. Experiments (H200; procedure in `ORACLE_V2_HANDOFF.md`)

Arms, all on campaign 14's workload and client (`ORACLE_HINTS=1` on every arm so request bodies are identical):

| arm                            | server, on top of`three_tier_wc`                                                                | isolates                               |
| ------------------------------ | ------------------------------------------------------------------------------------------------- | -------------------------------------- |
| `three_tier_wc_dl`           | `--hicache-storage-read-order deadline --hicache-storage-prefetch-pin-fraction 0.3`, oracle off | lever c alone: every deadline is "now" |
| `three_tier_wc_oracle_v2`    | the above +`--hicache-storage-next-turn-prefetch on`                                            | b + b' + c + d                         |
| `three_tier_wc_oracle_v2_nu` | the above +`--radix-eviction-policy next_use`                                                   | + a                                    |
| `three_tier_wc_wb`           | `--hicache-write-policy write_back`, oracle off                                                 | the mirror, as an ablation             |

Smokes first (§H4 of the handoff): each arm boots, the new events appear, no tracebacks, 32 sessions x 12 turns.
Then campaign 16: the four arms, 5 h each, against campaign 14's `three_tier_wc` and campaign 15's
`three_tier_wc_oracle`, with `scripts/steady_state.py` and `scripts/oracle_eval.py` as in campaign 15.

Evaluation rule (option e of v1): report throughput at the controller's backlog target and, in a second pass, TTFT at
a fixed user cap with two seeds, because the controller converts any latency gain into load.

---

## 6. Expected outcomes and go/no-go

All targets are at campaign 14's load (768 users, backlog target 32, x70 gaps); baselines are the measured campaign
14/15 wc numbers of §1.1. "estimate" rows are reasoned from §1 and §2; the replay row is measured on CPU.

| lever / arm      | metric                                          | baseline                         | target (go)                                                                                                   | why                                                                                            |
| ---------------- | ----------------------------------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| c (`_dl`)      | queued pages at the cap vs parked               | 12,457 queued / 0 parked (§1.2) | parked > queued in congested windows                                                                          | the cap now holds work done ahead (§1.5: 11,426 parked at D = 40)                             |
| c                | re-evictions of landed spans                    | 2,982 (to), 124 (wc)             | < 50 per 5 h                                                                                                  | pinned until admission                                                                         |
| c                | host pool held by queued reads                  | 39 %                             | < 10 %                                                                                                        | allocation at dispatch, window-bounded                                                         |
| b + b' (`_v2`) | late reads (admission within 50 ms of read end) | 76 %                             | < 25 % while the SSD is below 90 % busy; unchanged above                                                      | deadlines order completions; saturation still forces lateness                                  |
| b                | trigger bursts                                  | 18-42 per 2-9 s                  | none: dispatch rate equals the window drain                                                                   | no backlog term                                                                                |
| d                | load estimate error                             | -6.5 s p50                       | within +-1 s                                                                                                  | online ms per page                                                                             |
| a (`_nu`)      | host-hit share of returns                       | 10-17 %                          | >= 30 % (the §4.7 replay gives 27.5-41.2 % at the campaigns' rooms, against 5-9 % for LRU in the same model) | eviction by next use                                                                           |
| a                | SSD read volume                                 | 15.6-18.0 TB                     | -20 % at the same turns                                                                                       | each host hit saves ~1.5 GB                                                                    |
| all (`_v2_nu`) | turns/s while congested                         | 1.34 (c14), 1.57 (c15)           | >= 1.8                                                                                                        | waste removal: 16 % displaced-host reads, 63 % of re-fetched tokens never credited, FIFO order |
| all              | users finished in 5 h                           | 765-766                          | >= 820                                                                                                        | throughput                                                                                     |
| mirror (`_wb`) | host hits; admission stalls                     | 16 %; none                       | host hits up;`write_back` blocking per eviction < 50 ms p99                                                 | §1.4                                                                                          |

No-go: if `_dl` shows the cap still full of queued pages, the window is too large or the "now" bypass is firing on
every read (check `srs_dispatch` `inflight_pages`); if `_v2` is not better than `_dl` in congested bins, deadlines
carry no information at this load and the design reduces to c; if the §4.7 replay gives next_use less than 5 points
over LRU, skip the `_nu` arm and move the mirror lever up.

---

## 7. Risks

- A "now" read bypasses the budget; a burst of head-of-queue re-queries can still over-allocate host pages (bounded
  by the number of admission candidates per pass, today 1-3).
- The pin holds the whole landed path; a request that never admits (client disconnect without abort) holds it until
  the 600-s leak guard.
- The lead horizon pins anchors of parked lookups (the v1 anchor lock); 256 parked x ~20K tokens is 5M tokens of
  protected host paths at the extreme, above the pool; the parked cap must be tuned down if `srs_park` counts stay high.
- "Now" reads (a request's own lookup at the head of the queue) bypass the window and are counted in it, so under a
  steady stream of head-of-queue lookups the window can stay full and no timed read is dispatched; nothing reserves a
  share of the window for timed reads. If `srs_summary.py` shows timed parks waiting while `inflight_pages` sits at the
  window on every dispatch, raise `--hicache-storage-read-window-pages` or add a reserved share (not implemented).
- `next_use` keeps up to 65,536 registrations; past that the oldest are dropped, so on a very large tree an old but
  live session can fall back to plain LRU (tier 3) rather than its registered tier.
- `next_use` treats un-hinted traffic as ended-first; never deploy it on a mixed workload without hints.
- Deadlines use `time.perf_counter()`; TP/PP > 1 is refused at boot, as in v1.
- The to policy with v2: a dispatched read's clock starts at dispatch, so the cut-while-queued class disappears, but
  a read that is slow in I/O is still cut at 1.0 s + 0.25 s per 1,024 hit tokens; wc remains the policy of record.

---

## 8. Reference numbers (H200, Qwen3-30B-A3B bf16, 6 MiB pages)

| quantity                                   | value                                                            | source                                              |
| ------------------------------------------ | ---------------------------------------------------------------- | --------------------------------------------------- |
| SSD read drain under write-through overlap | 302 pages/s = 3.32 ms per page = ~2.0 GB/s                       | c15 wc oracle, 7,200-7,320 s window                 |
| SSD read service without overlap           | 43.5 ms + 2.477 ms x pages                                       | c11 wc, 1,199 reads (v1 §8)                        |
| prefetch capacity cap                      | 0.5 x host pool = 813,824 tokens = 12,716 pages = ~42 s of drain | `cache_controller.py` `prefetch_capacity_limit` |
| host mirror of the device pool             | 668,160 tokens, 41 % of L2                                       | `_is_host_leaf` inclusion                         |
| host-only room at the median read hold     | 318K (oracle) / 578K (baseline) tokens                           | c15`diagnosis.txt`                                |
| write-through churn                        | ~1,000 tokens/s                                                  | c15 d2h volume / wall                               |
| host hit vs SSD restore TTFT, p50          | 0.30-0.34 s vs 22-23 s                                           | c14/c15 window                                      |
| a returning turn's missing span            | p50 ~250 pages (16K tokens)                                      | c15`oracle_eval.csv` own pages                    |
| idle SSD bank before a congested window    | 36 % of the first 7,200 s                                        | §1.5                                               |

---

## 9. Status

| step                                             | state                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 4.1 flags                                        | done 2026-10-10;`test_oracle_v2_flags.py` 5 passed                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| 4.2 read scheduler                               | done on CPU 2026-10-10, budget double count fixed after review 2026-10-11 (`test_storage_read_scheduler.py` 6, `test_storage_read_dispatch.py` 15 passed); GPU smoke pending (handoff §H4)                                                                                                                                                                                                                                                                                                                                                                                             |
| 4.3 pin                                          | done on CPU 2026-10-10 (pin cases in`test_storage_read_dispatch.py`, admit-site release on a CPU tree); GPU smoke pending                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| 4.4 prefetcher deadline mode + online load model | done on CPU 2026-10-10, dead sessions end their registration after review 2026-10-11 (`test_next_turn_prefetch.py` 32 passed, `test_next_turn_estimators.py` 4); GPU smoke pending                                                                                                                                                                                                                                                                                                                                                                                                      |
| 4.5 next-use eviction                            | done on CPU 2026-10-10, overdue tier and bounded registry after review 2026-10-11 (`test_next_use_eviction.py` 6 passed); GPU smoke pending                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| 4.6 events +`scripts/srs_summary.py`           | done 2026-10-10                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| 4.7 next-use replay                              | done 2026-10-10; numbers in 4.7 and §6                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| 4.8 read-shift replay                            | done 2026-10-10; reproduces §1.5 (shift 0 within 0.21 s mean of the real completions)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| unchanged suites                                 | `test_storage_prefetch_lifecycle.py` 8, `test_resident_prefix_probe.py` 4, `test_scheduler_hicache_events.py` 5, `test_session_unified_radix_cache.py` 3, `test_unified_hicache_regressions.py` 13, `test_evict_policy.py` 23, `test_mem_cache_utils.py` 36, `test_server_args_namespaces.py` 4, `test_runtime_context_config_bags.py` 18, `test_server_args.py` 279 passed (one case needs the optional `soundfile` package), `test_unified_radix_cache_unittest.py` 1280 passed with 2 CUDA out-of-memory artifacts of the laptop's 8 GB GPU (2026-10-11, laptop) |
| review                                           | adversarial review of the engine diff (6 dimensions, 14 findings verified by 3 refuters each): 8 confirmed and fixed (stale next-use entries, budget double count, unbounded registry, five coverage gaps), 6 refuted, 20 low-ranked left unverified; the two environment-only findings in the unverified list (the`launched` field, pin fraction vs cap) were fixed as well                                                                                                                                                                                                              |
| server arms                                      | `scripts/start_server.sh`: `three_tier_wc_dl`, `three_tier_wc_oracle_v2`, `three_tier_wc_oracle_v2_nu`, `three_tier_wc_wb` (dispatch dry-run checked on CPU; never booted)                                                                                                                                                                                                                                                                                                                                                                                                        |
| 5 smokes + campaign 16                           | not started: needs the H200 (`ORACLE_V2_HANDOFF.md`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
