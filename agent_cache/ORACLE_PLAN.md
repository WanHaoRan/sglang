# Oracle admission-anchored SSD staging — implementation plan

Working document for the oracle prototype of admission-anchored staging (AAS, `results/analysis_gapscale/TAKEAWAYS.md`).
The user gives a go per step; the status table is at the end. Written 2026-10-01 on the Nebius H200 box.

**Code version.** Repo `git@github.com:WanHaoRan/sglang.git`, branch `main`. The engine (`python/`, `test/`) is the
upstream merge `ace8ac79a7` (contains #39283); `eddb52490c` adds results and step 1 and changes no engine file. Every
`file:line` below is at that engine tree. If lines drift, re-find them by the function names given next to them.

**Sources.** Read-only code checks and an adversarial review (2026-10-01), campaigns 7 (x70), 10 (realistic cadence) and
11 (x10 + #39283), and `scripts/read_backlog.py`. The whole document was then checked against the code, scripts and logs
(51 corrections applied, 2026-10-01). Run data is under `agent_cache/results/` (committed).

---

## 0. Picking this up on another machine

- `git pull` on `main`; `git log --oneline -3` must show the commit that added this expanded plan on top of
  `eddb52490c` (the 61-line plan in `eddb52490c` itself is the older short version).
- **Container:** create it as in `RUNBOOK.md` §2.3: image `lmsysorg/sglang:nightly-dev-20260907-30705c00`,
  `docker run -itd --name sglang_hicache --gpus all --ipc=host --network=host --privileged --ulimit memlock=-1:-1
  --ulimit nofile=1048576:1048576 -v <repo>:/sgl-workspace/sglang -v ~/.cache/huggingface:/root/.cache/huggingface
  -v /mnt/ssd:/mnt/ssd:rshared <image> /bin/zsh` (without `memlock` the host pool cannot be pinned). sglang is imported
  from `/sgl-workspace/sglang/python`, so code edits take effect on the next server boot.
- **Machine-specific checks:** `start_server.sh` `l3_prep` (lines 93-95) refuses every `three_tier_*` arm unless
  `/mnt/ssd/hicache_l3` is on `/dev/vdc` — edit the device name for the new SSD. `NAT160` needs ≥ ~170 GB of pinnable
  host RAM (the H200 box pins 196 GB); otherwise set `HOST_GB`.
- **Kernel pin:** the merged tree needs `sglang-kernel 0.4.7`; the image ships 0.4.6.post1.
  `pip install --no-deps sglang-kernel==0.4.7` (CUDA 13, torch 2.13 build); rollback `... ==0.4.6.post1`.
- **Model and pools used by campaigns 7, 10 and 11:** `MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 CTX=131072 KV_DTYPE=auto TEMPLATE=`
  (bf16 KV, stock template), level `NAT160`: L1 = the profiled device pool (668,160 tokens on an H200), L2 = 160 GB
  host (1,627,648 tokens, write-through), L3 = nixl POSIX on `/mnt/ssd`, cleaner at 70/60 %. 98,304 B/token, page = 64
  tokens = 6 MiB. TP=1, `schedule_policy=fcfs`, overlap scheduler on, `--max-running-requests 64`,
  `--chunked-prefill-size 8192`, `--enable-request-time-stats-logging` (from campaign 8 on; not in campaign 7),
  `SGLANG_LOG_MS=1` (ms log stamps; set by `start_server.sh`). Retry flags: campaigns 7-10 ran the pre-merge engine
  with paced retries off (poll 0 / max 4); campaign 11 and HEAD default to 8 / 8 — baselines quoted from c7/c10 come
  from a different retry configuration than a HEAD boot. A different GPU or SSD changes L1 and every timing seed below:
  re-measure (section 8).
- **What needs no GPU:** steps 2-4 and their unit tests, client dry runs, the stub-server client test (step 1). Live
  smokes (steps 2, 5-7) and experiments need the GPU box.
- **Repo rules that apply** (`.claude/rules/`, `.claude/skills/`): read `large-class-style` before editing `Scheduler`
  (collaborator + one-line delegates, `maybe_init_*` helper with the gate inside); new data types are `msgspec.Struct`,
  not `@dataclass`; no defensive `getattr`/`hasattr` (always define the attribute, None-check it); comments per
  `comment-style.md`; CPU unit tests under `test/registered/unit/...` with `register_cpu_ci(est_time=N,
  suite="base-a-test-cpu")` and `CustomTestCase`; every new test must pass `.claude/rules/unit-test-admission.md`
  (bug regression, derived property or bookkeeping; no tests that mirror the formula). Load the
  `sglang-runtime-context` skill before adding the server-arg field (step 2). Server flags rather than `SGLANG_*` env
  vars for this feature.
- **Running unit tests** (container): `docker exec -w /sgl-workspace/sglang sglang_hicache python3 -m pytest -q -p
  no:cacheprovider <test file> -k <pattern>`.
- **Booting / stopping a server:**
  `docker exec -it sglang_hicache bash -c 'MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 CTX=131072 KV_DTYPE=auto TEMPLATE= L3_CLEANER_PCT=70,60 SGLANG_TIMEOUT_KEEP_ALIVE=3600 bash /sgl-workspace/sglang/agent_cache/scripts/start_server.sh <arm> NAT160'`
  and `docker exec sglang_hicache bash /sgl-workspace/sglang/agent_cache/scripts/stop_server.sh`
  (`start_server.sh` alone defaults to Qwen3-32B-FP8, CTX 32768, fp8 KV and the cleaner at 30/20; for a cold L3 on a
  `three_tier_*` arm first run `find /mnt/ssd/hicache_l3 -mindepth 1 -delete` in the container — `run_compare.sh`
  does this itself). Arms:
  `hbm_lru | hbm_host | three_tier_to | three_tier_wc | three_tier_wc_norq` (`start_server.sh` header). Each boot gets
  `results/<UTC>_<arm>_<level>/` with `server.log`, `server_args.txt`, `run.txt`.

---

## 1. Why: the evidence the oracle builds on

| Regime | What limits SSD restores | Numbers |
|---|---|---|
| x70 (campaign 7) | One FIFO SSD read thread; restores arrive in bursts and queue behind each other | Waiting for the read thread = 58 % (`timeout`) / 80 % (`wait_complete`) of a restored turn's TTFT; own read 15-17 %. TTFT ≈ 1.0 s + GB queued ahead / 2.0 GB/s (`wait_complete`; `timeout` ≈ 1.1 s + GB / 1.3 GB/s; corr 0.96-1.0): the backlog drains slower than the 2.4 GB/s raw read rate (reads overlapping SSD writes are ~1.9× slower). A restored turn becomes the oldest waiting request exactly when its own read starts (`wait_complete`). Figures: `results/analysis_gapscale/read_backlog*.png` |
| Realistic cadence (campaign 10) | Own read only; no backlog | SSD serves only returns after long human pauses (7.7-33 min, median 22 min; ~2-2.6 % of returning turns): TTFT 1.02 s, of which the fetch 0.62 s (61 %). Storage-hit turns sit in the queue p50 0.62-0.68 s vs 0.03 s for other turns |
| x10 (campaign 11) | Queue-time eviction of the arrival match; #39283 re-queries at admission | Re-query arms: mean TTFT 116 → 68-69 s (paired `three_tier_wc` vs re-query off −47.8 s). In `three_tier_wc`, 1,145 of the 1,158 SSD restores came from the re-query, each costing re-issue → admission p50 3.8 s, p90 7.1 s (re-issue → data in host memory p50 ~2.0 s). 94 % of the remaining 1,339 recomputes (`timeout`: 93 % of 1,361) had exhausted the shared 8-attempt budget on paced polls of the never-stored new tail |

**What the oracle measures:** the upper bound of starting each SSD read at exactly the right time, given exact gaps.
Case 1 (read before the turn arrives) targets x70 and campaign-10 cadence; case 2 (refetch while queued) targets x10's
~4 s re-query latency and the budget-blocked recomputes. A real predictor can only do worse.

---

## 2. Design

### 2.1 Hint (step 1, done)
Turn t of conversation c (rid `<tag>-c<c>-t<t>`) carries, unless it is the last turn or the wall cap certainly cuts the
next one:
```json
"kv_hints": {"protocol_version": "1", "message_id": "<tag>-c<c>-t<t>",
  "actions": [{"action_id": "next_turn", "action_type": "sglang.next_turn", "action_version": "1",
               "payload": {"session": "<tag>-c<c>", "next_rid": "<tag>-c<c>-t<t+1>", "next_gap_s": 7.048}}]}
```
`next_gap_s` is the exact sleep the client will do after receiving turn t's last token and doing its bookkeeping. It is
relative seconds; never send client timestamps (different clock).

### 2.2 Predicted arrival
`P = t_commit + next_gap_s + δ̂`
- `t_commit`: `time.perf_counter()` when turn t's KV is committed (hook at the end of `cache_finished_req`; same moment
  as the decode-finish `set_completion_time`, `batch_result_processor.py:1359-1361`).
- `δ̂`: learned offset = (scheduler enqueue of t+1) − t_commit − gap. EWMA α = 1/8, seed 0.10 s; **discard** samples
  outside [0, 5] s (do not clip; outliers of 138 s and 162 s exist). Measured δ: sparse p50 0.088 s / p90 0.166 s;
  dense p50 0.125 s / p90 0.87 s; it grows with prompt length (template + tokenization). Always positive.

### 2.3 Admission time
- **Before arrival:** `A = P + Ŵ`. `Ŵ = 0` if the queue is empty, else `W_ql = (Q_now + 0.5) / λ̂` with the rate
  λ̂ = (admissions in the last 60 s) / 60 s. Log a second estimate `W_ewma` = EWMA (α = 1/8) of measured waits of admitted requests,
  **excluding requests credited with their own SSD read** (`storage_hit_length > 0`; not "retry attempts > 0", which
  would drop almost every long-waiting dense request and bias Ŵ low).
- **After arrival:** `A = now + (k + 0.5) / λ̂`, k = the request's index in `waiting_queue` (FCFS keeps arrival order;
  15-19 % of x10 requests overtake an earlier one, so position slightly over-counts).
- **Fallback** when the 60 s window holds < 5 admissions (campaign-11 `three_tier_wc`: 18 of 117 60-s windows had no
  admission, 33 had fewer than 5): use λ̂ = (admissions in 300 s) / 300 s; if that still has < 5, then after arrival fire
  case 2 when `k <= 1`, and before arrival (Q_now > 0) use `W_ewma` if it has samples, else treat Ŵ as unknown (no
  case-1 trigger; `est=none` in `ntp_due`). Log which estimator fired.
- Offline accuracy at x10 (campaign 11): W_ql MAE ≈ 12 s, last-wait/EWMA MAE 8-10 s (lags by one wait); true wait
  mean 66 s (p50 55 s, p90 142 s). The residual estimator, re-evaluated per admission, puts the median true residual on target: firing at
  estimate ≤ 4 s gives true residual p10/p50/p90 = 0.75/3.97/6.75 s.
- At sparse load W ≈ 0.03 s, so A ≈ P.

### 2.4 SSD load time
`L̂ = lag_alloc + b·(own_pages + backlog_pages) + a + lag_drain`
- `own_pages = (len(chain) − l12) / 64`: the session's committed pages that are no longer in GPU or host memory (step-3
  probe). Upper bound: L3 holes or cleaner deletions can only shorten the loadable span.
- `backlog_pages`: pages of reads already allocated and waiting on / in the read thread (step-3 helper).
- Seeds (campaign 11 `three_tier_wc`, 1,199 reads; campaign 7's aggregate throughput agrees, 2.38-2.47 GB/s):
  `a = 0.044 s`, `b = 2.5 ms/page` (OLS ms = 43.5 + 2.477 × pages; aggregate ≈ 2.4 GB/s; reads overlapping SSD
  writes are ~1.9× slower, 3.66 vs 1.93 ms/page), `lag_alloc = 0.5 s` (query answered → I/O start with an idle read
  thread, dense; sparse ~0.03 s), `lag_drain = 0.36 s` (I/O end → insert, dense; sparse ~0.02 s). The existence query
  is ~5 ms and is ignored.
- The backlog drained more slowly than `b` at x70 (campaign 7: ~2.0 GB/s `wait_complete`, ~1.3 GB/s `timeout`, i.e.
  3.1-4.7 ms/page, because restores overlap write-through writes). Consider a separate `b_backlog ≈ 3.1 ms/page`.
- Seeds are static in v1 and are dense-load values: at sparse load they overestimate L̂ by ~0.8 s (true load ≈ 0.85 s
  for ~300 pages). Validate them in observe mode (E1).

### 2.5 Trigger and cases
- Fire on the first scheduler loop with `now ≥ T = A − L̂ − margin` (margin 0 by default; decision 2 in section 6).
  Ticks are one loop iteration apart (≪ 1 ms idle; one forward under load, e.g. ~1.0 s, p90 1.5 s, for an 8192-token
  chunk at x10).
- **Case 1** (`next_rid` not yet seen in the queue): stage under the next turn's own handle `CacheRequestHandle(next_rid, 0)`.
  If the request arrives while the read runs, it joins it; if the read has finished, its pages are in host memory and
  are credited as storage at admission (section 3.3).
- **Case 2** (request already queued): if the probe shows ≥ 256 tokens of its chain evicted, call the existing
  `Scheduler._prefetch_kvcache(req)` (not `_retry_storage_prefetch`, which spends the retry budget).
- Sparse load expectation with the dense seeds: `T ≈ P − (0.9 s + 2.5 ms × ~300 pages) ≈ P − 1.7 s` (case 1), about
  0.8 s earlier than needed (the staged node then sits ~0.8 s longer as an evictable host leaf). Dense: `Ŵ` (66-120 s) ≫ `L̂`
  (2-4 s), the request arrives first and case 2 fires when its remaining wait ≈ L̂.

### 2.6 Modes (`--hicache-storage-next-turn-prefetch off|observe|on`)
- `off` (default): the prefetcher object is `None`; every hook returns at a None check; scheduling and caching are
  identical to HEAD (only step 2's chat-route `kv_hints` validation differs).
- `observe`: registers plans, estimates, logs every prediction and its error; only read-only cache calls; no prefetch.
- `on`: as observe, plus case 1 and case 2 actions.

### 2.7 New log events
Format (parsed by `delay_components.py:20` and `timeline.py:37-38`): `HICACHE_EVT <kind> rid=<rid> key=value ...` with
`rid` the first key; values match `[\w./-]+` (no `:`, `,`, `%`, `+`, spaces); times in integer ms.
`ntp_plan` (rid = next_rid; sess, gap_ms, chain, pred_arr_ms), `ntp_arrive` (arr_err_ms = arrival − P, delta_ms = raw
arrival − t_commit − gap, pos, qlen), `ntp_due` (case,
l3_pages, backlog_pages, load_ms, wait_ms, lead_ms, est), `ntp_stage` (tokens, matched, inflight), `ntp_refetch`
(issued, shrink, pos, resid_ms, load_ms), `ntp_reevict` (evicted, refetches, capped=0|1), `ntp_admit` (wait_ms, pred_wait_ql_ms, pred_wait_ewma_ms,
adm_err_ms, trig_lead_ms, case), `ntp_expire` (staged, reason).

---

## 3. Engine facts the implementation relies on (verified at HEAD)

### 3.1 The L3 (SSD → host) prefetch pipeline
1. **Arrival** `Scheduler._add_request_to_queue` (`scheduler.py:3247-3262`): `_prefetch_kvcache(req)` (3259), append to
   `waiting_queue` (3260), `set_wait_queue_entry_time` (3261). Retracted requests re-enter here with attempts and
   `last_match_len` reset (3251-3254; `is_retracted=True` at 4230). Preempted requests (4041-4043) re-enter, and
   grammar-ready requests (3819) first enter, through the same path without that reset.
2. **`_prefetch_kvcache`** (`scheduler.py:3110-3159`): `req.init_next_round_input(...)` = full `match_prefix` (3112);
   `matched_len = len(prefix_indices) + host_hit_length`, stored as `req.storage_prefetch_last_match_len` (3129-3130,
   **before** any gate); gate `is_backuped(last_host_node) or is_root(last_host_node)` (3132-3139); then
   `tree_cache.prefetch_from_storage(req.cache_request_handle, last_host_node,
   req.full_untruncated_fill_ids[matched_len:match_end], get_last_hash_value(last_host_node), prefix_keys (only if
   hicache_storage_pass_prefix_keys, default off), matched_prefix_tokens=req.full_untruncated_fill_ids[:matched_len],
   extra_key, cache_salt, storage_hit_end)` (3149-3159).
3. **Existence query**: single query thread (`cache_controller.py:1234-1268`), nixl `batch_exists` returns the leading
   run of hits; logs `HICACHE_EVT s2h_query`. p50 ~5 ms incl. queueing.
4. **Allocation**: on the scheduler thread, once per loop: `check_hicache_events` (`unified_radix_cache.py:3439`) →
   `_drain_storage_control_queues_impl` (2804) → nested closure `_drain_and_alloc_storage_hit` (2953) → nested
   `_try_alloc_storage_hit` (2837-2951): host alloc (may evict host leaves, LRU; 2885-2889), trims
   `operation.hash_value` (2945), sets `ongoing_prefetch[h].host_indices` (2947), puts the op on `prefetch_buffer` (2950).
5. **Read**: single FIFO I/O thread `prefetch_io_aux_func` (`cache_controller.py:1164-1187`), logs `HICACHE_EVT s2h_io
   rid= pages= ms=` (use `pages`, not the stale `tokens=`).
6. **Insert**: ack drained on a later loop; `_handle_prefetch_result` (`unified_radix_cache.py:2257`) `insert_host`s a
   new host-only node (unlocked, an evictable host leaf immediately), deletes the op from `ongoing_prefetch`, records
   `prefetch_loaded_tokens_by_reqid[h]` and its start (2360-2366), logs `HiCache prefetch success req= completed=`.
7. **Admission** (`scheduler.py:3933-4053`): `check_prefetch_progress(h)` (`unified_radix_cache.py:2228`) — while the
   op is in flight the request is **skipped** (`continue`, 3959-3965; `wait_complete` never terminates, `timeout`
   terminates past its deadline); then `pop_prefetch_loaded_span(h)` (3966-3976, credits storage), re-match (3978),
   #39283 re-check (3979-3982), `add_one_req` → `init_load_back` (H2D, p50 44 ms), leave the queue (4040),
   `set_forward_entry_time` (4053).

### 3.2 `prefetch_from_storage` contract (`unified_radix_cache.py:1977-2139`)
- Takes only a `CacheRequestHandle` (frozen, `base_prefix_cache.py:44-47`; a `Req` uses `(rid, 0)`), no `Req` needed.
  `@rank_consensus(same_params=["request.rid", "len(new_input_tokens)"])`.
- In order: cancels any paced retry for the rid (1994); logs `prefetch_start` (1999, **before** the dedupe); returns if
  the handle is already in `ongoing_prefetch` (2001-2005: one op per handle → the join point); declines spans shorter
  than `prefetch_threshold` = 256 tokens silently (2045-2051); when rate-limited (`prefetch_tokens_occupied ≥ 0.5 ×
  host pool` = 813,824 tokens, `cache_controller.py:576-583`) calls `poll_miss(rid)` and returns (2052-2056); host-locks
  the anchor (2060-2064); submits and registers the op with `host_indices=None` (2119-2126); charges occupancy (2139).
- Returns `None` outside PP mode: use `has_ongoing_prefetch(h)` (2169) to know whether an op was issued.
- Does **not** check that the anchor is backed up: with an un-backed non-root anchor the fetched data is dropped at
  `insert_host`. Callers must copy the scheduler's gate.
- The PP `get_prefetch_submission` early return (1995-1997) is dead at PP=1.

### 3.3 Joining, records and cleanup for a handle staged before its request exists
- **Arrives mid-read:** the request's own `_prefetch_kvcache` reaches `prefetch_from_storage`, logs a second
  `prefetch_start` under the same rid and returns at the dedupe: it joins. At admission it is skipped until the op
  finishes, then credited as storage. #39283 stays quiet (current match ≥ the arrival match).
- **Arrives after the read finished:** its arrival match includes the staged host node; it still looks up its new tail
  (normally a miss, `poll_miss`); `pop_prefetch_loaded_span` still returns the staged span → credited as storage. If its
  own lookup *hits*, line 2360 overwrites the record and the staged tokens count as host ("Replacing unresolved
  storage-hit accounting", 2491-2498). Metrics-only skew either way; the client's `cached_details` stay right.
- **Never arrives:** records left behind: `prefetch_loaded_*_by_reqid[h]`, metric accounting, and any
  `storage_prefetch_retries._pending[next_rid]` (a decline or miss armed `poll_miss`; `pop_ready` only walks the queue,
  so the entry is never popped and its `due_step=None` forces a queue scan every step, `storage_prefetch.py:50-54,
  68-69`). Cleanup: `tree_cache.finish(h, CacheRequestOutcome.ABORT)` → `release_aborted_request`
  (`base_prefix_cache.py:371-377`, `unified_radix_cache.py:2592-2643`) pops the records and cancels the retry. **Only
  when `has_ongoing_prefetch(h)` is False** (in the I/O phase it frees the fetched pages without inserting them) and
  **never after the request has been seen** (it owns the handle).
- While a staged op is in flight `is_fully_idle` is False (`scheduler.py:5030`): storage attach/detach are rejected
  ("scheduler is not idle"), and flush_cache fails unless called with `timeout_s > 0` (then deferred until idle).

### 3.4 `timeout` policy deadline
`_prefetch_timeout_check_linear_func` (`unified_radix_cache.py:2141-2146`): `time.monotonic() − op.start_time > 1.0 +
pages × 0.015625 s`; `start_time` is set when the op is created (`hybrid_cache_controller.py:133`). A staged op's budget
therefore runs from staging, not arrival; an op still in its query phase (`hash_value == []`) has 1.0 s. Nothing checks
progress before the request exists, so staged ops never time out pre-arrival, but a joined op can be cut at the first
admission check. v1 runs `wait_complete` only (decision 3).

### 3.5 #39283 and the shared retry budget
- Paced miss polls (`storage_prefetch.py:39-74`, `pop_ready`): every 8 scheduling passes (loop iterations); a paced
  poll pending for the queue head is cancelled, not deferred (45-49); polls go through `_retry_storage_prefetch`
  (`scheduler.py:3174-3192`, `attempts += 1`) → `_prefetch_kvcache`.
- `_prefetch_after_device_hit_loss` (`scheduler.py:3194-3221`): at admission, if `prefix + host_hit < last_match_len`,
  re-issue and skip — **unless `attempts ≥ max_attempts`** (8).
- At x10 the arrival lookup covers only the new tail (the context is still in L1/L2), it misses, the polls retry the
  same never-stored tail 8 times within seconds, and the budget is gone long before admission (~120 s later).
- Blind spot: `last_match_len` is set before the fetch (3130), so if an arrival-fetched span is evicted later, current
  ≥ previous and #39283 does not fire.
- `_prefetch_kvcache` itself checks no budget: calling it directly (case 2) works even in the `_norq` arm.

### 3.6 Matching side effects and residency
- `match_prefix` is not read-only: splits a partially matched node (`unified_tree_core.py:882`), stamps
  `last_access_time` along the whole matched path (947-951), returns cache actions. Calling it to "look" makes the
  session MRU and changes eviction. Use it only when actually issuing a read.
- `calc_priority` does not re-match queued requests under FCFS: it re-matches only when
  `supports_fast_match_prefix()` (`schedule_policy.py:271-277`), which is False for the Python tree core (interface
  default `unified_tree_core_interface.py:409-411`, URC delegates at 567-568). A queued request is re-matched only when
  the admission loop reaches it (`scheduler.py:3978`: every pass for requests ahead of the loop's break that are not
  skipped for an in-flight prefetch) or on a paced / #39283 re-issue (`_prefetch_kvcache`, 3112). Deeper requests'
  `prefix_indices` / `host_hit_length` are stale snapshots; a request holds no lock until admission.
- `UnifiedTreeCore._walk_span(key, end)` (`unified_tree_core.py:2563-2580`) is read-only (no split, no stamp) and stops
  at a missing child or `child.evicted and not child.backuped` (the rule `match` uses, 876), but it is private, copies
  the remaining key at every node, and is absent from the Rust core. `match_full_device_prefix` (905-929) is read-only
  but device-only.
- RadixKey for a walk must be page-aligned (else `IndexError` in `child_key_at`), hold `array('q')` token ids (else
  `match_at` asserts), and carry the session's `extra_key` and `cache_salt`.
- Under write-through, "not in L1/L2" usually means the L3 write was attempted, not that it succeeded (backup aborts
  under host pressure at `unified_radix_cache.py:1649-1652` are silent; the nixl cleaner deletes at watermarks).

### 3.7 Turn end
- All finishes go through `release_kv_cache` (`mem_cache/common.py:280-299`) → `UnifiedRadixCache.cache_finished_req`
  (`unified_radix_cache.py:955-1106`, `@rank_consensus`): token ids `(origin_input_ids + output_ids)[:owned_kv_len]`
  (975); committed chain `radix_key = RadixKey(token_ids, req.extra_key, is_bigram=..., cache_salt=req.cache_salt)
  .page_aligned(page_size)` (1014-1019), inserted at 1025; sub-page tail freed; `req.last_node =
  result.last_device_node` (1091-1092); session-refs block (1100-1106). `result` is initialised to None (980) and set
  only when `is_insert`; `radix_key` is bound only when `is_insert`. Retraction (`schedule_batch.py:2250`) and
  queued/chunked aborts (`scheduler.py:3557`) enter with `is_insert=False`, but a **running** request that is aborted
  gets `to_finish = FINISH_ABORT` (`scheduler.py:~5540-5555`) and finishes through the normal path with `is_insert=True`
  (`batch_result_processor.py:1353-1359`): `result is not None` does not exclude it, so the `FINISH_ABORT` filter is
  required.
- Turns finish by `FINISH_LENGTH` (the client sends `ignore_eos` + `max_tokens`): do not filter it out; skip
  `FINISH_ABORT` and unfinished (retracted) requests, empty keys and `result.rotation_tail_declined`.
- Don't keep NodeIds across ticks (host-evicted nodes are deleted); keep the RadixKey and re-match/probe.

### 3.8 The per-iteration tick and clocks
- `Scheduler._process_hicache_events` (`scheduler.py:3607-3617`) runs once per loop iteration, idle or not (no
  `--sleep-on-idle`), after the request ingest of that iteration and before admission; `check_hicache_events` there is
  the TP-consensus drain, then `_process_storage_prefetch_retries()` under `if self.enable_hicache_storage:`
  (3616-3617). New hook goes right after it, inside that `if` (storage can be attached/detached at runtime).
- Clocks: `req.time_stats` uses `time.perf_counter` (`observability/req_time_stats.py:67`); the timeout check uses
  `time.monotonic`. Both are `CLOCK_MONOTONIC` on Linux CPython. Use `time.perf_counter` in the prefetcher.
- Wall-clock decisions are rank-local while `prefetch_from_storage` and the drains are TP collectives: v1 must refuse
  TP/PP > 1, DP attention and disaggregation at startup.

### 3.9 Scheduler wiring conventions
- Optional collaborators: `maybe_init_<x>()` sets `self.<x> = None`, returns on the gate, else constructs with narrow
  kwargs (e.g. `maybe_init_dynamic_chunk_sizer`, `scheduler.py:1322-1340`); call sites `if self.<x> is not None:`.
- Closest pattern: `StoragePrefetchRetries` (`mem_cache/storage_prefetch.py`), declared `Optional` on `BasePrefixCache`
  (`base_prefix_cache.py:339`), driven by `Scheduler._process_storage_prefetch_retries` (`scheduler.py:3161-3171`).
- `self.init_running_status()` is at `scheduler.py:630`, `self.init_schedule_policy()` at 643 (tree cache exists by then).
- `test/registered/unit/managers/test_scheduler_hicache_events.py` builds a `Scheduler` with `__new__` and only the
  attributes in `setUp` (22-37) and asserts exact call lists: any new attribute read in `_process_hicache_events` must
  be added there.
- Avoid the name "staged prefetch" (taken by buffer mode: `StagedPrefetchPlan`, `Req.staged_prefetch_plan`); use
  `next_turn_prefetch` / `ntp`.

### 3.10 Analysis-script constraints
- `delay_components.py` keeps the first timestamp per (rid, kind) and times the L3 read from the last `prefetch_start`
  at or before the first `s2h_io`; `timeline.py` keys s2h data by rid with `.update`; `read_backlog.py` keeps the first
  event of each kind per rid (and puts every `s2h_io` in its global read list, which correctly counts staged reads as
  backlog for other turns). Staged reads log `prefetch_start/s2h_query/s2h_io` under `next_rid` **before that turn's send**, so all
  three must ignore events stamped before the turn's send (step 8).

---

## 4. Steps

### Step 1 — client hints (DONE, 2026-10-01)
`agent_cache/scripts/replay_agentic.py`: `--oracle-hints` (refused with `--gap-sample`), `Replayer.rid()`,
`Replayer.next_turn_hints()`, `chat_turn(..., kv_hints=None)` adds `body["kv_hints"]` only when given, `conversation()`
draws turn t+1's gap when sending turn t (`pending_gap`) and records `oracle_hint = {next_rid, next_gap_s, suppressed}`;
suppressed when `elapsed + gap ≥ --max-seconds`. `start_client.sh`: `ORACLE_HINTS=1`.
Verify: `cd agent_cache/scripts && python3 replay_agentic.py --dry-run --oracle-hints --trace
../traces/lmcache_agentic_trace_262k.json --num-conversations 2 --max-turns 4 --offset 32 --gap-scale 10 --gap-cap 1200
--tag cmp_x --dry-run-max-chars 1 | grep -E '^--- conv|gap: recorded|oracle hint'` — each hinted gap equals the next
turn's gap; output without the flag is identical apart from the hint lines. (A stub-server test also checked bodies,
envelope validity with the server's `decode_kv_hints_envelope`, `gap_slept` = hinted gap, and wall-cap suppression.)

### Step 2 — chat transport for `kv_hints` + the server flag (user is implementing)
Goal: `/v1/chat/completions` accepts the envelope and carries it to `Req.kv_hints`; add the mode flag. Nothing reads
either yet, so behaviour is unchanged except that a malformed `kv_hints` on chat is now rejected (400) instead of dropped.
The path after the chat model already exists: `GenerateReqInput` normalisation decodes it (`io_struct.py:567-571`) →
`TokenizedGenerateReqInput.kv_hints` (`tokenizer_manager.py:1528`) → `Req(kv_hints=...)` (`scheduler.py:2786`) →
`Req.kv_hints` (`schedule_batch.py:1347`).

1. `python/sglang/srt/entrypoints/openai/protocol.py`
   - import next to `from sglang.utils import convert_json_schema_to_str` (line 73):
     `from sglang.srt.managers.kv_hints import KvHintsEnvelope` (checked: no import cycle).
   - in `ChatCompletionRequest` (class at 849), after `custom_params: Optional[Dict] = None` at **line 951** (not 381,
     which is `CompletionRequest`):
     ```python
     # Versioned KV-hint envelope (see GenerateReqInput.kv_hints), forwarded untouched.
     kv_hints: Optional[KvHintsEnvelope] = None
     ```
   - Typed (not `Dict`) so pydantic rejects a malformed envelope with 400 before a stream starts. Known trap:
     `model_dump(mode="json")` / `model_dump_json()` of a chat request carrying `kv_hints` raises
     `PydanticSerializationError`; nothing in the server does that today (request logging uses python-mode dumps).
2. `python/sglang/srt/entrypoints/openai/serving_chat.py`: in `GenerateReqInput(...)` (1232-1275), after
   `session_id=request.session_id,` (**1256**) add `kv_hints=request.kv_hints,`.
3. `python/sglang/srt/arg_groups/fields/memory.py`: after `hicache_storage_prefetch_retry_max_attempts` (ends `] = 8`
   at **line 201**), before the `# Unified Radix Cache` banner:
   ```python
       hicache_storage_next_turn_prefetch: A[
           str,
           Arg(
               help=(
                   "Oracle next-turn prefetch from storage, driven by the "
                   "sglang.next_turn kv_hints action (session, next request id, gap). "
                   "'observe' predicts and logs only; 'on' also starts each prefetch at "
                   "the predicted admission time minus the estimated load time. "
                   "Experimental; needs the hierarchical cache with a storage backend."
               ),
               choices=["off", "observe", "on"],
           ),
       ] = "off"
   ```
   The CLI flag `--hicache-storage-next-turn-prefetch` is generated from the field name; read it later as
   `get_memory().hicache_storage_next_turn_prefetch` (`runtime_context.py:1288`).
4. (optional) `python/sglang/srt/arg_groups/field_order.py`: append `"hicache_storage_next_turn_prefetch",` as the last
   entry (after `"disaggregation_decode_host_receive_threshold",`, line 519). Unlisted fields are appended anyway
   (`arg_groups/fields/__init__.py:45-47`); never insert it near 366 (shifts the positional constructor order).
5. Tests, `test/registered/unit/entrypoints/openai/test_serving_chat.py`: imports `from pydantic import
   ValidationError` and `from sglang.srt.managers.kv_hints import KvHintsEnvelope`. In
   `test_convert_to_internal_request_single` (588): before `_convert_to_internal_request`, set
   `self.basic_req.kv_hints = KvHintsEnvelope(protocol_version="1", message_id="m-1")`; after it,
   `self.assertIs(adapted.kv_hints, self.basic_req.kv_hints)`. New test:
   ```python
   def test_chat_request_kv_hints_validated_at_the_boundary(self):
       msgs = [{"role": "user", "content": "Hi?"}]
       env = {"protocol_version": "1", "message_id": "replay-c0-t0",
              "actions": [{"action_id": "next_turn", "action_type": "sglang.next_turn", "action_version": "1",
                           "payload": {"session": "replay-c0", "next_rid": "replay-c0-t1", "next_gap_s": 7.048}}]}
       req = ChatCompletionRequest(model="x", messages=msgs, kv_hints=env)
       self.assertIsInstance(req.kv_hints, KvHintsEnvelope)
       self.assertEqual(req.kv_hints.actions[0].payload["next_rid"], "replay-c0-t1")
       with self.assertRaises(ValidationError):
           ChatCompletionRequest(model="x", messages=msgs, kv_hints={"actions": 3})
   ```
Acceptance:
- `pytest test/registered/unit/entrypoints/openai/test_serving_chat.py -k "convert_to_internal_request or kv_hints"`
  (baseline before the change: 4 passed; after: 5) and `test/registered/unit/managers/test_io_struct.py -k KvHints`
  (9 passed, unchanged).
- Boot any arm; `server.log` shows `'hicache_storage_next_turn_prefetch': 'off'`.
- curl (container): a valid envelope → 200; `"kv_hints": {"actions": 3}` → 400 (before step 2 both return 200):
  ```bash
  docker exec sglang_hicache bash -c 'U=http://127.0.0.1:30000/v1/chat/completions; M="\"model\":\"x\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":4"
    curl -s -o /dev/null -w "valid %{http_code}\n" $U -H "Content-Type: application/json" -d "{$M,\"kv_hints\":{\"protocol_version\":\"1\",\"message_id\":\"t-c0-t0\",\"actions\":[{\"action_id\":\"next_turn\",\"action_type\":\"sglang.next_turn\",\"action_version\":\"1\",\"payload\":{\"session\":\"t-c0\",\"next_rid\":\"t-c0-t1\",\"next_gap_s\":5.0}}]}}"
    curl -s -o /dev/null -w "malformed %{http_code}\n" $U -H "Content-Type: application/json" -d "{$M,\"kv_hints\":{\"actions\":3}}"'
  ```
- Live client smoke (deferred from step 1): every turn 200, `oracle_hint` in the JSONL:
  `docker exec sglang_hicache bash -c 'cd /sgl-workspace/sglang/agent_cache/scripts && MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 TRACE=/sgl-workspace/sglang/agent_cache/traces/lmcache_agentic_trace_262k.json NCONV=2 TURNS=4 C=2 GAP=1 OFFSET=32 CHECK_IDS=0 ORACLE_HINTS=1 TAG=smoke_hints bash start_client.sh'`
Notes: exposing `kv_hints` on chat diverges from upstream's note that it is set "by a trusted orchestrator ... never by
an application client" (`io_struct.py:362-367`). No docs update is required (`server_arguments.mdx` is not enforced).

### Step 3 — read-only residency probe + read-backlog helper
Files: `python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py`, `.../unified_tree_core_interface.py`,
`python/sglang/srt/mem_cache/unified_radix_cache.py`, new `test/registered/unit/mem_cache/test_resident_prefix_probe.py`.
- `UnifiedTreeCore.resident_prefix_len(self, key: RadixKey) -> tuple[int, int]` returning `(l1_len, l12_len)`, next to
  `match_full_device_prefix` (905-929), modelled on `match_prefix`'s helper (`_match_prefix_helper`, ~872-894) but
  read-only: apply `maybe_to_bigram_view` and `page_aligned` as `match_prefix` does; loop `while pos < len(key)`, look up
  the child with `key.child_key_at(pos, self.page_size)` and match with `child.key.match_at(key, pos,
  page_size=self.page_size)` (offsets, no slice copies — both default to `page_size=1`, which at page 64 finds no child
  and silently returns `(0, 0)`); stop at a missing child, a zero-length match, or `child.evicted and not
  child.backuped`; count a partially matched last node without splitting; accumulate `l12_len` over all walked tokens
  and `l1_len` while nodes are not evicted. No `last_access_time` stamp, no cache actions.
- Interface: a concrete (non-abstract) default that raises `NotImplementedError`, next to `match_full_device_prefix`
  (394-397). It is concrete for the same reason as `rotation_base_of` (196, whose default returns None): the Rust
  adapter (`rust_tree_core/adapter.py`) stays constructible.
- URC wrappers next to `has_ongoing_prefetch` (2169):
  - `resident_prefix_len(self, key: RadixKey) -> tuple[int, int]`: delegates; `(0, 0)` when `self.disable`.
  - `storage_read_backlog_pages(self) -> int`: `sum(len(info.operation.hash_value) for info in
    self.ongoing_prefetch.values() if info.host_indices is not None)` — ops allocated and queued on / running in the
    read thread (`host_indices` is None at registration, 2119-2126, and set at allocation, 2947). Scheduler-owned,
    rank-deterministic; never read `prefetch_buffer.qsize()`. Slight overcount: finished reads whose ack is not drained.
- CPU test (`register_cpu_ci`; CPU URC from `make_params(enable_session=False)` in
  `test/registered/unit/mem_cache/test_session_unified_radix_cache.py:27-60`). That fixture has HiCache off and
  `page_size=1`: call `cache.tree_core.set_hicache_enabled()`, fabricate host-only nodes (set
  `node.component_data[ComponentType.FULL].host_value`, clear `.value`) and dead nodes, and use `page_size > 1` (e.g. 2,
  as the CPU tests in `test_unified_cache_linker.py` do) so the page-size handling is exercised. Derived property to
  assert: on trees with GPU-resident, host-only and dead nodes, `l12_len == len(device_indices) + host_hit_length` from
  `match_prefix` for the same key, and the probe leaves the node count and every `last_access_time` unchanged
  (match_prefix would split and stamp).
Acceptance: the test passes; no caller yet, so no behaviour change.

### Step 4 — estimators module (pure) (DONE, 2026-10-09)
Files: `python/sglang/srt/mem_cache/next_turn_estimators.py` + `test/registered/unit/mem_cache/test_next_turn_estimators.py`.
State in `msgspec.Struct`; every time is a `time.perf_counter()` value the caller passes in (the module reads no clock).
- `ArrivalOffset`: `observe(offset_s) -> bool` folds `arrival − t_commit − gap` into an EWMA (α = 1/8, seed 0.10 s);
  samples outside [0, 5] s are dropped and counted (`discarded`), never clipped. `predict_arrival(t_commit=, gap_s=)` = P.
- `AdmissionRate`: `record_admission(now=, wait_s=, storage_hit=)` appends the detecting tick's time (must not decrease)
  and prunes stamps older than 300 s; `wait_ewma` (None until the first sample) skips storage hits, which still count
  as admissions. `rate(now) -> Optional[Rate]` = admissions in 60 s / 60 s, else in 300 s / 300 s, None below 5;
  `Rate.window_s` names the window used (log it as the estimator).
- Pure functions taking the tick's `Rate` (call `rate(now)` once per tick, not once per plan):
  `wait_at_arrival(queue_len=, rate=, wait_ewma=) -> (w_ql, w_ewma)`, both 0 for an empty queue, else
  `w_ql = (queue_len + 0.5) / λ̂`; `residual_wait(position=, rate=)` = `(k + 0.5) / λ̂`; None without a rate.
- `L3LoadModel` (frozen, kw_only): `read_fixed_s` 0.044 (a), `read_s_per_page` 0.0025 (b), `backlog_s_per_page` 0.0025
  (b_backlog; consider 0.0031, section 2.4), `alloc_lag_s` 0.5, `drain_lag_s` 0.36; `seconds(own_pages=, backlog_pages=)`.
- `trigger_time(admission_at=, load_s=, margin_s=0)` = A − L̂ − margin.
Tests (4, per unit-test-admission): outliers dropped, not clipped; the 60 → 300 s → None fallback and stamp pruning; no
estimate (not a division by zero) without admissions, but an empty queue needs none; storage hits count as admissions,
not as waits. Each test caught a deliberately injected bug (8 mutants: clipping, no fallback, no pruning, storage hits
left out of the rate or fed to the wait EWMA, EWMA starting at 0, empty queue needing a rate, no None guard). The load
model and `trigger_time` are formulas with no derived property, so they have no test.
Replay of campaign-11 three_tier_wc `ReqTimeStats` (scratch, not committed; 4,676 requests, 1,158 storage hits): true
wait mean / p50 / p90 = 65.5 / 55.2 / 141.4 s; W_ql MAE 14.2 s (bias +2.4 s; the offline estimate was ≈ 12 s), W_ewma
MAE 9.8 s (bias −2.2 s). At arrival the 60 s window always held ≥ 5 admissions (219 empty-queue arrivals, 18 without a
rate during warm-up), so the 300 s fallback is for sparse load.

### Step 5 — the prefetcher, observe mode, wired in, plus the arms to boot it
Files: new `python/sglang/srt/mem_cache/next_turn_prefetch.py` + test; small edits in `mem_cache/base_prefix_cache.py`,
`mem_cache/unified_radix_cache.py`, `managers/scheduler.py`; fixture line in
`test/registered/unit/managers/test_scheduler_hicache_events.py`; the two server arms in `agent_cache/scripts/start_server.sh`
(moved here from step 9, because steps 5-7 need a server booted with the flag and `start_server.sh` has no flag passthrough).
- Types (`msgspec.Struct`): `NextTurnHint(session, next_rid, next_gap_s)`; `TurnPlan(next_rid, session, key: RadixKey,
  t_commit, gap_s, P, state: "pending"|"arrived", arrival_t, last_probe_t, fired, staged, refetches)`.
  `read_next_turn_hint(kv_hints) -> Optional[NextTurnHint]`: pick the action with `action_type == "sglang.next_turn"`
  and `action_version == "1"`, `msgspec.convert` its payload; ignore other actions; one warning on a bad payload.
- `NextTurnPrefetcher.create(*, mode, tree_cache, page_size, enable_hicache_storage, tp_size, pp_size,
  enable_dp_attention, disaggregation_mode)`: `ValueError` unless storage is on, `tree_cache` is a `UnifiedRadixCache`
  in cache mode, the tree core is the Python `UnifiedTreeCore` (the probe raises on Rust), TP = PP = 1, no DP attention,
  no disaggregation, not eagle (pattern: `mem_cache/registry.py:262-268`). One startup banner line with the mode.
- `on_turn_committed(*, req, key, rotation_tail_declined)`: register only if `req.finished()`, `finished_reason` is not
  `FINISH_ABORT` (import lazily, as URC does at 1101; required: an aborted *running* request arrives here with
  `result is not None`, section 3.7), a hint is present, `len(key) > 0`, not declined. Keep the RadixKey (≤ 8 B/token),
  never the `Req`. Log `ntp_plan`.
- `tick(*, waiting_queue, now) -> list[Req]`:
  1. Diff the queue against the previous tick's `{rid: Req}` map (references held for one tick only). New rid with a
     plan → arrival: update δ̂, state = arrived, log `ntp_arrive` (`arr_err_ms` = arrival − P, `delta_ms` = raw
     offset). A departed Req with `time_stats.forward_entry_time > 0` → admission: λ̂ += 1, and if
     `req.storage_hit_length == 0` feed `forward_entry_time − wait_queue_entry_time` to the wait EWMA; for a plan log
     `ntp_admit` with both wait predictions and errors, drop the plan. Departed with `forward_entry_time == 0` → abort:
     drop. Dedupe by rid (retraction / preemption re-enter the queue).
  2. Pending plans: once `now ≥ P + Ŵ − L̂(whole chain)` (cheap window), probe `resident_prefix_len(plan.key)` at most
     every 0.5 s and compute T. Arrived plans: `A = now + residual_wait(position=k, rate=rate)` with
     `rate = admissions.rate(now)` computed once per tick (or fallback).
  3. When `now ≥ T`, log `ntp_due` once per plan (case, l3_pages, backlog_pages, load_ms, wait_ms, lead_ms, est).
  4. Expire **pending** plans only, at `P + max(60 s, gap)`; log `ntp_expire`. Arrived plans leave by admission/abort.
  5. Observe mode returns `[]` and calls no state-changing cache API. Keep the tick O(due plans) (heap by next check time).
- `base_prefix_cache.py`: class attribute `next_turn_prefetch: Optional["NextTurnPrefetcher"] = None` next to
  `storage_prefetch_retries` (339), with a `TYPE_CHECKING` import.
- `unified_radix_cache.py`, end of `cache_finished_req` after the session-refs block (1100-1106):
  ```python
  if self.next_turn_prefetch is not None and result is not None:
      self.next_turn_prefetch.on_turn_committed(
          req=req, key=radix_key, rotation_tail_declined=result.rotation_tail_declined
      )
  ```
  (`result is not None` implies `is_insert`, so `radix_key` is bound and retraction / queued aborts are skipped; an
  aborted running request still reaches the hook and is filtered by `FINISH_ABORT` inside.)
- `scheduler.py` (large-class-style):
  1. `maybe_init_next_turn_prefetch(self)`: `self.next_turn_prefetch = None`; return if
     `get_memory().hicache_storage_next_turn_prefetch == "off"`; else `NextTurnPrefetcher.create(mode=...,
     tree_cache=self.tree_cache, page_size=self.page_size, enable_hicache_storage=self.enable_hicache_storage,
     tp_size=get_parallel().tp_size, pp_size=get_parallel().pp_size, enable_dp_attention=self.enable_dp_attention,
     disaggregation_mode=DisaggregationMode(get_disagg().disaggregation_mode))` and set
     `self.tree_cache.next_turn_prefetch = self.next_turn_prefetch`. The Scheduler has no `self.tp_size`/`self.pp_size`,
     and `self.disaggregation_mode` is only set in `init_disaggregation()` (called at 661, set at 1451), so read them via
     `get_parallel()` / `get_disagg()` (`runtime_context.py:1247, 1304`). Call the helper in `__init__` right after
     `self.init_schedule_policy()` (643).
  2. `_process_next_turn_prefetch(self)` next to `_process_storage_prefetch_retries` (3161): return if None; else
     `for req in self.next_turn_prefetch.tick(waiting_queue=self.waiting_queue, now=time.perf_counter()):
     self._on_next_turn_refetch(req)` (step 7 fills that in; observe returns nothing).
  3. Call it right after `self._process_storage_prefetch_retries()` inside `if self.enable_hicache_storage:` (3616-3617).
- `test_scheduler_hicache_events.py` `setUp` (23-38): add `s.next_turn_prefetch = None` next to
  `s._process_storage_prefetch_retries = self.calls.retry` (31), otherwise every test there fails.
- New CPU test (fake tree cache, fake Reqs with `rid`, `time_stats.forward_entry_time`, `kv_hints`): registration
  filters (abort / no hint / empty key / rotation decline skip; FINISH_LENGTH registers); arrival / admission / abort
  classification; a retracted rid is not double-counted; observe returns `[]` and makes no state-changing calls.
- `start_server.sh` arms (the `case` at ~104-113; header docs 7-12; the arm list in the error message; `run.txt`):
  `three_tier_wc_ntpobs` (= `three_tier_wc` + `--hicache-storage-next-turn-prefetch observe`) and
  `three_tier_wc_oracle` (`on`), both `POLICY=wait_complete` and pinning `--hicache-storage-prefetch-retry-poll-interval 8
  --hicache-storage-prefetch-retry-max-attempts 8`. Boot check: print the mode from `server.log`; change the
  re-query-cap check (line ~149, an exact match on `three_tier_wc_norq` today) to
  `$(case "$ARM" in *_norq) echo 0;; *) echo 8;; esac)`. Names keep the `three_tier` prefix (`l3_prep`; `run_compare.sh`
  wipes L3 for `three_tier*`).
Acceptance: unit tests pass; live smoke:
```bash
docker exec -it sglang_hicache bash -c 'MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 CTX=131072 KV_DTYPE=auto TEMPLATE= L3_CLEANER_PCT=70,60 SGLANG_TIMEOUT_KEEP_ALIVE=3600 bash /sgl-workspace/sglang/agent_cache/scripts/start_server.sh three_tier_wc_ntpobs NAT160'
docker exec sglang_hicache bash -c 'cd /sgl-workspace/sglang/agent_cache/scripts && MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 TRACE=/sgl-workspace/sglang/agent_cache/traces/lmcache_agentic_trace_262k.json NCONV=2 TURNS=4 C=2 GAP=1 OFFSET=32 CHECK_IDS=0 ORACLE_HINTS=1 TAG=smoke_ntp bash start_client.sh'
```
Pass: boot prints mode `observe` and cap 8; one `ntp_plan` per hinted turn; `ntp_arrive` + `ntp_admit` for every hinted
rid; `|arr_err_ms|` p50 ≲ 50 (raw `delta_ms` p50 ≈ 90-125); the per-rid `prefetch_start` pattern identical to a
mode-off run.
**Implemented (2026-10-09).** `mem_cache/next_turn_prefetch.py`: `NextTurnHint`, `read_next_turn_hint`, `TurnPlan`,
`NextTurnPrefetcher` (`create`, `on_turn_committed`, `tick`); `BasePrefixCache.next_turn_prefetch`; the hook at the end of
`cache_finished_req`; `Scheduler.maybe_init_next_turn_prefetch` (after `init_schedule_policy`) and
`_process_next_turn_prefetch` (after the paced retries). Differences from the text above: `on_turn_committed` takes the
clock (`now=time.perf_counter()` from the hook); pending plans sit in a heap, and once a probe puts the trigger T before
the next 0.5 s check, the next check is scheduled at T itself (a fixed 0.5 s cadence fired up to 0.5 s late); due checks
are popped before any is re-queued, so "retry next tick" cannot loop within a tick; an abort while queued logs
`ntp_expire reason=abort`. Tests: `test_next_turn_prefetch.py` (factory refusals and the real `cache_finished_req` hook on
a CPU cache, registration filters, queue tracking, trigger timing, probe economy) and `test_scheduler_hicache_events.py`
(fixture line, tick after the paced retries). The two arms are in `start_server.sh` with step 9's.

### Step 6 — case 1: staging before arrival
File: `next_turn_prefetch.py` + tests. When a pending plan reaches T (mode `on`):
1. Probe; if `len(key) − l12 < tree_cache.prefetch_threshold` (256), nothing to fetch yet: re-probe every 0.5 s.
2. `h = CacheRequestHandle(next_rid, 0)`; skip if `tree_cache.has_ongoing_prefetch(h)`; if
   `tree_cache.cache_controller.prefetch_rate_limited()`, retry on the next tick (do not call into a decline: it arms
   a `poll_miss` for a rid that is not queued).
3. Re-match once: `m = tree_cache.match_prefix(MatchPrefixParams(key=plan.key))` (req None); `anchor =
   m.last_host_node`; `matched = len(m.device_indices) + m.host_hit_length`. (This stamps LRU on the session path:
   accepted at trigger time, logged.)
4. Gate: `tree_cache.is_backuped(anchor) or tree_cache.is_root(anchor)` (copy of `scheduler.py:3132-3139`).
5. `tokens = plan.key.token_ids` (page-aligned `array('q')`; `plan.key.extra_key` / `cache_salt` carry the namespace);
   `tree_cache.prefetch_from_storage(h, anchor, tokens[matched:], tree_cache.get_last_hash_value(anchor), None,
   matched_prefix_tokens=tokens[:matched], extra_key=plan.key.extra_key, cache_salt=plan.key.cache_salt,
   storage_hit_end=None)` (prefix_keys `None` because `hicache_storage_pass_prefix_keys` is off; pass them if it is on).
6. `inflight = has_ongoing_prefetch(h)`; log `ntp_stage`; mark staged.
Rules: never stage after the rid was seen (arrivals are ingested before the tick in the same iteration). Whenever a
staged op is no longer in `ongoing_prefetch` and the rid has not been seen, call
`tree_cache.storage_prefetch_retries.cancel(next_rid)` (covers miss, below-threshold and host-capacity endings, each of
which arms `poll_miss`; a no-op after a successful read). On expiry of a staged, never-arrived plan: if
`has_ongoing_prefetch(h)`, wait a tick; else `tree_cache.finish(h, CacheRequestOutcome.ABORT)` (pops the loaded-span
records, cancels the retry entry) and log `ntp_expire staged=1`.
Tests: fires only at `now ≥ T`, only for unseen rids, only above threshold; exact handle and arguments; anchor gate;
rate-limit defer; retry entry cancelled after a staged miss; expiry calls `finish(ABORT)` only when idle; never stages
after arrival.
Acceptance (small-pool smoke to force SSD reads):
```bash
docker exec -it sglang_hicache bash -c 'find /mnt/ssd/hicache_l3 -mindepth 1 -delete; MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 CTX=131072 KV_DTYPE=auto TEMPLATE= L3_CLEANER_PCT=70,60 SGLANG_TIMEOUT_KEEP_ALIVE=3600 L1_TOKENS=65536 HOST_GB=12 bash /sgl-workspace/sglang/agent_cache/scripts/start_server.sh three_tier_wc_oracle NAT160'
docker exec sglang_hicache bash -c 'cd /sgl-workspace/sglang/agent_cache/scripts && MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 TRACE=/sgl-workspace/sglang/agent_cache/traces/lmcache_agentic_trace_262k.json NCONV=8 C=8 TURNS=6 GAP=10 GAP_CAP=1200 OFFSET=32 CHECK_IDS=0 ORACLE_HINTS=1 TAG=smoke_stage bash start_client.sh'
```
Pass: `ntp_stage` lines; `s2h_io` for rid X stamped before X's `ntp_arrive`; those turns report
`cached_details.storage > 0` with `queue_duration` ≈ GPU-hit turns (~0.03 s) instead of ~0.6 s; "Replacing unresolved
storage-hit accounting" not above an observe-mode run of the same config (`three_tier_wc_ntpobs`).
**Implemented (2026-10-09)** as specified: `_check_pending` → `_stage` (prefix keys passed when
`hicache_storage_pass_prefix_keys` is on) and `_expire`. `ntp_stage` carries `tokens matched gate inflight`. Tests: the
`TestCase1Staging` class of `test_next_turn_prefetch.py`, one per rule above.

### Step 7 — case 2: refetch while queued
Files: `next_turn_prefetch.py`, the step-5 scheduler delegate + tests. For each arrived plan per tick (mode `on`):
1. `A = now + residual_wait(position=k, rate=rate)` (or fallback); probe `l12`; `evicted = baseline − l12` where `baseline =
   max(len(plan.key), req.storage_prefetch_last_match_len or 0)`; `T = A − L̂(evicted pages, backlog)`.
2. When `now ≥ T` and `evicted ≥ prefetch_threshold` and not `has_ongoing_prefetch(req.cache_request_handle)` and not
   `prefetch_rate_limited()` (else retry next tick): return the `Req`.
3. Scheduler delegate `_on_next_turn_refetch(req)`: `self._prefetch_kvcache(req)`, then
   `self.next_turn_prefetch.on_refetch_result(req=req, issued=self.tree_cache.has_ongoing_prefetch(req.cache_request_handle))`.
   Log `ntp_refetch issued=0|1`; keep the plan armed when `issued == 0`.
4. Re-eviction: `_prefetch_kvcache` lowers `last_match_len` to the shrunken length (3130), which blinds #39283 for that
   request. Keep the plan armed after a refetch: on later ticks re-probe, and if ≥ 256 tokens were evicted again before
   admission, log `ntp_reevict` and refetch again (cap 2 refetches per request; `ntp_reevict ... capped=1` at the cap).
Why `_prefetch_kvcache` directly: no attempt counted, works in the `_norq` arm; while the read runs, admission skips the
request (3959-3965), so no double fetch with #39283. Late trigger under `wait_complete` = the request waits the
remaining load time (skipped, others admitted).
Tests: a due queued plan is returned once; not returned below threshold, with a read in flight, or rate-limited
(deferred, then returned); `issued == 0` keeps it armed; re-eviction triggers a second refetch; pruned when the rid
leaves the queue.
Acceptance (dense smoke): boot as in step 6 (`three_tier_wc_oracle`, `L1_TOKENS=65536 HOST_GB=12`, cold L3); client
`MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 TRACE=/sgl-workspace/sglang/agent_cache/traces/lmcache_agentic_trace_262k.json
NCONV=32 C=32 TURNS=12 GAP=1 OFFSET=32 CHECK_IDS=0 ORACLE_HINTS=1 MAX_SECONDS=1200 TAG=smoke_refetch bash start_client.sh`.
Pass: each `ntp_refetch issued=1` whose `s2h_query` reports `storage_hit ≥ 256` is followed by `s2h_io` for the same rid
before its admission (log misses separately); fewer "device prefix shrank ... reissuing" lines and fewer cap-blocked
losses (rids with "reissue cap reached" admitted with a cached prefix below their arrival match) than observe mode on
the same config. The "reissue cap reached" count itself is expected to stay about the same: the paced polls spend the
budget within seconds of arrival, long before case 2 fires.
**Implemented (2026-10-09).** `_check_queued` (returns True to refetch) → `_wants_refetch`, `on_refetch_result`
(`RefetchDecision` carries the log fields), `MAX_REFETCHES = 2`; `Scheduler._on_next_turn_refetch`. Attempts are at least
a probe interval (0.5 s) apart, `refetches` counts only issued reads, and the capped `ntp_reevict` is logged once. Tests:
`TestCase2Refetch` (+ the call-order test in `test_scheduler_hicache_events.py`). A mutation check of steps 5-7 (24
single-point bugs: filters, dedupe, trigger timing, probe economy, gate, handle, limiter, cleanup, cap, observe purity)
fails at least one test each.
**Fixed after smoke C (uncommitted).** `_wants_refetch` judged a re-eviction by the last 0.5 s-cadence probe, which could
predate the landing of the read it judged: about half of C's re-evictions were landed reads, re-fetched as tails or
declined as too short, and each issued one spent the cap. It now probes again right before deciding
(regression test `test_a_landed_read_is_not_taken_for_a_re_eviction`; mutation check 26/26). Reruns C2/C3 (fixed) and
C4 (pre-fix, frozen copy): no tail-only lookups or too-short declines left, TTFT unchanged (README of
`compare_20261009_082335`). Next: hold a host lock on re-fetched and staged pages until admission (section 7).

### Step 8 — analysis scripts
- `delay_components.py` (`load_arm`, ~28-58): drop `prefetch_start`/`s2h_query`/`s2h_io` events stamped before the
  turn's send (`t0 + t_send`); time the L3 read only from events after the send; add a `staged_l3` column (`ntp_stage`
  → its `s2h_io`) so reads done before arrival are not charged as queue time.
- `timeline.py`: `elif` branches for `ntp_stage`, `ntp_refetch`, `ntp_admit` (102-133), an oracle lane; keep the first
  `prefetch_start` after the send in the rid-keyed s2h dict (123-127); colours for new arms (34-36).
- `crossscale.py`: new arms in `ARM_COLOR`/`ARM_MARK` (14-15), else they are silently skipped.
- `read_backlog.py`: the same pre-send filter for the per-rid view; new arms in `ARM_COLOR`.
- New `oracle_eval.py`: per run, from `ntp_*` lines + `client.jsonl`: arrival error; both wait-estimator errors;
  admission error; trigger lead (admission − T) vs load time; late fraction (read still running at admission); wasted
  staged reads (expired, or staged pages evicted before admission); tokens refetched vs recovered; re-evictions.
Acceptance: campaign-11 outputs unchanged (no `ntp` lines, no pre-send prefetches); `oracle_eval.py` matches a hand
grep of a few rids on the smoke logs.
**Implemented (2026-10-09).** One deviation: `timeline.py` keeps the last `prefetch_start` per rid as before (keeping the
first after the send would have changed campaign-11 outputs) and only moves a rid's events stamped before its send, or
every event of a planned rid that was never sent, into `s2h_staged` (oracle lane + `s2h_staged` rows). A turn's own
first lookup comes 16-19 ms or more after its send in campaigns 10-13 (35k turns), so the pre-send filter needs no slack.
`oracle_eval.py` writes `oracle_eval.csv` and `oracle_rids_<arm>.csv`; late = admission within 50 ms of the end of the last
read the oracle started; admission time = the `ntp_arrive` stamp + the measured wait. Verified: on campaign 11 every
printed line and every CSV of the four scripts is unchanged (`turn_components_*` gains an all-zero `staged_l3` column); a
synthetic oracle run (staged read before the send, refetch + re-eviction, an expired staged plan) checks each oracle path
by hand.

### Step 9 — `_norq` variants and the campaign driver
- `start_server.sh`: optional `three_tier_wc_ntpobs_norq` / `three_tier_wc_oracle_norq` arms pinning
  `--hicache-storage-prefetch-retry-max-attempts 0` (as `three_tier_wc_norq`); the step-5 cap check already expects 0
  for `*_norq`.
- `run_compare.sh`: export `ORACLE_HINTS=${ORACLE_HINTS:-0}`, record it in `config.txt`, pass it to `start_client.sh`.
  For oracle campaigns set `ORACLE_HINTS=1` for every arm (identical client bodies; only the server mode differs).
Acceptance (container, campaign knobs; the script defaults are a different model and workload):
`<section-5 common knobs> TRACE=/sgl-workspace/sglang/agent_cache/traces/lmcache_agentic_trace_262k.json NCONV=16 C=16
TURNS=8 GAP=10 GAP_CAP=1200 OFFSET=32 ARMS="three_tier_wc three_tier_wc_ntpobs" CLIENT_TIMEOUT=600 bash run_compare.sh`
writes a manifest; `config.txt` records `ORACLE_HINTS=1`; the observe arm logs `ntp_plan`, the baseline does not.
**Implemented (2026-10-09).** `start_server.sh`: the four arms `three_tier_wc_ntpobs | _oracle | _ntpobs_norq |
_oracle_norq` (wait_complete; retry flags pinned 8/8, or max attempts 0 for `_norq`); a guard refuses them on an engine
without `next_turn_prefetch.py` (the frozen `2abe005bb2` copy accepts the flag and silently ignores it); boot checks print
the re-query cap (0 for `*_norq`) and the next-turn mode with the prefetcher's startup banner; `run.txt` records
`next_turn_prefetch`. `run_compare.sh` exports `ORACLE_HINTS` and records it in `config.txt`. Smokes for steps 5-9:
`agent_cache/frozen/oracle_scripts_staging/oracle_smokes.sh` (A observe vs off, B case 1, C case 2).

### GPU smokes of steps 2 and 5-9 (2026-10-09, H200; engine `294c82b7fc`, scripts as applied)
Logs under `results/oracle_smokes_1009_0701/`; every arm ORACLE_HINTS=1, cold L3, observe vs on unless noted; per-rid
tables in each compare dir (`oracle_eval.csv`, `oracle_rids_<arm>.csv`). The frozen `2abe005bb2` engine is refused by the
new guard (checked).

| Run (compare dir) | Load | Result |
|---|---|---|
| A `compare_20261009_070112`, off vs observe | 16 x 8, x10, natural pools | Steps 2/5/9: boot checks (off/banner 0, observe/banner 1, cap 8); curl valid 200 / malformed 400; 109 hinted turns -> 109 `ntp_plan` / `ntp_arrive` / `ntp_admit`; arrival \|err\| p50 38 ms, p90 121; TTFT p50/p90 0.145/0.471 vs 0.142/0.466 s, tiers identical; `prefetch_start` per turn identical on all 109 hinted turns (12 first turns differ: a cold-start race on the shared prompt's backup, before any plan exists); `oracle_eval` rows match a hand grep |
| B `compare_20261009_074959` | the step-6 config, 8 x 6, x10, L1 65,536 / host 12 GB | No eviction at all: the 8 sessions' private KV (~48K tokens) fits in L1. Not a case-1 test |
| C `compare_20261009_082335` | 32 x 12, GAP 1, small pools | Step 7: TTFT mean 16.76 -> 13.13 s; SSD-restored turns 64 -> 157; recomputes 103 -> 45; "prefix shrank" 80 -> 45; cap hits 139 -> 104; 238 issued refetches: 202 SSD hits all read before admission, 36 misses; 144 re-evictions (40 capped), about half of them a stale probe (fixed after the smoke). Reruns C2/C3 (fixed) and C4 (pre-fix): no stale re-evictions left, TTFT unchanged (+0.07 s), run-to-run spread ~1 s; README in the compare dir |
| B2 `compare_20261009_084120` | 48 x 10, x10, small pools | Congested, so case 2 only: TTFT 21.99 -> 15.36 s; SSD turns 33 -> 228; recomputes 116 -> 28; 341 refetches: 256 hit + done before admission, 1 not, 84 misses |
| B3 `compare_20261009_092437` | 96 x 4, x70, small pools | Step 6: 120 staged reads, 116 landed before arrival, all 116 credited as storage with queue p50 0.016 s vs 5.27 s for the arm's other SSD-restored turns; TTFT 3.66 -> 3.13 s; SSD turns 198 -> 246; recomputes 1 -> 0 |

"Replacing unresolved storage-hit accounting" rises with the oracle (C 0 -> 107, B3 0 -> 39) but only on requests read
from SSD twice (a refetch after the request's own read); staged reads cause none. In B3 all 39, in C 84 of 107 re-credit
the identical span; the rest move the credit to the later read (C: net 5,952 of 1.2M tokens). Reporting only.
For E1: reads land ~0.8-1.0 s (p50) before admission and `L̂` overestimates the read by 0.7-0.9 s (the dense lag seeds),
the queue-length wait estimate is off by 4-7 s under load (B3: over, C/B2: under), re-evictions are frequent at small
pools (the cascade risk of section 7): after the stale-probe fix 52-61 real ones per C run, and a re-fetched context
can be evicted by another turn's read milliseconds before admission, which neither #39283 nor the 0.5 s re-probe sees.
Single runs differ by up to ~1 s mean TTFT on identical code (C vs C4): compare arms over repeated runs.

---

## 5. Experiments

Common knobs (as campaign 11): `MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 CTX=131072 KV_DTYPE=auto TEMPLATE=
LEVEL=NAT160 L3_CLEANER_PCT=70,60 SGLANG_TIMEOUT_KEEP_ALIVE=3600 ORACLE_HINTS=1`; launch from the host with
`docker exec -d sglang_hicache bash -c 'cd /sgl-workspace/sglang/agent_cache/scripts && <knobs> nohup bash run_compare.sh > ../results/compare_driver_<name>.log 2>&1'`.
`run_compare.sh` defaults (`GAP_CAP=600`, `OFFSET=29`, `NCONV=48`, ...) differ from every campaign: always set them.

- **E0 smoke** (after step 5): observe mode, 2 conversations × 4 turns. Pass: step-5 acceptance.
- **E1 estimator validation** (observe, after steps 5 and 8), two regimes, each vs a mode-off arm (`three_tier_wc`):
  - sparse = campaign-10 config: `TRACE=/sgl-workspace/sglang/agent_cache/traces/lmcache_agentic_trace_262k_ge45_tl_s1.json
    NCONV=283 C=80 ARRIVAL=0.1 TURNS=50 GAP=1 GAP_CAP=0 OFFSET=0 CLIENT_TIMEOUT=7200` (campaign 10's SSD hits come from
    pauses ≥ 7.7 min, so `GAP_CAP` must be 0; with 3600 s only 60 of its 185 storage-hit turns would be sent);
  - dense = campaign-11 config: `TRACE=/sgl-workspace/sglang/agent_cache/traces/lmcache_agentic_trace_262k.json
    NCONV=128 C=128 TURNS=40 GAP=10 GAP_CAP=1200 OFFSET=32 CLIENT_TIMEOUT=3600`.
  Pass: `|arr_err|` p50 ≤ 0.2 s, p90 ≤ 1 s; wait MAE near the offline numbers at x10; sparse mostly case 1 when SSD
  pages exist, dense mostly case 2; L̂ close to the measured #39283 re-issue → insert time ("HiCache prefetch success"),
  p50 ≈ 2.0 s, p90 ≈ 4.6 s in campaign 11 (re-issue → admission, p50 3.8 s, adds ~2 s from insert to the next admission
  pass, which L̂ does not model); TTFT and tier shares match the mode-off arm. Outcome: choose Ŵ, `b_backlog` and the margin.
- **E2 case 1** (sparse, the E1 sparse knobs, 7200 s): `three_tier_wc` (hints on, mode off) vs `three_tier_wc_oracle`.
  Metrics: queue duration and TTFT of storage-hit turns (baseline p50 0.62-0.68 s vs 0.03 s), storage-tier tokens,
  wasted staged reads, SSD read pages, host evictions. Also run at x70 (campaign-7 config:
  `TRACE=.../lmcache_agentic_trace_262k.json NCONV=128 C=128 TURNS=40 GAP=70 GAP_CAP=1200 OFFSET=32`), where the read
  backlog dominates.
- **E3 case 2** (dense, campaign-11 config, `CLIENT_TIMEOUT=9000`): `three_tier_wc` vs `three_tier_wc_oracle`
  (optionally `three_tier_wc_oracle_norq` vs `three_tier_wc_norq`). Metrics: TTFT (c11 wc mean 68 s, norq 116 s);
  cap-blocked losses (c11 wc: 1,214 requests admitted ≥ 256 tokens below their arrival match after the cap, 24.8 M
  tokens); "device prefix shrank" counts; refetch lead vs L̂; late fraction; re-evictions (eviction cascade); total SSD reads.
- **Campaign 15** (launched 2026-10-09 16:41Z, `results/campaign15/DECISIONS.md`): oracle vs baseline on campaign 14's
  workload (x70, 768 fresh users, backlog cap, 5 h per arm), `wait_complete` and `timeout`. Oracle arms only; the
  baselines are campaign 14's runs (user, 16:47Z). Done 2026-10-10 02:48Z; results and both policies in
  `results/campaign15/README.md`, the analysis below.
- **E4 later:** `timeout`-policy arms; misprediction (gap × 0.75 / 1.25, noise); online-learned load model; a
  CachedAttention-faithful baseline.

### Campaign 15, `wait_complete`: why the gain is small (2026-10-09)

`three_tier_wc_oracle` (5 h, finished 21:43Z) against campaign 14's `three_tier_wc`, single runs, window 2,700-16,800 s
after the client started (campaign 14's analysis window). Outputs in `frozen/oracle_scripts_staging/c15_pairs/wc/`
(git-ignored): `steady.csv` and `steady_paired.csv` from `scripts/steady_state.py --manifest <that dir>/manifest.txt
--results results --window-start 2700 --window-end 16800 --iostat <that dir>/iostat_c14_c15.log --out <that dir>`, and
`diagnosis.txt` from `frozen/oracle_scripts_staging/c15_diagnosis.py` (usage in its docstring).

| | c14 `three_tier_wc` | c15 `three_tier_wc_oracle` |
|---|---|---|
| TTFT mean / p50 / p90 / p99 (s) | 22.2 / 10.4 / 40.4 / 173.9 | 20.8 / 16.8 / 36.1 / 154.9 |
| queue wait, mean (s) | 21.6 | 20.2 |
| turns/s; users finished per hour | 1.217; 169 | 1.218; 169 |
| user cap, mean; live users | 171; 133 | 178; 130 |
| returns: device / host / SSD / recompute | 21 / 16 / 50 / 13 % | 21 / 10 / 58 / 11 % |
| uncached tokens per returning turn | 1,846 | 1,389 |
| SSD read / written | 15.6 / 3.4 TB | 18.0 / 2.7 TB |
| SSD ≥ 90 % busy (10-s iostat samples) | 49 % | 68 % |
| host pool held by in-flight SSD reads, p50 | 382K tokens (23 %) | 642K tokens (39 %) |

Same (conversation, turn): −1.41 s [−2.09, −0.73], p50 +0.01 s (16,594 pairs). The CI covers within-run noise only: two
runs of identical code differed by 0.98 s at 32 sessions (smoke C, section 4).

The x70 gaps gave the oracle time; the SSD and the host pool had no room:

1. **Reads waited in the queue; they were not slow.** A staged context is ~250 pages (~0.6 s of I/O), but it joined the
   single FIFO read queue behind a median 9,820 pages (~60 GB; p90 12,933). Case 1 fired a median 33 s before admission
   (`trig_lead`), and the read still ended a median 29 ms before it: 7,183 of 9,397 case-1 reads (76 %) were late
   (admission ≤ 50 ms after the read ended). Trigger to read end p50 32.8 s for ~10,070 pages ≈ 3.2 ms per queued page;
   L̂'s seed is 2.5 ms, so L̂ under-predicted the read by 6.5 s (p50, `oracle_eval` `load_err`). Case 2 (1,587 due): 52 s.
2. **When the read queue never empties, timing only reorders it.** The SSD was ≥ 90 % busy in 49 % of the samples
   without the oracle and 68 % with it (ceiling ~2.1 GB/s). Issuing one read earlier delays the reads behind it; issuing
   all of them earlier lengthens the queue. An earlier read gains only by filling idle SSD time, and the oracle's 9
   extra points of busy time match its 16 % extra read volume, not old reads moved into idle time.
3. **Early reads hold host memory.** A prefetch allocates its host pages when its lookup hits, before it joins the FIFO,
   and evicts cached host contexts if the pool is full (`_try_alloc_storage_hit`, `unified_radix_cache.py:2856-2970`),
   so a read that queues 30 s holds its pages 30 s. In-flight reads held a median 642K tokens (39 % of the pool; 23 %
   without the oracle) and at p90 811K, the rate limiter's ceiling (0.5 × pool). Under write-through the host also keeps
   copies of the 668K device-resident tokens, so the room for cached host-only contexts fell from ~578K to ~318K tokens.
   Host hits fell from 16 % to 10 % of returns, those turns read the SSD instead (+16 % read volume), and the median
   TTFT rose (10.4 → 16.8 s).
4. **The backlog cap fixes the mean TTFT.** The client steers the user cap to a backlog of requests awaiting their first
   token, so mean queue wait ≈ backlog / throughput (Little's law): 26.2 / 1.217 = 21.5 s (c14) and 24.6 / 1.218 = 20.2 s
   (c15), the measured waits. A better cache shows up as throughput, and the mean turns/s did not move. Split by phase
   (`steady_bins.csv`, 5-min bins), the oracle arm completed 17 % more turns/s while congested (TTFT median above 10 s:
   1.34 → 1.57; 13-31 % for thresholds of 3-30 s) and fewer while light (0.74 against 1.06), when fewer users were live
   (73 against 94). Light phases are demand-limited, so the mean hides a capacity gain under congestion, despite items
   1-3.

v1 is per-request timing: each turn triggers at `A − L̂` on its own, reads join one FIFO, and host eviction is LRU. Timing
is the wrong lever when SSD bandwidth and host space are both full. The hints also say which context is needed when,
which v1 uses for neither eviction nor read order.

### Campaign 15, `timeout` (2026-10-10)

`three_tier_to_oracle` against campaign 14's `three_tier_to`, same window; numbers in `results/campaign15/README.md` and
`results/campaign15/three_tier_to_oracle/`. The oracle clearly helps here: mean TTFT 53.3 → 41.1 s, p90 / p99 −27 %,
paired −7.28 s [−8.38, −6.16]; turns/s +13 % (0.988 → 1.114), 708 against 646 users finished; turns/s while congested
+22 % (0.64 → 0.78) and 34 % of the bins light against 28 %. The baseline left the SSD idle (54 % busy; ≥ 90 % in 22 % of
samples), because the timeout cut reads and 35 % of returns recomputed; the oracle's earlier reads used that bandwidth
(86 % busy, 18.0 against 11.0 TB read) and cut uncached tokens per turn by 12 %. The read queue at a trigger was short
(p50 1,203 / 2,941 pages for case 1 / 2), L̂ was close (read − L̂ p50 −0.3 s) and 34 % of the reads were late.
It is wasteful, as decision 3 expected: the timeout cut 5,018 of 9,663 oracle reads (52 %, 73.7M tokens short) against
11 % of the requests' own reads; 28 % of all reads were cut while still queued and skipped (campaign 14: 6 %); 37 % of
the 116M re-fetched tokens were credited from the SSD. `wait_complete` without the oracle is still better (22.2 s, 1.22
turns/s). Together with the `wait_complete` arm: the oracle gains where the SSD has idle bandwidth, and timing alone
cannot gain where it has none.

### Options after campaign 15 (decision 8, open)

Ordered by expected effect at campaign 14's load.

| | Change | Why (campaign 15 `wait_complete`) | Cost; first test |
|---|---|---|---|
| a | Evict by next use: when the host (or device) pool must free space, evict the context whose session's next turn is furthest away, ended sessions first (decision 7), instead of LRU; Belady's MIN with the hints | Fewer SSD reads is the only lever that adds capacity when the SSD is the bottleneck; host hits fell 16 → 10 % | Medium: an eviction priority in the tree core from the prefetcher's plans. First an offline replay of campaign 14's accesses through an L1 + L2 model, LRU vs next use (no GPU) |
| b | Order SSD reads by deadline (predicted admission) instead of FIFO; staging for far-off turns yields to queued turns | 76 % of case-1 reads late behind a ~30 s queue | Medium: the read queue (`prefetch_buffer`) becomes a priority queue fed by the prefetcher; speculative reads can starve |
| c | Allocate a read's host pages when it leaves the queue, or cap the share speculative reads may hold | In-flight reads held 39 % of the host pool | Medium-high: allocation timing in the cache controller (the transfer needs its pages; the rate limiter counts requested spans) |
| d | Online load model: measured ms per queued page (EWMA) instead of the 2.5 ms seed (decision 6) | L̂ under-predicted by 6.5 s | Small. Alone it moves every trigger earlier and adds no capacity in a full queue; (b) and the `timeout` arms need it |
| e | Evaluate capacity: users or turns per second at a fixed backlog target, or fixed-load arms without the cap, with repeated runs | The cap turns speed-ups into load; single runs vary ~1 s | Analysis or workload only |

---

## 6. Decisions

| # | Decision | State |
|---|---|---|
| 1 | Transport: `kv_hints` on the chat route (vs `custom_params`, or a literal `metadata` field, which chat lacks) | **taken** (go on step 1) |
| 2 | Trigger margin: exactly `A − L̂` (offline ~20-30 % of reads partly exposed) or + one inter-admission gap | open; decide after E1 |
| 3 | Prefetch policy: `wait_complete` arms only in v1 (a staged read's `timeout` deadline counts from staging) | proposed; campaign 15 measured it: `three_tier_to_oracle` beats campaign 14's `three_tier_to` (TTFT 53.3 → 41.1 s, +13 % turns/s) but the timeout cut 52 % of its reads (11 % of the requests' own), 28 % of all reads skipped while queued; `wait_complete` stays the better policy (section 5) |
| 4 | Case 2 scope: hinted requests only, or every queued request (needs no hint) | proposed: hinted only |
| 5 | Ŵ estimator: `W_ql` vs `W_ewma` (both logged in observe) | open; decide after E1 |
| 6 | Load model: static seeds (v1) vs online EWMA (needs an `io_seconds` field from the I/O thread); separate `b_backlog` | static in v1; `b_backlog` after E1. Campaign 15 `wait_complete`: the seeds under-predict the read by 6.5 s (p50) at campaign 14's load (≈ 3.2 ms per queued page vs 2.5) |
| 7 | Evicting ended sessions first (the oracle knows there is no next turn) | out of the headline; separate ablation only |
| 8 | Next lever after campaign 15: next-use eviction, deadline-ordered reads, host pages at dequeue, an online load model, capacity-based evaluation (section 5, options a-e) | open; the user decides |

---

## 7. Risks to watch

- Staged reads log SSD events under `next_rid` before that request exists: until step 8, `delay_components.py`,
  `timeline.py` and `read_backlog.py` misattribute them.
- Staged and refetched pages are unprotected after insert (an evictable host leaf): a too-early trigger wastes the read;
  at x10 a refetch for queue position k can evict position k+1's context (cascade). Measure re-evictions.
- One FIFO SSD read thread: oracle reads delay later, more urgent reads; reads ~1.9× slower during SSD writes. Campaign
  15 `wait_complete`: staged reads waited behind a median 9,820 pages (~30 s); 76 % of case-1 reads were late.
- Host pages are allocated when a lookup hits, before the read waits in the FIFO (`_try_alloc_storage_hit`), so a long
  read queue holds them. Campaign 15 `wait_complete`: in-flight reads held a median 39 % of the host pool (23 % without
  the oracle), and host hits fell from 16 % to 10 % of returns.
- Shared rate limit (0.5 × host pool): many staged reads push live arrivals into declined + paced retries.
- Prometheus storage counters skew: a request arriving after its staged read finished has its own new-tail miss resolve
  the staged hit as "dropped" (or, if its lookup hits, "Replacing unresolved storage-hit accounting"). Client numbers
  are right.
- `match_prefix` at trigger time makes the session MRU (behaviour change beyond the prefetch).
- Orphan plans (client cap, abort, run end) and late arrivals (offset outliers of 138-162 s): expiry + cleanup rules.
- Rank consistency: wall-clock triggers are rank-local; refuse TP/PP > 1 at startup.
- Plan memory: ≤ 8 B/token of committed chain per plan (~1 MB at 131k context; ≤ ~128 MB with 128 sessions).
- Upstream merges: exposing `kv_hints` on chat may conflict if upstream exposes it differently.

---

## 8. Reference numbers (H200 box, Qwen3-30B-A3B bf16, page 6 MiB; re-measure elsewhere)

| Quantity | Value | Source |
|---|---|---|
| SSD read I/O per page | p50 2.13 ms, p90 4.22 ms; fit ms = 43.5 + 2.477 × pages; aggregate ≈ 2.4 GB/s | c11 wc `s2h_io` (n 1,199); c7 aggregate 2.38-2.47 GB/s |
| Read size | p10/p50/p90 = 203/305/437 pages | c11 wc |
| Query → I/O start | idle read thread p50 0.51 s (dense alloc lag), queued p50 1.55 s | c11 wc |
| I/O end → insert | dense p50 0.36 s; sparse ~0.02 s | c11 wc; c10 |
| #39283 re-issue → insert | p50 ≈ 2.0 s, p90 ≈ 4.6 s ("HiCache prefetch success") | c11 wc |
| #39283 re-issue → admission | p50 3.83 s, p90 7.14 s ("reissuing storage lookup" → `load_back_init`, n 1,145 restored) | c11 wc |
| H2D load-back at admission | p50 44 ms for 17.7k tokens | c11 wc `h2d_done` |
| Arrival offset δ | sparse p50 0.088 s; dense p50 0.125 s, p90 0.87 s | c10, c11 |
| Queue wait | sparse p50 0.03 s; x10 p50 55 s (all requests; 73 s for returning turns), mean 66 s, p90 142 s | c10, c11 wc `ReqTimeStats` |
| Admission rate x10 | 60-s window p50 0.5/s; inter-admission p50 0.41 s, p90 3.4 s; 18 of 117 60-s windows empty | c11 wc |
| Restore TTFT vs backlog (x70) | ≈ 1.0 s + GB ahead / 2.0 GB/s (wc), 1.1 s + GB / 1.3 GB/s (to); wait = 58 % (to) / 80 % (wc) of TTFT | c7, `read_backlog.py` |
| Read queue at campaign 14's load (x70, 768 users, backlog cap) | ahead of a staged read p50 9,820 pages, p90 12,933; ≈ 3.2 ms per queued page; SSD ≥ 90 % busy in 49 % of 10-s samples (68 % with the oracle) | c14 wc, c15 wc_oracle (`c15_diagnosis.py`) |
| Retry budget | 2,162 of 4,660 client rids hit the cap ("reissue cap reached"; ≥ 9 `prefetch_start`) | c11 wc |

To re-measure on a new box: run the campaign-11 config for one `three_tier_wc` arm and use
`scripts/read_backlog.py`, `scripts/delay_components.py`, and `grep 'HICACHE_EVT s2h_io'` on its `server.log`.

---

## 9. Status

| Step | State | Date |
|---|---|---|
| 1 | done: client hints (`replay_agentic.py`, `start_client.sh` `ORACLE_HINTS`), verified offline (dry run, stub server); live smoke deferred to step 2 | 2026-10-01 |
| 2 | done (`2abe005bb2`): chat `kv_hints` + mode flag; live curl 200/400 and the `ORACLE_HINTS=1` client verified in smoke A (2026-10-09); open: lint (isort, trailing spaces, `[Haoran]` comments) | 2026-10-02 |
| 3 | done: residency probe + read-backlog helper (`fca11af340`, `ead87198a0`), `test_resident_prefix_probe.py`; open: two trailing spaces | 2026-10-06 |
| 4 | done: `next_turn_estimators.py` + `test_next_turn_estimators.py` (`294c82b7fc`), replayed against campaign 11 | 2026-10-09 |
| 5 | done: prefetcher in observe mode, wired into the Scheduler and `cache_finished_req` (`294c82b7fc`); CPU tests; GPU smoke A | 2026-10-09 |
| 6 | done: case 1 staging (`294c82b7fc`); CPU tests; GPU smoke B3 (116 of 120 staged reads before arrival) | 2026-10-09 |
| 7 | done: case 2 refetch + re-eviction cap (`294c82b7fc`); CPU tests; GPU smokes C and B2; stale re-eviction probe fixed (uncommitted), reruns C2-C4 | 2026-10-09 |
| 8 | done: analysis scripts + `oracle_eval.py` (uncommitted); campaign-11 outputs unchanged; hand-grep match on smoke A | 2026-10-09 |
| 9 | done: oracle arms incl. `_norq`, engine guard, `ORACLE_HINTS` in `run_compare.sh` (uncommitted); smoke A | 2026-10-09 |
| C15 | campaign 15 done (2026-10-10 02:48Z): both oracle arms against campaign 14's baselines, `results/campaign15/README.md`; analysis and options in section 5 (decision 8 open) | 2026-10-10 |
