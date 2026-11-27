# Review of the scheduler-side changes (2026-11-27)

Scope: the load hold (`scheduler_components/prefill_load_hold.py` and its wiring), the admission pass-over
(`scheduler_components/admission_pass_over.py`, new, not yet run on a GPU), the SSD cleaner's re-check before a delete
(`storage/nixl/nixl_cleaner.py`, `hicache_nixl.py`), the `queued_order` flag check. Three reviewers (hold, pass-over,
repo rules and tests), then one adversarial verifier for each of the six bugs and risks ranked first. Raw output:
the workflow journal `wf_eb286bf9-337`; reviewers' scripts in `review_sched/`.

None of the findings changes what campaigns 39 to 43 measure: they decode in CUDA graphs, never pause or flush,
keep the strict memory check off, read with `wait_complete`, and the prefill-cost fit feeds only switches that are off.

| Finding | Verdict | Where fixed |
|---|---|---|
| Strict busy-time memory check (`SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY`) does not count a held batch: false leak assert | confirmed | working tree and campaign 44's copy: `get_held_batch` in the invariant checker and the pool stats observer; test in `test_invariant_checker.py` |
| The hold's size test sums `host_loaded_length`, never reset: a batch with a small copy can be held | confirmed; 25 of 8,556 holds in campaign 39 (0.3 %), about one decode step each | working tree and campaign 44's copy: the adder counts what this walk's admissions copy (`PrefillAdder.host_load_tokens` -> `ScheduleBatch.hicache_load_tokens`); test in `test_scheduler_hicache_events.py` |
| `pause_generation(mode='retract')` leaves a held batch; a weight update with a flush then asserts | refuted as a hold defect: a pending load ack blocks that flush with or without the hold (pre-existing) | not changed; the held requests are not retracted on such a pause and run after `continue_generation` |
| Eager decode can wait on a held copy through the running batch's stale `hicache_consumer_index` | plausible, eager decode only | both: `prepare_for_decode` resets the index |
| The hold is booked as prefill time (`forward_entry_time` is the admission stamp); the meter's prefill-cost fit learns it | plausible; read only by switches that are off | documented in the hold's docstring; not changed (the admission estimates need the stamp) |
| The pass-over checks reads in flight behind a refused request, which ends them early under `timeout` and `best_effort` | plausible; not under `wait_complete` | both: such a candidate is skipped without the check; test |
| Priority scheduling without preemption passes the pass-over's gate | risk | both: the gate requires no priority scheduling |
| A refused `ignore_eos` candidate stays in the adder's state when the radix cache is disabled | risk | both: the gate requires the radix cache |
| `MAX_PASSES` was not the bound enforced (it was `MAX_PASSES - 1 + lookahead`) | style | both: the walk stops once the refused request has been passed `MAX_PASSES` times; test |
| The post-walk `batch_is_full` always used the HiCache rule | style | both: one helper reproduces the rule of the walk without the pass-over; test |
| The cleaner's re-check is off when only the stored-write skip refreshes mtimes (campaigns 34 to 39) | risk (pre-existing race) | both: the re-check is on whenever the store refreshes mtimes; test |
| Under `queued_order farthest`, turns first seen in one tick tie and fall back to LRU | risk; 8 % of arrivals share a tick in campaign 40 | working tree only, after campaign 44's copy was frozen: `Queued(since=arrival)`; test |
| `farthest` assumes FCFS admission | risk | both: a warning under any other schedule policy; test |
| Class-level defaults were shared mutable instances | risk (tests only) | both: lazy per-instance `cached_property` |
| `--enable-mixed-chunk` turned the hold off silently | risk | both: only a chunked request in progress stops a hold; a batch actually mixed has nothing to decode |
| Gates built inline in the scheduler; env vars in the storage section; formatter; docstrings; keyword calls | style | both |
| No test drove either component through the scheduler | test gap | both: wiring tests in `test_scheduler_hicache_events.py` (hold: 4 cases; pass-over: 5) |
| The cleaner test would hang, not fail, on one regression | test gap | both: the tick is bounded by a timer |
