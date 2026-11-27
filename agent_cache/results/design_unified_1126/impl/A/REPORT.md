# Implementation report, copy A: Fetch and Price (2026-11-26)

The implementer's structured report, saved by the main session (a subagent may not write report files).

## Summary

Built the FETCH and PRICE half of "Four actions, three prices" in the copy
/home/wanhr/sglang/agent_cache/frozen/impl_unified_A, in the planner's order (PR1, F5, F2 wiring, F3, F1, F4) with
amendments A1 to A5, including the starved-GPU rule (A3). Everything sits behind eight env switches, all off by
default.

Checked on CPU only: nothing was run on a GPU or a server, so the switches have no end-to-end run or measurement
behind them.
- All nine required test files pass on the final code; 76 unit tests were added to five existing files and every
pre-existing test passes unmodified (only two import lines grew).
- Switches off: a side-by-side replay against the unchanged prefetcher module (240 random scenarios of 600 events,
every mode, read order, timing, gate, hold, soft-pin, price-learning and eviction-policy combination) compared 40,628
log lines of all 17 pre-existing kinds, 48,226 answers and every cache call; all equal.
- Only the prices log on: the same holds apart from its own 26,603 `prices` lines and two fields at the end of
`ntp_admit`, so the log switch changes no decision.
- 33 one-line regressions were injected one at a time; each is caught by a new test (the first run missed one, and
that test was strengthened).

Two things need attention.
1. REPORT.md does not exist. The harness refused the write (subagents may not write report files) and I did not work
around it; this structured output is its content. My check scripts and their outputs are in
/home/wanhr/sglang/agent_cache/results/design_unified_1126/impl/A/checks/.
2. A merge was staged at 21:24:47 UTC (4:24 PM ET) into /home/wanhr/sglang/agent_cache/frozen/impl_unified_merged
while I was still working, so it is stale and needs to run again. Changed in copy A after it: code in
python/sglang/srt/mem_cache/unified_cache/components/full.py (`next_host_victim` breaks ties as the eviction heap
does), one test added in test/registered/unit/mem_cache/test_next_use_eviction.py, and comments only in environ.py,
next_turn_prefetch.py, unified_radix_cache.py, next_turn_meter.py and next_turn_prices.py.

Also for the merge: the working tree's environ.py gained SGLANG_HICACHE_PREFILL_LOAD_HOLD_STEPS at 20:37 UTC (not
mine) on the line my seven switches are inserted after, so both blocks must be kept there. I touched nothing outside
the copy and my notes folder, ran no git write and used no network.


## Files changed

- /home/wanhr/sglang/agent_cache/frozen/impl_unified_A/python/sglang/srt/mem_cache/next_turn_prices.py -- new, 337
  lines. Pure policy: DecayedMean, PriceBoard (c, mu, mu_for, kappa, lam), landing_quantile, AdmissionErrorTracker,
  PrefillCostModel (quadratic), quote_recompute.
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_A/python/sglang/srt/mem_cache/next_turn_meter.py -- new, 318
  lines. PriceMeter: feeds the board from the queue, host eviction, admissions and prefills; prices the next eviction
  victim; builds the prices line. ready_turns() is shared with the prefetcher.
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_A/python/sglang/srt/mem_cache/next_turn_switches.py -- new, 122
  lines. Switches.read(): reads the eight env values once, decides which act in this setup, raises ValueError on a
  missing prerequisite.
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_A/python/sglang/srt/environ.py -- +24 lines.
  SGLANG_LOG_HICACHE_PRICES_INTERVAL in the Logging section and seven SGLANG_HICACHE_NEXT_TURN_* switches after
  SGLANG_HICACHE_NEXT_TURN_ROOM_RETRY.
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_A/python/sglang/srt/mem_cache/next_turn_prefetch.py -- +536 -44,
  now 2,135 lines. Wiring of every step. New: _init_switched_changes, _follow_prices, _memory_price, _staging_price,
  _staged_start, _queued_baseline, _history_tokens, on_lookup_result, _recompute_quote, _relieve_starved_gpu,
  _hold_margin, _line_seconds; seven TurnPlan fields. Changed: tick, _on_departure (returns 1 for an admission),
  _check_queued, _check_pending_deadline, _check_pending_expected_cost, on_read_landed, _log_start_decision,
  _follow_room, prefetch_gate (optional third argument), read_not_before, _refresh_load_model, _stage, _expire.
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_A/python/sglang/srt/mem_cache/next_turn_policy.py -- +31 -2.
  GapDistribution.holding_rate, HOLDING_HORIZON_STEPS; the comment on RECOMPUTE_S_PER_TOKEN (now a seed).
  read_start(stall_weight=...) kept as received.
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_A/python/sglang/srt/mem_cache/next_turn_estimators.py -- +55.
  LagEwma, ReadPathLags, LAG_MIN_SAMPLES.
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_A/python/sglang/srt/mem_cache/next_use_eviction.py -- +46.
  holding_rate() on NextUseStrategy and ReturnChanceStrategy, HOLDING_MIN_HORIZON_S. queued_order kept as received.
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_A/python/sglang/srt/mem_cache/storage_read_scheduler.py -- +40.
  parked_reads, pages_due_by, pages_ahead_of.
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_A/python/sglang/srt/mem_cache/unified_radix_cache.py -- +99 -3.
  Class-level hooks prefetch_gate_prompt_fn, prefetch_lookup_fn, read_lags (all None); next_host_victim,
  storage_read_line_pages, parked_read_span, parked_read_pages_ahead, drop_parked_read, _observe_read_landing; the
  gate call in prefetch_from_storage; lag samples at park, dispatch and landing; the lookup report in the drain.
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_A/python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py
  -- +5. host_eviction_observer (None).
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_A/python/sglang/srt/mem_cache/unified_cache/components/full.py --
  +13. The observer call in drive_host_eviction; next_host_victim.
-
  /home/wanhr/sglang/agent_cache/frozen/impl_unified_A/python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py
  -- +4. PrefetchOperation.dispatched_at and ahead_pages.
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_A/test/registered/unit/mem_cache/test_next_turn_prefetch.py --
  +1216: 40 tests, the MeasuredCache fake and helpers.
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_A/test/registered/unit/mem_cache/test_next_turn_estimators.py --
  +396: 25 tests.
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_A/test/registered/unit/mem_cache/test_next_use_eviction.py --
  +115 -1: 4 tests (the removed line is an import that grew).
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_A/test/registered/unit/mem_cache/test_storage_read_dispatch.py --
  +178 -1: 5 tests (the removed line is an import that grew).
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_A/test/registered/unit/mem_cache/test_storage_read_scheduler.py
  -- +37: 2 tests.
- /home/wanhr/sglang/agent_cache/results/design_unified_1126/impl/A/checks/test_switches_off_equivalence.py -- notes
  folder, not part of the engine: the side-by-side replay against checks/baseline/next_turn_prefetch.py (a snapshot of
  the module before the change). Output in checks/equivalence.out.
- /home/wanhr/sglang/agent_cache/results/design_unified_1126/impl/A/checks/mutation_check.py -- notes folder: the 33
  injected regressions. Outputs in checks/mutation_check.out (M1-M32) and checks/mutation_check_after_last_edit.out
  (M12, M27, M28, M33 re-run after the last code edit). It edits the copy in place for about 12 s per mutation and
  restores it, so do not run it while a merge reads the copy.
- /home/wanhr/sglang/agent_cache/results/design_unified_1126/impl/A/checks/final_tests.out -- notes folder: the final
  run of the nine required test files.
- Not touched: managers/scheduler.py, managers/tokenizer_manager.py, model_executor/model_runner.py,
  managers/cache_controller.py, the real tree under /home/wanhr/sglang/python and /home/wanhr/sglang/test,
  agent_cache/scripts, agent_cache/data, agent_cache/frozen/engine_*.

## Switches

- Common to all eight: typed fields on Envs in environ.py, read once at construction (Switches.read). A switch with
  nothing to act on in a setup is silently off there (mode observe, no arrival gate, no arrival hold, fifo read order,
  another read timing); a switch that would act without another switch it needs raises ValueError at construction.
- SGLANG_LOG_HICACHE_PRICES_INTERVAL: gates the measuring and one HICACHE_EVT prices line per interval (c, mu,
  waiting, ready, queued, plans, kappa, late, parked, pend_w, lam, evict_tok, k_us, rec_a_us, rec_b_ns, rec_n,
  read_ms_page, err_n, err_q10_ms, err_q50_ms, err_q90_ms, rd_lag_ms, q20k, margin_ms, lookup_ms, lag_ms, whold_ms,
  win_holds, budget_holds, held, write_holds); ntp_admit gains reader= and land_ms= at its end. No decision reads it
  (checked side by side). Default EnvFloat(0.0) = off. Dependencies: none; works in mode on and observe.
- SGLANG_HICACHE_NEXT_TURN_MEASURED_PRICES: priced decisions take measured prices: host room at the eviction
  frontier's price lam (the hold's quantile; a staged read's rent, where a staging also pays the index of the holder
  it would evict next), and the harm of a refused read as mu x the fitted prefill cost instead of the constant x queue
  count. Default EnvBool(False). Acts in mode on only. Dependencies: raises unless
  --hicache-storage-next-turn-price-learning is off. Turns the measuring on.
- SGLANG_HICACHE_NEXT_TURN_FREE_EMPTY_LOOKUPS (F5): the plan's key is cut to what the turn's prompt repeats of it; a
  lookup that finds nothing gives back what the gate and the queue check charged (at most 8 times a turn) and marks
  where storage ends, so it is not asked again until the match shrinks below that. New gate answer absent, new line
  HICACHE_EVT ntp_lookup rid= start= asked= hit= refunded= lookups=. Default EnvBool(False). Acts only in mode on with
  the arrival gate. Dependencies: none.
- SGLANG_HICACHE_NEXT_TURN_LATENESS_WEIGHT (F2): read_start weighs a second of stall by c, the share of admissions
  that left no ready turn waiting; ec_start gains c= at its end. Default EnvBool(False). Acts only under read timing
  expected_cost. Dependencies: none; turns the measuring on itself.
- SGLANG_HICACHE_NEXT_TURN_READ_LINE (F3): read-time estimates count the parked reads that dispatch orders no later,
  the rest of a write hold for a timed read, and the read path's measured lags (after 8 samples of each) in place of
  the seeds. Applies to the staged start, the queue check, the hold, the recompute quotes, and horizon / just-in-time
  starts while c >= 0.5. ntp_due backlog_pages becomes the line's pages; ec_start gains line_pages= at its end.
  Default EnvBool(False). Acts only under the deadline read order. Dependencies: raises without LATENESS_WEIGHT when
  read timing is expected_cost. Turns the measuring on.
- SGLANG_HICACHE_NEXT_TURN_HOLD_QUANTILE (F1): a held read starts by minus the q-quantile of the measured admission
  error at its predicted wait, q = rent / (c + rent) clamped to [0.05, 0.5], instead of the fixed 2 s; the fixed
  margin stands until that wait's bin holds 30 admissions. ntp_hold gains margin_ms= and q= at its end. Default
  EnvBool(False). Acts only with the arrival hold. Dependencies: raises without MEASURED_PRICES.
- SGLANG_HICACHE_NEXT_TURN_RECOMPUTE_BY_PRICE (F4 with A2): at the gate a turn that arrives with nobody else queued
  and no ready turn waiting recomputes its missing history when c >= 0.8 and mu_for(recompute_s) x recompute_s < the
  read's line-aware time; the answer recompute spends no lookup and is priced again at the turn's next call. ntp_gate
  gains rec_ms= read_ms= mu= c= on priced calls. Default EnvBool(False). Acts only in mode on with the arrival gate.
  Dependencies: raises without MEASURED_PRICES, READ_LINE (which itself acts only under the deadline read order) or
  SGLANG_HICACHE_NIXL_SKIP_STORED_WRITES.
- SGLANG_HICACHE_NEXT_TURN_RECOMPUTE_WHEN_STARVED (A3): in tick, once the queue has held no ready turn for
  NTP_GPU_BOUND_S (0.5 s), the first queued turn with a plan and a read still parked gives that read up if recomputing
  its span is sooner than its landing (new cache method drop_parked_read; a dispatched read is left alone);
  plan.recompute then makes the gate decline the turn's lookups uncharged. One turn a pass. New line HICACHE_EVT
  ntp_starve rid= land_ms= rec_ms= tokens= parked=. Default EnvBool(False). Acts only in mode on with the arrival
  gate. Dependencies: the same three as RECOMPUTE_BY_PRICE.

## Tests added

- All in existing files under /home/wanhr/sglang/agent_cache/frozen/impl_unified_A/test/registered/unit/mem_cache/. 76
  tests. M-numbers are the injected regressions of checks/mutation_check.py that the test caught.
- test_next_turn_estimators.py :: TestReadPathLags.test_a_lag_is_published_only_after_the_minimum_samples -- red when
  a lag is published before 8 samples (one warm-up sample would set every read-time estimate).
- test_next_turn_estimators.py ::
  TestReadPathLags.test_a_timed_read_waits_what_is_left_of_a_write_hold_of_the_usual_length -- red when the write-hold
  remainder is added outside a hold, before one was measured, or not reduced by the time already held.
- test_next_turn_estimators.py :: TestStallWeight.test_weighted_start_is_the_cheapest_on_a_fine_grid -- red when the
  weight reaches only one of the two stall terms (arrival, late landing): a grid search then finds a cheaper start.
- test_next_turn_estimators.py :: TestStallWeight.test_free_lateness_never_reads_early -- red when weight 0 still
  starts a read before arrival.
- test_next_turn_estimators.py :: TestStallWeight.test_a_smaller_weight_never_starts_a_read_earlier -- red when the
  weight lands on the rent instead of the stall.
- test_next_turn_estimators.py :: TestPriceBoard.test_c_is_the_share_of_admissions_that_left_no_ready_turn_waiting --
  red when only one admission of a pass is counted, or the share is inverted.
- test_next_turn_estimators.py :: TestPriceBoard.test_c_forgets_a_saturated_episode -- red when c does not return
  toward 1 minutes after the queue drained.
- test_next_turn_estimators.py :: TestPriceBoard.test_mu_at_light_load_is_one_plus_the_time_average_of_ready_turns --
  red when ready turns are averaged per pass instead of per second.
- test_next_turn_estimators.py :: TestPriceBoard.test_at_saturation_a_gpu_second_delays_every_session_not_running --
  red when mu at saturation is read off the ready queue (M24), or mu_for drops the arrivals during the job.
- test_next_turn_estimators.py ::
  TestPriceBoard.test_room_is_free_until_something_is_evicted_and_again_once_evictions_stop -- red when lam is nonzero
  before an eviction or 30 s after the last, an unpriced victim counts as price 0, or lam passes its ceiling.
- test_next_turn_estimators.py :: TestPriceBoard.test_kappa_prices_late_reads_at_c_and_pending_writes_at_lambda -- red
  when kappa drops a term or prices late reads at 1 instead of c.
- test_next_turn_estimators.py ::
  TestLandingQuantile.test_cheap_room_lands_early_and_a_saturated_gpu_late_within_the_clamp -- red when q leaves
  [0.05, 0.5] or the floor on c is dropped.
- test_next_turn_estimators.py ::
  TestAdmissionErrorTracker.test_margin_is_minus_the_quantile_of_the_bin_the_prediction_falls_in -- red when the sign
  is wrong (M17), another bin answers, or a sorted copy outlives the bin's next sample.
- test_next_turn_estimators.py :: TestAdmissionErrorTracker.test_an_empty_bin_keeps_the_fixed_margin -- red when a bin
  with under 30 samples answers from other bins.
- test_next_turn_estimators.py :: TestAdmissionErrorTracker.test_readers_do_not_feed_the_estimate_of_the_gpu_turn --
  red when a reader's wait enters the error bins.
- test_next_turn_estimators.py :: TestPrefillCostModel.test_fit_recovers_both_terms_from_warm_and_cold_prefills -- red
  when the fit does not recover a and b from mixed warm and cold prefills.
- test_next_turn_estimators.py :: TestPrefillCostModel.test_the_seed_holds_until_enough_turns_were_seen -- red when
  the fit replaces the seed before 20 samples.
- test_next_turn_estimators.py ::
  TestPrefillCostModel.test_one_shape_of_request_gives_a_per_token_cost_not_a_negative_term -- red when a degenerate
  fit returns a negative coefficient.
- test_next_turn_estimators.py ::
  TestRecomputeQuote.test_a_recompute_must_also_beat_the_turns_that_arrive_while_it_runs -- red when the quote ignores
  arrivals during the recompute (M26).
- test_next_turn_estimators.py :: TestRecomputeQuote.test_history_deep_in_a_long_prompt_costs_its_position_too -- red
  when the recompute is priced per token whatever its depth in the prompt.
- test_next_turn_estimators.py :: TestRecomputeQuote.test_once_ready_turns_were_left_waiting_the_read_is_kept -- red
  when a recompute wins with c under 0.8.
- test_next_turn_estimators.py :: TestRecomputeQuote.test_at_triton_cost_the_read_wins_at_every_size -- red when the
  seed cost (110 us a token) ever recomputes.
- test_next_turn_estimators.py :: TestHoldingRate.test_falling_hazard_entries_lose_value_with_idle_time -- red when
  the rate is taken over the support points (it then jumps) or rises with idle time.
- test_next_turn_estimators.py :: TestHoldingRate.test_a_sharp_entry_is_worth_one_over_its_mean_until_it_is_due -- red
  when a rising-hazard entry is not worth 1 / E[gap].
- test_next_turn_estimators.py :: TestHoldingRate.test_no_mass_left_means_no_value -- red when a distribution with no
  survival left gives a rate or divides by zero.
- test_next_use_eviction.py ::
  TestHoldingRate.test_next_use_rate_is_one_over_the_time_left_and_zero_for_a_session_not_coming -- red when a queued
  or unregistered holder is priced 0 instead of left out, or a turn due now divides by zero.
- test_next_use_eviction.py :: TestHoldingRate.test_return_chance_rate_is_the_sessions_whichever_of_its_nodes_goes --
  red when a session's body (no registration of its own) gets no rate, or an exact hint's grace is ignored.
- test_next_use_eviction.py ::
  TestHostEvictionObserver.test_each_victim_is_reported_with_its_host_tokens_before_it_goes -- red when the observer
  is told after the eviction (M27) or of a leaf eviction did not take.
- test_next_use_eviction.py :: TestHostEvictionObserver.test_the_next_victim_is_the_leaf_eviction_then_takes_first --
  red when the next victim is named from the other end of the order (M33), or one is named with nothing evictable.
- test_storage_read_scheduler.py ::
  TestStorageReadScheduler.test_the_line_ahead_of_a_read_is_the_now_reads_and_the_timed_ones_due_no_later -- red when
  pages_due_by counts every parked read or none (M29).
- test_storage_read_scheduler.py ::
  TestStorageReadScheduler.test_ahead_of_a_parked_read_are_those_dispatch_takes_first -- red when pages_ahead_of
  counts a read of the same deadline parked later.
- test_storage_read_dispatch.py ::
  TestReadOwnerHooks.test_the_gate_that_reads_the_prompt_gets_it_whole_from_the_match_on -- red when the gate gets the
  prompt cut to a known hit (M9), or the two-argument gate is still asked.
- test_storage_read_dispatch.py ::
  TestReadOwnerHooks.test_every_answered_lookup_is_reported_once_with_what_it_asked_and_found -- red when a terminated
  lookup (M10) or one whose request is gone is reported, or one is reported twice.
- test_storage_read_dispatch.py ::
  TestReadLine.test_the_line_is_the_launched_reads_and_the_parked_ones_ordered_no_later -- red when the line leaves
  out launched or parked-ahead pages, or counts reads due later.
- test_storage_read_dispatch.py ::
  TestReadLine.test_only_a_parked_read_can_be_dropped_and_its_turn_is_then_free_to_admit -- red when a dispatched read
  is dropped (M23), or the dropped turn still has a read under way.
- test_storage_read_dispatch.py :: TestReadLine.test_the_lags_are_taken_at_park_dispatch_and_landing_once_installed --
  red when the fixed lag charges the pages ahead (M16), or a cache nobody asked measures.
- test_next_turn_prefetch.py ::
  TestOwnerSwitches.test_prices_are_measured_only_for_the_log_or_the_switch_and_decide_only_in_mode_on -- red when
  hooks are bound with everything off, observe decides on measured prices (M25), or price learning is accepted with
  measured prices.
- test_next_turn_prefetch.py ::
  TestMeasuredPrices.test_c_falls_only_when_admissions_leave_a_turn_with_nothing_to_read_waiting -- red when c ignores
  the ready turns left (M1), or a turn waiting on its own read counts as ready.
- test_next_turn_prefetch.py ::
  TestMeasuredPrices.test_a_prices_line_per_interval_carries_what_rebuilds_the_users_in_session -- red when waiting
  leaves out unplanned queued turns (M2), a field is missing or moved, more than one line is logged per interval, or a
  value leaves [\w./-]+.
- test_next_turn_prefetch.py ::
  TestMeasuredPrices.test_ntp_admit_keeps_its_fields_and_gains_the_reader_flag_and_the_landing -- red when the two
  fields are not last, or appear while nothing is measured.
- test_next_turn_prefetch.py ::
  TestMeasuredPrices.test_the_admission_error_is_fed_by_turns_that_read_nothing_with_readers_apart -- red when readers
  (M6) or retracted requests feed the error bins.
- test_next_turn_prefetch.py :: TestMeasuredPrices.test_host_eviction_prices_the_room_by_the_victim_worth_most -- red
  when the frontier takes the lowest rate (M4), or an unpriced victim sets a price.
- test_next_turn_prefetch.py :: TestMeasuredPrices.test_kappa_counts_the_parked_reads_past_their_latest_start -- red
  when lateness is judged by the deadline alone (M3).
- test_next_turn_prefetch.py :: TestMeasuredPrices.test_the_prefill_fit_is_fed_only_by_turns_admitted_at_once -- red
  when queued or retracted turns feed the fit (M5).
- test_next_turn_prefetch.py ::
  TestFreeEmptyLookups.test_a_reply_the_next_prompt_does_not_repeat_is_not_missing_history -- red when the key's
  unrepeated tail still costs a lookup, or history that did leave memory is no longer read.
- test_next_turn_prefetch.py ::
  TestFreeEmptyLookups.test_an_empty_lookup_is_given_back_and_not_asked_again_until_the_match_shrinks -- red when an
  empty lookup stays charged, is repeated from the same match, or stays closed once the match shrank (M30).
- test_next_turn_prefetch.py ::
  TestFreeEmptyLookups.test_a_turn_whose_lookups_found_nothing_still_reads_its_history_once_it_leaves_memory -- red
  when empty lookups use up the allowance and the gate answers capped (OPEN_ITEMS.md section 1).
- test_next_turn_prefetch.py :: TestFreeEmptyLookups.test_lookups_are_given_back_only_so_many_times -- red when
  refunds have no bound.
- test_next_turn_prefetch.py :: TestFreeEmptyLookups.test_the_staged_lookup_does_not_set_what_the_prompt_repeats --
  red when the staged lookup sets the shared length (M7).
- test_next_turn_prefetch.py :: TestFreeEmptyLookups.test_an_empty_re_read_of_a_queued_turn_is_not_a_re_eviction --
  red when the queue check's count is not given back (M8).
- test_next_turn_prefetch.py :: TestFreeEmptyLookups.test_the_switch_binds_only_with_the_gate_in_mode_on -- red when
  the hooks are bound without the gate or in observe.
- test_next_turn_prefetch.py ::
  TestLatenessWeight.test_after_a_saturated_minute_a_read_is_not_started_where_an_idle_gpu_starts_it -- red when c is
  not applied (M11), or c= is not the last field of ec_start.
- test_next_turn_prefetch.py ::
  TestLatenessWeight.test_a_minute_of_turns_waiting_on_their_own_reads_leaves_the_start_where_it_was -- red when a
  queue held up by the SSD lowers c.
- test_next_turn_prefetch.py :: TestLatenessWeight.test_the_weight_acts_only_on_reads_staged_by_expected_cost -- red
  when the switch measures or acts under another timing.
- test_next_turn_prefetch.py :: TestMeasuredRoomPrice.test_a_staging_pays_for_the_holder_it_would_evict_next -- red
  when a staging is priced by the frontier alone (M12), or a queued victim is priced as unknown (M28).
- test_next_turn_prefetch.py ::
  TestMeasuredRoomPrice.test_the_harm_of_a_refused_read_is_its_recompute_at_the_fitted_cost_times_mu -- red when the
  harm keeps the constant times the queue count.
- test_next_turn_prefetch.py ::
  TestReadLine.test_a_parked_read_due_earlier_moves_a_start_earlier_and_one_due_later_does_not -- red when the line is
  not counted (M13), or reads due later are (M32).
- test_next_turn_prefetch.py :: TestReadLine.test_once_ready_turns_mostly_wait_just_in_time_leaves_the_line_out -- red
  when just-in-time counts the line with c under 0.5 (M14).
- test_next_turn_prefetch.py ::
  TestReadLine.test_measured_lags_replace_the_seeds_only_where_the_line_is_counted_in_pages -- red when seeds and line
  are both counted, a negative residual becomes a lag, or lags are used before both were measured.
- test_next_turn_prefetch.py :: TestReadLine.test_a_held_read_starts_by_its_pages_and_the_measured_lags -- red when
  the hold keeps the seeded lags under the switch (M15), or changes with it off.
- test_next_turn_prefetch.py ::
  TestReadLine.test_the_switch_needs_parked_reads_and_under_expected_cost_the_lateness_weight -- red when the switch
  acts without the deadline order, or is accepted alone under expected-cost staging.
- test_next_turn_prefetch.py ::
  TestReadLineByExpectedCost.test_with_an_idle_gpu_a_line_ahead_starts_a_staged_read_earlier -- red when the staged
  start ignores the line, or line_pages= is not last.
- test_next_turn_prefetch.py :: TestReadLineByExpectedCost.test_after_a_saturated_spell_the_line_starts_nothing_early
  -- red when the line starts reads early while c is low.
- test_next_turn_prefetch.py ::
  TestHoldQuantile.test_free_room_lands_a_held_read_early_and_a_saturated_gpu_at_the_median -- red when the margin
  stays 2 s, takes the wrong quantile, or plans past the median.
- test_next_turn_prefetch.py :: TestHoldQuantile.test_ntp_hold_gains_the_margin_and_the_quantile_at_its_end_once --
  red when the fields are not last, or the line repeats every pass.
- test_next_turn_prefetch.py ::
  TestHoldQuantile.test_held_reads_still_all_go_when_the_margin_starts_them_past_the_estimate -- red when a negative
  margin leaves a parked read held at the head.
- test_next_turn_prefetch.py :: TestHoldQuantile.test_the_switch_needs_the_hold_and_measured_prices -- red when the
  switch is accepted on the knob's price, or acts without the hold.
- test_next_turn_prefetch.py ::
  TestRecomputeByPrice.test_both_recompute_switches_need_the_gate_and_refuse_what_they_cannot_price_on -- red when
  either switch is accepted without measured prices, the read line or skipped stored writes, or acts without the gate.
- test_next_turn_prefetch.py ::
  TestRecomputeByPrice.test_alone_at_an_idle_gpu_a_short_history_is_recomputed_and_a_long_one_read -- red when 8K
  tokens are read or 60K recomputed, a lookup is charged, or the priced fields are missing.
- test_next_turn_prefetch.py ::
  TestRecomputeByPrice.test_a_reply_the_prompt_does_not_repeat_is_not_priced_as_history_to_restore -- red when the
  quote prices the key's tail the prompt lacks (M31).
- test_next_turn_prefetch.py :: TestRecomputeByPrice.test_behind_another_turn_or_once_it_has_waited_the_turn_reads --
  red when a recompute is priced behind another turn (M18), or kept once the turn has itself waited.
- test_next_turn_prefetch.py ::
  TestRecomputeByPrice.test_a_falling_c_unmeasured_lags_and_the_staged_lookup_are_not_recomputed -- red when priced on
  the seeded lags (M19), under c 0.8, or for the staged lookup.
- test_next_turn_prefetch.py ::
  TestRecomputeWhenStarved.test_the_first_planned_turn_with_a_parked_read_recomputes_and_reads_no_more -- red when the
  rule fires before 0.5 s, touches a dispatched read, takes a turn with no plan, or the gate then allows or charges a
  lookup (M22).
- test_next_turn_prefetch.py :: TestRecomputeWhenStarved.test_one_turn_with_nothing_to_read_keeps_every_read -- red
  when the rule fires while a ready turn waits (M20).
- test_next_turn_prefetch.py ::
  TestRecomputeWhenStarved.test_only_the_first_such_turn_is_weighed_and_a_read_that_lands_sooner_is_kept -- red when
  reads parked later count as ahead (M21), or more than one turn a pass is taken.
- test_next_turn_prefetch.py :: TestRecomputeWhenStarved.test_with_the_switch_off_a_starved_queue_keeps_its_reads --
  red when anything is dropped or logged with the switch off.

## Test results

- test_next_turn_prefetch.py: 117 passed, 80 subtests passed (before the change: 77 passed, 55 subtests)
- test_next_turn_estimators.py: 39 passed (before: 14)
- test_next_use_eviction.py: 25 passed, 2 subtests passed (before: 21, 2 subtests)
- test_storage_read_scheduler.py: 11 passed (before: 9)
- test_storage_read_dispatch.py: 33 passed, 2 subtests passed (before: 28, 2 subtests)
- test_host_census.py: 7 passed, 9 subtests passed (unchanged)
- test_hicache_nixl_cleaner.py: 14 passed, 12 subtests passed (unchanged)
- test_hicache_nixl_storage.py: 21 passed, 1 skipped (unchanged)
- test_evict_policy.py: 23 passed (unchanged)
- No test file was added to the repository copy. The nine counts above are from the final run on the final code, saved
  in /home/wanhr/sglang/agent_cache/results/design_unified_1126/impl/A/checks/final_tests.out.
- /home/wanhr/sglang/agent_cache/results/design_unified_1126/impl/A/checks/test_switches_off_equivalence.py (notes
  folder, run on the final code): 3 passed. Switches off: 40,628 lines of 17 kinds and 48,226 answers equal to the
  unchanged module's. Only the prices log on: equal apart from 26,603 prices lines and two fields on 3,814 ntp_admit
  lines.
- /home/wanhr/sglang/agent_cache/results/design_unified_1126/impl/A/checks/mutation_check.py: 33 of 33 injected
  regressions caught. M1-M32 ran before the last code edit (4 lines in full.py); M12, M27, M28 and M33 ran after it. I
  did not repeat the full run afterwards because a merge had started reading the copy and the check edits files in
  place.
- Wider CPU run of 22 other mem_cache test files on the copy, before the last edits: 19 files pass.
  test_unified_radix_lock_ref.py (1 failed), test_pp_prefetch_ticket.py (2 failed, 9 passed) and
  test_hicache_pp_sync_drain.py (1 failed, 3 passed) fail with the same tests and counts on the unchanged tree: their
  fixtures lack attributes the cache already had.
- test_unified_radix_cache_unittest.py: cannot run here; it needs an accelerator and the GPU had to stay hidden.

## Deviations from the plan

- Gate hook: the prompt-reading gate is a second hook, prefetch_gate_prompt_fn, asked in place of prefetch_gate_fn
  when bound; the plan changed prefetch_gate_fn's signature. Why: the two-argument gate, its fixtures and its tests
  stay as they are, and with the switches off the cache calls exactly what it called before.
- Cache attributes: the three new ones (prefetch_gate_prompt_fn, prefetch_lookup_fn, read_lags) are class-level
  defaults, not set in __init__. Why: the dispatch fixture builds the cache with __new__, and existing tests had to
  pass unmodified.
- Module split: the measuring is a PriceMeter (next_turn_meter.py) and the switch reading a Switches struct
  (next_turn_switches.py); the plan put both in the prefetcher. Why: the prefetcher would have reached about 2,380
  lines. The meter builds the prices line and the prefetcher logs it; PriceBoard.snapshot was not needed.
- Ready-turn bookkeeping: the meter keeps its own record; the plan moved the prefetcher's into _follow_ready for
  everyone. Why: _queue_ahead updates its record only on passes that hold a plan, so one shared record fed every pass
  changed expected-cost decisions with only the log on. Kept apart, the log changes nothing (checked side by side).
- Read-path lags: ReadPathLags lives in next_turn_estimators.py with the write-hold remainder as its method, and the
  cache feeds it only once the meter installs it; the plan measured always, in cache fields. Why: with every switch
  off the cache does no new work and keeps no new state.
- New log fields are appended only while their switch (or the measuring) is on: ntp_admit reader/land_ms, ec_start
  c/line_pages, ntp_hold margin_ms/q. The plan appended them always. Why: that changes existing lines with every
  switch off, against the contract.
- Dependencies: a missing switch raises, a structure with nothing to act on makes the switch inert. LATENESS_WEIGHT
  and READ_LINE turn the measuring on themselves (the plan left LATENESS_WEIGHT silently off without the board).
  MEASURED_PRICES refuses any price learning. Why: a silently inert switch in a campaign arm is worse than an error,
  and a learned price would be logged and never used.
- mu at saturation is max(light-load value, sessions waiting), not sessions waiting alone (A1). Why: mu must not fall
  when c crosses 0.1 while few sessions are known to the prefetcher.
- lam and the next victim's price are capped at 1e-3, the learned price's ceiling. Why: a victim due within 0.5 s is
  worth 2 per second, which times the restore cost is far above any price the knob ever took and would stop all
  staging.
- Victims with no holding rate (a queued turn's history, un-hinted prefixes) count as evicted tokens and set no price;
  the plan dropped them. Why: evict_tok in the prices line would have under-reported eviction.
- Late reads and kappa are computed where they are used (the eviction fold, the prices line, the next victim's price),
  not every pass. Why: nothing reads them in between, and it costs one deadline per parked read.
- Admission error: a reader whose landing was not seen is skipped (the plan counted it as a non-reader); a retracted
  request (negative wait) is skipped here and in the prefill fit; landing times are kept only for planned or queued
  requests. Why: each of the first two would bias the estimate, and the third keeps that table bounded.
- The prices line carries lookup_ms, lag_ms and whold_ms beyond the plan (F3's measured lags; nothing else logs them),
  beside waiting, plans, queued, ready and the four hold counters (A1, A4).
- F5: the gate gets the prompt whole, captured before the cut to a known hit. Why: a retry that knows where the last
  hit ended cuts the tokens it asks storage for; handed that cut, the gate takes the prompt for one that leaves the
  key early.
- F5: a terminated lookup is not reported to the plan. Why: its hit count of 0 is no answer and would mark history
  absent and refund wrongly.
- F5: RadixKey.match_at gives the shared length; no new common_prefix_len helper. Why: it exists and handles bigram
  keys.
- F5: the free room retry clears lookup_charged. Why: its empty answer must not give back a lookup charged before it.
- F2: the plan's guard is built: a staging pays max(frontier price, the next victim's index). A queued turn's history
  as next victim is priced at the top rate, not as unknown, and the victim is priced at most every 0.5 s. Why: the
  frontier's price is zero until an eviction happens, so alone it would stage every plan into a full room; pricing
  costs one priority per evictable host leaf.
- F3: two load models, _load on the seeds as today and _line_load on the measured lags, instead of replacing _load.
  Why: the estimates that leave the line out must keep today's numbers.
- F3: horizon and just-in-time starts count the line only while c >= 0.5 (READ_LINE_MIN_C). Why: these timings weigh
  no cost against an early start, so at saturation the line (42 s at the median in campaign 34) would only pin more
  history early; the plan's remedy, c through F2, covers expected-cost only.
- F3: the line's deadline is the one read_deadline() gives the read, the queue check takes 'now' at the head, and the
  write hold's remainder is added for timed reads only. Why: the estimate then follows dispatch's own order, and 'now'
  reads are never write-held.
- F3: the fixed lag subtracts the pages completed, not those asked. Why: a read cut for host room completes fewer.
- F4: quote_recompute implements A2 and replaces the prototype's recompute_wins. Two more conditions: the lags must be
  measured, and the missing span is cut to the prompt's tail. Why: on the seeds a recompute wins at every size; and
  without F5 the key still ends with a reply the prompt does not repeat, which neither a read nor a recompute
  restores.
- A3: also needs measured lags; the pages ahead of the parked read are those dispatch takes before it (new
  pages_ahead_of), not pages_due_by; the queue check stops re-issuing a prefetch for a turn that gave its read up;
  cache helpers parked_read_span and parked_read_pages_ahead were added. Why: pages_due_by counts reads of the same
  deadline parked later, which made the read look later than it is and its turn give it up.
- Small: holding_rate takes queued= (default None, the plan's answer) so the next victim can be priced;
  next_host_victim breaks ties by node as the eviction heap does; the env comments carry no F-tags and name each
  switch's needs; parked_reads(), a one-line accessor, has no test of its own (the kappa test goes through it).

## Not done

- REPORT.md was not written at /home/wanhr/sglang/agent_cache/results/design_unified_1126/impl/A/REPORT.md: the Write
  tool refused it ('Subagents should return findings as text, not write report files'), and I did not work around the
  refusal. This structured output is its full content and can be saved to that path.
- Nothing was run on a GPU or a server: no end-to-end run, no measurement. The switches are exercised by CPU unit
  tests on fakes and on the real cache fixture only.
- The merge staged at 21:24:47 UTC (4:24 PM ET) in /home/wanhr/sglang/agent_cache/frozen/impl_unified_merged predates
  the copy's last edits and has to be run again. Changed after it: 3 code lines in
  python/sglang/srt/mem_cache/unified_cache/components/full.py, one test added in
  test/registered/unit/mem_cache/test_next_use_eviction.py, and comments only in environ.py, next_turn_prefetch.py,
  unified_radix_cache.py, next_turn_meter.py and next_turn_prices.py.
- environ.py insertion point: the working tree gained SGLANG_HICACHE_PREFILL_LOAD_HOLD_STEPS right after
  SGLANG_HICACHE_NEXT_TURN_ROOM_RETRY (20:37 UTC, not mine), the line my seven switches follow. Left for the merge;
  both blocks must be kept.
- next_turn_prefetch.py is 2,135 lines, over the ~2k guideline, and environ.py crossed 2,000 (2,013). Going under in
  the prefetcher means moving existing code (the hint payload parsing), which I left alone so the merge with copy B
  stays simple.
- Host eviction on a real tree with the observer, and UnifiedRadixCache.next_host_victim, have no CPU test:
  test_unified_radix_cache_unittest.py needs an accelerator. FullComponent's side is covered on a stub tree core.
- The scripts under /home/wanhr/sglang/agent_cache/scripts do not read the new lines (prices, ntp_lookup, ntp_starve)
  or the appended fields; they were not mine to touch.
- Constants picked, not tuned, each saying so in its comment: READ_LINE_MIN_C 0.5, NTP_MAX_FREE_LOOKUPS 8, LAM_CEILING
  1e-3, VICTIM_PRICE_INTERVAL_S 0.5, HOLDING_HORIZON_STEPS 48, LAG_MIN_SAMPLES 8, MIN_ERROR_SAMPLES 30, the prefill
  fit's 20 samples and 200-sample memory.
- The prefill fit starts from Triton's 110 us a token and holds it until 20 turns were admitted at once; until then
  RECOMPUTE_BY_PRICE keeps every read.
- The admission error of turns with no plan is recorded only on passes that hold a plan or run the arrival hold.
- waiting= counts a session whose next turn never comes until its plan expires, and misses a session without hints
  between its turns.
- F5's absent mark can cost a recompute of a span another session wrote meanwhile (the plan's own caveat).
- No documentation outside the code.
- The pre-existing failures in three other CPU test files (test_unified_radix_lock_ref, test_pp_prefetch_ticket,
  test_hicache_pp_sync_drain), identical on the unchanged tree, were not addressed.


# Fix round after the three reviews (2026-11-27 00:37Z)

## Summary

Every finding that is real is fixed in the copy /home/wanhr/sglang/agent_cache/frozen/impl_unified_A: the five bugs,
the three estimator weaknesses, the two default-off test gaps and all but one of the other test gaps. Three findings
are rejected, each with its reason. Checked on CPU only; nothing ran on a GPU or a server.  What the checks show on
the final code: - **Switches off:** the nine test files pass, the base snapshot's unmodified test files pass on the
fixed engine with the baseline counts, and the engineer's side-by-side replay still finds 40,628 log lines and 48,226
answers equal. - **Only the prices log on:** host eviction takes the same leaves as with it off, on a stub tree and on
a real CPU tree. The replay is equal apart from its prices lines and the two ntp_admit fields. - **Tests:** 34 added
(225 to 259 in the five changed test files). Of 93 injected regressions (16 behaviours fixed in this round, the
engineer's 33, the reviewers' 44), 90 are caught. The other 3 are two that change no behaviour and one corner left
without a test.  The fixes changed what four switches do: - **LATENESS_WEIGHT and READ_LINE** are now off in mode
observe, as the other decision switches are. - **MEASURED_PRICES** is refused at startup under an eviction policy
other than next_use or return_chance when reads are staged by expected cost or held by quantile. Its room price no
longer feeds on pending writes and counts a queued turn's history by the wait it has left; its admission error leaves
out turns that found the queue empty. - **RECOMPUTE_WHEN_STARVED** gives up at most one read per half second, and
weighs a turn only once it is registered as queued. - **FREE_EMPTY_LOOKUPS** drops its mark of where storage ends once
the match is past it.  Three fixes depart from the plan's pseudo-diffs; items 7, 8 and 10 of the fixed list say why.
Needs attention: 1. **REPORT.md was not updated by me.** A subagent may not write report files, and the main session
saved the engineer's report the same way; this output is the update. 2. **The staged merge is stale.**
/home/wanhr/sglang/agent_cache/frozen/impl_unified_merged (4:24 PM ET on 26 November) predates all of this and must be
run again before an engine copy is frozen from it. Copy A's change set now also holds evict_policy.py.  My check
scripts and their outputs are in /home/wanhr/sglang/agent_cache/results/design_unified_1126/impl/A/fix/. I wrote
nothing outside the copy and that folder, ran no git write and used no network.


## Fixed

- 1. Logic reviewer, finding 1 (bug). With READ_LINE on, the queue check walked the parked reads for every queued
  planned turn on every pass, for values that are only logged, and counted a turn's own parked read twice. The line is
  now counted once when the turn is first marked due and once when a refetch is handed back (`_queued_line_seconds` in
  next_turn_prefetch.py), and a read of the turn's own that is already parked stands in the line. With the reviewer's
  timing script, 60 queued planned turns and 43 parked reads cost 124 us a tick with READ_LINE against 83 us with the
  switches off (it was 800 against 115); 120 turns and 100 reads cost 210 against 148 (it was 3,070 against 200).
  Test: TestReadLine.test_the_queue_check_counts_the_line_once_a_turn_and_not_the_turns_own_read.
- 2. Logic 2, default-off 1, style 1 (bug). Pricing a host eviction victim adopted nodes into sessions, so the prices
  log alone changed which leaf eviction took under return_chance. `ReturnChanceStrategy.holding_rate` now resolves the
  session without adopting (`_session_registration(node, adopt=False)`). `FullComponent.next_host_victim` ranks
  through a new read-only `EvictionStrategy.peek_priority` (evict_policy.py; return_chance overrides it) and keeps the
  session-ref order. On the fixed copy the default-off reviewer's four checks pass, the real tree among them (its
  fifth test patches the old function signature and no longer applies), and the style reviewer's two pass. Tests:
  TestPricingLeavesEvictionAlone (the stub tree with the observer a prefetcher binds when only the log is on; naming
  the next victim; a real CPU tree),
  TestHostEvictionObserver.test_the_next_victim_follows_session_refs_as_eviction_does,
  TestHoldingRate.test_return_chance_rate_is_the_sessions_whichever_of_its_nodes_goes.
- 3. Logic 3 (bug). After the starved-GPU rule gave a read up, its 0.5 s wait did not start again when admission took
  the dropped turn within the pass. A drop now restarts the wait. The reviewer's script shows one drop at 1.600 s and
  none in the next 125 ms (it was five in 100 ms). Test:
  TestRecomputeWhenStarved.test_after_a_drop_the_queue_must_starve_for_another_half_second.
- 4. Logic 4 (bug). The starved-GPU rule ran before the pass's arrivals were registered, so a turn was weighed by the
  deadline of the staged read it had joined. `_gpu_starved` now keeps the wait before the queue loop and
  `_relieve_starved_gpu` acts after it, when every parked read has the deadline dispatch orders it by. The reviewer's
  turn x keeps its read whichever tick first sees it. Test:
  TestRecomputeWhenStarved.test_a_turn_is_weighed_by_the_deadline_it_has_once_it_is_queued.
- 5. Logic 5 (bug). The mark of where storage ends outlived a match that grew past it. The gate now drops the mark
  when the match is past it, and the queue check drops it when memory reaches past it (`_probe_missing`). The
  fifo-order shortcut before the probe uses the length without the mark (`_queued_chain`). Test:
  TestFreeEmptyLookups.test_a_match_that_grew_past_the_mark_opens_what_lies_beyond_it.
- 6. Logic 6 (nit). Turns that arrived to an empty queue (predicted wait 0 by rule) filled the error bin the hold
  reads for waits under 2 s. `PriceMeter.observe_admission` now leaves them out; readers still feed the landing lag.
  With the reviewer's numbers the margin at a predicted 1.5 s is +1.02 s at q = 0.05 and +0.72 s at q = 0.1 (it was
  +0.19 s and 0.00 s). The err_ fields of the prices line leave these turns out too. Test:
  TestMeasuredPrices.test_the_admission_error_is_fed_by_turns_that_read_nothing_with_readers_apart (rewritten).
- 7. Logic 7 (nit). The frontier price fed on itself through kappa's pending-write term. The restore cost behind lam
  is now price-free, c x (1 + late reads) x read seconds per token (`PriceBoard.kappa_of_reads`); the logged kappa
  keeps both terms. This departs from 01_PR1_prices.diff and follows DESIGN.md section 7. In the reviewer's
  scenario lam stays at the price-free value at all four loop gains (it reached the ceiling of 1e-3 at gain 1.95).
  Test: TestMeasuredPrices.test_the_room_price_does_not_feed_on_the_writes_it_prices.
- 8. Logic 8 (nit), in two parts. (a) A queued turn's history now sets a price at the frontier, by the wait it has
  left (`queued_holding_rate` in next_use_eviction.py). Under queued_order farthest the victim is the last arrival,
  which waits for the whole queue; under lru it is the turn nearest admission, so the top rate. I did not use the top
  rate for both, as the finding proposed: campaign 40's unified arm runs queued_order farthest, where the top rate
  overstates the index by wait / 0.5 s (40 times at a 20 s queue) and would push the hold's quantile to its upper
  clamp. This also departs from the plan, which left queued victims unpriced. (b) MEASURED_PRICES is refused at
  construction when the eviction policy gives no holding rates and a staged or held read would take the room price.
  Tests: TestMeasuredPrices.test_a_queued_turns_history_is_priced_by_the_wait_it_has_left, two new cases in
  TestHoldingRate, TestOwnerSwitches.test_measured_prices_need_an_eviction_policy_that_prices_its_holders.
- 9. Default-off 2 and 3 (test gaps). TestPricesLogOnly runs one script (a saturated spell, a held read, a staged
  read, a refused read) with and without the log and compares every start, hold time, harm charge and line. It also
  pins the switched-off field lists of ec_start and ntp_hold. It catches the reviewer's five mutants.
- 10. Default-off 4 and style 12. LATENESS_WEIGHT and READ_LINE are now gated on mode on, so the docstring of
  next_turn_switches.py is true. This departs from the pseudo-diffs, which left both ungated without meaning to: under
  03_F2 the prices log would have decided whether the lateness weight applies in observe. Tests:
  TestOwnerSwitches.test_in_mode_observe_only_the_prices_log_works; price learning 'on' is now refused in a test
  beside 'harm'.
- 11. Style 2 to 10 (test gaps): 27 more cases, each red on the reviewer's mutant. Free empty lookups: a match past
  the key, a hit under the threshold, the ntp_lookup line, a free room retry with ROOM_RETRY on. Measuring: a landing
  under next_use, landings of unknown requests, unplanned queued turns, a reader with no landing seen, aborts.
  Measured prices: the frontier price in a staging, the half-second re-pricing and the ceiling of the next victim,
  return_chance, stale queued registrations, the harm at a fitted cost other than the seed, the fit's forgetting. The
  prices line: its full ordered field list, hold counts per line, the fifo read order. The starved rule: a fresh ready
  turn, the wait restarting, unmeasured lags, the write hold, no refetch after a drop. The read line: the write hold
  for a timed read and not for one needed now, ntp_due's pages for pending and queued turns, a timed read due earlier
  under expected-cost staging. The gate: arrivals in mu, the line in the read time. TestGateOnTheRealCache: the prompt
  gate and the prices log on the real CPU cache.
- 12. Style 11, 14, 16, 18, 19. The unused `import re` is gone. The design is cited by title, without section numbers.
  The six functions named take their later arguments by keyword only. The three stale docstrings are current, and the
  prefetcher, not the meter, now binds the eviction observer and the lag measuring, so the meter only reads. 'today'
  reads 'without the switch'.
- 13. Style 15, in part. The helper docstrings named are cut to two lines (one keeps three, for a tuple of three
  values), the meter's two are now comments, and the tests added in this round carry two-line docstrings.
- 14. Style 17, in part. The cache's gate call and lookup report are two small methods (`_prefetch_gate_allows`,
  `_report_storage_lookup`), so the two long functions now grow by 2 lines and 1 line net.

## Rejected, with reasons

- Style 13 (the class-level attributes prefetch_gate_prompt_fn, prefetch_lookup_fn and read_lags): kept. Set in
  __init__ instead, the unmodified base test_storage_read_dispatch.py fails on the new engine, because its fixture
  builds the cache without __init__; that breaks 'every existing test passes unmodified'. A comment in
  unified_radix_cache.py now says why the three are class-level.
- Style 20 (no verb token in the seven switch names): kept. The names follow 00_environ.diff, the neighbouring
  SGLANG_HICACHE_NEXT_TURN_ROOM_RETRY and DESIGN.md section 11, and
  /home/wanhr/sglang/agent_cache/results/campaign35/unified.env and
  /home/wanhr/sglang/agent_cache/results/campaign40/settings.env already use them.
- Style 2, item 4, first half (a test for the free room retry clearing lookup_charged): not written, because the line
  cannot be reached with the mark set. A retry needs room_refused; only a refusal sets it; a refusal follows a found
  answer, and every reported answer clears the mark; any later charged call clears room_refused. The line is left in
  place.

## Test results

- All counts are from the final code, saved in
  /home/wanhr/sglang/agent_cache/results/design_unified_1126/impl/A/fix/final_tests.out. The test files are under
  /home/wanhr/sglang/agent_cache/frozen/impl_unified_A/test/registered/unit/mem_cache/.
- test_next_turn_prefetch.py: 144 passed, 90 subtests passed (117 and 80 before the fixes; 77 and 55 at the base).
- test_next_turn_estimators.py: 40 passed (39 before the fixes; 14 at the base).
- test_next_use_eviction.py: 31 passed, 4 subtests passed (25 and 2 before the fixes; 21 and 2 at the base).
- test_storage_read_scheduler.py: 11 passed (not changed in this round; 9 at the base).
- test_storage_read_dispatch.py: 33 passed, 2 subtests passed (not changed in this round; 28 at the base).
- test_host_census.py: 7 passed, 9 subtests passed. test_hicache_nixl_cleaner.py: 14 passed, 12 subtests passed.
  test_hicache_nixl_storage.py: 21 passed, 1 skipped. test_evict_policy.py: 23 passed. All four unchanged.
- No test file was added to the copy: the 34 new tests joined test_next_turn_prefetch.py (27),
  test_next_use_eviction.py (6) and test_next_turn_estimators.py (1).
- The base snapshot's unmodified test files (/home/wanhr/sglang/agent_cache/frozen/impl_unified_base) on the fixed
  engine: 77 (55 subtests), 14, 21 (2 subtests), 9, 28 (2 subtests), 7, 14, 21 and 1 skipped, 23. These are the
  baseline counts.
- Switches-off replay (the engineer's
  /home/wanhr/sglang/agent_cache/results/design_unified_1126/impl/A/checks/test_switches_off_equivalence.py, run on
  the final code; output in fix/equivalence_after_fixes.out): 3 passed. Switches off: 40,628 lines of 17 kinds and
  48,226 answers equal to the unchanged module's. Only the prices log on: equal apart from 26,603 prices lines and two
  fields on 3,814 ntp_admit lines.
- Mutation check (/home/wanhr/sglang/agent_cache/results/design_unified_1126/impl/A/fix/zz_fix_mutations.py; it
  re-compiles one function in memory and edits no file; output in fix/mutations.out): 93 regressions, 90 caught.
  Caught: all 16 behaviours from before this round's fixes, all 33 of the engineer's, all 7 of the default-off
  reviewer's and 34 of the style reviewer's 37. The 3 that survive were expected to: two change no behaviour, and one
  differs only after a terminated lookup (see what is left).
- Reviewers' reproduction scripts on the fixed copy (output in fix/reviewer_scripts_after_fixes.out and
  fix/fix_checks.out): eviction order equal with and without the log, in one pass and in two; one starved-rule drop
  where there were five; the arriving turn's read kept in both orders; lam at its price-free value at all four loop
  gains; the hold margin at a predicted 1.5 s back at +1.02 s and +0.72 s.
- 32 other mem_cache test files that touch the changed modules (fix/wider_copy.out): 24 pass and 1 is all skipped. The
  other 7 fail with the same tests on the base tree (fix/wider_base.out): test_hicache_pp_sync_drain (1),
  test_pp_prefetch_ticket (2), test_rust_tree_core (5), test_decode_retraction_backup (10),
  test_unified_radix_lock_ref (1), test_radix_cache_unit (1) and test_rust_tree_core_integration (collection error).
- Not run: test_unified_radix_cache_unittest.py (it needs an accelerator), and nothing on a GPU or a server.

## Files changed in this round

- /home/wanhr/sglang/agent_cache/frozen/impl_unified_A/python/sglang/srt/mem_cache/next_turn_prefetch.py -- this round
  +136 -80, now 2,191 lines. New: _gpu_starved, _queued_chain, _probe_missing (both replace _queued_baseline),
  _queued_line_seconds, _seconds_in_line. Changed: tick (the starved rule acts after the queue loop), _check_queued
  and _wants_refetch (the line is counted where it is logged), prefetch_gate (the absent mark), _relieve_starved_gpu,
  _init_switched_changes (binds the eviction observer and the lag measuring), docstrings.
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_A/python/sglang/srt/mem_cache/next_turn_meter.py -- +65 -40, now
  343 lines. A price-free restore cost, queued victims priced by their wait, admissions predicted 0 left out, follow()
  takes rate=, keyword-only arguments, and no hook is bound here any more.
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_A/python/sglang/srt/mem_cache/next_turn_prices.py -- +19 -14.
  PriceBoard.kappa_of_reads; mu_for and quantile_of take keywords; the design cited by title.
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_A/python/sglang/srt/mem_cache/next_turn_switches.py -- +33 -11.
  LATENESS_WEIGHT and READ_LINE gated on mode on; the new priced_eviction argument and its check; the docstring.
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_A/python/sglang/srt/mem_cache/next_use_eviction.py -- +27 -7.
  _session_registration(adopt=), ReturnChanceStrategy.peek_priority, a read-only holding_rate, queued_holding_rate.
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_A/python/sglang/srt/mem_cache/evict_policy.py -- +5, new in copy
  A's change set. EvictionStrategy.peek_priority, which defaults to get_priority.
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_A/python/sglang/srt/mem_cache/unified_cache/components/full.py --
  +10 -2. _peek_host_priority; next_host_victim ranks through it.
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_A/python/sglang/srt/mem_cache/unified_radix_cache.py -- +39 -19.
  _prefetch_gate_allows and _report_storage_lookup taken out of the two long functions; _observe_read_landing takes a
  keyword; the comment on the class-level attributes.
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_A/python/sglang/srt/environ.py -- +2 -2. The comment of
  SGLANG_HICACHE_NEXT_TURN_MEASURED_PRICES names what the switch needs.
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_A/test/registered/unit/mem_cache/test_next_turn_prefetch.py --
  +694 -54, now 3,880 lines. 27 tests added; two rewritten (the admission error, the price off host eviction); three
  fixtures changed (TestReadLine and TestReadLineByExpectedCost setUp, the harm test); `import re` removed.
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_A/test/registered/unit/mem_cache/test_next_use_eviction.py --
  +187 -4. 6 tests added, one changed, and the TestHostEvictionObserver fixture given an eviction strategy.
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_A/test/registered/unit/mem_cache/test_next_turn_estimators.py --
  +13 -2. 1 test added; mu_for called by keyword.
- Not changed in this round: hybrid_cache_controller.py, next_turn_estimators.py, next_turn_policy.py,
  storage_read_scheduler.py, unified_tree_core.py, test_storage_read_dispatch.py, test_storage_read_scheduler.py.
  managers/scheduler.py, managers/tokenizer_manager.py and model_executor/model_runner.py are byte-identical to the
  base copy.
- /home/wanhr/sglang/agent_cache/results/design_unified_1126/impl/A/fix/ -- notes folder, not part of the engine.
  zz_fix_mutations.py and zz_fix_checks.py (my checks); fixes.diff (this round's diff, 12 files); before/ (the 18
  changed files as I received them, 1 MB); the outputs final_tests.out, equivalence_after_fixes.out, mutations.out,
  reviewer_scripts_after_fixes.out, fix_checks.out, wider_copy.out, wider_base.out.

## Left

- REPORT.md at /home/wanhr/sglang/agent_cache/results/design_unified_1126/impl/A/REPORT.md was not updated by me (a
  subagent may not write report files). It still holds the engineer's report as the main session saved it at 22:53
  UTC; this output is the update.
- Nothing was run on a GPU or a server. The fixes are exercised by CPU unit tests on fakes, on the real cache fixture
  and on a real CPU tree only.
- The merge staged in /home/wanhr/sglang/agent_cache/frozen/impl_unified_merged predates this round and must be run
  again before frozen/engine_unified_20261126 (campaign 40's engine, not built yet) is frozen. Copy A's change set
  gained evict_policy.py. The places where A and B add at the same spot (the test files' imports, the lines before
  _end_session in next_turn_prefetch.py, environ.py after SGLANG_HICACHE_NEXT_TURN_ROOM_RETRY) still need a read after
  the union merge.
- Queue lines: MEASURED_PRICES needs --hicache-storage-next-turn-price-learning off (campaign 34's ec arms pass harm),
  and now also next_use or return_chance eviction wherever reads are staged by expected cost or held by quantile.
  Campaign 40's unified_load line meets both.
- File size (style 17): next_turn_prefetch.py is 2,191 lines (2,135 before this round), test_next_turn_prefetch.py
  3,880 and environ.py 2,013; prefetch_from_storage and the drain are still far over 100 lines. Splitting the
  prefetcher means moving code copy B also edits, so it is left for after the merge, as the reviewer suggested.
- One corner has no test: a refunded lookup clearing the queue check's cap matters only after a refetch whose lookup
  was terminated (no answer, so its mark stays), followed by another trigger's empty lookup and a match that later
  grows past the absent mark.
- Style 15, the rest: the engineer's 54 test docstrings of three or four lines are as they were; the file's older
  tests are written the same way.
- With anything measured (the prices log included) a tick costs about 40 to 60 us more than with the switches off at
  60 queued turns: one read check per queued turn, and the admission rate is worked out twice. Not reduced.
- The starved-GPU rule now gives the GPU at most one recompute per half second. Where the SSD is the bottleneck the
  GPU still idles between them; the wait is the plan's NTP_GPU_BOUND_S and was not tuned.
- The rate of a queued victim is an estimate. Under queued_order farthest it takes the wait of the queue's last turn,
  though the evicted turn can be an earlier arrival when the last one's history is not evictable. Under lru it takes
  the top rate, on DESIGN.md section 5.1's statement that lru takes the turn nearest admission; I did not measure
  that.
- An observe arm that shares the switch block now logs its would-be decisions without the lateness weight and the read
  line.
- The engineer's /home/wanhr/sglang/agent_cache/results/design_unified_1126/impl/A/checks/mutation_check.py no longer
  matches the code (it edits the copy in place by text); fix/zz_fix_mutations.py holds its 33 mutants on the present
  text and edits nothing. The logic reviewer's scripts a and b call the meter's old interface; fix/zz_fix_checks.py
  repeats their scenarios.
- Unchanged from the engineer's list: the scripts under /home/wanhr/sglang/agent_cache/scripts do not read the new
  lines or fields; the constants are picked, not tuned; the prefill fit starts from Triton's 110 us a token; an absent
  mark can still cost a recompute of a span another session wrote while the match stayed below it; there is no
  documentation outside the code.
