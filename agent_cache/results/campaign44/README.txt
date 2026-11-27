Campaign 44: the admission pass-over (Preempt on GPU room) on top of the whole unified design, ordered as part of the
author's request of 2026-11-26 ("consider everything no matter how tiny the gain is"; design notes:
agent_cache/results/design_unified_1126/DESIGN.md, section 7 and the lever table of section 10: admission stops at a
ready turn that does not fit, 2 % of the GPU pool idle behind such a head, 0.3 s on the simulator).
Setting: campaign 35's (128 users, 160 GB host pool, 2 h). settings.env is campaign 40's plus
SGLANG_PREFILL_PASS_OVER_LOOKAHEAD=8, so its one arm pairs with campaign 40's unified_load (everything without the
pass-over) by user, stream and index; the launcher takes switches per campaign folder.
Arm: queue.txt (unified_load_po, engine frozen/engine_unified_po_20261127: engine_unified_20261126 plus the
pass-over, scheduler_components/admission_pass_over.py, and class-level defaults of the two scheduler components).
Runs after the six arms released at 04:42Z on 2026-11-27 (after_chain.sh: the running launcher's folders get END, the
pre-flight preflight_passover.sh runs, then a launcher for this folder alone). No baseline is rerun: the comparison
is with campaigns 40 and 35.

2026-11-27 16:12Z: the pre-flight stopped at the unit tests: of the working tree's test files, the one test of the
tie fix (test_next_turn_prefetch.py::TestDeadlineMode::test_turns_first_seen_in_one_tick_are_queued_at_their_own_
arrivals) fails on this engine copy, as it must: the fix was kept out of the copy on purpose. The watcher then
started the launcher for campaigns 44 and 45 with this line held, so campaign 45 ran first (from 16:13:51Z).
after_c45.sh runs the pre-flight again with that test left out once campaign 45's arm ends, and releases this line.
2026-11-27 18:18Z to 18:20Z: the repeat by after_c45.sh stopped on the same test: pytest's --deselect did not match
the node id here, so the test ran. 18:20:50Z: the pre-flight run again by hand with the test left out by name
(PREFLIGHT_SKIP=own_arrivals, pytest -k "not own_arrivals"): every unit test file passes (test_next_turn_prefetch:
152 passed, 1 deselected); smoke from 18:22:41Z; release_after_preflight.sh releases this line on a pass. The GPU
idled from 18:18:03Z (campaign 45's end) to 18:22:41Z.
2026-11-27 18:37:21Z: pre-flight passed. Smoke (every switch on with the pass-over, cleaner at 90/80 %): 1,316 requests,
0 errors, no traceback; 544 pass_over lines, 642 requests admitted behind a refused one, the largest pass count 8
(the bound); 764 batches held; TTFT mean 13.50 s, p50 4.45, p90 36.44. Line released at 18:37:28Z.

2026-11-27 20:42:09Z: unified_load_po DONE: 17,735 requests, 0 errors, 3 cleaner passes. Results (compare_unified/
A_128u_160g/, checks/vs_c40/pair_windows.out):
  Mean TTFT 9.10 s on the 15,560 requests every arm sent: against the current design -1.87 s [-2.13, -1.61]; against
  campaign 40 (everything without the pass-over) -0.02 s [-0.22, +0.17] on 16,026 pairs: a tie. The pass-over moves
  the distribution, not the mean: p50 6.13 s against 7.17 s, p90 21.70 against 20.55, p99 55.07 against 49.91 s.
  By window against campaign 40: minutes 28-50 -0.31 s [-0.61, -0.01]; 50-70 -0.37 s [-0.82, +0.12]; 70-88 +0.12 s
  [-0.32, +0.59]; 88-110 +0.55 s [-0.17, +1.29]. It acted 7,629 times and admitted 8,789 requests behind a refused
  one; 383 refused requests were passed the full 8 times. Own requests 16,130, 146.6 a minute (146.4).
  Reading: letting small turns go ahead of a large one shortens the median and lengthens the large turns' waits; the
  2 % of the GPU pool it recovers does not show in the mean (the simulator's -0.3 s did not appear).
