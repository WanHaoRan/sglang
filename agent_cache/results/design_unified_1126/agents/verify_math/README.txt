verify_math: a hostile re-check of the three mathematicians' results on DESIGN.md (2026-11-26).
CPU only, read-only on the repository; everything here was written from scratch (no code shared with
math_fetch, math_retain or math_order_price). Different seeds, laws and edge cases from theirs.

Run any script with
  CUDA_VISIBLE_DEVICES= nice -n 10 /home/wanhr/sglang/agent_cache/baselines/continuum/.venv/bin/python -I -B <script>
The saved output is <script>.out next to it.

vlib.py        own helpers: two-piece lognormal (survival, hazard, stop-loss E[(G - t)^+] in closed form, capped),
               scipy laws with a mass at infinity, atoms; closed-form integrals of the survival function,
               cross-checked against quadrature (worst gap 3e-10)
vf_fetch.py    math_fetch R1-R6: J_F closed form and derivative (750 cells, exact standard errors, atoms, x from
               1 ms to 1,000 s, lambda = 0, c = 0, infinite-mean law); random queue wait, with the wait observed
               at arrival or not; threshold rule on random multi-mode laws; drop time and mass at infinity;
               Fetch-against-Retain identity and a counterexample to its item (4); landing quantile, dependent
               read time, fixed late charge, moving estimate
vf_trace.py    math_fetch R7-R9, R13: hazards of the two trace entries; life-table hazard of the replayed gaps;
               the J_F table on the two entries; the 16,118 hints of campaign 34 ec_full (all of them, no
               sampling); the engine's read_start loaded read-only by path
vo_queue.py    math_fetch R11, math_order_price 1, 2, 3, 7, 8: open-queue weight (Lindley), the convexity lower
               bound, the cost of a stall under pass-over (new closed form 1/E[exp(-lambda s)]), closed loop
               with one server, pool-bound batch server, the three-user counterexample
vo_finite.py   finite injections in the closed loop over 400 independent paths per cell
vr_retain.py   math_retain R1-R8: index against backward induction on 120 lattice laws x 29 prices, hull form,
               closed forms, the relaxed many-session problem solved as an LP, offline brute force under a hard
               capacity, the queue-aware index by simulation, the index with early re-fetch
vr_write.py    math_retain R9-R16: three-action backward induction for a dirty page, the several-gap probe,
               due date and lateness cost, order of writes against an exact assignment solver, the share of
               writes ever needed, mirror arithmetic, SSD chains and trees by enumeration
vo_sched.py    math_fetch R10, R12, math_order_price 4, 5, 6, 9, 10 and math_retain R6's simulation: device order
               by enumeration, read-or-recompute rule on every length, covering knapsack, room price as an LP
               dual, the stability model, many sessions under a hard capacity against the priced bound

What differs from the mathematicians' reports (details in the outputs):
- fetch R1 needs a finite mean gap, not only P(gap = infinity) = 0 (vf_fetch.out, R1 last lines)
- fetch R2: '(B) and (C) have the same best start' holds only if nothing about the queue wait is learned at
  arrival; if the wait is observed at arrival the start condition becomes
  Phi(s) >= q (1 + hazard(s) x E[(W - r)^+]) (vf_fetch.out, R2, the bimodal case)
- fetch R5 item (4) needs item (2)'s condition; counterexample in vf_fetch.out, R5
- fetch R9: on all 16,118 hints the hold-to-admission argmin loses 0.3 to 2.4 s (their 4,000-turn sample: 0.8
  to 2.5 s); the model's E[(gap - 2 s)^+] is 15.2 s, not 16.1 s (vf_trace.out)
- fetch R10: the rule 'D0 < 0, otherwise read iff L < ...' is wrong when mu gamma <= kappa/beta (40 of 192
  cells); then recompute wins at every length (vo_sched.out)
- retain R10: the write-or-not test is exact for one gap; over several gaps it only under-writes (39 of 300
  random cases, all in that direction) (vr_write.out)
- order_price 3: closed form for the short-stall cost in an open M/G/1 queue, 1/E[exp(-lambda s)]
  (1 + rho for exponential work, exp(rho) for equal work) (vo_queue.out)
Everything else reproduced.
