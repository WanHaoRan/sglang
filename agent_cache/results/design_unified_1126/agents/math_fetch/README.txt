math_fetch: checks of DESIGN.md sections 1.5, 2, 3 and result 1 of section 7 (2026-11-26). CPU only, read-only on the repository.

Run any script with
  CUDA_VISIBLE_DEVICES= nice -n 10 /home/wanhr/sglang/agent_cache/baselines/continuum/.venv/bin/python -I -B <script>
Each script prints to stdout; the saved output is <script>.out next to it.

fetchlib.py                    shared helpers: the engine's two-piece lognormal, stop-loss M(t), J_F, loader of next_turn_policy.py (read-only)
check_derivative.py            J_F with recourse: closed form and derivative, deterministic and random W, defective law with a drop time (Monte Carlo)
threshold_counterexamples.py   when the threshold rule is optimal: three crossings yet optimal; three crossings and not optimal; fail-fast tool
landing_check.py               after arrival: landing quantile, random read time, fixed late charge, the 2 s margin as a quantile, moving estimate
hazard_trace.py                hazard of the tool and human entries (engine model) and of the replayed gaps (campaign 34 ec_full client log)
jf_grid.py                     J_F on the two entries: L = 25 K, r = 2..6 s, W = 0/2/5 s, c = 1/0.3/0, lambda = 0/1e-6/2e-5/1e-4 -> jf_grid.csv
fetch_vs_retain.py             identity G_keep - G_fetch = K_net S(s) - (c + lamL) [M(s) - M(s+x)]; where Retain keeps and where a read pays
hints_replay.py                the 16,118 hints the replay sent: entry shapes, calibration, realized priced cost of each start rule
engine_artifact.py             the engine's 64-point support: read_start never returns None for x > 0; the cap's atom
read_or_recompute.py           K_read against K_rec, crossover lengths at mu = 1, 2, 5, 128; hybrid restore (recompute head while reading tail)
improvements.py                Weibull closed form; unimodal hazard; value of a hint against its sharpness; point forecast regret; block landing; life cycle
queue_prices.py                toy queues for section 1.5: mu in an open queue, c with pass-over, mu in the closed loop
order_check.py                 section 3.4 by enumeration: EDD, release times, c/size order, mixed case
