verify_evidence: independent re-checks of the four evidence analysts' numbers (2026-11-26). CPU only.
Scripts are my own (none of the analysts' code is imported). Run with
  nice -n 10 python3 -I -B scripts/<name>.py            (stdlib)
  CUDA_VISIBLE_DEVICES= nice -n 10 <continuum venv python> -I -B scripts/<name>.py   (p06, p13, p14, f05: numpy / engine policy by path)

Build order: build_extracts.py <camp:arm ...> -> runtab.py <camp:arm ...> -> t00_trace_extract.py -> chains.py <camp:arm ...>
  vlib.py        log readers; all times are seconds since the client's t_start (server log = UTC wall clock)
  runtab.py      per-run tables: requests, reads, rebuilt waiting queue (ready = no read of its own under way), GPU turn D and stall per reader
  chains.py      block chains per request from the trace; reproduces ideal_cached for every request
  ssd_replay.py  page-level replay of the SSD tier (writes, logged or modelled cleaner passes, logged lookups and reads)

Checks (output in out/<same name>.txt unless noted):
  c00_clock                  server and client clocks agree
  f01_hazard                 gap hazards, pooled and per key (replayed population and whole trace)
  f02 (inline, out/f02_ceiling_trace.txt)  trace-level exact-gap ceiling, now-or-arrival shares
  f03_regimes                queue rebuild check, c, readers, stall, pre-arrival ceiling by regime
  f04_hold_line              holds, pool at holds, held reads' lateness/pin, read line, planner read time
  f05_f3_alone               expected-cost start given the true read time (engine read_start by path)
  f06_admit_err              at-arrival wait estimate error
  f07_c26_hints              campaign 26 hint sharpness and gap hazards
  f08_who_reads, f09 (inline, out/f09_c33.txt), f10_margin_shift
  w01_writes_basic, w02_op_cost, w03_device_share, w04_false_ends, w05_c32_first_refusals, w06_hold_cost,
  w07_hold_what, w08_readback, w09_ecstart_price, w10_false_end_cost, w11_false_end_ttft
  s01_validate               the SSD replay against the logs (lookups, store size, losses)
  s03_policies               cleaning orders with the logged passes; s06_volumes with a volume model (x1, x0.5, x0.25)
  s04_store_stats, s05_first_holders, s07 (inline, out/s07_live_deleted.txt)
  p01_identity               campaign 32 against 34, closed-loop identity
  p02_lost_history, p03_lookups, p04_cap, p05_loss_classes, p06_costs, p07_f4, p08_jitp50, p09_mix,
  p10_lambda_inputs, p11_prefill_blocks, p12_kappa, p13_minute_throughput, p14_minute_lag, p15 (inline, out/p15_before_pass.txt)
extracts/: parsed server logs (ev_*.pkl), run tables (runtab_*.pkl), chains (chains_*.pkl), trace_gaps.jsonl.gz, trees_played.jsonl.gz
