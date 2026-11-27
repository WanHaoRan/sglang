ev_prices: checks of DESIGN.md sections 1.5, 2 and 6 on campaigns 34, 33 and 32 (CPU only, read-only on the runs).

Run from this folder's scripts/ directory. PY3 = python3 -I -B (stdlib); NP = the numpy venv:
  NP="/home/wanhr/sglang/agent_cache/baselines/continuum/.venv/bin/python -I -B"
Every command: prefix with  CUDA_VISIBLE_DEVICES= nice -n 10

lib.py            loaders; `PY3 lib.py c34:ec_full ...` streams <run>/server.log once and caches ../ext/<camp>_<arm>.server.pkl;
                  lib.requests() joins client.jsonl with the server's lines per request (../ext/*.reqs.pkl)
qstate.py         the waiting queue rebuilt per request (queue, READY queue, users in session); `PY3 qstate.py c34:ec_full`
                  prints the check against the logged #queue-req
readers.py        per request: pass-over time, due (when the GPU could have admitted it), stall, tokens read
prefill_fit.py    NP; per-run FA3 prefill cost, idle and loaded -> ../out/prefill_fit.json, prefill_fit.out
part1_prices.py   PY3; per-minute mu, c, kappa counts, host turnover -> ../out/prices_<camp>_<arm>.csv, part1_*.out
part1_regimes.py  PY3; the same by regime (minute ranges) -> part1_regimes.out
part1_lambda.py   PY3; lambda as the index of the marginal holder, against ec_price -> part1_lambda_*.out
part2_identity.py PY3 (needs prefill_fit.json); Little's-law terms per 5-minute window and run pairs -> part2_identity.out
part2b_minutes.py NP; minute-by-minute regressions of dTTFT on d(recompute GPU s) and d(stall) -> part2b_minutes.out
part3_f4.py       PY3; read or recompute per reading request, rules scored -> part3_f4_*.out, f4_readers_<camp>_<arm>.csv
part3b_externality_check.py  PY3; delay of turns arriving during a long prefill (natural experiment) -> part3b_*.out
part45_recompute.py PY3; prefill tokens by cause, lookups and the cap per half hour -> part45_*.out, lost_requests_*.csv
part2c_queue_split.py        PY3; queue = ready + reading turns per phase and arm (campaign 34) -> part2c_queue_split.out
part3_state_summary.py       PY3; readers by GPU state at arrival, the c readers saw (from f4_readers_*.csv; run part3_f4.py on the
                             whole runs first, which rewrites the CSVs with every reader) -> part3_state_summary.out
part3c_loadback.py           PY3; host-to-GPU load time after admission at light load -> part3c_loadback.out
part4_empty_lookup_stall.py  PY3; queue wait added by a lookup that finds nothing, light load -> part4_empty_lookup_stall.out
Order: lib.py (extract) -> prefill_fit.py -> the rest. Outputs are in ../out/, caches in ../ext/ (safe to delete; rebuilt).
