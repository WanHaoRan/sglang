The run check of defect 2's fix (OPEN_ITEMS.md §2; commit 94c3dd0c85, engine frozen/engine_queuedfix_20261122):
campaign 29's jit_exact arm (TraceLab, the same users and seed, Triton like campaign 29) for its first 30 minutes on
the fixed engine. Started by chain.sh after the two oracle arms; run by baselines/common/run_baselines.sh qfix_check.

Result 2026-11-22 (05:38 to 06:14Z, 949 requests, 0 errors): passed. Among requests that reached the server in the
first 30 minutes, allowance used up 55 to 70 s after arrival: 88 in campaign 29, 6 here; of them refused a read: 57
and 1; refused a read in all: 62 and 23. Paired TTFT on the requests both sent: -1.1 s [-2.0, -0.3] before 1,200 s,
-6.0 s [-7.7, -4.4] before 1,500 s.

Files: signature.py (the counts; usage in its docstring) and its output signature.txt; compare/ holds the pair's
settings.env and manifest.txt for frozen/oracle_scripts_staging/c28_users/compare_arms.py, output in
compare/compare_arms.txt. Host side, from agent_cache/:
  python3 -I -B results/qfix_check/signature.py results/<run>/server.log 1800
  python3 -I -B frozen/oracle_scripts_staging/c28_users/compare_arms.py results/qfix_check/compare [--cut 1500]
