# Campaign 20: the p50 of the online table at x10 (first step of DIST_HINT_DESIGN.md)

Requested 2026-10-31: "do the smoke test and evaluation with only the x10 gap with 5h timeout -- the same as the previous
campaign 19 workload. Leave out the x70 case."

**The question:** how much of the oracle's gain at x10 survives when the exact gap is replaced by the p50 of the client's
online table?

**The arm.** `three_tier_wc_oracle_v2_nu` on the frozen engine `frozen/engine_5df71c4bf0_v2/python`, as in campaign 19. The
only change is the client: `HINTS=dist ORACLE_HINTS=0` (`HINT_SOURCE=online`, the default, tool + first word keys, explicit
ends). The engine reads the entry's p50 as the version 1 `next_gap_s` and ignores the version 2 table. No new baseline or oracle
arm: campaign 19's (`../campaign19/`) are the comparison, same users, seed and arrival times.

**The workload.** Campaign 19's: `USERS=768 TURNS=50 ARRIVAL=0.1 GAP=10 C=128 SEED=0 CLIENT_TIMEOUT=18000` on
`traces/lmcache_agentic_trace_262k.json`.

**Cookbook.** `docs/cookbook/autoregressive/Qwen/Qwen3.mdx` read before running. This run stays pinned to campaign 19's server
for comparison, which differs from the page in the usual ways: no `--tool-call-parser` or `--reasoning-parser` (the replay sends
no tools and thinks off), `--context-length 131072`, one GPU, and the HiCache flags the page does not cover.

**Launcher.** `frozen/oracle_scripts_staging/c20_dist/c20_dist_x10.sh smoke|full` (log `launcher.log`). Smoke acceptance is
`c20_dist/check_smoke.py`: 0 errors and tracebacks, every hint with a point has a plan with that gap, no plan without one, and
the list grows during the run.
