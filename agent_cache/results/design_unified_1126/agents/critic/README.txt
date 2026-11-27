critic: completeness check of DESIGN.md (2026-11-26). CPU only, read-only on the runs and the repository.

Run from scripts/. PY3 = python3 -I -B (stdlib); NP = /home/wanhr/sglang/agent_cache/baselines/continuum/.venv/bin/python -I -B
Every command: prefix with  CUDA_VISIBLE_DEVICES= nice -n 10

xlib.py            loaders. `PY3 xlib.py c34:ec_full ...` streams <run>/server.log once and caches ../ext/<camp>_<arm>.pkl
                   (every HICACHE_EVT line, the batch lines, ReqTimeStats, cleaner passes). Arms come from the campaigns' manifests.
gpu_time.py        NP. Copy times as logged (h2d, d2h, GPU eviction) per window, and an OLS of the 40-step decode-log
                   intervals on decode steps, prefill batches and copied tokens.      -> out/gpu_time_c34_ec_full.out
load_block.py      NP. Per prefill pass that loads history: pass time (h2d_submit to the 'Prefill batch' line) against new
                   tokens and the copy's own time; blocked time per window.            -> out/load_block_<camp>_<arm>.out
load_block2.py     NP. The model 'pass = max(d, C) + F' checked on the decode-log intervals. -> out/load_block2_c34_ec_full.out
gpu_budget.py      NP. Wall time at saturation = decode steps + prefill passes; what the prefill forward must cost per new
                   token if the copy blocks fully, half or not at all.                 -> out/gpu_budget.out
pool_use.py        PY3. GPU pool use on batch lines with a queue; shape of the h2d copy rate. -> out/pool_use_c34.out
hol.py             PY3. Unused GPU pool tokens against the ready queued turns that would fit (open loop). -> out/hol_c34.out
gpu_sticky.py      PY3. Turns that return before the next admission after their previous finish. -> out/gpu_sticky_c34.out
lookups.py         PY3. Lookup latency by window, result and size.                    -> out/lookups_c34.out
lost_bins.py       PY3. Lost history per request by size; first requests' shared prefixes by tier. -> out/lost_bins_c34.out
misc_counts.py     PY3. Pin ends and durations, cuts, refusals, window and budget state at dispatch, census holders,
                   prefix-shrank events, over-eviction, re-evictions, gate decisions. -> out/misc_counts_c34.out
cleaner_pass.py    PY3. Decode step, 40-step interval, read and write ms per page before, during and after each cleaner pass.
                                                                                       -> out/cleaner_pass_c34.out
step_overhead.py   NP. Pure decode step per arm at the same batch shape.              -> out/step_overhead_c34.out
(inline)           out/wear.out (bytes written per run), out/subagents.out (subagent streams, hint coverage)

ext/ holds the caches (about 210 MB; rebuilt by xlib.py).
