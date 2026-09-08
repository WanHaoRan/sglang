# Pending after the main sweep

1. `rerun_exp2_baseline.sh qwen8b` — the qwen8b baseline sweep ran before the
   writeload SIGTERM fix, so `achieved_rps` is 0 on all 21 rows. Every other
   column in it is valid; only the rate axis needs re-measuring. The 32B and
   70B legs already have the fix.
2. Revert the C2 patch to `python/sglang/srt/managers/cache_controller.py`
   (`git checkout` that one file). Diff archived at `exp2/C2_backup_skip.patch`.
3. C1 redo (best effort, low priority): `--hicache-size 30`, a 40 GiB tmpfs,
   and a write rate whose total volume over the block stays under the tmpfs
   size. The specified form is not runnable here -- see FINDINGS_LIVE.md F11.
   C2 already answers C1's question without the confound, so this is a
   nice-to-have rather than a gap in the evidence.
4. The qwen8b baseline + timeout sweeps predate the paired recompute-under-load
   control, so they lack it. The re-run in item 1 covers baseline; the timeout
   sweep would need one too if the paired number matters there.
5. Re-run C3 for all three models with `c3_pyspy.sh`. The in-driver version
   failed twice: `py-spy top --duration` is not a py-spy 0.4.2 flag, and it
   attached to the launcher pid (all threads idle) rather than the scheduler
   subprocess. The replacement uses `record -s --gil`, which answers the plan's
   question directly. Needs a live server, so run it per model.
6. Harness fix (not blocking): `exp3.py` server tag should include MODEL_KEY --
   the three models currently share `exp3_<condition>/` and overwrite each
   other's server logs. Results are unaffected. See FINDINGS_LIVE.md F20.
7. **HIGHEST PRIORITY** — re-run Exp 2 with a non-replaying write load.
   `writeload.py` defaults `--seed 7` / `--pool 64` and `exp2.py` passes
   neither, so each rate point replays earlier points' prompts, which are by
   then in L3. The "write load" becomes a read load at high rates (51 prefetch
   hits of 4032 tokens = 4.09 GB/s during the R=8 probe, with iostat_w at
   0.02 MB/s). Add `--seed` per invocation and an `--idx-offset`. Until then the
   high-rate end of the Exp 2 rate axis conflates write rate with accumulated
   L3 read demand. See FINDINGS_LIVE.md F23.
8. Re-size the Exp 3 client working set past L1+L2 so L3 is actually read.
   As specified, Exp 3/4 cannot show a write-policy benefit on any model
   because dead fraction is 1.0 for capacity reasons (F21).
9. C3 re-run on 32B and 70B with `c3_dump.sh` (the 8B has 12 samples; the
   `c3_pyspy.sh` record-based version hangs and should not be used).
