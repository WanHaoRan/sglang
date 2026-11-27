Campaign 38: the unified four-action design against the current one where the SSD's cleaner works for most of the
run, ordered as part of the author's request of 2026-11-26 (design notes in agent_cache/results/design_unified_1126/).
Setting: campaign 35's (128 users, 160 GB host pool, 2 h), with the cleaner's marks at 50 and 40 % of the SSD volume
instead of 90 and 80 %: the first pass comes near minute 50 instead of minute 88. In campaigns 34 and 35 two thirds
of the history the best arms still recompute comes from pages the cleaner deleted under live sessions, all of it in
the last 22 minutes; here that phase is an hour long.
The marks are set by the last flag of each queue line, which overrides the launcher's l3_extra.json (same keys, two
values changed). run.txt therefore still says l3_cleaner_pct=90,80; the server log's "HiCacheL3Cleaner started" line
has the marks in force.
Arms: queue.txt, ec_full first (the engine and flags of campaign 35's). Run by campaign35/chain_multi.sh.
2026-11-26 21:57:43Z: ec_full done (16,789 requests, 0 errors, 8 cleaner passes, the first at 20:46:29Z, minute 50
of the client's run; "HiCacheL3Cleaner started: high=50.0% low=40.0%"). Against campaign 35's ec_full (same users,
same GPU unit, cleaner from minute 88): mean TTFT 11.56 s against 10.97 s; paired on 15,356 requests +0.77 s
[+0.61, +0.92]; minutes 60-110 22.34 s against 20.80 s; prompt tokens recomputed there 7.0 % against 6.2 %
(checks/vs_c35/compare_arms.out). This is the cleaner's cost measured directly: the two runs differ only in when
the cleaner starts.

2026-11-27 16:08:43Z: unified DONE (the cache rules without the hold): 17,209 requests, 0 errors, 7 cleaner passes.
Results (compare_unified/B_cleaner50/, checks/unified_vs_ec_full/pair_windows.out):
  Mean TTFT 10.56 s against 11.55 s for ec_full on 15,366 paired requests: -0.99 s [-1.18, -0.81] (the simulator:
  -0.93 s). By window: minutes 0-28 -0.02 s; 28-50 +0.00 s; 50-70 -1.01 s [-1.19, -0.83]; 70-88 -2.14 s [-2.37,
  -1.91]; 88-110 -2.31 s [-2.97, -1.65]: the gain starts with the cleaner at minute 50. With the hold (campaign 41)
  -2.49 s; the hold on top of the rules -1.45 s [-1.62, -1.28].
