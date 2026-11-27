Campaign 35: the unified four-action design (Fetch, Preempt, Retain, Price over reads, eviction and writes; design
notes in agent_cache/results/design_unified_1126/) against the current design, ordered by the author on 2026-11-26.
Setting: campaign 34's (settings.env copied), so requests pair with campaigns 32 and 34 by user, stream and index.
Arms: queue.txt. ec_full first, unchanged from campaign 34, as the baseline on this campaign's GPU unit and as a
repeat. Run by chain.sh (log chain.log), which starts run_campaigns.sh campaign35, reads the weights before each
boot and runs arm_checks.sh after each arm. No END file while arms are still to be added: the launcher waits.
2026-11-26 18:3xZ: END written so the first launcher exits after ec_full; chain_multi.sh then starts one launcher
over campaigns 35, 36 and 37 (it removes END again, so that launcher waits for the unified arm's line here and runs
the other two campaigns' baselines meanwhile). Campaign 36: 64 users, 88 GB, 1 h. Campaign 37: 128 users, 112 GB.
2026-11-26 18:5xZ: chain_multi.sh restarted (it was still waiting) over campaigns 35, 38, 37 and 36 in that order.
Campaign 38: campaign 35's setting with the SSD cleaner's marks at 50/40 % (cleaner from about minute 50).
Campaign 37: 128 users, 112 GB host pool. Campaign 36: 64 users, 88 GB, 1 h.
2026-11-26 19:50:34Z: ec_full done (17,093 requests, 0 errors, 4 cleaner passes; arm checks 15 ok). As a repeat of
campaign 34's ec_full (another GPU unit, three days and one host restart later) it gives the run-to-run spread:
mean TTFT 10.97 s against 10.96 s, paired on 15,521 requests +0.01 s [-0.09, +0.11] (checks/vs_c34/compare_arms.out;
by window 0.14/0.14, 0.54/0.50, 6.20/6.18 and 20.80/20.82 s). chain_multi.sh started the launcher over campaigns 35,
38, 37 and 36 at 19:53:22Z; campaign 38's ec_full launched 19:53:47Z.
Pre-flight of the unified engine: preflight_unified.sh <engine folder under frozen/> (unit tests on the frozen copy,
tier check and a 10-minute smoke on small pools; needs unified.env and unified.flags here), run with the GPU free.
2026-11-26 20:47Z: the ec_full lines of campaigns 37 and 36 are commented out (#HOLD) so that the launcher idles
after campaign 38's ec_full (due about 21:58Z); END files then end it, the load-hold engine is pre-flighted
(campaign39/preflight_loadhold.sh) and a new chain_multi.sh starts over campaigns 39, 35, 38, 37 and 36.

2026-11-27 14:03:34Z: unified DONE (the cache rules without the hold): 17,270 requests, 0 errors, 3 cleaner passes.
Results (compare_unified/A_128u_160g/, checks/unified_vs_ec_full/pair_windows.out):
  Mean TTFT 10.58 s against 10.96 s for ec_full on 15,548 paired requests: -0.38 s [-0.51, -0.25]. By window:
  minutes 0-28 -0.02 s; 28-50 -0.14 s [-0.21, -0.08]; 50-70 -0.38 s [-0.50, -0.26]; 70-88 -0.46 s [-0.65, -0.27];
  88-110 -1.03 s [-1.60, -0.40]. Own requests 15,708, 142.8 a minute (141.5).
  Against the hold alone (campaign 39) +1.20 s; with the hold (campaign 40) the rules add -0.22 s instead of -0.38 s:
  the two parts are slightly sub-additive (-1.61 and -0.38 apart, -1.79 together).
