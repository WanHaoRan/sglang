Campaign 40: the whole unified four-action design against the current one, ordered as part of the author's request of
2026-11-26 (design notes: agent_cache/results/design_unified_1126/DESIGN.md; slides: "Four Actions, Three Prices").
Setting: campaign 35's setting (128 users, 160 GB host pool, 2 h); settings.env is that campaign's with the load hold
(SGLANG_HICACHE_PREFILL_LOAD_HOLD_STEPS=16) and the ten cache-rule switches added. The launcher takes switches per
campaign folder, which is why this arm has a folder of its own; its baseline is campaign 35's ec_full, campaign 39's ec_full_load and campaign 35's unified.
Arm: queue.txt (unified_load, engine frozen/engine_unified_20261126).

Pre-flight of frozen/engine_unified_20261126 (preflight.log; started by after_c39.sh when campaign 39's arm ended):
2026-11-27 04:00:57Z to 04:22:52Z, first attempt. Unit tests pass (the ten CPU files, the flag check and the GPU
  write/read cases of the unified cache). Tier check with every switch on, GPU pool 100,000 tokens, host 30 GB: the
  server granted every Retain switch (track_stored=1 skip_ended_writes=1 touch_interval_s=300
  retire_ended_pages=1); quiet 5 of 5; with six generations decoding 5 of 5, two batches held and released with the
  copy landed, host and SSD stages exact. Smoke with every switch off (ec_full's flags, cleaner at 3/2 % of the
  volume so that passes run: 18 in ten minutes): 899 requests, 0 errors, no traceback while serving, TTFT mean
  22.47 s (campaign 39's smokes ran the cleaner at 90/80 %, so their 14 s does not compare). The attempt stopped on
  one traceback logged after the server began to exit: the SSD cleaner was mid-pass when the interpreter shut down
  ("cannot schedule new futures after shutdown"; the cleaner's loop is the baseline engine's, line for line). The
  pre-flight now counts tracebacks logged before "Gracefully exiting" only.
2026-11-27 04:29:20Z, second step: the switches-on smoke only (resume_unified.sh; the earlier stages passed and the
  baseline smoke is not repeated). The GPU idled from 04:22:52Z to 04:29:20Z.
2026-11-27 04:42:17Z: pre-flight passed. Switches-on smoke (all eleven switches, the unified flags, cleaner at 3/2 %):
  915 requests, 0 errors, no traceback; 242 batches held; 69 prices lines (the last: c=0.021, mu=46, lam=2.8e-7,
  kappa=0.013 with 46 users waiting); Retain: 48 l3_touch lines, 18 sessions ended by their hint (each retired:
  l3_retire, 0 pages waiting for an ack), 174 sessions forgotten rather than ended (ntp_end explicit=0
  why=no_entry: their next turn may still come), 47 writes of an ended session's last turn skipped, 8 groups kept
  by the cleaner's re-check; 146 SSD reads failed on pages the cleaner had deleted (156 with the switches off). TTFT
  mean 22.14 s against 22.47 s with the switches off (one ten-minute run each, cleaner-bound: not a result).
  Released: unified_load here and in campaigns 41 to 43, unified in campaigns 35 and 38 (their settings.env gained
  the ten cache-rule switches of campaign35/unified.env).
2026-11-27 06:46:50Z: unified_load DONE: 17,636 requests, 0 errors, 3 cleaner passes. Results (compare_unified/
A_128u_160g/: compare_arms.out, mechanisms.out, timeline.png, ttft_cdf.png; checks/vs_c39/pair_windows.out):
  Mean TTFT 9.18 s on the 15,560 requests every arm of setting A sent before minute 110: against the current design
  (campaign 35) -1.79 s [-2.00, -1.59]; against the hold alone (campaign 39) -0.22 s [-0.32, -0.11] on 15,932 pairs.
  Own requests: 16,107, 146.4 a minute (141.5 current, 144.9 hold alone); TTFT p50 7.17, p90 20.55, p99 49.91 s.
  The cache rules on top of the hold, by window (paired with campaign 39): minutes 0-28 -0.001 s; 28-50 -0.03 s;
  50-70 +0.16 s [+0.06, +0.28]; 70-88 -0.14 s [-0.29, +0.00]; 88-110 -1.16 s [-1.58, -0.70].
  Where it comes from (mechanisms.out): once the cleaner deletes (minutes 90-110), recomputed history is 0.32 % of
  prompt tokens against 1.27 % with the hold alone and 1.22 % in the current design, and 1 SSD read failed against
  10; the cleaner deleted as much (81,150 groups against 81,800), but pages in use had been refreshed (109 l3_touch
  lines in the window) and ended sessions' own pages went first (107 retired). Writes fell from 5.3-5.5 to 4.8 M tokens
  a window (183-203 last turns of ended sessions not written). Before the cleaner the rules read more from the SSD
  (9.0 against 5.4 M tokens in minutes 50-70) and recompute slightly more (0.24 % against 0.10 %). The +0.16 s is
  not from those extra reads: campaign 45, with the fixed price, does not read them and costs the same there.
  Measured prices once saturated: mu 62 to 107, c 0.01 to 0.07, lam 1e-7 to 7e-7 per token-second (the current
  design's fixed price of host room is 5e-5).
  The simulator's prediction for the cache rules at these users was -0.28 s [-0.40, -0.17] (seed 0, patched); the
  hold and the cache rules add up (-1.61 and -0.22, against -1.79 together).
