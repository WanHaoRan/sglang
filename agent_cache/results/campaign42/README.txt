Campaign 42: the whole unified four-action design against the current one, ordered as part of the author's request of
2026-11-26 (design notes: agent_cache/results/design_unified_1126/DESIGN.md; slides: "Four Actions, Three Prices").
Setting: campaign 37's setting (128 users, 112 GB host pool, 2 h); settings.env is that campaign's with the load hold
(SGLANG_HICACHE_PREFILL_LOAD_HOLD_STEPS=16) and the ten cache-rule switches added. The launcher takes switches per
campaign folder, which is why this arm has a folder of its own; its baseline is campaign 37's ec_full.
Arm: queue.txt (unified_load, engine frozen/engine_unified_20261126).

2026-11-27 10:56:07Z: unified_load DONE: 17,727 requests, 0 errors, 3 cleaner passes. Results (compare_unified/C_112g/,
checks/vs_c37/pair_windows.out):
  Mean TTFT 9.41 s against 12.80 s for the current design (campaign 37) on 15,449 paired requests: -3.39 s
  [-4.09, -2.74] (26 %). Own requests 16,270, 147.9 a minute (140.5). The distribution changes shape: p50 2.99 s
  against 0.26 s, p90 36.7 s against 63.3 s, p99 78.1 s against 113.0 s; 44 % of paired requests are faster. The
  current design served 43 % of the history from the GPU and kept readers waiting behind those requests (13 % of
  requests over 30 s); the unified design serves 22 % from the GPU, 54 % from host memory, and readers sooner.
  By window: minutes 0-28 -0.005 s; 28-50 -0.76 s [-0.95, -0.58]; 50-70 -5.61 s [-7.21, -4.21]; 70-88 -7.25 s
  [-9.48, -5.34]; 88-110 -4.41 s [-6.42, -2.55]; the median of the paired differences is +1.5 to +2.8 s in the last
  three windows.
  Measured prices: c 0.06 to 0.24 and lam 1.5e-6 to 1.6e-5 (both higher than at 160 GB, where host room is looser).
