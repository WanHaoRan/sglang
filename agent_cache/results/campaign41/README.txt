Campaign 41: the whole unified four-action design against the current one, ordered as part of the author's request of
2026-11-26 (design notes: agent_cache/results/design_unified_1126/DESIGN.md; slides: "Four Actions, Three Prices").
Setting: campaign 38's setting (campaign 35's with the SSD cleaner's marks at 50 and 40 %); settings.env is that campaign's with the load hold
(SGLANG_HICACHE_PREFILL_LOAD_HOLD_STEPS=16) and the ten cache-rule switches added. The launcher takes switches per
campaign folder, which is why this arm has a folder of its own; its baseline is campaign 38's ec_full and unified.
Arm: queue.txt (unified_load, engine frozen/engine_unified_20261126).

2026-11-27 08:51:32Z: unified_load DONE: 17,619 requests, 0 errors, 7 cleaner passes (the first at minute 55; the
current design's at minute 50). Results (compare_unified/B_cleaner50/, checks/vs_c38/pair_windows.out):
  Mean TTFT 9.07 s against 11.56 s for the current design (campaign 38) on 15,370 paired requests: -2.49 s
  [-2.76, -2.21]. Own requests 16,040, 145.8 a minute (139.7); p50 7.20, p90 21.01, p99 50.07 s.
  Recomputed history once the cleaner deletes: 0.27 / 0.20 / 0.51 % of prompt tokens in minutes 50-70 / 70-90 /
  90-110, against 1.37 / 1.38 / 1.75 % in the current design; 1 SSD read failed against 9. The cleaner deleted as
  many groups (about 80,000 a pass).
  By window (paired): minutes 0-28 -0.02 s; 28-50 -0.89 s [-1.00, -0.78]; 50-70 -3.16 s [-3.42, -2.89]; 70-88
  -4.83 s [-5.08, -4.59]; 88-110 -4.44 s [-5.12, -3.76].
