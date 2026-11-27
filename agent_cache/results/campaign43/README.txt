Campaign 43: the whole unified four-action design against the current one, ordered as part of the author's request of
2026-11-26 (design notes: agent_cache/results/design_unified_1126/DESIGN.md; slides: "Four Actions, Three Prices").
Setting: campaign 36's setting (64 users, 88 GB host pool, 1 h); settings.env is that campaign's with the load hold
(SGLANG_HICACHE_PREFILL_LOAD_HOLD_STEPS=16) and the ten cache-rule switches added. The launcher takes switches per
campaign folder, which is why this arm has a folder of its own; its baseline is campaign 36's ec_full.
Arm: queue.txt (unified_load, engine frozen/engine_unified_20261126).

2026-11-27 11:59:27Z: unified_load DONE: 8,363 requests, 0 errors, no cleaner pass. Results (compare_unified/
D_64u_88g/, checks/vs_c36/pair_windows.out):
  Mean TTFT 0.54 s against 0.68 s for the current design (campaign 36) on 6,998 paired requests: -0.14 s
  [-0.20, -0.08] (21 %). p50 0.10 / p90 1.40 / p99 7.26 s against 0.10 / 1.59 / 10.64 s; first requests 1.04 s
  against 1.51 s. The simulator's prediction was -0.14 s (-0.18 with the hold's free copy). Prices stay at their
  light-load values (mu 1.0 to 1.2, c 0.95 to 1.0): no regression where the GPU pool fills only now and then.
