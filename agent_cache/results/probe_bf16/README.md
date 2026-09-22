Campaign-6 preflight (2026-09-22 01:45Z): Qwen/Qwen3-32B bf16 weights, fp8_e5m2 KV, ctx 40960, no HiCache, natural pool.
- p_measure/ttft_by_tier.csv: recompute sweep, 7 lengths x 3 reps -> P = 2,920 tok/s (fit), T_rec quadratic in DECISIONS.md of compare_20260922_015454
- warm_prefix.jsonl + probe.log: one_batch_server B=1/4/8/16 at 25.6K ctx, 96 % warm -> decode 30.8/31.0/29.0/20.3 tok/s per sequence
- boot: L1 = 468,288 tokens, weights 61.04 GB, READY in 66 s (warm), no Marlin line
