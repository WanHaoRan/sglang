# HiCache backend A/B, Exp 0 and Exp 1 rerun on an A100 box: Qwen3-8B

Run 2026-09-17 on GCP `a2-ultragpu-1g` (A100-SXM4-80GB, local NVMe). Reference: `20260907_203035/nixl`.
Every number is in `COMPARISON.md` / `comparison.csv` (`scripts/compare_campaigns.py`, same statistics on both
sides; against the old data alone it reproduces the old P, bar, L3 rate and paired-difference CI exactly).
What differs from the old run is in `DEVIATIONS.md`. The three-model picture is in
`../20260917_a100_gcp_32b70b_fp8kv/REPORT.md`.

**Validity.** 84 Exp 1 rows (21 per tier), 0 discards, exact prompt lengths, every tier probe a full hit on the
intended tier, no contaminated recompute row; Exp 0 attribution 4032 / 4032 / 4032 with 128 L3 files totalling exactly
4096 x 147456 B; the nixl cleaner never fired.

| | old (H100, virtio disk) | new (A100, local NVMe) |
|---|---:|---:|
| device pool / host pool (tokens) | 374,784 / 678,208 | 376,000 / 678,208 |
| P, marginal prefill (tok/s) | 15,011 | 8,858 |
| recompute bar b*P (GiB/s) | 2.061 | 1.216 |
| L1 hit at 16384 tokens (s) | 0.159 | 0.184 |
| L2 delivered (GiB/s), and over the bar | 12.87, 6.2x | 7.38, 6.1x |
| L3 delivered (GiB/s), and over the bar | 3.277, 1.59x | 0.625, **0.51x** |
| L3 hit / recompute at 16384 and 32512 tokens | 1.17, 0.66 | 2.44, 1.93 |
| backend A/B, L3 read: nixl / file (GiB/s) | 2.84 / 0.083 (34x) | 0.655 / 0.344 (1.9x) |

**Result.** On the old box an L3 hit beat recompute only at the longest length (0.66x at 32512 tokens; it
still lost 1.17x at 16384), although the two fitted lines cross at about 9k tokens. Here it never does: the
slower GPU and the attention-kernel change it forced (flashinfer instead of fa3, not separated here) lowered the
bar by 1.7x, but L3 delivers 5x less than it did on the old box (0.625 vs 3.277 GiB/s; this SSD saturates at
about 0.68 GiB/s read), so L3 delivers half the bar and a hit is 1.8x to 2.9x slower than recomputing at every
length. L2 is as worthwhile as before (6x the bar). The old 34x advantage of nixl over the reference file backend
shrinks to 1.9x. nixl is capped by the device (0.655 GiB/s against that 0.68 GiB/s ceiling). The file backend
got 4x faster (0.083 to 0.344 GiB/s) although this disk is slower than the old one: it is software-bound on both
boxes (here a hit with a warm page cache still takes 1.46 s of the 1.61 s), and the cause of its speed-up was not
isolated.

Exp 2, 3 and 4 were not run (cancelled before they started; `DEVIATIONS.md` D8).
