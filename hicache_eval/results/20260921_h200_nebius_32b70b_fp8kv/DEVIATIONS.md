# Deviations from campaign 5 (70B and 32B, fp8_e5m2 KV, nixl) in this rerun

Rerun on 2026-09-21 on a Nebius H200 instance, 20:46-21:55 UTC. Reference: `20260917_a100_gcp_32b70b_fp8kv`
(campaign 5). Same image tag (`nightly-dev-20260907-30705c00`), same commit under `python/`, same server flags
except D5, same client parameters and seeds, same stage order (70B recompute sweep, 70B Exp 0 / Exp 1 / L2 pass,
then the 32B's three stages), same checkpoint snapshots (32B `aa55da1e...`, 70B `64d25562...`). Driver:
`scripts/run_h200.sh`.

| # | what differs | why | effect on the comparison |
|---|---|---|---|
| D1 | H200 (SM90, 143,771 MiB, PCIe Gen5 x16) instead of A100-SXM4-80GB (SM80, Gen4); 16 vCPU / 196 GB RAM instead of 12 / 167; driver 580.173.02 instead of 580.178.04 | the box | this is the comparison |
| D2 | L3 on a 2.27 TiB attached virtio SSD (`/dev/vdc`, ext4, O_DIRECT through nixl POSIX) instead of a local NVMe | the box | the device sustains 1.88 GiB/s in both directions (campaign 5's NVMe: 0.68 read / 0.38 write); L3 delivers 2.8-3.0 GiB/s by the Exp 1 fit here vs 0.62-0.63 there |
| D3 | Qwen3-32B-FP8 runs native FP8 (DeepGEMM W8A8, JIT pre-compile at first boot) | SM90 has native FP8; `can_auto_enable_marlin_fp8()` is `80 <= sm < 89`, so no Marlin fallback and no flag | the 32B's recompute difference is GPU plus weight kernel, in the opposite direction to campaign 5 (its A100 ran the slower W8A16 Marlin path). The 70B runs `awq_marlin` on both boxes |
| D4 | `--attention-backend triton` still passed explicitly | same flag as campaign 5; on SM90 the `fa3` default would be rewritten to `triton` for fp8_e5m2 KV anyway | none; the launch line is identical to campaign 5's |
| D5 | `--hicache-size 160` instead of 100 (host pools 976,576 / 1,220,736 tokens instead of 610,368 / 762,944) | the profiled device pools are 2.5-2.7x larger here (531,008 / 705,856 vs 194,816 / 281,216) and `exp1.py`'s L2 filler budget (`min(hi, 1.15*maxtot)`, else `lo`) needs `0.92*host - probes > 1.05*maxtot`, which a 100 GB pool no longer satisfies: probes plus fillers would have overflowed L2 into L3 and tripped the discard/retry path (README, "Deviation from campaign 5") | none on any rate (L2's figure is a copy rate, not a pool-size property); both L2 passes ran with 0 discards. A first attempt at 100 GB (started 20:15Z) was stopped 2 min into `exp1_70b` and is not part of this campaign; its `p_measure` agreed with the rerun |
| D6 | nixl cleaner watermarks pinned to 30 / 20 % (`l3_extra.json`) instead of the 80 / 70 defaults | 80 % of 2.27 TiB would never fire; 30 % is still far above what a pass writes (0.5-0.7 GB for Exp 0, tens of GB for an Exp 1 pass) | none: the cleaner never logged in any stage |
| D7 | no device-side disk telemetry: every `iostat.log` is empty | `sysstat` was not installed in the container until after the campaign (it is now) | the L3 rate is the Exp 1 fit only; the 1.88 GiB/s device ceiling comes from the 2026-09-21 fio/iostat measurement outside the campaign, and `pcie.log` was captured as usual |
| D8 | one L3 lookup each for the 70B and 32B was warm-cache in Exp 0 (step 6) as prescribed; page cache dropped for step 5 through `/proc/sys/vm/drop_caches` (container is `--privileged`) | same procedure as campaign 5 | none |
| D9 | the 8B (`all3`) was not run | the box was needed for the agent_cache campaign; the third block of the three-model figure is left for a later pass | the figure has two models on the H200 row |

`--reasoning-parser qwen3` is passed for Llama as well, exactly as campaigns 2 and 5 launched it; with `max_tokens=1` it cannot affect TTFT.
