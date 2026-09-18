# Deviations from campaign 2 (70B and 32B, fp8_e5m2 KV, nixl) in this rerun

Rerun on 2026-09-17 on GCP `a2-ultragpu-1g`. Reference: `20260908_llama70b_awq_fp8kv`. Same image tag
(`nightly-dev-20260907-30705c00`), same commit under `python/`, same server flags, same client parameters and
seeds, same stage order (70B recompute sweep, 70B Exp 0 / Exp 1 / L2 pass, then the 32B's three stages), same
checkpoint snapshots (32B `aa55da1e...`, 70B `64d25562...`). Driver: `scripts/run_a100_rerun_c2.sh`.

| # | what differs | why | effect on the comparison |
|---|---|---|---|
| D1 | GPU A100-SXM4-80GB (SM80) instead of H100 PCIe; 12 vCPU / 167 GB RAM instead of 26 / 221; PCIe Gen4 | the box | this is the comparison |
| D2 | L3 on a local NVMe (`/mnt/nvme/hicache_l3`, ext4, O_DIRECT through nixl POSIX) instead of a virtio disk | the box | L3 delivers about 0.63 GiB/s through nixl here, at the device's read ceiling (iostat: about 0.68 GiB/s at 97 to 99 % util); the old disk delivered about 3 GiB/s |
| D3 | `--attention-backend triton` passed explicitly | campaign 2 resolved to triton on every boot: fa3 rejects `fp8_e5m2` KV and sglang rewrote it. SM80 defaults to flashinfer and never reaches that rewrite | same attention kernel family on both boxes; the launch line differs by this one flag |
| D4 | Qwen3-32B-FP8 weights run through weight-only FP8 Marlin (W8A16, bf16 activations) | SM80 has no native FP8 compute; the H100 ran DeepGEMM W8A8. sglang selects Marlin automatically, no flag | the 32B's recompute and L1 differences are GPU plus weight kernel. The 70B runs `awq_marlin` on both boxes |
| D5 | 32B device pool 281,216 tokens (old 284,224, -1.1 %); 70B 194,816 (old 193,728, +0.6 %); host pools identical (762,944 / 610,368) | Marlin group scales take about 0.5 GiB on the 32B | changes the Exp 0 filler count by at most one request |
| D6 | one extra, unmeasured boot: `preflight/boot_32b` | settles whether the FP8 checkpoint runs on SM80 (it does: tier round trip 4032/4032/4032, 128 L3 files totalling 536,870,912 B, correct generations) and absorbs the 32B's one-time JIT compile (boot 10.6 min) | `exp0_32b` boots JIT-warm here; on the H100 it was the 32B's first boot. The 70B's JIT is absorbed by the recompute sweep on both boxes |
| D7 | `HICACHE_FLUSH_TIMEOUT=1800`, `START_TIMEOUT_S=2400` | slower disk behind the flush in Exp 0 step 5; slower JIT-cold boots on 12 vCPUs | none on timings; both waits are outside every probe |
| D8 | every pass has telemetry and metrics scrapes, a verified L3 wipe before each boot, a store-directory assertion after it, and a failed boot ends its stage | the old drivers captured telemetry for Exp 0 only and would have run a client against a dead server | none on emitted rows |
| D9 | the L2 passes ran against a cold store with their own write-through draining | same procedure as campaign 2 (its drivers wiped the store the server used) | none |

`--reasoning-parser qwen3` is passed for Llama as well, exactly as campaign 2 launched it; with `max_tokens=1` it cannot affect TTFT.
