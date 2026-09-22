# Campaign 6 — Exp 0 / Exp 1 on the H200 box

**Status: complete** (run 2026-09-21 20:46-21:55 UTC, 7 stages, driver `scripts/run_h200.sh`). Results: `REPORT.md`
(the finding), `COMPARISON.md` / `comparison.csv` (every number, both boxes), `DEVIATIONS.md`, `versions.txt`, the
`ttft_by_tier_3models*.png` figures. The rest of this file is the plan as written before the run.

## The box

| | campaign 5 (reference) | campaign 6 (here) |
|---|---|---|
| host | GCP `a2-ultragpu-1g` | Nebius `computeinstance-u00jtv5xqvxttejgvw` |
| GPU | A100-SXM4-80GB, SM80, PCIe Gen4 x16 | **H200, 143,771 MiB, SM90, PCIe Gen5 x16** |
| driver | 580.178.04 | 580.173.02, CUDA 13.0 |
| CPU / RAM | 12 vCPU, 167 GB | 16 vCPU (Xeon 8468), 196 GB |
| L3 disk | 375 GiB ephemeral local NVMe, `/mnt/nvme` | **2.27 TiB persistent attached SSD `/dev/vdc`, `/mnt/ssd`** |
| L3 ceiling | 0.68 GiB/s read, 0.38 GiB/s write | **1.88 GiB/s, symmetric** (measured 2026-09-21, `iostat` steady state at 100 % util) |
| FP8 weight kernel | weight-only FP8 Marlin (W8A16) — SM80 has no native FP8 | **native FP8** (`can_auto_enable_marlin_fp8()` is `80 <= sm < 89`) |
| attention backend | `triton`, pinned over the SM80 `flashinfer` default | `triton`, and now the *same* value the SM90 default resolves to (`fa3` → rewritten by the fa3+fp8_e5m2 rule) |

Everything else is held fixed on purpose — same image, same commit under `python/`, same flags, client
parameters, seeds, checkpoint snapshots and stage order — so that a difference between the two campaigns is
attributable to the box.

## Why this campaign exists

Two of the three quantities in the admission criterion `bandwidth(tier) > b * P` move here, **in opposite
directions**:

- `P` (marginal prefill rate) goes **up**: native FP8 on a newer SM, against the A100's W8A16 Marlin
  fallback. That **raises** the bar a tier must clear.
- L3 delivery goes **up** too: 1.88 GiB/s against 0.68. That **raises** L3's side.

On the A100 the 32B cleared the bar 4.2x and the 70B 5.0x, while the 8B lost at 7 of 7 lengths. Which way each
model lands here is not predictable from either campaign and is the question this one answers.

## Stages and artefacts

Run inside `sglang_hicache` (see `agent_cache/RUNBOOK.md` §2.3 for the container):

```bash
# container: sglang_hicache, bash
cd /sgl-workspace/sglang/hicache_eval/scripts
bash run_h200.sh preflight_32b     # unmeasured: does the FP8 checkpoint run on SM90, and what pool does it profile?
bash run_h200.sh all               # p_measure_70b, then Exp 0/1/1-L2 for the 70B and the 32B
# bash run_h200.sh all3            # the same plus the 8B, which completes the three-model figure
```

Per stage, `<tag>/` gains: `server.log`, `server.pid`, `server_args.txt`, `startup_facts.txt`,
`cold_check.txt`, `client.jsonl`, `run.log`, `metrics_before.txt`, `metrics_after.txt`, `iostat.log`,
`pcie.log`, and for Exp 1 `ttft_by_tier.csv`. `run_h200.log` is the driver's own transcript.

Then, on the host or in the container:

```bash
# COMPARISON.md + comparison.csv, every statistic computed identically on both sides
RESULTS=/sgl-workspace/sglang/hicache_eval/results/20260921_h200_nebius_32b70b_fp8kv \
OLD_C2=/sgl-workspace/sglang/hicache_eval/results/20260917_a100_gcp_32b70b_fp8kv \
OLD_LABEL="A100 SXM4, local NVMe (campaign 5)" NEW_LABEL="H200, attached SSD (campaign 6)" \
  python3 ../scripts/compare_campaigns.py

# the figure: one row per box, one column per model; rows with no data are skipped
python3 ../scripts/plot_tier_ttft.py               # light
python3 ../scripts/plot_tier_ttft.py --mode dark   # dark
```

`compare_campaigns.py` is unchanged from campaign 5 and reproduces that campaign's published `COMPARISON.md`
byte-for-byte when run against it (verified 2026-09-21), so a difference here is data, not method.

## Deviation from campaign 5: `--hicache-size 160`, not 100

Campaign 5 ran a 100 GB host pool against L1 = 194,816 (70B) / 281,216 (32B), so L2/L1 was 3.1x / 2.7x and
`exp1.py`'s L2 filler budget was bounded by the model pool (`min(hi, 1.15*maxtot)`) — the intended design.

This box profiles **L1 = 531,008** for the 70B (measured 2026-09-21) and ~700,000 for the 32B, while a 100 GB
pool is only 610,368 tokens for the 70B (b = 163,840). The budget rule
(`exp1.py`: `lo, hi = 1.05*maxtot, 0.92*host_total - probe_tokens; budget = min(hi, 1.15*maxtot) if hi > lo else lo`)
then falls back to `lo`, and probes + fillers exceed the host pool:

| | L1 | L2 @100 GB | lo | hi | probes+fillers | 0.92*L2 | |
|---|---:|---:|---:|---:|---:|---:|---|
| 70B | 531,008 | 610,368 | 557,558 | 367,234 | 751,862 | 561,538 | **exceeds** |
| 32B | ~700,000 | 762,944 | 735,000 | 507,604 | 929,304 | 701,908 | **exceeds** |

The probes would drop through to L3 instead of staying in L2, miss `exp1.py`'s 0.9 `cached_host` threshold, and
trigger the discard/retry path (each retry re-sends 0.35 x maxtot of fillers, up to 3 attempts x 21 probes).

**160 GB** restores the campaign-5 structure: the threshold for `hi > 1.15*maxtot` is 143.4 GB (70B) / 142.4 GB
(32B), so 160 leaves ~12 % margin, and it clears the server's own guard
(`pool_host/base.py:host_memory_budget_bytes` = available - 10 GiB reserve, ~182 GB at boot). It pins 149 GiB of
the 182 GiB of RAM, leaving ~33 GiB.

**This does not affect the measured L2 rate**, which is a copy rate, not a pool-size property. It affects only
whether the filler pass can stage the probes into L2 at all. Set `HICACHE_GB=100` to reproduce campaign 5's
literal flag.

**First attempt aborted.** A run started 20:15:29Z with `--hicache-size 100` completed `p_measure_70b` and
`exp0_70b` and was stopped 2 min into `exp1_70b`; its output is kept under `aborted_hicache100/` and is not part
of this campaign. `p_measure` there is unaffected by the fix (no HiCache) and agrees with the re-run.

## Written after the run

`REPORT.md`, `DEVIATIONS.md` and `versions.txt`, in the shape of `../20260917_a100_gcp_32b70b_fp8kv/` (2026-09-22).
The 8B (`all3`) was not run, so the table has two models on the H200 row (`DEVIATIONS.md` D9).

## Expected, so a wrong number is recognisable

Not predictions to be quoted — the point is to measure them. They are here so that a result far outside them
is investigated rather than believed.

| quantity | A100 (campaign 5) | expected here |
|---|---|---|
| profiled device pool, 32B @ mem-fraction 0.85 | 281,216 tok | ~700,000 tok (HBM 143,771 MiB vs 81,920) |
| `Weight-only FP8 ... Marlin` line in the 32B boot log | present | **absent** — the driver warns if it appears |
| L3 delivered, all three models | 0.62-0.63 GiB/s | higher, bounded by 1.88 GiB/s; **if it lands near 0.62 the nixl path, not the disk, is the limit** |
| `P`, 32B | 1,226 tok/s | higher (native FP8) |
| cleaner watermarks in the boot log | `high=80.0% low=70.0%` | **`high=30.0% low=20.0%`** — pinned, because 80 % of 2.27 TiB would never fire |
| Exp 0 tier round trip | 4032/4032/4032 tokens, 128 L3 files | identical (it is an attribution test, not a rate test) |
