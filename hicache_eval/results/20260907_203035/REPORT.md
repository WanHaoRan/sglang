# HiCache Multi-Tier KV Cache Evaluation

**Model** Qwen/Qwen3-8B (BF16) · **GPU** 1× H100 PCIe 80 GB · **Host** 221 GB RAM, 26 vCPU
**sglang** `f3ccd1c0e4` · torch 2.13.0+cu130 · driver 580.173.02 · run `20260907_203035`

---

## 0. Summary

The evaluation plan fixes `--hicache-storage-backend file`. That backend turns
out to be a **reference implementation, not a production path**, and measuring
HiCache through it produces a picture of the L3 tier that is an artefact of the
backend rather than of the hardware or the design.

The headline, on the same disk, same model, same probes:

| | `file` | **`nixl`** |
|---|---|---|
| L3 hit, 4096 tokens | 11.58 s | **0.271 s** |
| L3 read bandwidth (fitted) | 0.058 GB/s | **3.277 GB/s** |
| vs the recompute bar | **35× below** | **1.59× above** |
| L3 vs recompute at 32,512 tokens | 75.0 s vs 2.17 s | **1.46 s vs 2.22 s — L3 wins** |
| L3 vs recompute at 4,096 tokens | 9.35 s vs 0.27 s | 0.44 s vs 0.25 s — recompute still wins |

**On `file`, an L3 hit is never worth taking at any prompt length. On `nixl` the
tier becomes viable, but only for long contexts: it beats recompute above
~32 K tokens and loses below that**, because it carries a ~0.19 s fixed
overhead (§4).

Everything measured on `file` is archived under `file_backend_reference/` with a
README explaining what it is good for. This report leads with `nixl`.

---

## 1. Environment and measured constants

| Quantity | Value | How |
|---|---|---|
| `KV_BYTES_PER_TOKEN` | **147,456 B** (144 KiB) | 2 × 36 layers × 8 kv_heads × 128 head_dim × 2 B; confirmed to the byte (4096 tokens → 603,979,776 B backed up) |
| `PAGE_SIZE` / `PAGE_BYTES` | 64 tokens / 9,437,184 B | 64 files per 4096-token prompt, exactly |
| L1 device pool | **374,784 tokens** | `max_total_num_tokens` |
| L2 host pool | **678,208 tokens** (100 GB pinned) | `sglang:hicache_host_total_tokens` |
| Prefill throughput **P** | **15,011 tok/s** (nixl run) · 15,745 (file run) | Exp 1 recompute fit |
| **Recompute bar b·P** | **2.061 GB/s** (nixl run) · 2.162 (file run) | 147,456 B × P |

P was fitted independently in each Exp 1 run; the two agree to 4.7 %, which is
the run-to-run spread on this quantity. Comparisons below use **each run's own
bar** — 2.061 for nixl figures, 2.162 for file figures — rather than mixing them.

### The disk

L3 lived on `/var/hicache_l3` (and `/var/hicache_nixl`) inside the container →
docker overlay → **`/dev/vda1`, the VM's local virtio-blk disk**. The virtiofs
mount `/lambda/nfs` held only the repo and result files; no KV page touched it.
That was the right choice — virtiofs serves the same pattern at 0.096 GB/s.

Measured with fio, 9 MiB random reads (the actual L3 access pattern):

| access pattern | GB/s |
|---|---|
| buffered, QD1 — *what `file` does* | 0.643 |
| O_DIRECT, QD1 | **6.19** |
| O_DIRECT, QD8 ×4 | **22.5** |
| sequential write, O_DIRECT QD1 → QD32 ×4 | 1.15 → **12.7** |

The device is **not** slow. Its `rotational=1` flag is meaningless — virtio-blk
does not report rotation, and under concurrency it reaches 40 GB/s read. An
earlier draft characterised it from buffered QD1 `dd` (1.7 GB/s) and drew a
wrong conclusion from it; see `DEVIATIONS.md` D14.

---

## 2. Why `file` is not the reference backend

Both directions land at ~0.26 % of what the disk gives a concurrent O_DIRECT
reader (0.058 / 22.5 GB/s), and `iostat %util` never exceeded **5.8 %** during any L3 work. The tier
is software-bound, and three independent factors compose:

| factor | penalty | cause |
|---|---|---|
| buffered instead of O_DIRECT | **9.6×** | `open(path,"rb",buffering=0)` (`hicache_storage.py:484`) — `buffering=0` disables only Python's buffer |
| serial QD1 instead of concurrent | **3.6×** | `batch_set` is a Python `for` loop over `set` (`:568`) — one page per syscall |
| Python per-page work + GIL | **11×** | `os.path.exists` + evictor reserve + `tofile` + `os.replace` per 9 MiB page, contending with a scheduler spinning in `check_hicache_events` |

`py-spy` during a backup drain, with the disk at 1–2 % utilisation:

```
MainThread (active):  check_hicache_events (unified_radix_cache.py:3002)
Thread-4  (active):  set (hicache_storage.py:540) -> batch_set (:568)
```

`nixl`, by contrast, uses `O_DIRECT` by default
(`SGLANG_HICACHE_NIXL_USE_DIRECT_IO`, `environ.py:742`), submits one io_uring /
libaio batch per 128 pages, and shards into `xx/` subdirectories instead of one
flat directory. Driving the repo's own NIXL classes directly at that batch size
reaches **14–20 GB/s read, 9.8 GB/s write**, with O_DIRECT proven by `mincore`
(direct=True → 0/18432 pages resident after write *and* read; direct=False →
18432/18432).

Of the eleven registered backends, `nixl` is the only drop-in alternative that
runs here with no setup: `mooncake`/`hf3fs`/`aibrix`/`eic`/`simm`/`mori` need
absent services or libraries and most are network/DRAM stores, `shm` performs no
file I/O at all, and `dynamic` is a loader with nothing shipped to load.

```bash
export SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR=/var/hicache_nixl
--hicache-storage-backend nixl        # instead of file
```

**Caveat:** `nixl` writes in place with `O_CREAT` and no temp+rename, so a crash
mid-write leaves a torn page where `file` would not. It also requires
`--hicache-mem-layout page_first` (or `page_first_direct`) and a 4096-aligned
host buffer, or zero-copy disengages and every page pays a 9 MiB memcpy.

---

## 3. Experiment 0 — tier attribution: **PASS** on both backends

Each tier forced deterministically and verified by per-tier attribution
(`return_cached_tokens_details: true`; the field lands as a top-level
`sglext_cached_tokens_details`).

| Step | Tier | **`nixl`** | `file` |
|---|---|---|---|
| 1 | recompute, cold *(warm-up inflated — see below)* | 0.391 s | 0.373 s |
| 3 | L1 (GPU HBM) | **0.059 s** | 0.059 s |
| 4 | L2 (host DRAM) | **0.067 s** | 0.068 s |
| 5 | **L3, page cache evicted** | **0.271 s** | 11.581 s |
| 6 | L3, page cache warm | 0.259 s | 8.120 s |
| 7 | recompute reference | — | 0.343 s |

All plan pass criteria met on both: attributions correct at every step, L3 file
count matched pages written exactly (64 files, 603,979,776 B on `file`; same
bytes as 128 K/V-split files on `nixl`), tier ordering strict, and page-cache
eviction demonstrably changes the measurement.

Two results worth separating out:

1. **~~The tier ordering is healthy: L3 (0.271) < recompute (0.391).~~
   CORRECTED — that comparison is a sampling artifact; do not cite it.**
   Exp 0 is n = 1 per step and its step-1 recompute is the *first* long prefill
   after server start, so it carries a warm-up penalty. The same penalty is
   visible directly in Exp 1: at L = 512 the three recompute reps are
   **0.288 / 0.043 / 0.043 s** — a 6.7× first-request cost. Exp 0's recompute
   (0.391 s) sits outside Exp 1's entire L = 4096 recompute range
   (0.218–0.252 s, median 0.248), while its L3 sample (0.271 s) sits at the fast
   end of Exp 1's L3 spread (0.247–0.515 s). So Exp 0 compared an inflated
   recompute against a lucky-fast L3, one sample each.
   **Exp 1 is authoritative: at 4096 tokens nixl L3 (0.442 s median) is 1.8×
   *slower* than recompute (0.248 s).** L3 only wins above ~32 K tokens (§4).
   Exp 0's role is tier *attribution*, which it establishes correctly; its
   absolute timings should not be used for tier comparison.
2. **The page-cache confound disappears on `nixl`.** Evicting the page cache
   changes TTFT by **1.05×** on nixl (0.271 vs 0.259) against **1.43×** on file
   (11.58 vs 8.12), because O_DIRECT never populates the cache. That is an
   independent behavioural confirmation of the `mincore` proof.

`cached_tokens` reads 4032 not 4096 throughout: the trailing partial 64-token
page is never cached. Page alignment, as the plan predicts.

---

## 4. Experiment 1 — TTFT per tier vs context length

N = 3 per point, `wait_complete`, single in-flight probe, page cache evicted
before every L3 probe, attribution verified on every sample, **0 discards**.

### Cost curves on `nixl`

| tier | slope (s/token) | intercept (s) | effective bandwidth | vs bar |
|---|---|---|---|---|
| recompute | 6.70e-5 | −0.033 | 2.061 GB/s (= b·P) | — |
| L1 | 7.0e-6 | +0.036 | **19.82 GB/s** | 9.6× above |
| L2 | 1.10e-5 | +0.031 | **12.87 GB/s** | 6.2× above |
| **L3** | 4.20e-5 | +0.192 | **3.277 GB/s** | **1.59× above** |
| *L3 on `file`* | *2.353e-3* | *−0.400* | *0.058 GB/s* | *37× below its own bar (2.162)* |

Median TTFT (s):

| L | recompute | L1 | L2 | **L3 (nixl)** | *L3 (file)* | nixl speed-up |
|---|---|---|---|---|---|---|
| 512 | 0.043 | 0.030 | 0.034 | 0.199 | *0.347* | 1.7× |
| 1024 | 0.090 | 0.035 | 0.038 | 0.204 | *0.493* | 2.4× |
| 2048 | 0.137 | 0.035 | 0.046 | 0.159 | *3.42* | 21.6× |
| 4096 | 0.248 | 0.080 | 0.076 | 0.442 | *9.35* | 21.2× |
| 8192 | 0.428 | 0.113 | 0.128 | 0.521 | *21.9* | 42.1× |
| 16384 | 0.920 | 0.159 | 0.213 | 1.072 | *39.1* | 36.5× |
| 32512 | 2.219 | 0.251 | 0.372 | **1.459** | *75.0* | **51.4×** |

The speed-up grows with length because `file`'s per-page Python cost scales with
page count while `nixl` batches 128 pages per submission.

### Where each tier starts paying

- **L1 above ~1,160 tokens**, **L2 above ~1,140** — both limited by a ~30–36 ms
  intercept, not by bandwidth. Below that, recomputing is cheaper than a lookup.
- **L3: read the empirical crossover, not the fit.** The linear fit says
  L ≈ 9,100, but the measured curves cross later — nixl L3 is still 1.26× slower
  than recompute at 8,192 and 1.14× at 16,384, and only wins at **32,512**
  (1.459 s vs 2.219 s, a **1.52× saving**). L3 carries a ~0.19 s fixed overhead
  that the linear fit smears across the range.

Figures: `nixl/exp1_nixl/ttft_vs_len.png` (four tiers, linear and log-log),
`exp1_backend_comparison.png` (file vs nixl L3 against the recompute bar, with
the sustained crossover marked). Table: `exp1_backend_comparison.csv`.

### How to read the "effective bandwidth" numbers

The `effective bandwidth` column is **not a device bandwidth**. It is
`147,456 B / slope`, where slope is d(TTFT)/dL — an *end-to-end service rate*
that includes the tier's data movement plus all the per-token bookkeeping the
request pays anyway (index gather, attention page-table construction, scheduler
work) plus the prefill of the uncached tail. Comparing it directly to HBM or
DRAM bandwidth is a category error, most obviously for L1:

**L1 moves no KV at all.** A device hit reuses pages already resident in the
device pool; nothing is copied. Its 19.82 GB/s is therefore not a memory rate —
it is `144 KiB ÷ per-token overhead`, i.e. a measure of how fast the engine can
*account for* cached tokens. That it looks like a bandwidth is an artifact of
dividing bytes by a cost that has nothing to do with bytes.

To recover something that *is* comparable to hardware, take **differential
slopes** — each tier minus the one above it, which cancels the shared overhead:

| | rate | against the right ceiling |
|---|---|---|
| L1 (no transfer) | 19.82 GB/s | — this is pure per-token bookkeeping |
| **L2 − L1** = host→device | **36.72 GB/s** | PCIe Gen5 ×16, ~50–55 GB/s practical → **71 %** |
| **L3 − L2** = disk read | **4.40 GB/s** | disk O_DIRECT QD1 6.19 GB/s → **71 %**; concurrent ceiling 22.5 → 20 % |

Both land at ~71 % of their single-queue ceiling, which is a plausible efficiency
for a real transfer path and a far more sensible reading than "L2 runs at 12.9
GB/s when DRAM does hundreds". The remaining L3 headroom (20 % of the concurrent
ceiling) is queue depth: `nixl` submits 128-page batches, but the engine issues
one prefetch at a time per request.

**Which number to use depends on the question:**

- *Should this hit be taken instead of recomputing?* — use the **absolute**
  service rate. That decision is about end-to-end TTFT, and overhead is real.
- *Is the hardware being used well?* — use the **differential** rate.
- *Below ~4 K tokens, use neither.* The intercept dominates: at L = 512 the fixed
  cost is **85–91 %** of TTFT for every tier (L1 0.036 s of 0.040 s; L3 0.192 s
  of 0.213 s). A "bandwidth" fitted through that region is meaningless, which is
  also why the fitted L3 break-even (~9.1 K) disagrees with the measured
  crossover (~32 K) — the fit is being dragged by points where bandwidth is not
  what is being measured.

### Prefetch policy

Repeating the L3 curve under `--hicache-storage-prefetch-policy timeout` on the
`file` backend (21 rows, all 7 lengths):

| L | `timeout` | `wait_complete` | recompute | deadline truncations |
|---|---|---|---|---|
| 4096 | 1.40 s | 9.35 s | 0.268 s | 0/3 |
| 8192 | 2.14 s | 21.9 s | 0.415 s | 2/3 |
| 16384 | 4.10 s | 39.1 s | 0.941 s | 3/3 |

`timeout` is 2–10× better than `wait_complete` on a slow backend precisely
because it *gives up*: the prefetch abandons the read at its deadline and the
remainder is recomputed. Truncation frequency rises with prefix length. It is
still 1.5–4× worse than simply recomputing — a faster policy over a broken tier
is not a fix.

---

## 5. Experiment 2 — write-through interference (file backend)

L = 16,384 probes, `wait_complete`, `write_through`. Nine distinct prompts, each
probed once so every probe is a first touch from L3.

| requested R | TTFT median | `%util` | control |
|---|---|---|---|
| 0 | 36.77 s | ≤ 2.74 | baseline |
| 2 | 24.84 s | ≤ 3.06 | baseline |
| 8 | 8.51 s | ≤ 5.80 | baseline |
| 8 | **9.13 s** | 4.04 | **C1 — L3 on tmpfs** (device removed) |
| 8 | **9.02 / 9.13 / 9.24 s** | ≤ 0.14 | **C2 — `backup_skip`** (writes removed) |

**Neither control moves the number.** Per the plan's own decision rule —
degradation removed by C2 but not C1 is the SSD; degradation surviving C2 is
PCIe/host/GIL — it survives both, so it is neither the device nor the write
traffic.

**Two corrections that matter for reading this table:**

1. **The original write load was broken.** `writeload.py` appended its uniqueness
   marker as a *suffix*, so all 64 pooled bodies shared their whole prefix and
   every 64th request hit the radix cache instead of inserting nodes. Re-running
   the endpoints with the marker as a prefix: `write_gbps_actual` went
   **0.0003 → 1.647 GB/s (~5000×)** and `storage_prefetch_unfulfilled_tokens`
   went **0 → 32,640**. Under a *correct* load at R = 8, only **1 of 3** probes
   still received its L3 data; the other two never did. Their low TTFT is "the
   prefetch gave up and the model recomputed", not "the read went faster".
2. **So the corrected reading is that write pressure makes L3 hits *unreliable*,
   not slower** — at 1.65 GB/s of concurrent backup traffic the tier stops
   delivering within the request's life, under `wait_complete`, where by
   construction it should have waited.

The plan's headline question — the write rate at which an L3 hit becomes slower
than recompute — has no answer on `file`, because an L3 hit is already 17–37×
slower than recompute at **zero** write load.

**This experiment has not been repeated on `nixl`, and should be.** Its whole
subject is contention in a path that `nixl` replaces.

---

## 6. Experiment 3 — write amplification on a multi-turn reasoning workload

`bench_multiturn.py --api-format openai`, `enable_thinking: true`, 64 requests
per condition, fresh server and verified-cold L3 each time. **All on `file`.**

| condition | host pool | write policy | **written to L3** | **read from L3** | dead | hit rate |
|---|---|---|---|---|---|---|
| wt100 | 100 GB | write_through | **17.99 GB** | **0** | **100 %** | 0.732 |
| wt30 | 30 GB | write_through | 17.77 GB | 0 | 100 % | 0.743 |
| wts100 | 100 GB | write_through_selective | **11.08 GB** | 0 | 100 % | 0.733 |
| wts30 | 30 GB | write_through_selective | 11.32 GB | 0 | 100 % | 0.718 |
| nohicache | — | GPU radix only | 0 | 0 | — | 0.693 |

`prefetched_tokens`, `storage_prefetch_hit_tokens_total` and
`hicache_dropped_tokens_total` were **zero in every condition**.

1. **18 GB written, 0 bytes read.** Every hit was served by L1 or L2.
2. **Shrinking the host pool 100 → 30 GB changed nothing.** A 121 K-token
   working set still fits a 203 K-token pool, so L3 is never consulted.
3. **`write_through_selective` cuts L3 writes 38 %** at equal hit rate — it backs
   up on first *reuse* rather than on insert, so single-use blocks are never
   written down. Strictly better than `write_through` here.
4. **HiCache produced no measurable latency benefit.** Median TTFT 0.110–0.128 s
   and throughput 344–359 tok/s across all seven conditions — and the *fastest*
   median belongs to `nohicache`. The +4 points of hit rate did not convert into
   user-visible speed.

**Why the dead fraction is 100 %, verified not assumed.** With
`--reasoning-parser qwen3` the client appends only `content` to history, and
Qwen3's template never re-feeds prior thinking. Measured directly
(`file_backend_reference/exp3/caveat13_thinking_check.json`): turn 1 generated
**300 tokens**; turn 2's prompt grew by **21** — the new user message alone.
**279 of 300 generated tokens were dropped from the conversation.** Compounding
that, whatever *is* reusable stays resident in L2, so the L3 copy is redundant
even for the live fraction.

**Backend dependence:** the *dead fraction* is a property of the workload and
carries over to `nixl` unchanged. The *byte volumes* and the write cost are
file-specific and would need re-measuring.

---

## 7. Experiment 4 — oracle upper bound for a class-aware write policy

`--strip-thinking-cache` skips caching reasoning output on finish — for this
workload the ideal class decision, hence an upper bound.

| condition | written to L3 | vs `write_through` | read | hit rate |
|---|---|---|---|---|
| wt100 (reference) | 17.99 GB | — | 0 | 0.7324 |
| wts100 (selective) | 11.08 GB | −38.4 % | 0 | 0.7333 |
| **oracle100** | **11.32 GB** | **−37.1 %** | 0 | **0.7324** |
| **oracle30** | 11.23 GB | −36.8 % | 0 | 0.7333 |

1. **The oracle removes ~37 % of L3 writes at exactly zero hit-rate cost** —
   oracle100 and wt100 report identical hit rates to five decimals, because the
   tokens it declines to write are tokens nothing asks for again.
2. **`write_through_selective` already captures essentially all of it** — 11.08
   vs 11.32 GB, within 2 %, from opposite mechanisms. A learned class-aware
   policy would be competing with a shipped flag for the last ~2 %.
3. **The oracle does not fix the real problem.** It reduces dead writes from
   100 % of 18 GB to 100 % of 11 GB. No write policy makes L3 useful here; only
   a *placement* policy that declines the tier can.

---

## 8. What this means

**1. Evaluate HiCache's L3 on `nixl`, not `file`.** The plan's premise — an SSD
tier sitting *at* the recompute bar, tipped by write pressure — is untestable on
`file`, whose delivered bandwidth is an implementation artefact. On `nixl` the
premise becomes testable and the tier is genuinely marginal-to-positive: 1.59×
above the bar, winning above ~32 K tokens.

**2. The admission rule falls out of the measurements.** Admit a block to tier
*T* only if measured `bandwidth(T) > b·P`. Here: L1 19.8 ✓, L2 12.9 ✓,
L3-on-nixl 3.28 ✓, L3-on-file 0.058 ✗. That single rule would have avoided all
18 GB of dead writes in Exp 3 without knowing anything about thinking tokens.

**3. For a class-aware *write* policy the headroom is ~2 %.** Selective and the
ideal oracle land within 2 % of each other. The large lever is placement — which
tier, or none — not classification.

**4. The dead-write finding is the workload's, not the backend's.** 100 % of L3
writes went unread because thinking is never re-fed and the live set fits L2.
A faster L3 makes those writes cheaper, not more useful.

**5. Long contexts are where L3 earns its place.** L3's fixed ~0.19 s overhead
means it only pays above ~32 K tokens on this model. For an 8B model with 144
KiB/token, that is the regime to target; models with lower b·P (larger, or
MLA-compressed KV) would cross earlier.

---

## 9. Status against the plan, caveats, and artifacts

An independent audit extracted **213 requirements** from the plan and checked
each against the artifacts (`CONFORMANCE.md`): **99 SATISFIED · 25
DEVIATED_AND_RECORDED · 59 PARTIAL · 29 NOT_DONE**. This report does **not**
claim the plan was executed in full.

**Not executed** (all disclosed in `DEVIATIONS.md` D16.3): Exp 2 control C3
(GIL check as a formal artifact), Exp 2 timeout sweep at R = 8 (R = 0 and 2
captured), Exp 1 background-load L3/L2 tiers (recompute and L1 captured), the
Exp 3 iostat write cross-check, `read_L2_tokens` (unrecoverable — no per-tier
hit counter exists at this commit), and the optional Exp 4 stretch oracle.

**Principal caveats.** N = 3, not the plan's 7. Exp 2's write axis is unreliable
for the original sweep (see §5). Exp 3/4 median generation was 678–725 tokens,
below the plan's 800-token trigger, and the prescribed remedy was not applied.
Client co-resident on the same 26 vCPUs — a real confound given the central
finding is CPU/GIL contention. Four harness bugs were found mid-run and fixed;
each had changed a published number, and all are itemised in D16.1.

### Artifacts

```
REPORT.md                       this file
CONFORMANCE.md                  213 plan requirements, audited
DEVIATIONS.md                   D1–D16, including four harness bugs
constants.json                  measured constants
device_bandwidth.json           fio matrix for /dev/vda1
preflight/                      §0 audit trail: model config, flags, deps, container
nixl/exp0_nixl/                 Exp 0 on nixl, 7 steps
nixl/exp1_nixl/                 Exp 1 on nixl, 84 probes, 4 tiers, fits + figure
nixl/exp1_nixl_l2/              the dedicated L2 pass
nixl/ab_nixl/                   nixl side of the A/B
backend_ab/                     file vs nixl A/B summary
exp1_backend_comparison.{png,csv}   L3 file vs nixl vs recompute
missing_runs_summary.json       timeout-policy and background-load results
file_backend_reference/         ALL file-backend runs + README explaining
                                why they are archived, and REPORT_v1
```
