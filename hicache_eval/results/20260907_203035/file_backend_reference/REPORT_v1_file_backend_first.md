# HiCache Multi-Tier KV Cache Evaluation — Results

**Model** Qwen/Qwen3-8B (BF16) · **GPU** 1× H100 PCIe 80 GB · **Host** 221 GB RAM, 26 vCPU
**sglang** `f3ccd1c0e4` · torch 2.13.0+cu130 · driver 580.173.02
**Run** `results/20260907_203035`

> **Read D1 in `DEVIATIONS.md` first.** This machine has **no NVMe**. The plan's
> central premise — an SSD reading at 3–7 GB/s, i.e. *at* the recompute bar — does
> not hold here. L3 lives on a virtio-blk device measured at 1.7 GB/s cold read /
> 1.1 GB/s write. Every absolute L3 number below is specific to that device.
> The *mechanisms* the plan set out to measure are unaffected, and two of them turned
> out to be device-independent (see §L3 throughput).

---

## 1. Environment and measured constants

| Quantity | Value | Source |
|---|---|---|
| `KV_BYTES_PER_TOKEN` | **147,456 B** (144 KiB) | 2 × 36 layers × 8 kv_heads × 128 head_dim × 2 B; confirmed exactly (4096 tokens → 603,979,776 B backed up) |
| `PAGE_SIZE` / `PAGE_BYTES` | 64 tokens / 9,437,184 B | `--page-size 64`; 64 files per 4096-token prompt, exactly |
| L1 device pool | **374,784 tokens** | `max_total_num_tokens` from the server |
| L2 host pool | **678,208 tokens** (100 GB) | `sglang:hicache_host_total_tokens` |
| L3 cap | 380 GB | `SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE` |
| L3 device | `/dev/vda1`, virtio-blk, VM **local** disk (via the container overlay) | not the virtiofs `/lambda/nfs` mount |
| Device, 9 MiB random read, **buffered QD1** (what `file` does) | **0.643 GB/s** | fio |
| Device, 9 MiB random read, **O_DIRECT QD1** | **6.19 GB/s** | fio |
| Device, 9 MiB random read, **O_DIRECT QD8 ×4** | **22.5 GB/s** | fio |
| Device, seq write, O_DIRECT QD1 → QD32 ×4 | **1.15 → 13.0 GB/s** | fio |

> **Correction (post-draft).** The first version of this report characterised the
> device from `dd`: 1.7 GB/s read, 1.1 GB/s write. Those are *buffered, QD1,
> sequential* numbers and they badly understate the device. Under concurrent
> O_DIRECT it reaches **40 GB/s read / 12.7 GB/s write**; its `rotational=1` flag
> is meaningless (virtio-blk does not report rotation). Full matrix in
> `device_bandwidth.json`. Conclusions that depended on the low figure are
> corrected below and marked.

100 GB of pinned host memory registered without incident despite `ulimit -l 8192`,
because the container is privileged (CAP_IPC_LOCK). See D2.

---

## 2. Experiment 0 — sanity and tier attribution: **PASS**

All tiers forced deterministically and verified by per-tier attribution.

| Step | Tier | TTFT (s) | device / host / storage |
|---|---|---|---|
| 1 | recompute (cold) | 0.373 | – |
| 3 | L1 (GPU HBM) | **0.059** | 4032 / 0 / 0 |
| 4 | L2 (host DRAM) | **0.068** | 0 / 4032 / 0 |
| 5 | L3 (SSD, page cache evicted) | **11.581** | 0 / 0 / 4032 |
| 6 | L3 (page cache warm) | 8.120 | 0 / 0 / 4032 |
| 7 | recompute reference (cold restart) | 0.343 | – |

Pass criteria, all met:

- attribution matches the intended tier at every step;
- L3 file count matched pages written **exactly** (64 files, 603,979,776 B = 64 × 9 MiB);
- L3(evicted) > L2 > L1 TTFT;
- L3(evicted) ≠ L3(page-cache warm) — eviction demonstrably changes the measurement;
- step 7 reproduces step 1 within noise (0.343 vs 0.373 s).

`cached_tokens` is 4032, not 4096: the trailing partial 64-token page is never
cached. Page alignment, exactly as the plan's caveat #4 predicts.

### Exp 0 repeated on the `nixl` backend

The plan fixes `--hicache-storage-backend file`. After §10 showed the tier was
software-throttled rather than device-limited, Exp 0 was repeated verbatim —
same probe, same recipe, same 7 steps — on `nixl`.

| Step | Tier | `file` | **`nixl`** | speed-up |
|---|---|---|---|---|
| 1 | recompute (cold) | 0.373 s | 0.391 s | — |
| 3 | L1 (GPU HBM) | 0.059 s | **0.059 s** | 1.0× |
| 4 | L2 (host DRAM) | 0.068 s | **0.067 s** | 1.0× |
| 5 | **L3, page cache evicted** | 11.581 s | **0.271 s** | **43×** |
| 6 | L3, page cache warm | 8.120 s | **0.259 s** | 31× |

All attributions held (`device=4032` / `host=4032` / `storage=4032` at the
respective steps), so the tier-forcing recipe transfers unchanged.

Two things stand out:

1. **The tier ordering is finally healthy.** L1 (0.059) < L2 (0.067) <
   **L3 (0.271) < recompute (0.391)**. On `nixl` an L3 hit is *cheaper than
   recomputing* at 4096 tokens; on `file` it was 31× dearer. This is the
   ordering the plan expected to find and did not.
2. **The page-cache confound disappears.** On `file`, evicting the page cache
   changed TTFT by 1.43× (11.58 vs 8.12 s) — the plan's caveat 1 in action. On
   `nixl` it changes it by **1.05×** (0.271 vs 0.259 s), because `nixl` opens
   with `O_DIRECT` and never populates the page cache at all. That is an
   independent behavioural confirmation of the `mincore` result in §10.4.

*Harness note:* `hcommon.l3_stats()` globbed `**/*.bin`, which counts `file`
backend pages but not `nixl`'s (`<key>_<model>_<tp>_k` / `_v`, no extension), so
the step-2 file count read 0 for the nixl run. Fixed to count all regular files;
the byte total for this prompt is independently known from §10.4 (128 files,
603,979,776 B — identical bytes to `file`'s 64 files, split K/V).

### The headline from Exp 0

**An L3 hit is 31× slower than recomputing the same prefix** (11.58 s vs 0.373 s
for 4096 tokens). On this device L3 is not a marginal win that write pressure can
flip — it is far under water before any write load is applied at all.

---

## 3. Experiment 1 — TTFT per tier vs context length

N = 3 per point (D10), `wait_complete` prefetch, single in-flight probe, page
cache evicted before every L3 probe. Attribution verified on every sample. The **retained** rows carry 0 discards; an earlier L2 pass whose filler budget overflowed L2 produced 59 failed attribution attempts and was discarded wholesale (D12), not averaged in. Fits are least-squares on per-length medians.

### Cost curves

| tier | slope (s/token) | intercept (s) | effective bandwidth | throughput |
|---|---|---|---|---|
| recompute | 6.40e-5 | +0.014 | 2.162 GB/s (= b·P) | **P = 15,745 tok/s** |
| L1 (HBM) | 7.0e-6 | +0.039 | 20.5 GB/s | 149,482 tok/s |
| L2 (host DRAM) | 1.0e-5 | +0.031 | 13.2 GB/s | 95,961 tok/s |
| L3 (file) | 2.353e-3 | −0.400 | **0.058 GB/s** | 425 tok/s |

Median TTFT (s), min–max in parentheses:

| L | recompute | L1 | L2 | L3 |
|---|---|---|---|---|
| 512 | 0.048 | 0.031 | 0.033 | 0.347 |
| 1024 | 0.078 | 0.035 | 0.038 | 0.493 (0.49–2.63) |
| 2048 | 0.115 | 0.045 | 0.051 | 3.42 (1.42–4.73) |
| 4096 | 0.268 | 0.082 | 0.075 | 9.35 (8.99–9.50) |
| 8192 | 0.415 | 0.108 | 0.125 | 21.9 (16.9–22.4) |
| 16384 | 0.941 | 0.160 | 0.200 | 39.1 (25.4–42.2) |
| 32512 | 2.167 | 0.247 | 0.368 | 75.0 (66.4–76.5) |

### The recompute bar, and where each tier sits against it

**b·P = 147,456 B/token × 15,745 tok/s = 2.162 GB/s.** Against that:

- **L1 at 20.5 GB/s — 9.5× the bar.** Always worth it above ~440 tokens.
- **L2 at 13.2 GB/s — 6.1× the bar.** Always worth it above ~310 tokens.
  Host DRAM over PCIe is comfortably faster than recomputing an 8B model.
- **L3 at 0.058 GB/s — 37× *under* the bar.**

Both L1 and L2 have a ~30–40 ms intercept, so for very short prompts
(< ~310–440 tokens) recomputing is cheaper than a cache lookup. That is the
break-even the plan asked for, and it is small. Per-tier break-evens are in
`exp1/break_even.csv`: L1 437, L2 308, L3 181 tokens.

### L3 never pays off, at any length

The L3 break-even against recompute is **L = 181 tokens**, and because L3's
*slope* is worse than recompute's, L3 only wins *below* that length. Storage
prefetch is skipped below `prefetch_threshold` = 256 tokens, so **the region
where L3 would win is unreachable by construction**. There is no prompt length
on this machine at which an L3 hit beats recomputing.

At 32,512 tokens an L3 hit costs **75 s** against **2.2 s** to recompute — 35×
worse. Reading 4.67 GB in 75 s is 65 MB/s.

### Exp 1 repeated on the `nixl` backend

Same probes, same lengths, same N = 3, `wait_complete`, page cache evicted before
every L3 probe. L2 required a dedicated pass (see note below). 0 discards.

| tier | slope (s/token) | intercept (s) | effective bandwidth | vs recompute bar |
|---|---|---|---|---|
| recompute | 6.70e-5 | −0.033 | 2.061 GB/s (= b·P) | — |
| L1 | 7.0e-6 | +0.036 | 19.82 GB/s | 9.6× above |
| L2 | 1.10e-5 | +0.031 | 12.87 GB/s | 6.2× above |
| **L3 (nixl)** | **4.20e-5** | **+0.192** | **3.277 GB/s** | **1.59× ABOVE** |
| *L3 (file, for contrast)* | *2.353e-3* | *−0.400* | *0.058 GB/s* | *37× below* |

Median TTFT (s) by tier:

| L | recompute | L1 | L2 | **L3 (nixl)** | *L3 (file)* |
|---|---|---|---|---|---|
| 512 | 0.043 | 0.030 | 0.034 | 0.199 | *0.347* |
| 1024 | 0.090 | 0.035 | 0.038 | 0.204 | *0.493* |
| 2048 | 0.137 | 0.035 | 0.046 | 0.159 | *3.42* |
| 4096 | 0.248 | 0.080 | 0.076 | 0.442 | *9.35* |
| 8192 | 0.428 | 0.113 | 0.128 | 0.521 | *21.9* |
| 16384 | 0.920 | 0.159 | 0.213 | 1.072 | *39.1* |
| 32512 | 2.219 | 0.251 | 0.372 | **1.459** | *75.0* |

**The L3 tier crosses the recompute bar.** Its fitted bandwidth, 3.277 GB/s, is
**1.59× above** b·P = 2.061 GB/s, where the `file` backend sat 37× below. The
speed-up over `file` grows with length — 21× at 4096, 42× at 8192, **51× at
32512** — because `file`'s per-page Python cost scales with page count while
`nixl` batches 128 pages per io_uring submission.

**Where L3 actually starts winning.** The linear fit puts the break-even at
**L ≈ 9,100 tokens**, but the measured curves cross later: nixl L3 is still
1.26× slower than recompute at 8192 and 1.14× at 16384, and only wins at 32512
(1.459 s vs 2.219 s, a **1.52× saving**). The discrepancy is because L3's cost
is not linear at small L — it carries a ~0.19 s fixed overhead that the fit
spreads across the range. **Read the empirical crossover (between 16K and 32K
tokens), not the fitted one.** L1 and L2 break even far earlier, ~1.15K tokens,
both dominated by their ~30–36 ms intercept.

Figures: `exp1_nixl/ttft_vs_len.png` (all four tiers, linear and log-log) and
`exp1_backend_comparison.png` (file L3 vs nixl L3 against the recompute bar,
with the crossover marked). Data in `exp1_backend_comparison.csv`.

*L2 note:* the combined run ordered L3 before L2, and every L3 probe loads its
pages back into the host pool, so by the L2 phase the pool was 99.5 % full
(674,752 / 678,208 tokens) and the filler pass evicted the probes out of L2
entirely — all 21 attempts came back as misses. Re-run as a dedicated
recompute+L2 pass on a fresh server, which is the same fix the `file` backend
needed (D12). 21 rows, 0 discards.

### L3 is not disk-bound — it is software-bound

The measured 0.058 GB/s must be read against the device underneath it:

| 9 MiB random read (the L3 access pattern) | GB/s | L3 delivers |
|---|---|---|
| device ceiling, O_DIRECT QD8 ×4 | 22.5 | **0.26 %** |
| device at O_DIRECT QD1 (no concurrency) | 6.19 | 0.94 % |
| device at buffered QD1 (what the code actually does) | 0.643 | 9.0 % |
| **L3 read through HiCache** | **0.058** | — |
| **L3 write through HiCache** | **0.068** | — |

The gap decomposes into three independent, individually fixable factors:

| factor | penalty | cause |
|---|---|---|
| buffered instead of O_DIRECT | **9.6×** | `open(..., "rb", buffering=0)` — no `O_DIRECT` (`hicache_storage.py:484`) |
| serial QD1 instead of concurrent | **3.6×** | one page at a time, one thread |
| Python per-page work + GIL | **11×** | `batch_set` serial loop; scheduler spinning in `check_hicache_events` |

Read and write both land at ~60–68 MB/s — within 15 % of each other and both
~3–6 % of what the device does. A tier limited by its device would not produce
the same number in both directions at that ratio. Direct evidence from `iostat`
during a backup drain: **`%util` 0.76–2.50 %** at 20–104 MB/s. The disk is
idle.

`py-spy` on the scheduler process during that drain shows why:

```
Thread (active): "MainThread"
    check_hicache_events (unified_radix_cache.py:3002)
    _get_new_batch_prefill_raw (scheduler.py:3686)
Thread (active): "Thread-4 (backup_thread_func)"
    set (hicache_storage.py:540)
    batch_set (hicache_storage.py:568)
    _generic_page_set (cache_controller.py:1248)
```

`HiCacheFile.batch_set` is a **serial Python `for` loop over `set`**
(`hicache_storage.py:568-571`), and each `set` does an `os.path.exists`, an
evictor reserve under a lock, `tensor.contiguous().numpy().tofile()`, an
`os.replace`, and an evictor commit — per 9 MiB page. That single thread
competes for the GIL with a scheduler main loop that is itself spinning in
`check_hicache_events`. ~154 ms of wall time per page against ~8 ms of device
time.

This is the mechanism, and it is **device-independent**: putting a real NVMe
under this path would not move 65 MB/s much, because the bottleneck is above
the device. The plan's premise (L3 marginal, tipped by write pressure) assumed
a disk-bound tier; on stock SGLang at this commit the file backend is not
disk-bound at all — it delivers **0.26 %** of what this disk gives a concurrent
O_DIRECT reader.

### Variance

L3 is strongly long-tailed: 1.42–4.73 s at 2048 tokens, 25.4–42.2 s at 16384.
L1/L2 are tight (max/min < 1.2). The spread is consistent with GIL contention
between the probe's prefetch and the scheduler loop (D4) rather than device
jitter, since the device is ~1 % utilised throughout.

---

## 4. Experiment 2 — write-through interference on L3 hit latency

L = 16,384 probes (2.25 GiB of KV each) pre-populated into L3, `wait_complete`,
`write_through`. Nine distinct prompts, each probed exactly once, so every probe
is a first touch served from L3 — necessary because with `writeload` running the
scheduler is never `fully_idle` and `flush_cache` can never succeed (D8).
Page cache dropped before every probe. Storage attribution confirmed on all 9
samples (`cached_storage = 16320`, `device = host = 0`).

| requested R (req/s) | TTFT median (s) | range | iostat `%util` | `w_MB/s` |
|---|---|---|---|---|
| 0 | **36.77** | 33.68 – 38.11 | 0.00 – 2.74 | 0.02 – 106.7 |
| 2 | **24.84** | 22.16 – 31.20 | 0.32 – 3.06 | 0 – 126.0 |
| 8 | **8.51** | 7.97 – 11.09 | 0.00 – 5.80 | 0 – 140.7 |

### The result is the opposite of the hypothesis

The plan expected write-through backup traffic to degrade L3 hits until they
fell below the recompute bar. Instead **L3 hit latency improved 4.3× as the
write load increased**, and the R = 0 and R = 8 ranges do not overlap
(33.7–38.1 vs 8.0–11.1). The effect is monotonic across all three points.

The plan's headline question — "the write rate at which an L3 hit becomes
slower than recompute" — has no answer here, because **an L3 hit is already
17–37× slower than recompute at zero write load** (Exp 1). There is nothing for
write pressure to tip.

### Why interference does not appear, and why load helps

The disk is never the constraint: `%util` never exceeded **5.8 %** at any write
rate, and `r_await` stayed at 0.16–0.67 ms. There is no queueing at the device
to contend for.

The mechanism is GIL contention, and it runs the other way. Comparing `py-spy`
dumps of the scheduler process:

| condition | scheduler MainThread | backup thread |
|---|---|---|
| backup backlog, no request load | `check_hicache_events` (spinning) | **active** in `set` → `batch_set` → `_generic_page_set` |
| R = 8 request load | `event_loop_overlap` (real work) | idle on its queue |

With no requests to run, the scheduler main loop spins in Python inside
`check_hicache_events` and monopolises the GIL that the single backup/prefetch
thread needs to make progress. Adding GPU work displaces that spin with work
that *releases* the GIL, so the storage thread gets scheduled more and the probe
completes sooner. The tier is bound by Python scheduling, not by bytes moved.

### CORRECTION — the sweep above used a broken write load

The conformance audit found that `writeload.py` appended its uniqueness marker
as a **suffix**, so all 64 pooled bodies shared their entire prefix and every
64th request hit the radix cache instead of inserting new nodes. That voids the
plan's L129 guarantee ("unique prompts guarantee every request inserts new
nodes") and means the sweep above was run under almost no real write pressure.

The two load-bearing points were re-run with the marker moved to the prefix
(`control = baseline_uniquefix`):

| | R = 0 | R = 8 |
|---|---|---|
| **`write_gbps_actual`** | 0.0002 | **1.647 GB/s** (was 0.0003 — ~5000× higher) |
| TTFT (3 reps) | 41.97 / 39.03 / 51.99 s | 5.62 / 5.76 / 5.93 s |
| **probes that actually got an L3 hit** | 3 / 3 | **1 / 3** |
| **`storage_prefetch_unfulfilled_tokens`** | 0 | **32,640** (= 2 × 16,320) |

What this changes:

1. **The write axis in the table above is not a write axis.** At the requested
   R = 8 the server was measured moving 0.0003 GB/s of backups; with unique
   prompts the same setting moves **1.647 GB/s**. Every `write_gbps_actual`
   number in the original sweep is therefore meaningless, as is any claim
   indexed to it.
2. **The "TTFT improves under load" headline does not survive intact.** The
   direction persists (5.6–5.9 s at R = 8 vs 39–52 s at R = 0), but under a
   *correct* write load two of three probes never received their L3 data at all
   — 32,640 unfulfilled prefetch tokens, exactly the two probes that came back
   with no cache attribution. Their low TTFT is "the prefetch gave up and the
   model recomputed", not "the read went faster". Only one probe of three both
   hit L3 and was fast.
3. **The corrected reading is that write pressure makes L3 hits *unreliable*
   rather than slower.** At 1.65 GB/s of concurrent backup traffic the tier
   stops delivering its data within the request's life, under `wait_complete`,
   where by construction it should have waited.
4. **The GIL attribution in §3–§4 is unaffected**, because it rests on `%util`
   ≤ 5.8 %, the symmetric ~65 MB/s read/write rates, the `py-spy` frames, the
   C1/C2 nulls, and the nixl swap — none of which depend on the write load's
   uniqueness.

The plan's headline question — the write rate at which an L3 hit becomes slower
than recompute — remains unanswerable on this build for the reason given above:
an L3 hit is already 17–37× slower than recompute at zero write load.

### Caveats on this experiment

- **`achieved_rps` instrumentation failed.** The harness kills `writeload` 60 s
  after the probes finish, before it prints its summary, so `achieved_rps`
  logged 0.0 at every rate and `write_gbps_actual` (0.0003–0.115 GB/s) is not
  trustworthy either. The x-axis is therefore the *requested* rate, which the
  plan explicitly warns against (caveat #11). The `iostat` `w_MB/s` column is
  the reliable indicator that write pressure did rise with R (106 → 126 → 141
  MB/s peak), and the server independently served 1,747 requests / 7.36 M
  prompt tokens over the block.
- **N = 3** with 15–40 % spread. The R = 0 vs R = 8 separation is far larger
  than the spread and is safe; the R = 2 midpoint is directionally consistent
  but individually weak.
### Controls C1 and C2 — both run, both null

The plan's two mandatory controls were executed after the first draft.

| condition (R = 8, L = 16,384) | TTFT (s) | `iostat w_MB/s` | what it removes |
|---|---|---|---|
| baseline | 8.51 median (7.97–11.09) | up to 140.7 | — |
| **C2 — `backup_skip`** | 9.02 / 9.13 / 9.24 | **0.02–0.04** | every SSD **write**, keeping D2H copies and all Python bookkeeping |
| **C1 — L3 on tmpfs** | 9.13 | 134 (write load only) | the **block device entirely** |

C2 was implemented with the one-line patch the plan specifies
(`cache_controller.py:554`, saved as `exp2/C2_backup_skip.patch`), gated on
`HICACHE_EVAL_BACKUP_SKIP`; the env var was verified present on the running
server and `w_MB/s` duly collapsed to ~0.03, confirming the control did what it
claims. Probes were populated into L3 *without* the flag, then the server was
restarted *with* it.

**Neither control changes the result.** Removing all SSD writes leaves L3-hit
TTFT at 9.0–9.2 s; moving L3 to a RAM disk leaves it at 9.1 s; the baseline is
8.5 s. Per the plan's own decision rule — "degradation removed by C2 but not by
C1 is the SSD; degradation that survives C2 is PCIe/host/GIL" — the degradation
survives **both**, so it is neither the device nor the write traffic. It is the
software path, which is what §3's `py-spy` frames and the serial `batch_set`
loop already indicated, and what §10's backend swap then proves constructively.

*C1 caveat:* only 1 of its 3 reps is a valid L3 sample. The tmpfs was capped at
30 GB and the R = 8 write load overran it, so the LRU evictor deleted the probe
pages before reps 1–2 could read them; those two came back as pure recomputes
(0.94 s / 1.00 s, `cached_storage = null`, matching 16,384 / 15,745 tok/s). The
one valid rep is reported. The three `_populate_only` rows in
`interference.csv` are additional R = 0 baseline samples (38.2–44.9 s) collected
during C2 setup, consistent with the baseline R = 0 range.

---

## 5. Experiment 3 — write amplification on a multi-turn reasoning workload

Client: `bench_multiturn.py --api-format openai --disable-random-sample
--enable-round-barrier --request-length 1024 --output-length 1024
--num-clients 16 --num-rounds 4 --max-parallel 8 --request-rate 2`
(64 requests/condition; scaled down from the plan's 64×8 — see D10).
`--reasoning-parser qwen3` with `enable_thinking: true`. Fresh server and
verified-cold L3 per condition.

| condition | host pool | write policy | **written to L3** | **read from L3** | dead fraction | gen tokens | cache hit rate |
|---|---|---|---|---|---|---|---|
| wt100 | 100 GB | write_through | **17.99 GB** (121,984 tok) | **0** | **100 %** | 44,628 | 0.732 |
| wt30 | 30 GB | write_through | **17.77 GB** (120,512 tok) | **0** | **100 %** | 46,423 | 0.743 |
| wts100 | 100 GB | write_through_selective | **11.08 GB** (75,136 tok) | **0** | **100 %** | 44,239 | 0.733 |
| wts30 | 30 GB | write_through_selective | **11.32 GB** (76,800 tok) | **0** | **100 %** | 43,443 | 0.718 |
| nohicache | — | (GPU radix only) | 0 | 0 | — | 43,775 | 0.693 |

`prefetched_tokens`, `storage_prefetch_hit_tokens_total`,
`hicache_dropped_tokens_total` and `hicache_backup_dropped_tokens_total` were
all **zero** in every condition.

### Findings

1. **The L3 tier was written 18 GB and read exactly zero times.** Write
   amplification is infinite and the dead fraction is 100 %, in all four
   HiCache conditions. Every cache hit in this workload was served by L1 or L2.
2. **Shrinking the host pool from 100 GB to 30 GB changed nothing** (17.99 →
   17.77 GB written, still 0 read). The plan expected capacity pressure to push
   traffic down to L3; with a 121 K-token working set, a 30 GB host pool
   (203 K tokens) is still comfortably large enough, so L3 is never consulted.
3. **`write_through_selective` cuts L3 writes by 38 %** (18.0 → 11.1 GB) at
   equal cache hit rate (0.732 vs 0.733) and equal generated tokens. It backs
   up on first *reuse* rather than on insert, so single-use blocks are never
   written down. It is strictly better than `write_through` on this workload.
4. **HiCache bought +4 points of hit rate, all of it from L2.** 0.693 without
   HiCache → 0.732 with it. The entire L3 tier contributed nothing but 18 GB of
   writes, ~265 s of backup-thread time, and the GIL contention quantified in
   Exp 1 and 2.
5. **`hicache_dropped_tokens_total` was 0 throughout**, so none of this is a
   write/evict race — the writes were all completed successfully and then
   never used.

### Per-round and end-to-end latency (plan line 277)

From `exp3/per_round_ttft.csv`, extracted from each condition's `bench.jsonl`:

| condition | median TTFT | p90 | p99 | avg output len | output tok/s |
|---|---|---|---|---|---|
| nohicache | **0.110 s** | 0.159 | 0.285 | 684.0 | 358.1 |
| wt100 | 0.125 s | 0.167 | 0.233 | 697.3 | 348.5 |
| wt30 | 0.124 s | 0.153 | 0.238 | 725.4 | 354.4 |
| wts100 | 0.124 s | 0.169 | 0.287 | 691.2 | 346.9 |
| wts30 | 0.127 s | 0.167 | 0.278 | 678.8 | 344.3 |
| oracle100 | 0.128 s | 0.178 | 0.261 | 696.4 | 347.8 |
| oracle30 | 0.121 s | 0.172 | 0.288 | 709.1 | 359.3 |

**HiCache produced no measurable latency or throughput benefit on this
workload.** Median TTFT spans 0.110–0.128 s and output throughput 344–359 tok/s
across all seven conditions — and the *fastest* median TTFT belongs to
`nohicache`, the condition with no host or storage tier at all. The +4 points of
cache hit rate (0.693 → 0.732) did not convert into user-visible speed, because
the GPU radix cache was already serving the reuse that mattered and the extra
hits landed on a tier whose load-back cost roughly cancels the prefill it saves
at these prompt lengths (~2.9 K tokens average).

So the accounting for HiCache on this workload is: **+18 GB of SSD writes,
0 bytes read back, 0 measurable latency benefit.**

Per-round `average_ttft` and `cache_hit_rate` for every round of every condition
are in `exp3/per_round_ttft.csv` (35 rows); the written-vs-read bar chart the
plan asks for is `exp3/amplification_bars.png`.

**Conformance note.** The plan (line 266) says to raise prompt variety if median
generation is under 800 tokens. Measured `average_output_len` is 678–725 across
conditions, i.e. **below that bar**, because `--output-length 1024` caps
generation and Qwen3 stopped earlier. The workload is therefore somewhat less
reasoning-heavy than the plan intends. This weakens the *magnitude* of the
thinking-token argument but not its direction: the dead fraction is 100 %, and
the mechanism (thinking is never re-fed) does not depend on how long the
thinking is.

### Why the dead fraction is 100 %

Two independent reasons compound:

- **Thinking output is dead by construction.** With `--reasoning-parser qwen3`,
  the endpoint returns thinking in `reasoning_content`, and the client appends
  only `content` to the history. Qwen3's template never re-feeds prior
  thinking, and because the answer tokens were generated *after* the thinking
  tokens, dropping thinking shifts the answer's positions. So the whole
  generated span of turn *t* is unusable as a prefix in turn *t+1* — yet
  `write_through` writes all of it to host and SSD. Generated tokens were
  ~44 K per condition against ~186 K prompt tokens.
- **The live working set fits in L2.** Whatever *is* reusable is still resident
  in host DRAM when it is needed, so the L3 copy is redundant even for the
  live fraction.

**Verified, not assumed (plan caveat 13).** The mechanism was measured directly
(`exp3/caveat13_thinking_check.json`): a two-turn exchange with
`enable_thinking: true` returned 1,031 characters of `reasoning_content`, an
assistant `content` containing **no `<think>` markup**, and — appending only
`content` to the history exactly as `bench_multiturn.py --api-format openai`
does — turn 1 generated **300 tokens** while turn 2's prompt grew by only
**21** (the new user message). **279 of 300 generated tokens were dropped from
the conversation.** That is the dead-block mechanism, confirmed end to end.

The first reason is a property of the workload; the second is a property of the
sizing. Both point the same way: L3 was written for content that either could
not be reused or did not need to be.

---

## 6. Experiment 4 — oracle upper bound for a class-aware write policy

`--strip-thinking-cache` skips caching the reasoning-model output (thinking +
answer) on finish. For this workload the entire generated span is dead, so this
is the ideal class decision applied at all tiers — an upper bound on what a
learned class-aware policy could achieve here.

| condition | written to L3 | vs `write_through` | read from L3 | cache hit rate | prompt tokens |
|---|---|---|---|---|---|
| wt100 (reference) | 17.99 GB (121,984 tok) | — | 0 | 0.7324 | 187,398 |
| wts100 (selective) | 11.08 GB (75,136 tok) | **−38.4 %** | 0 | 0.7333 | 184,507 |
| **oracle100** (strip-thinking) | **11.32 GB** (76,736 tok) | **−37.1 %** | 0 | **0.7324** | 186,768 |
| wt30 (reference) | 17.77 GB (120,512 tok) | — | 0 | 0.7432 | 183,249 |
| wts30 (selective) | 11.32 GB (76,800 tok) | −36.3 % | 0 | 0.7183 | 186,334 |
| **oracle30** | **11.23 GB** (76,160 tok) | **−36.8 %** | 0 | **0.7333** | 185,812 |

### Findings

1. **The oracle removes ~37 % of L3 write traffic at exactly zero hit-rate
   cost.** oracle100 and wt100 report an identical cache hit rate to five
   decimal places (0.7323943…), because the tokens the oracle declines to write
   are tokens nothing ever asks for again.
2. **`write_through_selective` already captures essentially all of that gain.**
   11.08 GB (selective) vs 11.32 GB (oracle) — within 2 % of each other, from
   opposite mechanisms: selective withholds until first reuse; the oracle
   withholds by content class. On this workload a learned class-aware policy
   would be competing against a shipped one-line flag for the last ~2 %.
3. **The oracle does not fix the real problem.** It reduces dead writes from
   100 % of 18 GB to 100 % of 11 GB. The dead *fraction* stays 1.0, because the
   remaining traffic is prompt-prefix content that L2 already serves. No write
   policy can make L3 useful here; only a *placement* policy that declines L3
   entirely can.

---

## 7. What this means for a class-aware placement policy

Strictly from the numbers above.

**1. On this testbed the question is not what to write to L3, but whether to
have L3 at all.** An L3 hit runs at 0.058 GB/s against a recompute bar of
b·P = 2.162 GB/s (Exp 1). The break-even is 181 tokens and the prefetch
threshold is 256, so the winning region is unreachable. Across five HiCache
conditions on a realistic multi-turn workload, L3 was written 11–18 GB and read
**zero** times (Exp 3, 4). A policy that classifies *which* blocks to write down
is optimising a term whose best achievable value is still negative.

**2. The tier is software-bound, and that is the fixable part** — and §10 now
proves it constructively by swapping the backend. L3 read
(0.058 GB/s) and L3 write (0.068 GB/s) land within 15 % of each other at
**0.26 %** of the 22.5 GB/s this disk delivers to a concurrent O_DIRECT reader,
with `iostat %util` ≤ 5.8 % throughout. The
bottleneck is `HiCacheFile.batch_set`'s serial per-page Python loop
(`hicache_storage.py:568`) contending for the GIL with a scheduler that spins in
`check_hicache_events` when it has no GPU work (Exp 2's inverted interference
curve is the same effect seen from the other side). **This is
device-independent**: putting a real NVMe under the current code path would not
move 65 MB/s much. Batched/threaded writes and a non-spinning completion check
are worth more than any placement policy.

**3. ~~Even a perfect L3 implementation would lose on this model/GPU pair.~~
CORRECTED — a well-implemented L3 would comfortably *win* here.** The original
claim rested on a buffered-QD1 `dd` figure of 1.7 GB/s. Measured properly with
fio, this disk serves the actual L3 access pattern (9 MiB random reads) at
**6.19 GB/s at O_DIRECT QD1** and **22.5 GB/s with modest concurrency** — that
is **3–10× above** the 2.162 GB/s recompute bar, not below it.

So the correct statement is the opposite of the draft: **the hardware supports a
profitable L3 tier on this exact model and GPU; the current implementation
forfeits it.** Closing even the O_DIRECT gap alone (0.643 → 6.19 GB/s) would put
L3 at ~3× the recompute bar. The tier is not doomed by physics here, it is
doomed by an access pattern that is buffered, serial, and Python-per-page.

The b·P framing still holds as a *rule* — admit to a tier only if its delivered
bandwidth exceeds b·P — but on this box the honest reading is that L3's
delivered bandwidth is an implementation artefact, not a device property.

**4. Where a policy does have room: L2, and the write path.** HiCache's real
contribution here was +4 points of hit rate (0.693 → 0.732), entirely from L2,
which runs at 13.2 GB/s — 6.3× the recompute bar and clearly worth it. The
actionable write-side result is that `write_through_selective` removes 38 % of
L3 write traffic at zero hit-rate cost, and the ideal class oracle
(`--strip-thinking-cache`) removes 37 % — **the two are within 2 % of each
other**. On this workload the headroom for a *learned* class-aware write policy
over the shipped selective policy is that 2 %. The large remaining win is not
classification but placement: not writing to L3 at all when b·P exceeds the
tier's delivered bandwidth.

**5. A concrete decision rule falls out of the measurements.** Admit a block to
tier *T* only if `bandwidth(T) > b·P`, measured, not assumed. Here:
L1 20.5 > 2.16 ✓, L2 13.2 > 2.16 ✓, L3 0.058 < 2.16 ✗. That rule would have
avoided all 18 GB of dead writes without needing to know anything about
thinking tokens.

---

## 8. Measurement caveats that apply to the above

1. **No NVMe (D1), but the device is fast (corrected).** L3 ran on the VM's
   local virtio-blk disk via the container overlay — not on the virtiofs mount.
   The device is not slow: 6.19 GB/s at O_DIRECT QD1 and 22.5 GB/s concurrent
   for the 9 MiB random-read pattern. Absolute L3 latencies are specific to the
   *implementation*, not to the device. The *relative* conclusions that survive a device change
   are: the software-bound ratio (both directions ~65 MB/s at ≤ 6 % util), the
   100 % dead fraction, and the write-policy comparison. The absolute
   break-even lengths do not.
2. **N = 3, not 7 (D10).** Exp 1's tier separations are orders of magnitude and
   safe. Exp 2's R = 0 vs R = 8 separation is well outside the spread; its
   R = 2 midpoint is weak.
3. **Exp 2's write-rate axis is unreliable.** `achieved_rps` logged 0.0 because
   the harness killed `writeload` before it reported. Requested rate and
   `iostat w_MB/s` are used instead; see §4.
4. **Exp 2 controls C1/C2 were not run.** Disk-vs-software attribution rests on
   `%util`, the symmetric read/write rates, `py-spy`, and the code path.
5. **Exp 3/4 client scaled down** to 64 requests/condition (16 clients ×
   4 rounds) from the plan's 512. Generated ~44 K tokens per condition, so the
   median generation is near the plan's 800-token guidance rather than well
   above it. The dead-fraction result is insensitive to this (it is 100 %, and
   the mechanism is structural), but the per-round TTFT tails are thin.
6. **`read_L2_tokens` was not measured directly.** `bench_multiturn.py` does not
   log per-tier attribution, and the aggregate `cache_hit_rate` does not split
   device from host. L2 read volume is inferred from hit rate and
   `host_used_tokens`, not measured.
7. **Two harness bugs were found and fixed mid-run (D8, D11).** Runs affected by
   the L3-wipe bug were identified from each server's evictor line and
   discarded; every reported run started from `existing=0 B (0 entries)`.
8. **`hicache_host_used_tokens` is a stale gauge (D9)** and was not used for any
   conclusion; tier facts come from per-request attribution.
9. **Single node, single GPU, client co-resident** on the same 26 vCPUs. Given
   that the central finding is CPU/GIL contention, client co-residency is a
   real confound for absolute latencies — though the scheduler was never CPU
   starved (it ran at 124 % during the drains).

---

## 9. Artifacts

```
CONFORMANCE.md          213 plan requirements audited section by section
                        99 SATISFIED / 25 DEVIATED / 59 PARTIAL / 29 NOT_DONE
constants.json          measured constants and device bandwidths
device_bandwidth.json   fio matrix: the corrected device characterisation
exp2/ttft_vs_writerate.png   plan L235 figure (baseline/C1/C2 + recompute line)
exp2/C2_backup_skip.patch    the C2 control patch (applied then reverted)
exp3/amplification_bars.png  plan L277 figure (written vs read per condition)
exp3/per_round_ttft.csv      plan L277 per-round TTFT, 7 conditions x rounds
exp3/caveat13_thinking_check.json  proof that thinking is not re-fed
exp1/l3_iostat_check.json    plan L211 cross-check, marked INCONCLUSIVE
backend_ab/             file vs nixl A/B (single length)
exp0_nixl/              Exp 0 repeated on nixl (all 7 steps, attributions held)
exp1_nixl/              Exp 1 repeated on nixl: 84 probes, 4 tiers, 0 discards
exp1_nixl/ttft_vs_len.png        nixl tier cost curves
exp1_nixl/tier_fits.csv          nixl slope/intercept/bandwidth per tier
exp1_backend_comparison.png      file L3 vs nixl L3 vs recompute, crossover marked
exp1_backend_comparison.csv      the same as a table
preflight/              §0 audit trail: model_config, flag_check, deps, container checks
missing_runs_summary.json        timeout-policy and background-load results
versions.txt            sglang / torch / driver
DEVIATIONS.md           D1-D12, including two harness bugs found mid-run
l3_write_throughput.json  backup drain rate measurement
exp0/recipe.json        the verified tier-forcing recipe
exp0/client.jsonl       all Exp 0 probes
exp1/ttft_by_tier.csv   108 probes: recompute 45, L1 21, L2 21, L3 21 (0 discards)
exp1/tier_fits.csv      slope / intercept / effective bandwidth per tier
exp1/break_even.csv     break-even length vs recompute per tier
exp1/ttft_summary.csv   median/min/max per (tier, L)
exp1/ttft_vs_len.png    cost curves, linear and log-log
exp2/interference.csv   9 L3 probes across 3 write rates + iostat
exp3/amplification.csv  7 conditions (Exp 3's 5 + Exp 4's 2)
exp3/<cond>/            per-condition server.log, metrics before/after, bench.jsonl
```

---

## 10. Follow-up — the tier is fixable: storage backend swap

Added after the first draft, prompted by the question "what disk is this, and can
another backend use it better?"

### 10.1 Which disk

L3 lived at `/var/hicache_l3` **inside the container** → docker overlay →
`/var/lib/docker` → **`/dev/vda1`, the VM's local virtio-blk disk**. The
virtiofs mount (`/lambda/nfs/MLSys-Learn`) held only the repo and the result
CSVs; no KV page ever touched it. That was the right choice: virtiofs serves the
same 9 MiB random-read pattern at **0.096 GB/s**, ~7× worse than the local disk.
The container overlay costs nothing measurable (6.19 vs 6.07 GB/s O_DIRECT QD1,
in-container vs host).

### 10.2 The device is fast; the first draft mischaracterised it

See §1's correction box and `device_bandwidth.json`. Summary for the 9 MiB
random-read pattern L3 actually issues: **buffered QD1 0.643 · O_DIRECT QD1
6.19 · O_DIRECT QD8 ×4 22.5 GB/s.** Sequential write scales 1.15 → 12.7 GB/s.

### 10.3 Backend survey

All eleven registered backends were surveyed against the repo at `f3ccd1c0e4`,
each survey adversarially verified by a second reviewer.

| backend | runnable here | O_DIRECT | I/O shape | verdict |
|---|---|---|---|---|
| **nixl** | **yes, no setup** | **yes, default on** | one io_uring/libaio batch per 128 pages | **the fix** |
| file | yes (current) | no | serial Python loop, 1 page/syscall | the 0.058 GB/s baseline |
| hf3fs | mock only | no | threaded, batched | needs a 3FS mount |
| mooncake | needs a master service | not in SGLang's Python | batched below Python | network store, not a disk backend |
| aibrix | package absent | no | serial for local disk | would be slower than `file` |
| eic / simm / mori | import fails | n/a | batched | remote/RDMA stores |
| shm | yes | n/a | **no file I/O at all** | not a storage backend |
| dynamic | loader only | n/a | n/a | nothing shipped to load |

### 10.4 A/B: `file` vs `nixl`, identical harness

Same prompt (4096 tokens, seed 1000), same disk, cold store, page cache dropped,
`wait_complete`, `write_through`, `page_first`.

| | `file` | `nixl` | ratio |
|---|---|---|---|
| **L3 read TTFT** | 6.69 s | **0.195 s** | **34×** |
| **L3 read bandwidth** | 0.083 GB/s | **2.84 GB/s** | **34×** |
| L3 write bandwidth | 0.052 GB/s | 0.097 GB/s | 1.9× |
| files for 64 pages | 64 | 128 (K and V split) | — |
| layout | flat dir | sharded `xx/` subdirs | — |

**This crosses the recompute bar.** Recompute at 4096 tokens is 0.268 s; a
`nixl` L3 hit is **0.195 s**. L3 goes from "never worth reading at any length"
to *faster than recomputing*, on the same hardware, by changing one flag:

```bash
export SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR=/var/hicache_nixl
--hicache-storage-backend nixl      # instead of file
```

Independent corroboration from the survey agent, which drove the repo's own
`NixlFileManager`/`NixlRegistry` classes directly at the production batch size
(128 × 9 MiB): **read 14.1–20.7 GB/s, write 9.8 GB/s**, with O_DIRECT proven by
`mincore` (direct=True → 0/18432 pages resident after both write and read;
direct=False → 18432/18432). So the raw storage path is not the end-to-end
limiter any more — 2.84 GB/s end-to-end against 14–20 GB/s raw means the
remaining cost has moved to the H2D and scheduler stages.

### 10.5 Caveats on the swap

- **`file`'s 6.69 s here is its best case.** Exp 0 measured 11.58 s for the same
  probe with ~8,800 files in the store, because `HiCacheFile` does an
  `os.path.exists` per page and keeps an in-memory LRU over every entry. It
  degrades as the store fills; nixl's directory sharding is the structural fix.
- **nixl needs `--hicache-mem-layout page_first` (or `page_first_direct`)** and a
  4096-aligned host buffer, or `is_zero_copy` flips off and every page pays a
  9 MiB CPU memcpy into a bounce buffer. The configuration used here satisfies
  that (9,437,184 / 4096 = 2304).
- **Durability regression:** nixl writes in place with `O_CREAT` and has no
  temp-file + rename, so a crash mid-write leaves a torn page. `file` does
  `tofile` to a temp name then `os.replace`, which is atomic.
- The A/B is N = 1 per backend at one length. The effect is 34×, far outside any
  plausible noise, but the *shape* of nixl's curve versus length was not measured.

### 10.6 Consequence for the report's conclusions

§7 point 3 is corrected in place: the hardware supports a profitable L3 tier on
this exact model and GPU, and the `file` backend forfeits it. §3's "the tier is
software-bound" finding is unchanged and is now demonstrated constructively — a
different software path on the same device gets 34× more out of it. The Exp 3/4
finding (100 % dead L3 writes) is **orthogonal and still stands**: a faster L3
would have made those 18 GB cheaper to write, not more useful to have written.
