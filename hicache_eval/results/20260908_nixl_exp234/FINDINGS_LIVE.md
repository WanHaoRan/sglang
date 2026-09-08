# Live findings (appended as stages land; verify before the report)

## F1 — WITHDRAWN (was: `wait_complete` shows 9x L3 read amplification)

**Disproved by F9 below. There is no read amplification.** The raw table
here is still correct; the interpretation was not. Kept for the record.

Qwen3-8B, nixl, L=16384 probe whose KV is 2394 MB. `probe_read_mb` brackets the
probe call with /proc/diskstats.

| R | wait_complete read | timeout read | wait_complete TTFT | timeout TTFT |
|---|---|---|---|---|
| 0 | 2304 MB | 2301 MB | 0.781 s | 0.509 s |
| 2 | 2307 MB | 2305 MB | 0.850 s | 1.457 s |
| 8 | **20961 MB** | 2305 MB | **6.643 s** | 5.213 s |
| 16 | 6607 MB | - | 5.377 s | - |

Same probe, same write load, same device. The write load cannot be the source:
the timeout arm reads the correct 2.3 GB under identical pressure. The
amplification is specific to `wait_complete` at saturation, and it accounts for
the TTFT: 21 GB at the device's loaded read rate is ~6 s.

Working hypothesis: under L2 pressure the prefetch's pages are evicted before
the operation completes and are re-read, and `wait_complete` retries instead of
giving up. TO VERIFY against `prefetched_tokens_total` vs
`storage_prefetch_hit_tokens_total` deltas and the server log.

## F2 — the timeout deadline discards an entire paid-for read

Timeout policy at R=8: `storage_prefetch_unfulfilled_tokens_total` delta is
16320 -- exactly the probe length -- and one of three reps came back with no
storage hit at all and only 89 MB read, i.e. it fell back to recompute. This is
the plan's "how often the deadline turns a paid-for read into a recompute", and
the answer at saturation is: all of it.

## F3 — Exp 2 crossover, Qwen3-8B nixl

L3-hit TTFT crosses the recompute bar (0.920 s at L=16384) between R=4
(0.931 s) and R=8 (6.643 s). Flat at ~0.9 s from R=0 to R=4.

## F4 — C1 (tmpfs): the SSD accounts for only 12% of the degradation

Qwen3-8B, R=8, L=16384.

| L3 location | TTFT (3 reps) | median |
|---|---|---|
| `/dev/vda1` (SSD) | 6.165 / 6.643 / 6.648 | 6.643 s |
| tmpfs (RAM) | 5.816 / 5.687 / 5.837 | 5.816 s |

Removing the storage device *entirely* recovers 12% of a 7x degradation. Per the
plan's decision rule ("if it persists, the cause is PCIe/host memory or GIL"),
the cause is not the SSD.

### CAVEAT — C1 is contaminated, redo queued

The 60 GiB tmpfs ran 100% full. `HiCacheL3Cleaner` (high=80%, low=70%) was
evicting throughout, and all three C1 probes returned `cached_storage: null` --
their L3 pages were very likely unlinked before the probe read them. At an
achieved 6.4 req/s x 4096 tok x 147456 B the write load produces ~3.9 GB/s, so
it overruns any tmpfs this host can hold within ~15 s.

The *direction* survives the confound (a store that evicts can only make tmpfs
look worse, and it still beat the SSD), but "an L3 read served from RAM takes
5.8 s" is not established. Redo with --hicache-size 30 and a ~150 GiB tmpfs,
logging L3 file count around each probe.

## F5 — the confound C1 exposes applies to the whole of Exp 2

At R=8 the server is ~2x oversubscribed (capacity P/4096 = 3.7 req/s), so probe
TTFT carries queueing delay that would hit a recompute exactly as hard. The
plan's headline compares a *loaded* L3 hit against the *idle* recompute bar from
Exp 1 (0.920 s), which is not a like-for-like comparison.

exp2.py now sends a paired never-seen prompt immediately after each L3 probe,
under the same load. If L3-hit and recompute converge at high R, the degradation
is queueing, not storage. Live from the C2 stage onward; the baseline sweeps
need the re-run already queued in PENDING.md.

## F6 — the dead-write result is a workload property, not a backend artifact

Qwen3-8B, nixl, 4-round thinking workload. Compare with the `file` backend run
of the same conditions:

| condition | nixl written | nixl read back | `file` written | `file` read back |
|---|---|---|---|---|
| wt100 | 16.92 GiB | **0** | 16.75 GiB | **0** |
| wt30 | 16.84 GiB | 0 | 16.55 GiB | 0 |
| wts100 | 10.50 GiB (-38%) | 0 | 10.32 GiB (-38%) | 0 |
| wts30 | 10.43 GiB | 0 | 10.55 GiB | 0 |
| nohicache | 0 | 0 | 0 | 0 |

Write amplification is infinite on both. This matters because on the `file`
backend the result was dismissible -- that backend is 34x too slow to read from,
so of course nothing read back. nixl reads at 3.28 GB/s and is well above this
model's recompute bar, and *still* not one byte is ever read back. The dead
write is caused by the workload's reuse structure (Qwen3 thinking output is
never re-fed, and shifts the positions of the answer tokens behind it), not by
the storage path.

`write_through_selective` cuts writes 38% at both backends. `nohicache` has the
*highest* cache hit rate of all five conditions (0.7432 vs 0.7333 for wt100):
at this working-set size L1 alone holds the reuse, so every byte HiCache moves
below the GPU is pure overhead.

## F7 — the oracle saves nothing over the policy that already ships

Exp 4's premise is that `--strip-thinking-cache` is the *ideal* class decision
for this workload (the whole generated span is known-dead), so it upper-bounds
what a learned class-aware write policy could achieve.

| condition | L3 written | vs wt100 | cache hit rate |
|---|---|---|---|
| wt100 (`write_through`) | 16.92 GiB | - | 0.7333 |
| wts100 (`write_through_selective`) | 10.50 GiB | **-37.9%** | 0.6986 |
| **oracle100 (`--strip-thinking-cache`)** | **10.47 GiB** | **-38.1%** | 0.7013 |
| nohicache | 0 | -100% | **0.7432** |

The oracle beats the shipping heuristic by 0.2 percentage points. On this
workload the entire headroom for a class-aware write policy is already captured
by `write_through_selective`, which needs no class signal at all. Reproduces the
`file`-backend result (-38.1% vs -37.9% there too), so it is not backend-specific.

The oracle is also not free: it lowers the cache hit rate slightly (0.7013 vs
0.7333) because it frees L1 earlier, and `nohicache` still beats every HiCache
condition on hit rate.

## F8 — with the paired control, the L3 advantage is stable under write load

Qwen3-32B-FP8, nixl, L=16384. Both columns measured seconds apart under
identical load; the recompute column is a never-before-sent prompt
(`cached_storage` NaN on every row, so the control is honest).

| requested R | achieved R | L3 hit | recompute | L3 speed-up |
|---|---|---|---|---|
| 0 | 0.00 | 0.982 s | 4.200 s | 4.3x |
| 0.25 | 0.27 | 0.815 s | 4.179 s | 5.1x |
| 0.5 | 0.52 | 0.845 s | 4.193 s | 5.0x |
| 1 | 1.09 | 0.676 s | 4.182 s | 6.2x |
| 2 | 1.82 | 1.136 s | 4.272 s | 3.8x |

The recompute line is flat at ~4.2 s across the whole sweep and the L3 line
stays near 1 s, so the advantage does not erode with write pressure -- it holds
at 4-6x. At R=4 one rep saturated (L3 16.6 s, recompute 21.3 s) and L3 *still*
won.

Harness cross-check: the R=0 recompute of 4.200 s matches the independent
Exp 1 fit for this model at L=16384 (4.2098 s) to 0.2%.

### This reframes F3/F5 rather than contradicting them

On Qwen3-8B the idle L3 advantage is only ~1.8x, so a few seconds of queueing
swamps it and L3 appears to "lose" at R=8. On the 32B the advantage is 4-6x and
survives the same treatment. The variable is not the write load -- it is how
much margin the model's recompute cost gives L3 to begin with, which is the b*P
criterion from the earlier report reappearing under load.

## F9 (RESOLVED by F23) — probe_read_mb is an observation window, not amplification; and a pure write load generates GB/s of reads

`iostat_r_mbps` samples the 5 s *before* each probe, with the write load running
and no probe in flight. It is therefore background device read traffic alone:

| R | qwen8b bg reads | qwen32b bg reads |
|---|---|---|
| 0 - 2 | ~0 MB/s | ~0 MB/s |
| 4 | **2382 MB/s** | **2117 MB/s** |
| 8 | **4083 MB/s** | - |
| 16 | 2262 MB/s | - |

So `probe_read_mb` = the probe's own ~2.3 GB + (background rate x probe
duration). For qwen8b at R=8 that predicts 2.3 + 4.08 x 6.6 = 29 GB against 21 GB
observed -- right mechanism, right order. The implied read rate
(`probe_read_mb / ttft_s`) is 1.2-3.6 GB/s at *every* point in both models,
i.e. the device simply reads at its usual rate for however long the probe's
wall-clock window happens to be. **F1 is withdrawn.**

### Open question: why does a write-only load produce reads at all?

The load sends unique prompts with `--out 1`, so it should be pure write. Yet at
R=4 on qwen8b the background read rate (2382 MB/s) is close to the implied write
rate (4 req/s x 4096 tok x 147456 B = 2.4 GB/s) -- roughly 1:1 read:write.
Candidates: a read-modify-write in the nixl write path, a read-back in the
existence check during prefix lookup, or L3 cleaner activity. Not resolved here;
needs a dedicated run with the probe removed entirely. Recorded as an open item
rather than guessed at.

## F10 — `timeout` beats `wait_complete` at low load, on both models

L=16384, R=0, same server config apart from the prefetch policy:

| model | `wait_complete` | `timeout` | ratio |
|---|---|---|---|
| Qwen3-8B | 0.781 s | 0.509 s | 1.5x |
| Qwen3-32B-FP8 | 0.982 s | 0.518 s | 1.9x |

Reproducible across two models. The policy that is *willing to give up* is
consistently faster than the one that waits, at loads where neither ever
actually gives up (`unfulfilled_tokens` = 0 on every 32B row). So the cost is
not the deadline -- it is something `wait_complete` does while waiting. Worth a
look at the retry-poll path
(`hicache_storage_prefetch_retry_poll_interval` defaults to 0,
`..._retry_max_attempts` to 4).

Qwen3-32B timeout sweep, with the paired control:

| R | achieved | L3 | recompute | speed-up | unfulfilled |
|---|---|---|---|---|---|
| 0 | 0.00 | 0.518 s | 4.206 s | **8.1x** | 0 |
| 0.5 | 0.51 | 0.638 s | 4.194 s | 6.6x | 0 |
| 2 | 1.75 | 8.192 s | 11.550 s | 1.4x | 0 |

The deadline never discarded a read on this model at any rate tested, unlike
Qwen3-8B where R=8 lost the whole 16320-token prefetch (F2). At R=2 both arms
saturate together and the advantage compresses to 1.4x -- the same
compression-under-saturation seen in the baseline sweep.

## F11 — C1 (tmpfs) is not runnable on this host; C2 answers its question better

Qwen3-32B, R=2, L=16384:

| L3 location | rep 0 | rep 1 | rep 2 | median |
|---|---|---|---|---|
| `/dev/vda1` (SSD) | - | - | - | **1.136 s** |
| tmpfs (RAM) | 6.608 | 11.176 | 20.386 (evicted) | **11.176 s** |

RAM came out **10x slower than the SSD**, and the third probe lost its pages
entirely (`cached_storage` NaN). That is not a measurement of RAM; it is a
measurement of memory pressure. A 60 GiB tmpfs plus a 100 GiB pinned host pool
is 160 GiB of 221 GiB, and the write load then overruns the tmpfs (1.07 GB/s at
R=2 fills it in ~56 s), so `HiCacheL3Cleaner` thrashes on top of that. Both
models' C1 runs are contaminated the same way.

**Conclusion: C1 as specified cannot be run on this host.** The plan anticipates
this ("if neither is possible, skip C1 and record the deviation").

C2 answers the same question strictly better, with no confound: with L3 writes
disabled the device sits at **0.1% util with no write traffic**, the probe reads
exactly its 2.31 GB with no amplification, and TTFT is still 5.09 s. The device
is provably idle while the latency persists, which is what C1 was supposed to
establish by removing the device.

## F12 — C2 (`backup_skip`) behaves inconsistently across models; treat its absolute numbers with caution

`HICACHE_EVAL_BACKUP_SKIP=1` is confirmed present in the live scheduler
environment, and `iostat_w_mbps` drops to ~0 in both runs, so the control does
what it says. But the effect on TTFT is opposite on the two models:

| model | R | baseline TTFT | C2 TTFT | C2 achieved rps | C2 device writes |
|---|---|---|---|---|---|
| Qwen3-8B | 8 | 6.643 s | **5.094 s** (0.77x) | 6.15 | ~0 MB/s |
| Qwen3-32B | 2 | 1.136 s | **11.708 s** (10.3x) | 1.74 | ~0 MB/s |

On the 32B, disabling L3 writes made TTFT 10x *worse* at a slightly *lower*
achieved load with an idle device. The three reps climb monotonically
(5.7 / 11.7 / 17.1 s) and the server log shows the queue growing
(`#running-req` 2 -> 11), so the block is sitting on the saturation knee:
measured batched prefill is ~7745 tok/s = 1.9 req/s, against 1.74 achieved.

A plausible mechanism is that the ack path still fires with the write skipped
(`ack_backup_queue.put` runs outside the `if not self.backup_skip` guard,
cache_controller.py:1288-1297), and its consumer releases the host lock
(`dec_host_lock_ref`, unified_radix_cache.py:2571-2581), so L2 entries turn over
faster than they otherwise would. NOT VERIFIED -- stated as a candidate only.

### Correction to what I said earlier

I claimed C2 "answers C1's question strictly better". That was too strong. What
survives is the direct observation, which does not depend on interpreting
`backup_skip`: **in C2 the device is idle (0.02-0.1% util, ~0 MB/s) while TTFT
is seconds.** Latency without device activity is established. The *size* of the
C2 effect is not trustworthy, and C2 is not a clean substitute for C1.

## F13 — dead-write result confirmed on a second model (3 independent runs)

| run | wt100 written | read back | wts saving |
|---|---|---|---|
| Qwen3-8B, `file` | 16.75 GiB | **0** | -38% |
| Qwen3-8B, nixl | 16.92 GiB | **0** | -37.9% |
| Qwen3-32B-FP8, nixl | 16.53 GiB | **0** | -35.7% |

Two models, two backends 34x apart in read bandwidth, three runs: not one byte
is ever read back from L3, and `write_through_selective` saves 36-38% every
time. `nohicache` remains competitive on hit rate (0.7375 vs 0.7209 for wt100
on the 32B).

The write volume is also near-identical across models despite different KV
sizes per token (147456 vs 131072 B), because the workload -- not the model --
sets how many tokens get written.

## F14 — Exp 4's oracle adds nothing, on both models

L3 bytes written vs `write_through` at hicache-size 100:

| model | `write_through_selective` | `--strip-thinking-cache` (oracle) | oracle's edge |
|---|---|---|---|
| Qwen3-8B | -37.9% | -38.1% | 0.2 pp |
| Qwen3-32B-FP8 | **-35.7%** | **-35.7%** | **0.0 pp** |

On the 32B the two are identical to three significant figures (10.63 vs
10.62 GiB). Exp 4 was designed to upper-bound what a *learned* class-aware write
policy could achieve, by giving the policy perfect knowledge of which spans are
dead. That upper bound is already reached by a heuristic that ships today and
uses no class signal.

Taken with F13, the practical conclusion for this workload is that the
interesting decision is not *which* blocks to write down but *whether* to write
them down at all: `nohicache` writes nothing and still holds the best or
near-best hit rate in both models.

## F15 — Llama-70B shows the largest L3 advantage measured, and it survives load until saturation

L=16384, paired control, nixl. Partial (rates 0.5/1/2 still running):

| requested R | achieved R | L3 hit | recompute | speed-up |
|---|---|---|---|---|
| 0 | 0.00 | 0.900 s | 14.089 s | **15.7x** |
| 0.1 | 0.09 | 0.838 s | 14.027 s | **16.7x** |
| 0.25 | 0.26 | 8.516 s | 19.684 s | 2.3x |

Harness cross-check: the R=0 recompute of 14.089 s matches the independent
Exp 1 fit for this model at L=16384 (14.0575 s) to 0.2% -- the same agreement
seen on the 32B, from a completely separate code path.

The 70B saturates at a far lower write rate than the smaller models (already
degrading at an achieved 0.26 req/s, against ~1.8 for the 32B), because each
4096-token write-load request costs ~4.5 s of prefill at 920 tok/s. So its
usable-load window is narrow -- but inside that window L3 is worth 16x, the
largest advantage anywhere in this evaluation.

### The three models line up on one axis

Idle L3 speed-up at L=16384: 8B 1.8x, 32B 4.3x, 70B 15.7x. This tracks the b*P
recompute bar from the earlier report (2.06 / 0.285 / 0.140 GiB/s) rather than
anything about the storage path, which delivers ~3 GB/s for all three.

## F16 — rep index confounds with queue depth; the median across reps is not a stable statistic

Llama-70B baseline, raw TTFT per rep (probes are ~10 s apart inside a 150 s
block of sustained write load):

| R | L3 reps | recompute reps |
|---|---|---|
| 0 | 0.6 / 1.2 / 0.9 | 14.1 / 14.1 / 14.1 |
| 0.1 | 0.8 / 1.1 / 0.7 | 14.0 / 14.0 / 14.0 |
| 0.25 | 0.7 / 8.5 / **46.2** | 14.0 / 19.7 / **31.9** |
| 0.5 | 1.0 / 0.8 / **24.0** | 14.2 / 14.0 / **49.2** |
| 1 | 0.8 / 1.2 / **33.2** | 14.0 / 14.0 / **107.7** |
| 2 | 1.1 / 0.8 / **94.1** | 14.2 / 14.0 / **108.0** |

Reps 0 and 1 are stable at every rate (L3 ~0.8-1.2 s, recompute ~14 s, i.e.
14-16x). Rep 2 blows up at every rate above 0.1. This is the queue growing over
the block, not the write rate: the third probe simply arrives later, behind more
backlog. So rep index is confounded with elapsed time under load, and with only
three reps the median lands on whichever value the ramp happens to reach.

That is why R=0.25 looks anomalous in the medians table (8.5 s, "2.3x") while
0.5 / 1 / 2 look fine (14.6x / 11.7x / 12.7x) -- the ordering is an artifact of
where each triple's middle value fell.

**Corrects the reading of F15's R=0.25 row.** The honest summary for this model
is: below saturation L3 is worth ~15x, and once the queue builds both arms
degrade together with L3 still ahead (3.2x at the worst point, R=1 rep 2:
33.2 s vs 107.7 s).

Two consequences for the report:
- Report reps 0-1 separately from rep 2, or report the minimum rather than the
  median, and say so.
- `achieved_rps` reads 0.000 at R=2 because no write-load request completed
  inside the window at all -- the server was fully backlogged.

Control sanity for this leg: 0 of 18 recompute rows reported a storage hit.

## F17 — `timeout` beats `wait_complete` at idle on all THREE models

| model | `wait_complete` | `timeout` | ratio |
|---|---|---|---|
| Qwen3-8B | 0.781 s | 0.509 s | 1.53x |
| Qwen3-32B-FP8 | 0.982 s | 0.518 s | 1.90x |
| Llama-3.3-70B-AWQ | 0.900 s | 0.560 s | 1.61x |

Three models, same direction, 1.5-1.9x, at a load where the deadline never
fires (`unfulfilled_tokens` = 0 on every row). Promotes F10 from an
observation to a reproducible result: the default `wait_complete` costs ~1.6x
on an idle L3 hit for no benefit. Best single actionable config finding here.

Llama-70B, `timeout`, R=0: L3 **0.56 s** vs recompute **14.12 s** = **25.2x**,
the largest L3 advantage measured in this evaluation.

## F18 — the paired control is only valid in steady state (limitation of my own fix)

The L3 probe and its recompute twin are sequential, not simultaneous. When the
queue is draining they are not exposed to the same conditions, and the pairing
inverts:

| | L3 probe | recompute probe (runs second) | "speed-up" |
|---|---|---|---|
| 70B timeout R=0.25 rep2 | 40.57 s | 23.31 s | **0.57x** |
| 70B timeout R=1.0 rep2 | 93.22 s | 75.87 s | **0.81x** |

A recompute cannot genuinely beat an L3 hit of the same prefix on this model
(idle: 14.1 s vs 0.56 s, a 25x gap). The later probe is simply advantaged
because it runs after some backlog has drained.

Likely mechanism: `writeload --duration 150` expires while the probes are still
running. On the 70B a single probe can take 40-100 s, so rep 2 often executes
after the load has stopped, against a draining backlog. The load window is sized
for the 8B and is too short for the slower models.

**Trustworthy subset:** R=0 rows (no load at all), and reps 0-1 at rates where
the system is still in steady state. Rep-2 rows at high rates measure the drain
transient, not the write rate. The fix for any re-run is to scale `--duration`
to the measured probe time rather than fixing it at 150 s.

### F11 addendum — C1 invalid on all three models

Llama-70B, R=1: SSD reps 0.78 / 1.20 / 33.17 s (3/3 storage hits) vs tmpfs reps
97.40 / 104.88 / 43.48 s (1/3 storage hits). RAM measured ~100x slower than the
SSD, and two of three probes lost their pages to the cleaner. C1 is contaminated
on 8B, 32B and 70B alike; it is a memory-pressure measurement, not a storage
one. Recorded as a plan deviation, not a result.

Incidental confirmation of F18: C1 rep2's recompute returned 14.25 s -- exactly
the *idle* recompute figure for this model -- proving the write load had already
stopped by the time that probe ran.

## F19 — C2 (`backup_skip`) is not a valid control either; it induces a pathology that scales with model size

Steady-state reps (0-1), L3-hit TTFT with L3 writes disabled vs the baseline at
the same rate:

| model | R | baseline | C2 | ratio |
|---|---|---|---|---|
| Qwen3-8B | 8 | 6.404 s | 5.336 s | **0.83x** (helps) |
| Qwen3-32B-FP8 | 2 | 0.902 s | 8.700 s | **9.65x** (hurts) |
| Llama-3.3-70B-AWQ | 1 | 0.991 s | **94.896 s** | **95.75x** (hurts) |

The 70B's C2 rep0 is 96.5 s immediately after the 30 s warm-up, against 0.78 s
for baseline rep0 at the same rate and the same point in the block. The only
difference is `HICACHE_EVAL_BACKUP_SKIP=1`, and the damage scales monotonically
with KV bytes per request.

Candidate mechanism (NOT verified): with the write skipped, pages are never
persisted, so L2 fills with data that can never be demoted to L3, and the write
path stalls waiting for host memory. Bigger model -> more KV per request -> L2
fills sooner. The ack still fires outside the skip guard
(cache_controller.py:1288-1297), so the controller does not observe the
shortfall.

**Both Exp 2 controls are therefore unusable on this host:** C1 measures memory
pressure (F11), C2 induces an L2 pathology. Retracting my earlier claim that C2
substitutes for C1 -- it does not, and its absolute numbers should not be
quoted.

### What survives, and it is stronger than either control

The paired recompute probe *is* the control the experiment needed. Recompute
touches no storage at all, yet it degrades in lockstep with the L3 hit at every
rate on every model (70B baseline: L3 0.78/1.20/33.17 s against recompute
14.0/14.0/107.7 s; both stable early, both blowing up on rep 2). If storage were
the bottleneck the recompute line would be flat. It is not.

That argument needs neither C1 nor C2, rests on the trustworthy steady-state
subset, and reproduces on three models. It is the conclusion Exp 2 should
report.

Clean idle numbers from the C2 populate stage, Llama-70B:
L3 0.74 / 0.93 / 1.19 s vs recompute 13.94 / 14.00 / 14.00 s = 12-19x.

## F20 — harness defect: Exp 3 server logs are shared across models and were overwritten

`exp3.py` derives the server tag from the condition name alone
(`tag = f"exp3_{name}"`), so all three models write into the same
`$RESULTS/exp3_wt100/`, `exp3_wt30/`, ... directories and each model overwrites
the previous one's `server.log` / `server_args.txt` / `startup_facts.txt`.

**Data is unaffected**: `amplification.csv` and `summary.json` are written under
per-model `exp3_<key>/` directories and are complete for all three models. Only
the server-side logs for the 8B and 32B Exp 3 runs are lost.

Verified against the live process rather than the overwritten file that the 70B
Exp 3 server is configured correctly: model `casperhansen/llama-3.3-70b-instruct-awq`,
`--kv-cache-dtype fp8_e5m2`, `--hicache-storage-backend nixl`,
`--hicache-size 100`, and correctly NO `--reasoning-parser` and NO
`--default-chat-template-kwargs enable_thinking`.

One-line fix for any future run: `tag = f"exp3_{os.environ.get('MODEL_KEY','')}_{name}"`.

## F21 — CORRECTION: the dead writes are a capacity result, not a thinking-token result

I attributed F6/F13 to Qwen3 thinking output never being re-fed. Llama-3.3-70B
has no reasoning channel at all -- its generated answers *are* re-fed -- and the
dead fraction is still exactly 1.0.

| model | thinking | wt100 written | read back | dead frac | `wts` saving | nohicache hit | wt100 hit |
|---|---|---|---|---|---|---|---|
| Qwen3-8B | yes | 16.92 GiB | 0 | 1.0 | **-37.9%** | 0.7432 | 0.7333 |
| Qwen3-32B-FP8 | yes | 16.53 GiB | 0 | 1.0 | **-35.7%** | 0.7375 | 0.7209 |
| Llama-70B-AWQ | **no** | 15.29 GiB | 0 | 1.0 | **-11.0%** | 0.7882 | 0.8105 |

### The correct decomposition

**Dead fraction 1.0 is a capacity result.** Nothing is read back from L3 because
the workload's live working set fits in L1, so L2 and L3 are never consulted.
The proof is in the hit rates: `nohicache` (GPU radix cache only) scores within
1-3 points of every HiCache condition on all three models. If L1 alone achieves
the same hit rate, L3 cannot be contributing, and every byte written below the
GPU is dead by construction. Thinking has nothing to do with it.

**The `write_through_selective` saving IS a thinking result.** It saves 36-38%
on the two thinking models and only 11% on Llama -- a 3.4x difference. That is
the signature of write-through-selective declining to persist generated spans
that are never reused: on Qwen3 the whole thinking+answer span is dead, on Llama
the answer is genuinely re-fed and legitimately cacheable.

So the earlier claim was right about *which knob helps and why*, and wrong about
*why nothing is read back*. Exp 4's null result (F14) is unaffected and is in
fact explained by this: with L3 never read, no write policy can improve hit rate,
so the oracle has nothing to win.

### What this means for the evaluation

Exp 3/4 as specified cannot show a write-policy benefit at this working-set
size, on any model. To make L3 reads happen the working set has to exceed
L1+L2 -- which is what Exp 1 did deliberately (256 groups x 4306 tokens =
2.7x L1+L2) and what Exp 3's `bench_multiturn` parameters do not. That is a
plan-level gap, not a run failure.

## F22 — CORRECTION and the real Exp 2 result: write interference degrades L3 specifically, but only where queueing does not mask it

Qwen3-8B baseline re-run, steady-state reps, paired control (0/21 recompute rows
reported a storage hit):

| R | achieved | L3 hit | recompute | speed-up |
|---|---|---|---|---|
| 0 | 0.00 | 0.786 s | 0.967 s | 1.23x |
| 0.5 | 0.50 | 0.888 s | 0.938 s | 1.06x |
| 1 | 1.09 | 0.911 s | 0.952 s | 1.05x |
| 2 | 2.05 | 0.973 s | 0.933 s | **0.96x** |
| 4 | 4.08 | 1.134 s | 0.981 s | **0.86x** |
| 8 | 5.97 | **7.903 s** | **1.919 s** | **0.24x** |
| 16 | - | 6.701 s | 1.724 s | 0.26x |

**Recompute barely moves** (0.97 -> 1.92 s, 2x) while **L3 blows up 10x**
(0.79 -> 7.90 s). The write load is not degrading the server generally; it is
degrading L3 reads specifically. L3 falls behind recompute from R=2 and is 4x
*worse* at R=8.

### This corrects F19

I wrote that recompute "degrades in lockstep with the L3 hit at every rate on
every model". That is true on the 70B and false on the 8B. The two models differ
because of what saturates first:

- **Qwen3-8B**: prefill is fast (15k tok/s, recompute of 16320 tokens ~1 s) and
  the server keeps up (5.97 achieved of 8 requested). Nothing queues, so the
  write path's interference with L3 reads is exposed cleanly. **Storage
  contention is real and is the dominant effect.**
- **Llama-70B**: prefill is slow (920 tok/s, recompute ~14 s), the server
  saturates at ~0.2 req/s, and queueing delay swamps everything. Both arms
  degrade together and the storage effect is invisible underneath.

So Exp 2's original hypothesis is *confirmed* -- on the model where it can be
seen. My earlier "it's all queueing" reading generalised the 70B to all three,
which the 8B re-run refutes.

### Net answer to the plan's headline question

"The write rate at which an L3 hit becomes slower than recompute", measured
against a paired control rather than an idle bar:

| model | crosses at | at max rate tested |
|---|---|---|
| Qwen3-8B | **R = 2 req/s** | 0.24x (4x worse than recompute) |
| Qwen3-32B-FP8 | never in range | 3.99x faster |
| Llama-70B-AWQ | never in range | 12.71x faster |

The crossover exists and is low for the model with the smallest idle margin, and
is never reached for the models with large margins -- the same b*P story, now
measured under load.

## F23 — RESOLVES F9, and corrects F22's mechanism: the "write load" becomes a READ load at high rates

`writeload.py` defaults to `--seed 7` / `--pool 64`, and `exp2.py` never passes a
seed. Every rate point spawns a fresh writeload whose `idx` restarts at 0 and
whose 64-body pool is regenerated identically. So each rate point **replays the
previous points' exact prompts**, which by then are sitting in L3.

Direct proof, `finish_qwen8b/server.log`, the 8 s of the R=8 rep0 probe:

    51 x  "HiCache prefetch success ... completed=4032 matched=0 loaded=4032"   <- writeload prompts
     1 x  "HiCache prefetch success ... completed=16320 matched=0 loaded=16320" <- the probe

221,952 tokens loaded from storage in 8 s = 27,744 tok/s = **4.09 GB/s of L3
reads**, matching the independent `iostat_r_mbps` of 3340 MB/s. `matched=0`
confirms these missed L1/L2 and came off disk.

This is exactly F9's unexplained "write-only load produces GB/s of reads".
Answer: the load was not write-only. **F9 is resolved.**

### Corrected regime decomposition (Qwen3-8B rerun, rep<2)

| R | bg writes | bg reads | regime | L3 TTFT | vs idle | recompute |
|---|---|---|---|---|---|---|
| 0 | 230 MB/s | 0.04 | - | 0.786 s | 1.00x | 0.967 s |
| 0.5 | 377 MB/s | 0.21 | **write-dominated** | 0.888 s | 1.13x | 0.938 s |
| 1 | 244 MB/s | 0.28 | **write-dominated** | 0.911 s | 1.16x | 0.952 s |
| 2 | 397 MB/s | 0.04 | **write-dominated** | 0.973 s | 1.24x | 0.933 s |
| 4 | 606 MB/s | 1474 | mixed | 1.134 s | 1.44x | 0.981 s |
| 8 | 0.02 MB/s | 3340 | **read-dominated** | 7.903 s | 10.05x | 1.919 s |

Two distinct effects, previously conflated:

1. **Write interference is real but modest.** In the genuinely write-dominated
   regime (R<=2, up to ~400 MB/s of committed L3 writes), L3 TTFT rises
   monotonically 1.13x -> 1.16x -> 1.24x while the paired recompute stays flat
   (0.94-0.95 s). Clean signal, small magnitude.
2. **L3 read-path contention is severe.** At R=8 the probe's 16,320-token
   prefetch is one of ~52 concurrent prefetches saturating the storage path at
   ~4 GB/s. L3 10x, recompute 2x.

The dramatic 10x in F22 is therefore mostly *read* contention, not write
interference. F22's direction (L3-specific degradation) stands; its attribution
to write traffic does not.

### Harness fix required for any re-run

Pass a per-invocation seed and index offset to `writeload.py` so rate points do
not replay each other:
`--seed $((7 + rate_index))` plus an `--idx-offset` argument. Without that, the
Exp 2 rate axis conflates write rate with accumulated L3 read demand.

## F24 — C3 (GIL attribution): the HiCache background threads never held the GIL

Method note: `py-spy record -s` HANGS INDEFINITELY here (4.5 h, killed) because
`-s` walks the launcher's children and the scheduler carries ~96 GB RSS.
`py-spy dump --pid <scheduler> --nonblocking` returns in under a second and
already annotates threads `active+gil` / `active` / `idle`, which is what C3
needs. Use `c3_dump.sh`, not `c3_pyspy.sh`.

12 samples on the Qwen3-8B scheduler under R=8 write load (achieved 6.08 req/s):

| thread state | count |
|---|---|
| idle | 62 |
| active | 7 |
| **active+gil** | **3 (all MainThread)** |

Frames holding the GIL: `to_dec_params` (base_prefix_cache.py:148),
`os.encode`, `check_hicache_events` (unified_radix_cache.py:3002).

`backup_thread_func` and `prefetch_io_aux_func` never appear as GIL holders in
any sample; the HiCache worker threads are idle in 62 of the observations. The
plan's hypothesis -- "the scheduler thread is waiting on the GIL held by
backup/prefetch" -- is NOT supported.

Weak evidence (12 samples over ~25 s, only 3 with any GIL holder), so treat as
directional. It is nonetheless the only C3 data collected, and it points the
same way as F23: the bottleneck is the storage read path, not Python contention.
