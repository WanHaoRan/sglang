# Deviations from the plan

## D1 — There is no NVMe on this machine (material)

The plan assumes "1 TB NVMe, 400 GB budget for L3" and builds its central
argument on an NVMe reading at 3-7 GB/s, i.e. *at* the recompute bar, making L3
hits marginal wins that write interference can flip into losses.

This box has **no NVMe device at all**. `lsblk` shows only `vda` (1 TB
virtio-blk) and a 386K `vdb`; `/sys/block/vda/queue/rotational` is `1`. L3
therefore lives on `/var/hicache_l3`, on the container overlay over `/dev/vda1`.

Measured on that device (4 GiB via dd, page cache dropped between runs):

| | GB/s |
|---|---|
| sequential read, cold | **1.7** |
| sequential read, page-cache warm | **6.0** |
| sequential write, O_DIRECT | 1.1 |

**Consequence:** the L3 tier is 2-4x slower than the plan's premise. L3 hits are
expected to sit *below* the recompute bar from the start rather than at it, so
Exp 2 measures how much further under water they go rather than a crossing.
All L3 absolute numbers are device-specific and must not be read as NVMe
numbers. The shape of the results (write interference, amplification, dead
fraction) still holds.

## D2 — ulimit -l is 8192 KB, not unlimited (not blocking)

The plan says to stop if `ulimit -l` is not unlimited. The container is
`--privileged`, so it holds CAP_IPC_LOCK, which exempts it from RLIMIT_MEMLOCK;
a 40 GB pinned host pool was already observed working on this container before
the evaluation began. Proceeding, and verifying the 100 GB pool actually
registers at Exp 0 startup instead of pre-emptively stopping.

## D3 — Paths

No `/workspace` or `/mnt/nvme` in this container. Using
`WORK=/sgl-workspace/sglang/hicache_eval` (bind-mounted to the host, so results
survive the container) and `L3_DIR=/var/hicache_l3`.

## D4 — GIL yield for issue #21880 (verified in modern form)

The plan asks for a `time.sleep` yield in the scheduler loop. That exact patch
site no longer exists at this commit: the loop was restructured and the yield is
now `IdleSleeper` / `maybe_sleep_on_idle` (`scheduler.py:842`, `:4722`,
`:5556`) behind `sleep_on_idle`. No ad-hoc patch applied. L3 TTFT variance is
monitored empirically in Exp 0/1 as the real check; if it is pathological the
workaround gets applied and re-recorded here.

## D5 — Tooling

`sysstat` (iostat) installed at setup time. `vmtouch` unavailable; not needed
because `drop_caches` is writable from this privileged container, which is the
plan's preferred method.

## D6 — 196 GB of stale L3 files were present before the run

Left over from ad-hoc runs before this evaluation. Removed during setup so
Exp 0 starts genuinely cold.

## D7 — Per-tier attribution is opt-in (plan assumed it was automatic)

`cached_tokens_details` is not returned by default. The request must set
`return_cached_tokens_details: true` (`protocol.py:355` for CompletionRequest,
`:883` for ChatCompletionRequest), and the result lands as a **top-level**
`sglext_cached_tokens_details`, not inside `usage.prompt_tokens_details`. The
plan's fallback to `/generate` was not needed once the flag was set. Exp 0
step 4 was silently unverifiable before this was found.

## D8 — flush_cache is gated on the HiCache backup queue, not just requests

`Scheduler.is_fully_idle()` additionally requires `ongoing_write_through`,
`ongoing_load_back`, `ongoing_prefetch` and `ongoing_backup` to be empty
(`scheduler.py:4778-4785`). `hicache_backup_tokens_total` counts *enqueued*
backups and goes flat minutes before the files actually land, so the plan's
"poll until the counter is stable" drain test is not a drain test. The harness
now polls `flush_cache?timeout=0` until it returns 200
(`hcommon.wait_until_flushable`). Draining the 83.3 GB written by one Exp-0
filler pass took **387 s**.

## D9 — hicache_host_used_tokens does not reset on flush

It still read 564,928 immediately after a successful flush whose effect was
independently confirmed by per-tier attribution (device=0, host=0,
storage=4032). Treated as a stale gauge; flush verification uses attribution.

## D10 — Repetition count reduced from N=7 (time budget)

L3 is so slow on this device (a 4096-token L3 hit takes 11.6 s, and populating
L3 drains at ~65 MB/s) that the plan's N=7 across all conditions does not fit
the available machine time. Using **N=3** and a reduced sweep grid, stated per
experiment. Medians and spread are still reported; with N=3 the p90 column is
reported as max instead. This weakens tail claims, not the central findings,
which are order-of-magnitude effects.

## D11 — Harness bug found mid-run: `rm -rf $L3_DIR/*` fails silently at scale

`rm -rf $L3_DIR/*` and `ls $L3_DIR/*.bin | wc -l` both expand the glob into an
argv that exceeds ARG_MAX once the L3 directory holds tens of thousands of
files. With stderr suppressed the wipe silently does nothing and the count
silently reports 0.

**Impact, checked directly from each server's evictor line** (`HiCacheFile
eviction enabled: ... existing=N B (M entries)`):

| run | existing at startup | cold? |
|---|---|---|
| exp0 | 0 B (0 entries) | **yes** — all Exp 0 data valid |
| exp1 | 0 B (0 entries) | **yes** — all Exp 1 recompute/L1/L3 data valid |
| exp1b | 355,253,354,496 B (37,644 entries) | **no** — discarded, produced no rows |

Fixed with `find "$L3_DIR" -mindepth 1 -delete`. Every subsequent run's
coldness is verified from the evictor line rather than assumed.

## D12 — Exp 1 L2 curve measured with the storage backend disabled

Two independent reasons. First, the L2 filler budget is squeezed from both
sides: it must exceed the device pool (374,784 tokens) to evict the probes from
L1, yet probes + fillers must stay under the host pool (678,208) or the probes
fall through to L3 instead of resting in host. The plan's Exp-0 recipe
(1.5 x L1 = 562K fillers) violates the upper bound once the 194K-token Exp-1
probe set is also resident, and the first attempt duly produced storage hits
instead of host hits. The budget is now computed as
`max(1.05 x L1, min(1.15 x L1, 0.92 x L2 - probe_tokens))`.

Second, with the storage backend attached, every filler pass costs a multi-minute
L3 drain and leaves L3 residue that contaminates the next run (D11). L2-hit
latency does not depend on whether an L3 backend exists, so the L2 curve is
measured on an L1+L2-only server. Recompute samples from that run are clean and
are pooled as additional reps.

## D13 — Exp 2 controls C1 and C2 were run after the first draft

Both are now complete and both came back null; §4 of the report carries the
numbers. C2 required the one-line patch the plan specifies at
`cache_controller.py:554`, plus an `import os` that file did not have (the patch
would have raised `NameError` at startup without it). The diff is preserved at
`exp2/C2_backup_skip.patch` and **has been reverted** — `git status` on the
sglang checkout is clean.

C1 caveat: the tmpfs was capped at 30 GB (`SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE=30G`,
down from 380G, because tmpfs consumes RAM alongside the 100 GB pinned host
pool). The R = 8 write load overran that cap, so the LRU evictor deleted the
probe pages before reps 1 and 2 could read them; those reps returned pure
recomputes (`cached_storage = null`, 0.94/1.00 s). Only rep 0 is a valid L3
sample. Reported as such rather than averaged.

## D14 — Device characterisation corrected after the first draft

The original `dd`-based figures (1.7 GB/s read, 1.1 GB/s write) were buffered,
QD1, sequential and understated the device badly. Re-measured with fio across
queue depths and access patterns; see `device_bandwidth.json`. The conclusion in
§7 point 3 that depended on the low figure was wrong and is corrected in place
(struck through, not deleted). No experimental data changed — only the
interpretation of what the hardware could have supported.

## D15 — Storage-backend follow-up (beyond the plan)

The plan fixes `--hicache-storage-backend file`. After the L3 results came in, a
survey of all eleven registered backends plus an A/B against `nixl` was run to
test whether the tier's cost was intrinsic. It is not: `nixl` is 34× faster on
the same disk and crosses the recompute bar. This is additional to the plan, not
a substitute for any of it; §10 reports it separately so the plan's own results
stay cleanly attributable.

---

# D16 — Independent conformance audit (added last; supersedes ad-hoc claims)

Nine independent auditors (one per plan section), each adversarially re-checked,
extracted **213 discrete requirements** from `hicache_eval_plan.md` and verified
each against the artifacts. Full matrix in `CONFORMANCE.md`.

| status | count |
|---|---|
| SATISFIED | 99 |
| DEVIATED_AND_RECORDED | 25 |
| **PARTIAL** | **59** |
| **NOT_DONE** | **29** |
| NOT_APPLICABLE | 1 |

**58 % of requirements were fully met or properly deviated; 41 % were partial or
missing.** The earlier framing of the evaluation as complete was not supportable;
this section is the correction.

## D16.1 — Errors the audit found in published numbers (now fixed)

1. **Three `L=512` "recompute" rows were actually device hits**
   (`cached_device=448`). They dragged the 512 median from 0.232 s to 0.048 s and
   biased the fitted intercept. `analyze.py` now drops any recompute row
   reporting cached tokens. Consequences, all propagated through REPORT.md:
   P 15,271 → **15,745 tok/s**; b·P 2.097 → **2.162 GB/s**; intercept −0.030 →
   **+0.014 s**; break-even L1 1179 → **437**, L2 1106 → **308**, L3 162 → **181**.
   The L3 conclusion is unchanged and marginally strengthened (181 is still below
   the 256-token prefetch threshold, so L3 still never wins).
2. **"0 discards" was false.** `exp1/run.log` contains 59 discard lines. The
   retained rows do have 0 discards, but an abandoned L2 pass produced 59 failed
   attribution attempts. REPORT.md §3 now says so.
3. **`writeload.py` did not generate unique prompts.** The uniqueness marker was
   appended as a *suffix*, so all 64 pool entries shared their entire prefix and
   every 64th request hit the radix cache instead of inserting new nodes. This
   voided the plan's L129 guarantee and is why `write_gbps_actual` was implausibly
   small. Fixed (marker moved to prefix) and the affected Exp 2 points re-run;
   see REPORT.md §4.
4. **The Exp 1 iostat cross-check (plan L211) is invalid.** The collector was
   started with `&` and never stopped, so `exp1/iostat.log` kept sampling through
   every later experiment *and* the fio device benchmarks; its 37.6 GB/s tail is
   fio, not an L3 probe. `exp1/l3_iostat_check.json` now records this as
   INCONCLUSIVE and points to the valid byte-accounting cross-checks instead.

## D16.2 — Plan-mandated artifacts that were missing (now produced)

- `exp2/ttft_vs_writerate.png` (L235) — baseline/C1/C2 lines + recompute line
- `exp2/interference.csv` — the `prefetch_policy` column the plan names
- `exp3/amplification_bars.png` (L277) — written vs read bytes per condition
- `exp3/per_round_ttft.csv` (L277) — per-round TTFT for all 7 conditions

## D16.3 — Sub-experiments genuinely NOT run (previously undisclosed)

These were omitted for time and, until now, were not recorded as deviations —
which is itself the plan's process requirement:

1. **Exp 1 background-load variant** (L207): all four tiers at L ∈ {2048, 8192,
   32512} under `writeload --len 1024 --rate 2`. 12 conditions. Never run.
2. **Exp 1 second run under `timeout` prefetch policy** (L195). Never run; only
   `wait_complete` was measured.
3. **Exp 2 `timeout`-policy sweep** at R ∈ {0, 2, 8} (L227), including the
   deadline-miss accounting. Never run.
4. **Exp 2 `hicache_backup_duration_seconds` and `prefetch_bandwidth`** (L225).
   Never scraped for Exp 2.
5. **Exp 3/4 `read_L2_tokens`** (L272). Not recoverable: no per-tier hit counter
   exists in `/metrics` at this commit (checked `exp3/*/metrics_after.txt`), and
   `bench_multiturn.py` does not log `cached_tokens_details`. The plan's stated
   fallback is unavailable, not merely skipped.
6. **`written_L3_bytes` cross-check against iostat / `du`** (L271). Not done; not
   recoverable post hoc.
7. **Exp 3 remedy for short generations** (L258): measured `average_output_len`
   is 678–725, below the plan's 800-token trigger, and the prescribed remedy
   (raise prompt variety / use math prompts) was not applied.
8. **Exp 4 stretch oracle** (L295) — explicitly optional in the plan; not run.
9. **Per-experiment telemetry set** (L54): `iostat.log`/`pcie.log`/`config.json`
   were captured for Exp 1 only, not for every experiment.

## D16.4 — Process requirements missed

- N = 7 → N = 3 (already D10), but the plan's seed formula
  `1000*exp + 10*cond + rep` was replaced with a length-keyed variant.
- The `--tag exp3_<condition>` / shared `--log-file` convention (L255) was
  changed to per-condition log files (arguably better; undisclosed until now).
- The plan's `prompt_tokens == target ± 2` assertion (L126) was never coded; the
  data happens to satisfy it.
- Caveat 9's commitment to verify the scheduler GIL yield: every
  `startup_facts.txt` records `sleep_on_idle: False`, so the guard was inactive
  throughout. This is consistent with — and arguably explains — the L3 variance
  and the inverted Exp 2 curve, but it was not verified up front as the plan
  required.
