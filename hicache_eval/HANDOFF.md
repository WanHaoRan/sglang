# HiCache evaluation — handoff

Written 2026-09-08 after two evaluation campaigns; updated 2026-09-21 for the port to the H200 box
(campaign 6) and the script cleanup that went with it. Read this before touching `hicache_eval/`. It exists
because most of the effort in this work went into discovering that measurements were wrong, not into taking
them.

**2026-09-21, two structural changes.**
1. **`scripts/` is now 14 files, not 48.** Everything the Exp 0 / Exp 1 path does not run moved to
   `scripts/archive/`, which has a README saying what each group was and when to bring it back. Nothing was
   deleted and `git log --follow` still works; a pre-2026-09-21 report that cites `scripts/exp3.py` means
   `scripts/archive/exp3.py`.
2. **The active box is an H200, not the A100.** `.current_results` points at
   `results/20260921_h200_nebius_32b70b_fp8kv/`, whose README has the box table and the run commands. The
   driver is `scripts/run_h200.sh`. **Every GPU-derived constant in §2 below was measured on an H100 or an
   A100 and does not transfer**: SM90 runs the FP8 checkpoints on native FP8 rather than the A100's
   weight-only Marlin, so `P` — the denominator of the whole admission criterion — is a different number.
   The disk changed too (1.88 GiB/s symmetric, against the A100's 0.68 read / 0.38 write).

---

## 1. What is here

| path | what |
|---|---|
| `hicache_eval_plan.md` | the original plan (Exp 0-4). Still the spec; several parts are not runnable on this host, see §5 |
| `scripts/` | the harness. `hcommon.py` is the shared library; everything else builds on it |
| `results/20260907_203035/` | campaign 1: Qwen3-8B, `file` vs `nixl`, Exp 0-4. Has `REPORT.md`, `CONFORMANCE.md` (213 requirements), `DEVIATIONS.md` (D1-D16) |
| `results/20260908_llama70b_awq_fp8kv/` | campaign 2: Exp 0/1 on Llama-70B-AWQ + Qwen3-32B-FP8, fp8 KV |
| `results/20260908_nixl_exp234/` | campaign 3: Exp 2/3/4 on nixl, all three models. `FINDINGS_LIVE.md` is the running log (24 findings), `PENDING.md` is the todo |
| `../bench-local/CODEMAP.md` | 78 verified code anchors for the radix tree + HiCache movement paths |
| `results/20260917_a100_gcp_qwen8b/` | campaign 4: Qwen3-8B rerun on a GCP A100 box with L3 on a local NVMe. Backend A/B, Exp 0, Exp 1 (Exp 2-4 not run). `REPORT.md`, `COMPARISON.md` (old vs new, same statistics both sides), `DEVIATIONS.md` |
| `results/20260917_a100_gcp_32b70b_fp8kv/` | campaign 5: Exp 0/1 for the 70B and the 32B on the same A100 box. Its `REPORT.md` has the three-model table for both boxes |
| `results/20260921_h200_nebius_32b70b_fp8kv/` | **campaign 6 (complete 2026-09-21; REPORT.md, DEVIATIONS.md, versions.txt written 2026-09-22)**: the same stages on the H200 box. Its `README.md` is the plan, the box table and the run commands |
| `scripts/run_h200.sh` | the campaign-6 driver. Every box-specific setting lives in it; the shared scripts keep their old behaviour when the env vars are unset |
| `scripts/archive/` | the 34 scripts the Exp 0/1 path does not run: Exp 2/3/4, the superseded per-campaign drivers, the one-off analyses. `archive/README.md` says why each group is there and what to do before reviving it |
| `scripts/compare_campaigns.py` | side-by-side old vs new tables (`CAMPAIGN=8b` or `c2`); run on an old campaign alone it must reproduce that campaign's published numbers — re-verified byte-for-byte against campaign 5 on 2026-09-21 |
| `scripts/plot_tier_ttft.py` | the three-model figure: TTFT by tier vs prompt length, **one row per box**, rows and panels with no data skipped so it can be made mid-campaign. `--boxes a100,h100` reproduces the two-row figure published in campaign 5 |

`.current_results` holds the active results dir; `scripts/env.sh` reads it (a pre-set `RESULTS` wins).
`.frozen_results` lists finished campaigns; `env.sh` refuses to run a driver against one, so a rerun
cannot append to or overwrite old evidence. Start a new campaign by pointing `.current_results` at a new dir.

---

## 2. The results that are solid

These reproduced across models, backends, or independent code paths. Trust them — with one 2026-09-21
qualification: **the numbers in the table below are H100 measurements, and the bar `b * P` is GPU-specific.**
The criterion is what reproduced; the values did not survive the move to the A100 (campaign 5 re-measured
every one) and will not survive the move to the H200 either. Read the table as "here is what the criterion
looked like on one box", never as constants.

**The admission criterion.** A tier is worth reading only when it delivers KV
faster than the GPU regenerates it: `bandwidth(tier) > b * P`, where `b` = KV
bytes/token and `P` = marginal prefill tokens/s. Measured:

| model | b (B/tok) | P (tok/s) | recompute bar | L3 delivered | idle L3 speed-up @16k |
|---|---|---|---|---|---|
| Qwen3-8B bf16 KV | 147456 | 15011 | 2.06 GiB/s | 3.28 GiB/s | 1.2x |
| Qwen3-32B-FP8 fp8 KV | 131072 | 2336 | 0.285 GiB/s | 3.18 GiB/s | 4.6x |
| Llama-3.3-70B-AWQ fp8 KV | 163840 | 920 | 0.140 GiB/s | 2.95 GiB/s | **15.1x** |

L3 bandwidth is model-independent (it is a disk). The bar moves 15x. **Whether
disk KV offload pays is decided by how slow the model is to prefill.**

**Backend choice dominates.** `file` vs `nixl` on identical hardware: L3 read
0.083 vs 2.84 GB/s (34x). `file` has no O_DIRECT and a serial Python
`batch_set`; it is a reference implementation, not a production path. Use `nixl`.
Everything in campaign 1's `file_backend_reference/` is archived for that
comparison only.

**`timeout` beats `wait_complete` at idle, on all three models** (1.53x / 1.90x /
1.61x), at loads where the deadline never fires. Best actionable config finding
here. Suspect the retry-poll path
(`hicache_storage_prefetch_retry_poll_interval` defaults to 0,
`..._retry_max_attempts` to 4).

**Nothing is ever read back from L3 in the Exp 3 workload.** 16-17 GiB written,
0 read, on 8B/`file`, 8B/nixl, 32B/nixl and 70B/nixl. `write_through_selective`
saves 36-38% of writes on thinking models, 11% on Llama.

**Cross-validation.** Campaign 3's measured idle recompute at L=16384 matches
campaign 2's independent Exp 1 fits to 0.2% on both the 32B (4.200 vs 4.210 s)
and the 70B (14.089 vs 14.057 s). Different code path, different day. If you
change the harness, re-run this check.

---

## 3. Traps. Every one of these produced a wrong number that was believed for a while

**Measure a paired control, not an idle baseline.** The single highest-value
change made. For every L3 probe, immediately send a *never-before-sent* prompt
of the same length under the same load; it must recompute. Comparing a loaded L3
hit against an idle recompute bar is not like-for-like and will invent
crossovers that do not exist. Both plan-mandated controls (C1, C2) turned out
invalid; this one carries the whole Exp 2 conclusion.

**Salt the recompute control's seed per process.** Fixed seeds get written to L3
by an earlier stage, and the "recompute" control silently becomes an L3 hit
(`cached_storage=16320` on a row that should show `NaN`). Always assert
`cached_storage.isna()` on control rows. `analyze_exp234.py` / `make_report.py`
now flag this automatically.

**`writeload.py` replays itself.** `--seed 7` and `--pool 64` are defaults and
`exp2.py` passes neither, so every rate point regenerates the same 64 bodies and
restarts `idx` at 0 -- replaying earlier points' prompts, which are by then in
L3. The "write load" therefore becomes a *read* load at higher rates: at R=8 the
server logged 51 prefetch hits of 4032 tokens (4.09 GB/s of L3 reads) while
`iostat_w_mbps` was 0.02. **Any Exp 2 re-run must pass a per-invocation seed and
an index offset.** This is `PENDING.md` item 7 and it invalidates the high-rate
end of the current rate axis.

**Rep index is confounded with queue depth.** Probes are ~10 s apart inside a
150 s block; by rep 2 the queue has grown and TTFT is 10-100x rep 0. With three
reps the median lands wherever the ramp happened to be. Use rep<2, or report
minimum, and say which.

**Scale the load duration to the probe time.** `--duration 150` is sized for the
8B. On the 70B a single probe takes 40-100 s, so the load expires mid-block and
later probes run against a *draining* backlog -- which is how a "recompute"
probe ends up faster than an L3 hit of the same prefix (it isn't; it just ran
later).

**`iostat_*` columns are background-only.** `iostat_sample(5)` runs *before* the
probe, so it never sees the probe's own I/O. `probe_read_mb` (via
`/proc/diskstats`) brackets the whole probe wall-clock, so it includes all
concurrent traffic -- it is an observation *window*, not the probe's bytes. An
L3 hit of 16320 tokens reads ~2.3 GB; anything larger is other traffic.

**`rm -rf $L3_DIR/*` silently fails past ~10k files** (ARG_MAX). Use
`find "$L3_DIR" -mindepth 1 -delete`. A "cold" run that was not cold cost a
whole experiment once.

**`flush_cache` is gated on full idleness** (`scheduler.py:4778-4785`), and
`hicache_backup_tokens_total` goes flat long before files land. Poll the flush
itself (`wait_until_flushable`), never the counter.

**`pkill -f <pattern>` matches your own shell.** Cost two self-kills. Collect
explicit PIDs first. The same holds for waiting: `while pgrep -f name; do sleep 5; done`
passed through `bash -lc '...'` matches its own command line and never exits (cost an idle
server and a 20-minute telemetry tail in campaign 4). Wait on a marker file or a log line, or
use the bracket form `pgrep -f '[n]ame'`.

**`py-spy record -s` hangs forever** on the scheduler (~96 GB RSS; `-s` walks
children). Killed after 4.5 h. Use `py-spy dump --pid <sglang::scheduler>
--nonblocking` -- sub-second, and it already annotates `active+gil`. See
`scripts/c3_dump.sh`.

**`HiCacheL3Cleaner` evicts under pressure** (high=80%, low=70% of the store
size). It will quietly delete the pages you pre-populated. Check
`cached_storage == 16320` on every probe row before believing a latency.

---

## 4. Claims that were made and then disproved. Do not re-derive them

| claim | why it is wrong |
|---|---|
| "`wait_complete` re-reads 9x its data" | `probe_read_mb` is a window, not amplification. Withdrawn (F1/F9) |
| "write traffic evicts L3 entries" | 111 of 114 valid probe rows are full hits; the one partial is at R=16 where the load generator had collapsed (`achieved_rps=0`) |
| "reads queue behind writes at the device" | at the operating point `iostat_w_mbps` was 0.02 while reads were 3.3 GB/s; the device was read-saturated |
| "L3 crosses recompute at R=2 on the 8B" | pooled paired difference over R=0.5-4 is +0.026 s, 95% CI [-0.095, +0.146]. A dead heat. Only R>=8 is decisive |
| "recompute degrades in lockstep on every model" | true on the 70B (queueing-dominated), false on the 8B (L3 10x vs recompute 2x) |
| "the dead writes are caused by Qwen3 thinking" | Llama-70B has no reasoning channel and still shows dead fraction 1.0. It is a *capacity* result: the working set fits in L1, so L3 is never read. `nohicache` scores within 1-3 points of every HiCache condition on all three models |
| "C2 substitutes for C1" | C2 induces an L2 pathology scaling with model size (0.83x on the 8B, 95.75x on the 70B) |

---

## 5. Parts of the plan that are not runnable here

- **C1 (L3 on tmpfs).** A tmpfs big enough plus a 100 GiB pinned host pool
  exceeds this host's 221 GiB. Measured RAM as ~100x *slower* than the SSD --
  that is memory pressure plus the cleaner evicting its own probes, not storage.
  Invalid on all three models. The plan anticipates skipping it.
- **C2 (`backup_skip`).** Not a clean "writes disabled" control on a single-rank
  server: the ack fires outside the skip guard
  (`cache_controller.py:1288-1297`) and its consumer releases the host lock, so
  L2 turns over abnormally. Damage scales with KV bytes/request.
- **Exp 3/4 cannot show a write-policy benefit at the specified working-set
  size**, on any model, because L3 is never read (see §4). To make Exp 3
  meaningful the client's working set must exceed L1+L2 -- Exp 1 does this
  deliberately (2.7x L1+L2); `bench_multiturn` with the plan's parameters does
  not. **This is the single most important change to make before re-running
  Exp 3/4.**
- **GPUDirect Storage is unavailable** (cuFile compat mode, no `nvidia_fs`, no
  `/dev/nvme`, overlay fs).
- **No NVMe.** L3 lives on `/dev/vda1`, a virtio-blk VM disk. Absolute L3
  bandwidth is a floor, not a datacenter-representative number.
  *(2026-09-21: this paragraph describes the H100 box. Campaign 5 ran L3 on a
  local NVMe at 0.68 GiB/s; campaign 6 runs it on a 2.27 TiB attached SSD that
  sustains 1.88 GiB/s in both directions. GPUDirect Storage is still
  unavailable — no `nvidia_fs` module on the H200 box either.)*

---

## 6. Where to pick up

**2026-09-22: campaign 6 is done.** `bash scripts/run_h200.sh all` ran 2026-09-21 (7 stages, 68 min), then
`compare_campaigns.py` and `plot_tier_ttft.py`; `results/20260921_h200_nebius_32b70b_fp8kv/REPORT.md` has the finding
(L3 clears the recompute bar 13.4x for the 70B and 7.5x for the 32B on this box; the 32B's L3 crossover moved up to
4096 tokens because native FP8 makes short recomputes 4-5x faster). Two follow-ups are open there: the 8B block
(`bash scripts/run_h200.sh all3`, DEVIATIONS.md D9) and device-side disk telemetry (`sysstat` is installed now; every
campaign-6 `iostat.log` is empty, D7). A new box needs the container and the model snapshots first
(`agent_cache/RUNBOOK.md` §2.3, §2.6).

Afterwards, `results/20260908_nixl_exp234/PENDING.md` is the live todo. In priority order:

1. **Re-run Exp 2 with a non-replaying write load** (per-invocation seed +
   index offset). This is the only way to separate write interference from L3
   read-path contention. Current best decomposition: in the genuinely
   write-dominated regime (R<=2, ~400 MB/s committed) L3 rises 1.13x -> 1.24x
   while the paired recompute stays flat; the 10x at R=8 is read contention.
   ~25 min on the 8B.
2. **Re-size Exp 3's working set** past L1+L2 so L3 is actually read, then
   re-run Exp 3/4. Without this the write-policy question cannot be answered.
3. C3 has only 12 samples (`c3_dump.sh`, 8B only). Directionally the HiCache
   worker threads never held the GIL, but it is thin. Re-run on all three.
4. Exp 2 `--duration` should scale with measured probe time, not be fixed at 150.

## 7. Environment

*(2026-09-21: the paragraph below describes the H100 box. On the current H200 box the measurement container
is `sglang_hicache` from `lmsysorg/sglang:nightly-dev-20260907-30705c00` — the image every A100 measurement
used, pinned so the compiled dependencies stay comparable. The repo is bind-mounted at
`/sgl-workspace/sglang`, `/mnt/ssd` is bind-mounted at the same path, and container writes are `root:root`
on the host, so `sudo chown -R wanhr:wanhr` anything you need to edit. `agent_cache/RUNBOOK.md` §2 is the
bring-up.)*

Dev happens inside `lmsysorg/sglang:dev` (`sudo docker exec -it sglang_dev
/bin/zsh`). The repo is bind-mounted at `/sgl-workspace/sglang` and is the same
inode as `/lambda/nfs/MLSys-Learn/sglang`. Files written by the container are
root-owned -- `chown -R ubuntu:ubuntu` anything you need to edit from the host.
The repo root `.gitignore` ignores `*.log`, `*.csv`, `*.png`, `*.jsonl`;
`hicache_eval/.gitignore` negates those so results are tracked.

**Never leave an eval patch in `python/`.** One was committed to `main` by
accident (`91d4805730`) and had to be removed in a follow-up. If you patch a
core file for a control, archive the diff under the results dir and revert
before committing.
