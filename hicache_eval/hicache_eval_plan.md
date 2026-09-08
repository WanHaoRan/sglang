# HiCache Multi-Tier KV Cache Evaluation Plan (H100, single node)

**Purpose.** Quantify, on stock SGLang + HiCache, (1) the TTFT of a prefix-cache hit at each tier (GPU HBM / host DRAM / local SSD) versus recompute, (2) how HiCache's `write_through` backup stream degrades L3 (SSD) hit latency, (3) how many bytes HiCache writes below the GPU that are never read back on a realistic reasoning/multi-turn workload, and (4) an upper bound on the gain from a class-aware write policy, using existing knobs as an oracle.

**Audience.** An autonomous Claude Code agent executing on the server. Follow the steps in order. Every experiment has explicit pass/fail checks; do not proceed past a failed check. Record everything under `$RESULTS` (defined below). When something deviates from this plan, write the deviation to `$RESULTS/DEVIATIONS.md` and continue only if the deviation does not invalidate the measurement.

---

## 0. Hardware, software, and fixed constants

### 0.1 Hardware (given)

| Resource | Value | Use |
|---|---|---|
| GPU | 1× H100 80 GB (PCIe Gen5 to host) | L1 tier = engine KV pool |
| vCPU | 26 | scheduler + HiCache threads + client |
| Host RAM | 221 GB | L2 tier (pinned) + OS + page cache + client |
| Local SSD | 1 TB NVMe, 400 GB budget for L3 | L3 tier (`file` backend) |

### 0.2 Model and derived constants

Model: `Qwen/Qwen3-8B`, BF16. Record these in `$RESULTS/constants.json` after verifying them from `config.json` (`num_hidden_layers=36`, `num_key_value_heads=8`, `head_dim=128`).

| Constant | Value | Derivation |
|---|---|---|
| `KV_BYTES_PER_TOKEN` | 147,456 B (144 KiB) | 2 (K,V) × 36 layers × 8 kv_heads × 128 head_dim × 2 B |
| `PAGE_SIZE` | 64 tokens | `--page-size 64` |
| `PAGE_BYTES` | 9,437,184 B (9 MiB) | 64 × 147,456 |
| GPU KV pool | ~50 GB ≈ 365K tokens | `--mem-fraction-static 0.85` on 80 GB minus 16.4 GB weights; **read the actual value from the server log** (`KV cache size` / `max_total_num_tokens`) and record it |
| Host pool `HICACHE_SIZE_GB` | 100 GB ≈ 730K tokens | leaves ≥ 100 GB for OS, model loading, client, and page-cache slack |
| L3 `L3_MAX_SIZE` | 380G | under the 400 GB budget so the file-backend LRU evictor has headroom |
| Recompute bar (b·P) | ~4 GB/s | 144 KiB/token × ~28K prefill tok/s on H100 for an 8B model. **Measure P in Exp 1; do not assume.** |

Why this testbed is interesting: a single NVMe reads at roughly 3–7 GB/s, which is *at* the recompute bar for this model/GPU pair. L3 hits are marginal wins here, so write interference is expected to flip them into losses. That is the effect Exp 2 must capture.

### 0.3 Software pinning

1. Use the dev container image `lmsysorg/sglang:latest` **or** the repo checkout already on the machine. Record `git rev-parse HEAD` of the sglang checkout and `python -c "import sglang; print(sglang.__version__)"` into `$RESULTS/versions.txt`, along with `nvidia-smi` driver/CUDA versions and `torch.__version__`.
2. Verify the scheduler/prefetch GIL fix from sglang issue #21880 is present: `grep -n "sleep" python/sglang/srt/managers/scheduler.py | head` should show a yield in the scheduler loop when storage prefetch is enabled (or the issue is closed as fixed at the pinned commit). If not present, apply the 1-line `time.sleep(0.001)` workaround from the issue in the scheduler step when `hicache_storage_backend` is set and the waiting queue is non-empty, and record this as a deviation. L3 numbers are meaningless without it.
3. Confirm these flags exist at the pinned commit (they exist on main as of 2026-09-07): `--enable-hierarchical-cache`, `--hicache-size`, `--hicache-write-policy {write_through,write_through_selective,write_back}`, `--hicache-io-backend {direct,kernel}`, `--hicache-mem-layout {layer_first,page_first,page_first_direct,...}`, `--hicache-storage-backend file`, `--hicache-storage-prefetch-policy {best_effort,wait_complete,timeout}`, `--hicache-storage-backend-extra-config`, `--radix-eviction-policy {lru,lfu,slru,priority}`, `--enable-cache-report`, `--enable-metrics`, `--reasoning-parser qwen3`, `--default-chat-template-kwargs`, `--strip-thinking-cache`, `--max-total-tokens`. Run `python -m sglang.launch_server --help | grep -c <flag>` for each; if any is missing, stop and report.
4. Python client deps inside the container: `aiohttp`, `requests`, `numpy`, `pandas`, `transformers`, `prometheus_client` (parsing only), `matplotlib`. Install missing with `pip install --break-system-packages`.
5. `iostat` (`sysstat`) and `nvidia-smi` must be available. `vmtouch` is optional (see §1.5).

### 0.4 Directory layout

```
export WORK=/workspace/hicache-eval          # or wherever the container has persistent, writable space
export SGLANG_REPO=/path/to/sglang           # the cloned repo (record it)
export L3_DIR=/mnt/nvme/hicache_l3           # MUST be on the 1 TB NVMe; verify with `df -h`
export RESULTS=$WORK/results/$(date +%Y%m%d_%H%M%S)
mkdir -p $WORK/scripts $RESULTS $L3_DIR
```

Every experiment writes: `server.log`, `client.jsonl` (one JSON per probe/request), `metrics_before.txt` / `metrics_after.txt` (raw scrape of `http://localhost:30000/metrics`), `iostat.log`, `pcie.log` (`nvidia-smi dmon -s t`), and `config.json` (the exact server and client arguments).

### 0.5 Container requirements (verify before anything else)

The container must have been started with at least:

```
--gpus all --ipc=host --shm-size 32g --ulimit memlock=-1:-1 -v /mnt/nvme:/mnt/nvme
```

Checks, in the container:

- `ulimit -l` prints `unlimited` (pinned host memory for the 100 GB L2 pool). If not, HiCache will fail at startup with a `cudaHostRegister`/pinning error. Stop and report.
- `df -h $L3_DIR` shows the NVMe device with ≥ 450 GB free.
- `free -g` shows ≥ 200 GB total.
- `nvidia-smi` shows the H100 with no other processes.
- `cat /proc/sys/vm/drop_caches` is readable; try `sync; echo 3 > /proc/sys/vm/drop_caches`. If it fails with permission denied, page-cache eviction must use the fadvise method in §1.5 (no root needed). Record which method works in `constants.json` as `page_cache_evict_method`.

---

## 1. Common procedures (write these as scripts in `$WORK/scripts/` once, reuse everywhere)

### 1.1 `start_server.sh <tag> [extra server args...]`

Base server command (all experiments share this; experiments add/remove flags as stated):

```bash
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=$L3_DIR
export SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE=380G
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B \
  --host 0.0.0.0 --port 30000 \
  --page-size 64 \
  --context-length 32768 \
  --chunked-prefill-size 8192 \
  --mem-fraction-static 0.85 \
  --enable-cache-report \
  --enable-metrics \
  --reasoning-parser qwen3 \
  "$@" 2>&1 | tee $RESULTS/<tag>/server.log
```

HiCache flags are added per experiment. The **full three-tier** set is:

```
--enable-hierarchical-cache --hicache-size 100 \
--hicache-write-policy write_through --hicache-io-backend kernel \
--hicache-mem-layout page_first \
--hicache-storage-backend file --hicache-storage-prefetch-policy timeout \
--radix-eviction-policy lru
```

Readiness: poll `GET http://localhost:30000/health` until 200 **and** the log contains `The server is fired up and ready to roll`. Then send one warm-up request (`max_tokens=1`, 64-token prompt) and discard it. Record from the log: `max_total_num_tokens` (GPU pool tokens), the HiCache host pool size line, and the storage backend attach line. Abort if the storage backend line is missing when it should be present.

Stop: `pkill -f sglang.launch_server`; wait until `nvidia-smi` shows 0 processes and port 30000 is free. **Always restart the server between configurations.** Never reuse a server across experiments.

### 1.2 Cache-state controls

| Desired state | How | Verification |
|---|---|---|
| Clear L1 + L2, keep L3 | `POST http://localhost:30000/flush_cache` (this resets the radix tree and host pool but **does not** delete L3 files — verified in `hiradix_cache.reset()` / scheduler `flush_cache`) | `sglang:hicache_host_used_tokens` metric drops to ~0; `ls $L3_DIR | wc -l` unchanged |
| Clear L3 | stop server, `rm -rf $L3_DIR/*`, start server | `ls $L3_DIR | wc -l` == 0 before first request |
| Fully cold | both of the above | — |
| Evict page cache for L3 files | §1.5 | — |

### 1.3 Probe client `probe.py`

A small Python script that sends **one** request and returns a JSON line with: `prompt_tokens`, `ttft_s` (time from send to first streamed token), `latency_s`, `cached_tokens`, and `cached_tokens_details` (`device`, `host`, `storage`), plus the experiment metadata passed on the command line.

Requirements:
- Use `/v1/chat/completions` with `stream=true`, `stream_options={"include_usage": true}`, `max_tokens=1`, `temperature=0`. `--enable-cache-report` makes the server return `usage.prompt_tokens_details.cached_tokens` and (with HiCache on) `cached_tokens_details` with per-tier counts (`device`/`host`/`storage`, defined in `srt/entrypoints/openai/protocol.py`). **In Exp 0, confirm the exact field path in the response and record it.** If the chat endpoint does not surface per-tier details at the pinned commit, fall back to the native `/generate` endpoint with `input_ids` and read `meta_info.cached_tokens` (+ details if present), and record the deviation.
- Pass `chat_template_kwargs={"enable_thinking": false}` for Exp 0–2 (probes must not think; `max_tokens=1` makes it irrelevant but keep it explicit).
- Prompt construction must be **deterministic and length-controlled**: build the user message from a seeded random selection of words from a fixed word list, then adjust until the tokenizer (`transformers.AutoTokenizer` for Qwen3-8B, applying the same chat template with `add_generation_prompt=True`) reports exactly the target token count. Target counts must be multiples of 64 (page alignment: partial pages are never cached). Record the actual `prompt_tokens` returned by the server and assert it equals the target ± 2; if not, fix the constructor before proceeding.
- Distinct prompts must differ from the first token (different seed → different first words), so no accidental prefix sharing.

### 1.4 Write-load generator `writeload.py`

Sends a stream of **unique** prompts at a fixed rate (Poisson arrivals, `--rate` req/s), each of `--len` tokens (multiple of 64), `max_tokens` = `--out` (default 1), `ignore_eos=true` when `--out > 1`, `temperature=0.7`. Unique prompts guarantee every request inserts new nodes → under `write_through` each request produces `len × 144 KiB` of D2H + L3 writes (plus `out × 144 KiB` if `out > 1`). Runs for `--duration` seconds, logs per-request latency to jsonl, and prints achieved req/s and the implied write rate in GB/s (`achieved_rps × (len + out) × 147456 / 2^30`). The generator must use asyncio with a concurrency cap (`--max-inflight`, default 32) so a slow server does not create an unbounded queue in the client.

### 1.5 Page-cache eviction for L3 files

The `file` backend reads pages with buffered I/O (`open(..., "rb", buffering=0)` + `readinto` into pinned memory; the kernel still caches the file). With 221 GB RAM, L3 reads would otherwise be served from DRAM and Exp 1/2 would be invalid. Before every L3 probe:

- Preferred (root): `sync; echo 3 > /proc/sys/vm/drop_caches`.
- Fallback (no root): evict just the L3 files:
  ```python
  import os, glob
  for p in glob.glob(os.path.join(L3_DIR, "*.bin")):
      fd = os.open(p, os.O_RDONLY); os.fsync(fd)
      os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
  ```
  or `vmtouch -e $L3_DIR`.
- Verify: `vmtouch $L3_DIR` (if available) shows ~0% resident, or `grep -E "^(Cached|Buffers)" /proc/meminfo` drops by roughly the L3 size.

Do the eviction **after** the writes have been flushed (`sync`) and **immediately before** the probe.

### 1.6 Telemetry capture

Start before each experiment block and stop after:

```bash
iostat -x -d 1 <nvme_device> > $RESULTS/<tag>/iostat.log &
nvidia-smi dmon -s t -d 1 > $RESULTS/<tag>/pcie.log &     # PCIe RX/TX MB/s
```

Scrape `http://localhost:30000/metrics` to `metrics_before.txt` and `metrics_after.txt` around each block. Counters of interest (names as of main, verify with `grep sglang: metrics_after.txt`):

`sglang:hicache_backup_bytes_total`, `sglang:hicache_backup_tokens_total`, `sglang:hicache_backup_duration_seconds`, `sglang:hicache_backup_dropped_tokens_total`, `sglang:hicache_dropped_tokens_total`, `sglang:hicache_host_used_tokens`, `sglang:hicache_host_total_tokens`, `sglang:prefetched_tokens_total`, `sglang:prefetch_bandwidth`, `sglang:prefetch_pgs`, `sglang:storage_prefetch_hit_tokens_total`, `sglang:storage_prefetch_unfulfilled_tokens_total`, plus the standard `sglang:cache_hit_rate`, `sglang:time_to_first_token_seconds` histogram, and `sglang:prompt_tokens_total`.

Compute deltas (`after − before`) per block and store them in `metrics_delta.json`.

### 1.7 Repetition and statistics

Every probe condition is measured **N = 7 times** (fresh prompt per repetition unless the condition requires the same prompt), and reported as median, p90, and max. Every throughput/steady-state condition runs for ≥ 120 s after a 30 s warm-up that is excluded. Fix seeds: probe seed = `1000 × exp_id + 10 × condition_id + rep`.

---

## 2. Experiment 0 — Sanity and tier attribution (must pass before anything else)

**Goal.** Prove that (a) each tier can be forced deterministically, (b) the per-tier `cached_tokens_details` attribution is reported and correct, (c) the L3 file backend actually stores and serves pages (guard against the low-hit-rate bug in sglang issue #27619), and (d) the page-cache eviction works.

**Server.** Full three-tier set with `--hicache-storage-prefetch-policy wait_complete` (so an L3 hit is complete, not partial). Fully cold start.

**Steps.**

1. Probe A: prompt P1 of 4096 tokens, cold. Expect `cached_tokens ≈ 0`. Record TTFT (this is a recompute sample).
2. Wait 5 s (let write-through complete). Check `hicache_backup_tokens_total` increased by ≈ 4096 and `ls $L3_DIR/*.bin | wc -l` increased by ≈ 64 files (4096/64) of 9 MiB each. If the file count is 0, the storage backend is not attached or is failing: read `server.log` and fix before continuing.
3. Probe A again immediately. Expect `cached_tokens ≈ 4096` with `details.device ≈ 4096` → **L1 hit**.
4. Force L2: send filler prompts (unique, 8192 tokens each) until the GPU pool has turned over (≥ 1.5 × `max_total_num_tokens` tokens of fillers), then probe A. Expect `details.host` ≈ 4096 (or host+device split), `details.storage == 0`. If `device` is still 4096, the GPU pool is larger than assumed; increase fillers. If the tree evicted from host too (host==0, storage>0), the fillers overflowed L2 — reduce filler volume.
5. Force L3: `POST /flush_cache`, `sync`, evict page cache (§1.5), probe A. Expect `details.storage ≈ 4096`, `device == host == 0` at request time. TTFT must be **noticeably larger** than step 3's; if it is close to step 3's, the read came from page cache — the eviction did not work; fix §1.5 before proceeding.
6. Repeat step 5 without page-cache eviction, once, and record the TTFT as `l3_pagecache_hit` for reference (this is what a naive measurement would have reported).
7. Recompute reference: stop server, `rm -rf $L3_DIR/*`, start, probe A. TTFT should match step 1 within noise.

**Pass criteria.** All expected attributions hold; L3 file count matches pages written; L3-with-eviction TTFT > L2 TTFT > L1 TTFT; L3 TTFT ≠ page-cache TTFT. Write the tier-forcing recipe that worked (exact filler volume, eviction method) to `$RESULTS/exp0/recipe.json`; Exp 1 and 2 must use it verbatim.

---

## 3. Experiment 1 — TTFT per tier versus context length (the cost curves)

**Goal.** For each tier, TTFT as a function of prompt length; extract effective bandwidth (slope) and fixed overhead (intercept); locate the break-even between each tier and recompute. Also measure prefill throughput P to compute the recompute bar b·P.

**Server.** Full three-tier set; run twice: once with `--hicache-storage-prefetch-policy wait_complete` (pure tier latency), once with `timeout` (default deadline config) to show what production sees. The L2 and L1 curves are identical between the two runs and need not be repeated.

**Conditions.** Prompt lengths `L ∈ {512, 1024, 2048, 4096, 8192, 16384, 32768 − 256}` tokens (≥ 512 because storage prefetch needs ≥ `prefetch_threshold` = 256 tokens; the last is below `--context-length` after template overhead). Tiers: `recompute`, `L1`, `L2`, `L3`. Concurrency: single in-flight probe (no other load), then repeated at moderate background load (see below).

**Procedure per (L, tier), N = 7 reps.**

- `recompute`: unique prompt each rep, fully cold L3 not required as long as the prompt is unique (it cannot hit). Record TTFT.
- `L1`: send prompt once (discard), probe again immediately.
- `L2`: send prompt once, apply the Exp 0 L2-forcing recipe, probe.
- `L3`: send prompt once, wait until `hicache_backup_tokens_total` shows it landed, `/flush_cache`, `sync`, evict page cache, probe.
- After each rep, verify the attribution matches the intended tier (from `cached_tokens_details`); **discard and redo** any rep that does not, and count discards in the results.

**Background-load variant.** Repeat the four tiers at `L ∈ {2048, 8192, 32768−256}` while `writeload.py` runs with `--len 1024 --out 1 --rate 2` (light, ~0.3 GB/s writes — enough to fill prefill batches, not enough to saturate the SSD). This shows how batching lowers the per-token recompute cost while tier loads do not benefit.

**Prefill throughput P.** From the `recompute` series: `P = L / (TTFT − intercept)` at L = 32768−256, single request; also compute from the background-load run. Record both. Compute `RECOMPUTE_BAR_GBPS = 147456 × P / 2^30`.

**Outputs.** `exp1/ttft_by_tier.csv` (`tier, L, rep, ttft_s, cached_device, cached_host, cached_storage, prefetch_policy, background_load`), and a plot `exp1/ttft_vs_len.png` with one line per tier plus recompute; a table of slope (effective GB/s), intercept (s), and break-even L per tier. Report the `iostat` read MB/s during L3 probes as an independent check on the slope.

**Expected shape (for sanity, not for tuning).** L1 slope ≈ 0 (indices only). L2 slope ≈ 144 KiB × L / (20–45 GB/s). L3 slope ≈ 144 KiB × L / (2–6 GB/s), with a visible intercept from per-page file opens. Recompute slope ≈ L / P. If L3 is faster than L2, the page-cache eviction failed.

---

## 4. Experiment 2 — Write-through interference on L3 hit latency

**Goal.** Measure L3-hit TTFT as a function of the concurrent backup (write) rate that `write_through` generates, and attribute the degradation to SSD contention vs. PCIe/host contention vs. Python/GIL contention using two controls.

**Server.** Full three-tier set, `--hicache-storage-prefetch-policy wait_complete` (so the probe measures the full read rather than giving up), `--hicache-write-policy write_through`.

**Probe.** A fixed set of 7 prompts of L = 16384 tokens (≈ 2.25 GiB of KV each) pre-populated into L3 before the block (send once each, wait for backup, verify files exist). Each measurement: `/flush_cache`, `sync`, evict page cache, probe one of the 7 (round-robin), verify `details.storage ≈ L`.

**Write-rate sweep.** `writeload.py --len 4096 --out 1` at rates `R ∈ {0, 0.5, 1, 2, 4, 8, 16}` req/s, run continuously for the whole block. Implied write rate ≈ `R × 4096 × 144 KiB` = `R × 0.56 GB/s` (so 16 req/s ≈ 9 GB/s of intended writes, far above any single NVMe's write capacity). At each R: 30 s warm-up, then interleave 7 probes spaced ≥ 10 s apart (page-cache eviction each time), then read `metrics_delta`. Record `achieved_rps` (the server may not sustain the higher rates — the achieved rate, not the requested one, is the x-axis), `hicache_backup_bytes_total` delta / block duration (actual write GB/s), `hicache_backup_duration_seconds` delta (time the backup thread spent), `storage_prefetch_unfulfilled_tokens_total` delta, `prefetch_bandwidth`, and iostat `w_MB/s`, `r_MB/s`, `r_await`, `w_await`, `%util`.

**Also run** the same sweep with `--hicache-storage-prefetch-policy timeout` at `R ∈ {0, 2, 8}` and record how many probe tokens were *not* loaded in time (`cached_storage` < L, and `storage_prefetch_unfulfilled_tokens_total`), i.e. how often the deadline turns a paid-for read into a recompute.

**Controls (mandatory).**

- **C1 — disk removed:** point `SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR` for the *write load* at a tmpfs? Not possible per-request (one backend). Instead: run the R = 8 condition with the storage directory on a tmpfs (`mount -t tmpfs -o size=120G tmpfs /mnt/tmpfs_l3`, or `/dev/shm` if large enough; if neither is possible, skip C1 and record the deviation). Pre-populate the probe prompts there too. Page-cache eviction is meaningless on tmpfs (skip it). If probe TTFT degradation vs. R = 0 largely disappears, the cause is the SSD; if it persists, the cause is PCIe/host memory or GIL.
- **C2 — L3 write path skipped:** run R = 8 with the backup to storage disabled but D2H writes kept. `backup_skip` is not a CLI flag (it is set only for MLA non-rank-0). Apply a temporary one-line patch in `python/sglang/srt/managers/cache_controller.py` after `self.backup_skip = (...)`: `self.backup_skip = self.backup_skip or os.environ.get("HICACHE_EVAL_BACKUP_SKIP") == "1"`. Start the server with that env var set. Pre-populate probes **without** the env var (restart), then restart with it set and run the sweep point. This keeps GPU→host copies and Python bookkeeping but removes SSD writes. Degradation that survives C2 is PCIe/host/GIL; degradation removed by C2 but not by C1 is the SSD.
- **C3 — GIL check (optional but cheap):** with R = 8 and `--out 1`, record `py-spy dump --pid <scheduler pid>` a few times, or use `py-spy top` for 20 s, to see whether the scheduler thread is waiting on the GIL held by `backup_thread_func` / `prefetch_io_aux_func`. Record the top 10 frames.

**Outputs.** `exp2/interference.csv` (`requested_rps, achieved_rps, write_gbps_actual, prefetch_policy, control, rep, ttft_s, cached_storage, unfulfilled_tokens, iostat_r_mbps, iostat_w_mbps, r_await_ms, util_pct`), plot `exp2/ttft_vs_writerate.png` with lines for baseline, C1, C2, plus a horizontal line at the recompute TTFT for L = 16384 from Exp 1. The headline number is the write rate at which an L3 hit becomes slower than recompute.

---

## 5. Experiment 3 — Write amplification on a realistic reasoning workload

**Goal.** On a multi-turn workload with Qwen3 thinking enabled, measure how many tokens/bytes HiCache writes to L2 and L3 versus how many are ever read back from each tier, and the resulting per-tier hit-rate under capacity pressure.

**Why this workload.** With `--reasoning-parser qwen3`, the OpenAI-compatible endpoint returns thinking in `reasoning_content`, and `bench_multiturn.py --api-format openai` appends only `content` to the conversation history. Qwen3's chat template therefore never re-feeds prior thinking, and because the answer tokens were generated *after* the thinking tokens, their positions shift when thinking is dropped — so **the entire generated output of turn t (thinking + answer) is dead for prefix caching in turn t+1**. HiCache still writes all of it through to host and SSD under `write_through`.

**Server.** Full three-tier set, `--hicache-storage-prefetch-policy timeout`, plus `--default-chat-template-kwargs '{"enable_thinking": true}'`. Two host-pool sizes: `--hicache-size 100` and `--hicache-size 30` (capacity pressure). Fully cold start for each.

**Client.** From `$SGLANG_REPO/benchmark/hicache`:

```bash
python3 bench_multiturn.py --model-path Qwen/Qwen3-8B --port 30000 \
  --api-format openai --disable-random-sample \
  --request-length 1024 --output-length 2048 \
  --num-clients 64 --num-rounds 8 --max-parallel 16 --request-rate 4 \
  --ready-queue-policy random --disable-auto-run --enable-round-barrier \
  --tag exp3_<condition> --log-file $RESULTS/exp3/bench.jsonl
```

Note `--output-length` is `max_tokens`; the model may stop earlier. Record the actual generated token counts from the server's `sglang:generation_tokens_total` delta. If the median generation is < 800 tokens, the workload is not reasoning-heavy enough; raise `--request-length` variety or use prompts that induce longer reasoning (e.g., math word problems from a fixed list), and record the change.

**Conditions (each a fresh server, cold L3):**

1. `write_through`, `--hicache-size 100`
2. `write_through`, `--hicache-size 30`
3. `write_through_selective`, `--hicache-size 100`
4. `write_through_selective`, `--hicache-size 30`
5. No HiCache (`--enable-hierarchical-cache` omitted) — GPU-only radix cache baseline.

**Measurements per condition** (from `metrics_delta`, `bench.jsonl`, `iostat`):

- `written_L2_tokens` = `hicache_backup_tokens_total` delta (host writes are what feed L3; if a separate host-write counter exists at the pinned commit, record both).
- `written_L3_bytes` = `hicache_backup_bytes_total` delta; cross-check with `iostat` cumulative writes and `du -sh $L3_DIR`.
- `read_L3_tokens` = `storage_prefetch_hit_tokens_total` delta; `read_L2_tokens` from the sum of `cached_tokens_details.host` over all requests (requires `--enable-cache-report` and a client that records it; if `bench_multiturn.py` does not log it, wrap its request function or run a parallel counter via `/metrics` `sglang:cache_hit_rate` and the per-tier gauges).
- **Write amplification** = `written_L3_bytes / (read_L3_tokens × 147456)` and the L2 analogue; **dead fraction** = 1 − read/written.
- Per-round TTFT (from `bench.jsonl` round metrics), p50/p99 TTFT, input token throughput.
- Total generated tokens vs. total prompt tokens (to compute the expected dead fraction analytically and compare).

**Outputs.** `exp3/amplification.csv` and a bar chart of written vs. read bytes per tier per condition; a table of per-round TTFT per condition.

---

## 6. Experiment 4 — Oracle upper bound for a class-aware write policy

**Goal.** Estimate the gain available from not writing dead blocks below the GPU, using an existing knob as the oracle for this workload.

**Oracle.** SGLang has `--strip-thinking-cache`: "Skip caching reasoning-model output (thinking + answer) in the radix tree on finish; keep only the prompt prefix." For the Exp 3 workload this is exactly the ideal class decision (the whole generated span is dead), applied at all tiers. It is therefore an upper bound on what a learned class-aware policy could achieve on this workload, and a slight over-estimate of a policy that only changes the *write-down* decision (it also frees L1 earlier).

**Conditions (fresh server, cold L3, same client command as Exp 3):**

1. `write_through` + `--strip-thinking-cache`, `--hicache-size 100`
2. `write_through` + `--strip-thinking-cache`, `--hicache-size 30`
3. (Reference) Exp 3 conditions 1–4 already measured.

**Measurements.** Same as Exp 3. Report deltas versus Exp 3 condition 1/2: write bytes per tier, L3 read latency (`prefetch_bandwidth`, `iostat r_await`), per-round TTFT p50/p99, throughput, and L2/L3 hit tokens.

**Stretch (only if time permits, and only after 1–2 are complete).** A finer oracle for *non*-thinking mode, where answers are re-fed and live but tool outputs/long generations are not: patch `HiRadixCache._inc_hit_count` to skip `write_backup` for nodes whose tokens were decode-generated (tag nodes at insert time in `cache_finished_req` by comparing the node's token range with `len(req.origin_input_ids)`), and rerun Exp 3 with `enable_thinking: false`. Record the patch as a diff in `$RESULTS/exp4/oracle_patch.diff`. This is optional; do not let it block the report.

---

## 7. Report

Produce `$RESULTS/REPORT.md` with:

1. Environment table (versions, pool sizes actually observed, measured P and b·P).
2. Exp 1 figure + table: per-tier slope/intercept/break-even; single vs. background-load.
3. Exp 2 figure: L3-hit TTFT vs. actual write GB/s, with C1/C2 lines and the recompute line; the crossover write rate; attribution conclusion (disk / PCIe / GIL) with the evidence.
4. Exp 3 table: written vs. read bytes per tier per condition, dead fraction, per-round TTFT.
5. Exp 4 table: oracle vs. write_through/selective deltas.
6. A "what this means for a class-aware placement policy" paragraph, strictly derived from the numbers above, and a list of measurement caveats that apply.

Attach all CSVs and PNGs. Keep raw jsonl/logs in place.

---

## 8. Caveats and failure modes (read before starting; re-read when a number looks surprising)

1. **Page cache masquerading as SSD.** The single most likely way to get wrong-but-plausible L3 numbers. Always evict before an L3 probe; verify in Exp 0 that eviction changes the TTFT.
2. **`timeout` prefetch policy blends tiers.** Under `timeout`, a partial L3 load + recompute of the rest is reported as one TTFT. Use `wait_complete` for pure-tier latency; use `timeout` only where the plan says so, and always record `cached_tokens_details` to know what actually happened.
3. **`/flush_cache` does not clear L3.** That is what makes L3-only probes possible; it also means "cold" runs need `rm -rf $L3_DIR/*` with the server stopped.
4. **Page alignment.** Only full 64-token pages are cached/matched. Use prompt lengths that are multiples of 64 and check `prompt_tokens` from the response; chat-template overhead shifts counts, so build prompts through the same template with the tokenizer.
5. **Storage prefetch threshold.** Prefetch from L3 is skipped below 256 tokens (`prefetch_threshold`). Probes must be ≥ 512 tokens.
6. **Write-through admission semantics.** `_inc_hit_count` runs on *insert*; the first insert counts as hit 1. So `write_through` backs up on request finish (before any reuse) and `write_through_selective` backs up on the first reuse. The storage write is unconditional once the host copy acks. Expect L3 file counts to track inserted pages exactly under `write_through`.
7. **Unbounded L3 growth.** Without `SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE` the file backend never evicts and will fill the disk. Keep it at `380G`; watch `df` in long runs.
8. **Pinned memory.** `--hicache-size 100` pins 100 GB; startup may take a minute for registration. `ulimit -l` must be unlimited. If startup OOMs, lower to 80 and record.
9. **GIL contention (issue #21880).** Without the scheduler yield, TTFT variance under L3 prefetch is huge and every L3 number is noise. Verify in §0.3.
10. **File-backend metadata cost.** `batch_exists` is an `os.path.exists` per page; with hundreds of thousands of files this becomes a visible intercept in Exp 1 and grows over long runs. Note file counts in each report; do not compare L3 intercepts across runs with very different file counts.
11. **Achieved vs. requested write rate.** In Exp 2, the server will not sustain the highest rates; always plot against the *measured* write GB/s (`hicache_backup_bytes_total` delta / seconds) and `iostat w_MB/s`, not the requested req/s.
12. **Backup queue lag.** After a burst, backups keep draining for a while. Wait for `hicache_backup_tokens_total` to stop increasing (poll every 2 s until stable for 10 s) before flushing/probing in Exp 1/2.
13. **Thinking mode and history.** The dead-block argument in Exp 3 depends on `--reasoning-parser qwen3` being active so that `reasoning_content` is separated from `content` and not appended to history. Verify on one request that the returned assistant `content` has no `<think>` text and that the next turn's `prompt_tokens` ≈ previous prompt + answer tokens only.
14. **Output-length 1 hides decode.** Exp 1/2 deliberately use `max_tokens=1` to isolate prefill; Exp 3/4 include real decode. Do not generalize Exp 2's absolute numbers to decode-heavy load without Exp 3.
15. **Variance and tails.** Storage latency is long-tailed; report medians and p90/p99, never means alone; N = 7 minimum per point; discard and re-run reps whose tier attribution is wrong, and report how many were discarded.
16. **One server per configuration.** Never change HiCache flags on a running server for these experiments (the runtime attach/detach API exists but leaves state that confounds counters). Restart, and confirm the metrics endpoint counters reset to zero (or record the baseline scrape).
17. **Client on the same box.** 26 vCPUs are shared by scheduler, HiCache threads, tokenizer, and the client. Keep client concurrency modest (`--max-parallel ≤ 16`, writeload `--max-inflight ≤ 32`) and pin nothing; if `top` shows the client starving the scheduler (scheduler process < 100% CPU while load is high), reduce client concurrency and record it.
18. **Model download.** First start downloads ~16 GB from Hugging Face; keep `HF_HOME` on the NVMe, outside `$L3_DIR`, and set `HF_HUB_ENABLE_HF_TRANSFER=1` if available.
19. **Disk budget.** L3 380 GB + model 16 GB + HF cache + results must fit in 1 TB. Check `df` before Exp 3 and after each condition.
20. **Do not tune while measuring.** The plan fixes flags on purpose. If a run looks bad, diagnose (logs, py-spy, iostat) and record; do not change flags to make it look better unless the plan explicitly allows it.

---

## 9. Execution order and rough time budget

| Step | Wall time (est.) |
|---|---|
| §0 setup, model download, checks | 30–45 min |
| §1 scripts + Exp 0 | 1–1.5 h |
| Exp 1 (two prefetch policies, background variant) | 2–3 h |
| Exp 2 (sweep + C1 + C2 + timeout points) | 3–4 h |
| Exp 3 (5 conditions × ~15–20 min each) | 2 h |
| Exp 4 (2 conditions) | 1 h |
| Report | 30 min |

Total ≈ 10–13 h of machine time. Run Exp 0 → 1 → 2 → 3 → 4 strictly in order; each later experiment depends on artifacts (recipe, P, recompute line) from earlier ones.
