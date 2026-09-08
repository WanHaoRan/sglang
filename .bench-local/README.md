# HiCache multi-tier evaluation — commands

Everything here runs **inside the sglang dev container**, in two shells:

```bash
sudo docker exec -it sglang_dev /bin/zsh
cd /sgl-workspace/sglang/.bench-local
```

> **Paste safety.** Do not retype these as `\`-continued lines. A trailing
> space after a backslash makes it an escaped space, the newline then ends the
> command, and the rest of your flags get run as separate commands
> (`zsh: command not found: --dataset-name`). The arrays below have no
> continuations at all, so they paste cleanly into bash and zsh alike.

---

## Shell 1 — server

```bash
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/var/hicache
export SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE=200Gi
export SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE=100Gi
mkdir -p /var/hicache ~/bench-logs
```

```bash
SERVER=(
  --model-path Qwen/Qwen3-8B
  --port 30000
  --page-size 64
  --context-length 32768
  --chunked-prefill-size 8192
  --mem-fraction-static 0.85
  --max-total-tokens 131072
  --enable-hierarchical-cache
  --hicache-size 40
  --hicache-write-policy write_through
  --hicache-io-backend kernel
  --hicache-mem-layout page_first
  --hicache-storage-backend file
  --hicache-storage-prefetch-policy timeout
  --radix-eviction-policy lru
  --enable-metrics
)
```

Foreground:

```bash
taskset -c 0-19 python3 -m sglang.launch_server "${SERVER[@]}"
```

Or backgrounded with a log, then block until it answers:

```bash
taskset -c 0-19 python3 -m sglang.launch_server "${SERVER[@]}" > ~/bench-logs/server.log 2>&1 &
```

```bash
until curl -sf -o /dev/null http://127.0.0.1:30000/health; do sleep 3; done; echo ready
```

First launch downloads ~16 GB of weights. `taskset -c 0-19` leaves cores 20-25
for the client so its own scheduling latency stays out of your TTFT numbers.

Stop it:

```bash
pkill -f "sglang.launch_server.*--port 30000"
```

---

## Shell 2 — client

```bash
CLIENT=(
  --backend sglang
  --host 127.0.0.1
  --port 30000
  --model Qwen/Qwen3-8B
  --dataset-name generated-shared-prefix
  --gsp-num-groups 256
  --gsp-prompts-per-group 8
  --gsp-system-prompt-len 4096
  --gsp-question-len 128
  --gsp-output-len 128
  --gsp-group-distribution zipf
  --gsp-zipf-alpha 1.0
  --max-concurrency 32
  --warmup-requests 0
  --seed 42
  --output-details
)
```

```bash
taskset -c 20-25 python3 -m sglang.benchmark.serving "${CLIENT[@]}" --output-file results/PHASE.jsonl
```

Use `sglang.benchmark.serving`, not `sglang.bench_serving` — the latter is a
deprecated shim. `--warmup-requests 0` matters: the default of 1 would warm the
cache before a cold-phase measurement. `--num-prompts` is ignored by this
dataset; the count is `gsp-num-groups x gsp-prompts-per-group` = **2048**.

---

## What the benchmark actually sends

`generated-shared-prefix` is a **synthetic generator, not a corpus**. Every
request is:

```
<system prompt: N random tokens>\n\n<question: M random tokens>
```

`gen_prompt` (`benchmark/datasets/common.py:79-83`) samples token ids uniformly
with replacement from the tokenizer vocabulary (151,669 usable ids for
Qwen3-8B) and decodes them. A real 32-token sample:

```
他 minusprecatedMOOTH Simpson_ctxChristian answering.gc fluent triggered숱WO毐
soapchildNodes쓩 đánhReject要比筱 spoon prerequisites Worship score延伸.game薁
Inspectable degrade pneumonia.False
```

- one system prompt per group, shared by every request assigned to it — this is
  the reusable prefix
- one unique question per request slot
- requests are shuffled unless you pass `--gsp-ordered`
- `--gsp-range-ratio 1.0` (default) makes every length exact instead of sampled

**Gibberish is fine for this experiment.** Radix-cache hit/miss is decided by
token-id prefix matching alone; content never enters the mechanism. Random
tokens buy exact length control and guarantee zero accidental sharing between
groups.

**What it does not model:** real traffic shares boilerplate across nominally
distinct prefixes, so its true cache footprint is smaller than the sum of its
prompts. This dataset is the no-incidental-sharing case — an upper bound on
distinct-prefix pressure. For realistic reuse structure use
`--dataset-name mooncake`, which replays a real trace and reconstructs prompts
from its `hash_ids` so the sharing structure matches production.

### Re-tokenization drift: prompts land ~5% longer than requested

`gen_prompt` samples token ids, decodes to text, and the prompt is sent **as
text** — the server re-tokenizes it, and decode -> encode is not a round trip.
Measured on Qwen3-8B:

| asked | re-encoded | drift |
|---|---|---|
| 4096 (system prompt) | 4286, 4338, 4293, 4309 — mean **4306** | +5.1% |
| 128 (question) | 130 | +1.6% |
| assembled prompt | mean **4441** | |

bench_serving reports the true re-encoded lengths, so its own statistics are
honest — but the working set is correspondingly bigger than
`groups x prefix-len`. See Sizing below.

At `--page-size 64` only whole pages are shared, so ~4224 of those ~4306 tokens
form the cacheable common prefix; the tail sits in a partial page that already
contains question tokens.

---

## The four phases

| phase | prepare | measures |
|---|---|---|
| 0 · cold | stop server, wipe L3, restart | no-cache prefill baseline |
| 1 · populate | nothing | L3 write path |
| 2 · **L3 read** | flush L1+L2, drop page cache | **disk read path, isolated** |
| 3 · warm | nothing | L1/L2 hit path |

```bash
# phase 0 — stop the server first, then wipe, then relaunch (shell 1)
pkill -f "sglang.launch_server.*--port 30000"
rm -rf /var/hicache/*
taskset -c 0-19 python3 -m sglang.launch_server "${SERVER[@]}" > ~/bench-logs/server.log 2>&1 &
until curl -sf -o /dev/null http://127.0.0.1:30000/health; do sleep 3; done; echo ready
```

```bash
# phase 0 (shell 2)
taskset -c 20-25 python3 -m sglang.benchmark.serving "${CLIENT[@]}" --output-file results/cold.jsonl
```

```bash
# phase 1
taskset -c 20-25 python3 -m sglang.benchmark.serving "${CLIENT[@]}" --output-file results/populate.jsonl
```

```bash
# phase 2 — the measurement
curl -s -X POST http://127.0.0.1:30000/flush_cache
sync && echo 3 > /proc/sys/vm/drop_caches
taskset -c 20-25 python3 -m sglang.benchmark.serving "${CLIENT[@]}" --output-file results/l3read.jsonl
```

```bash
# phase 3
taskset -c 20-25 python3 -m sglang.benchmark.serving "${CLIENT[@]}" --output-file results/warm.jsonl
```

**Why phase 2 is the one that matters.** `POST /flush_cache` clears L1 and L2
but leaves the L3 files on disk — `UnifiedRadixCache.reset()` ->
`_reset_full()` (`unified_radix_cache.py:343-377`) only touches the radix tree
and the host pool.
That cold-L1/L2, warm-L3 state is the only one where a prefix hit *must* come
off disk. L3 also survives a full server restart; the evictor re-indexes
existing files at startup.

**Wipe L3 only while the server is stopped.** The evictor builds its LRU byte
index at startup, so deleting files underneath a live server leaves it
counting bytes that no longer exist.

**`drop_caches` is not optional for the `file` backend.** It opens with
`buffering=0` (`hicache_storage.py:484`), which only disables Python's own
buffer — there is no `O_DIRECT`. With 221 GB of RAM the page cache would serve
most of L3 and you would publish memcpy numbers as disk numbers.

---

## Reading the counters

Run this before and after each phase and diff the two:

```bash
curl -s http://127.0.0.1:30000/metrics | awk '/^sglang:(prefetched_tokens_total|storage_prefetch_hit_tokens_total|storage_prefetch_unfulfilled_tokens_total|hicache_backup_tokens_total|hicache_backup_bytes_total|hicache_host_used_tokens|hicache_host_total_tokens)/ {n=$1; sub(/\{.*/,"",n); s[n]+=$NF} END {for (k in s) printf "%-52s %.0f\n", k, s[k]}' | sort
```

It sums each metric family across its label sets. Sample output from a real run:

```
sglang:hicache_backup_bytes_total                    915406848
sglang:hicache_backup_tokens_total                   6208
sglang:hicache_host_total_tokens                     271296
sglang:hicache_host_used_tokens                      6208
sglang:storage_prefetch_unfulfilled_tokens_total     0
```

A family missing from the output has not been touched yet — the labelled series
is only created on first use, so no `storage_prefetch_hit_tokens_total` line
means zero L3 reads so far.

| counter | reads as |
|---|---|
| `hicache_host_used_tokens` / `_total_tokens` | L2 occupancy — **check this first**; if it never saturates, nothing reached L3 |
| `hicache_backup_tokens_total` / `_bytes_total` | L3 write path |
| `storage_prefetch_hit_tokens_total` | L3 read hits — the number the whole experiment is for |
| `storage_prefetch_unfulfilled_tokens_total` | prefetch misses, labelled by `reason` |

L3 size on disk:

```bash
du -sh /var/hicache
```

---

## Sizing

Measured on this box with Qwen3-8B (36 layers, 8 KV heads, 128 head_dim, bf16):
**147,456 B per token** — confirmed exactly, 6,208 backed-up tokens =
915,406,848 bytes.

| tier | setting | tokens |
|---|---|---|
| L1 device | `--max-total-tokens 131072` | 131,072 |
| L2 host | `--hicache-size 40` | 271,296 |
| L3 file | `200Gi` | ~1,456,000 |
| **L1+L2** | | **402,368** |

A working set must exceed L1+L2 before anything is read back from L3. The
client's working set is `gsp-num-groups x <re-encoded prefix length>` — note
the *re-encoded* length, not the requested one: `256 x 4306 = 1,102,464`
tokens, clearing the threshold by **2.7x**. (Using the requested 4096 would
understate it as 1,048,576 / 2.6x.) Drop below the threshold and phase 2
measures an empty path.

Check the resolved numbers against a live server:

```bash
curl -s http://127.0.0.1:30000/get_server_info | python3 -c 'import json,sys; print("L1:", json.load(sys.stdin)["max_total_num_tokens"])'
```

---

## Second arm — `nixl`, with O_DIRECT

Same everything, two changes: swap the env var and the backend flag.

```bash
export SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR=/var/hicache
```

Then replace `--hicache-storage-backend file` with
`--hicache-storage-backend nixl` in `SERVER`. `nixl` uses `O_DIRECT` by default
(`SGLANG_HICACHE_NIXL_USE_DIRECT_IO`), so it bypasses the page cache that
`file` depends on.

There is no local SSD on this box — L3 lands on the virtio root disk either
way. Running both arms turns that limitation into the measurement: the gap
between them is the page cache's contribution.

---

For the code behind what these commands measure — radix tree construction,
eviction, and the HiCache L1/L2/L3 movement, with line anchors verified against
the tree — see [CODEMAP.md](CODEMAP.md).

Optional wrappers around all of the above: `hicache-server.sh` and
`hicache-phase.sh` in this directory, both with `--help` and `--dry-run`. The
commands above are complete on their own and need neither.
