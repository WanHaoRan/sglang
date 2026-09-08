# Storage backend A/B — `file` vs `nixl`

Identical harness (`scripts/backend_ab.py`), identical prompt (4096 tokens,
seed 1000), identical disk, cold store each time, page cache dropped before the
L3 probe, `wait_complete` prefetch, `write_through`.

| | `file` | `nixl` | ratio |
|---|---|---|---|
| **L3 read TTFT** | 6.69 s | **0.195 s** | **34× faster** |
| **L3 read bandwidth** | 0.083 GB/s | **2.84 GB/s** | **34×** |
| L3 read, page-cache warm | — | 0.118 s (4.8 GB/s) | — |
| **L3 write bandwidth** | 0.052 GB/s | **0.097 GB/s** | 1.9× |
| backup drain for 4096 tok | 11.4 s | **5.8 s** | 2.0× |
| files written | 64 | 128 (K and V split) | — |
| bytes written | 603,979,776 | 603,979,776 | identical |
| on-disk layout | flat dir, one file per page | **sharded `xx/` subdirs** | — |
| O_DIRECT | **no** (`buffering=0` only) | **yes** (`SGLANG_HICACHE_NIXL_USE_DIRECT_IO`, default True) | — |

## Why this matters

The recompute bar for this model/GPU is **2.10 GB/s** (Exp 1).

- `file` at 0.083 GB/s is **25× below** the bar → an L3 hit can never pay.
- `nixl` at 2.84 GB/s is **1.35× above** the bar → an L3 hit pays.

At 4096 tokens: recompute 0.268 s, `nixl` L3 hit **0.195 s**, `file` L3 hit
6.69 s. Switching backend flips L3 from "never worth reading" to "worth
reading", on the same hardware, with no code change.

## Caveat on the `file` number

`file` measured 6.69 s here against 11.58 s in Exp 0 step 5. The difference is
store size: 64 files here versus ~8,800 in Exp 0. `HiCacheFile.set/get` does an
`os.path.exists` per page and the evictor keeps an in-memory LRU over every
entry, so its cost grows with the number of files — the plan's caveat #10.
`nixl` shards into `xx/` subdirectories, which is the structural fix for that.

Both `file` numbers are far below the bar, so the conclusion is unchanged; but
`file` degrades further as the store fills, while this A/B measured it at its
most favourable.
