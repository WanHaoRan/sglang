# file-backend runs (archived reference, not the headline result)

Everything here was measured on `--hicache-storage-backend file`, which the
evaluation plan fixed as the backend. The measurements are sound; the backend
is not representative of what the hardware can do.

`HiCacheFile` is a **reference implementation**, not a production path:

| | |
|---|---|
| I/O | `open(path,"rb",buffering=0)` — buffered, **no O_DIRECT** (`hicache_storage.py:484`) |
| batching | `batch_set` is a serial Python `for` loop over `set` (`:568`) — one page per syscall |
| per page | `os.path.exists` + evictor reserve under a lock + `tofile` + `os.replace` + commit |
| layout | one flat directory, one file per 9 MiB page; cost grows with file count |
| delivered | **0.058 GB/s read, 0.068 GB/s write** = ~0.26 % of what this disk gives a concurrent O_DIRECT reader |

Consequences visible in this data: L3 hits 17–37× slower than recompute, a
100 % dead-write fraction on the multi-turn workload, and an inverted
write-interference curve caused by GIL contention rather than device pressure.

**Keep this for:** the evidence that the default backend is not production-ready,
and the Exp 3/4 write-amplification result (whose *dead fraction* is a property
of the workload, not the backend — only the byte volumes are file-specific).

**Do not cite these numbers as HiCache's L3 performance.** Use `../nixl/`.
