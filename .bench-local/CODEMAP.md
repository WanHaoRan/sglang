# HiCache prefill-cache code map

Where to read in the sglang tree for the multi-tier prefill-cache evaluation:
radix tree construction, eviction, and HiCache L1/L2/L3 movement.

Paths are relative to `python/sglang/srt/`. **Every line number here was
verified against the tree at commit `f3ccd1c0e4`** — re-check after a rebase,
this file drifts.

> **The class that actually runs is `UnifiedRadixCache`.** `HiRadixCache`
> (`mem_cache/hiradix_cache.py:82`) is defined but **never instantiated
> anywhere**, and plain `RadixCache` (`mem_cache/radix_cache.py:324`) is only
> built by `create_simulated()` (`:373`) for the waiting-queue estimate in
> `managers/schedule_policy.py`. Do not read either one expecting to see what
> your server does.

---

## Read these four first

The prefill-cache story is four functions deep. Everything else is plumbing.

| # | what it answers | where |
|---|---|---|
| 1 | how a cache hit is decided | `mem_cache/unified_cache/unified_tree_core.py:730` `_match_prefix_helper` |
| 2 | demote to L2 vs drop outright | `mem_cache/unified_cache/unified_tree_core.py:1368` `evict_device_leaf` |
| 3 | the L2 -> L3 handoff | `mem_cache/unified_radix_cache.py:1455` `_finish_write_through_ack` |
| 4 | the L3 -> L2 return path | `managers/cache_controller.py:964` `prefetch` |

---

## A. Initialization

| step | where |
|---|---|
| Scheduler asks for the cache | `managers/scheduler.py:577` `kv_cache_builder.build_kv_cache(...)`; result unpacked at `:611` |
| Backend selection chain | `mem_cache/registry.py:80` `default_radix_cache_factory` |
| Dispatch | `mem_cache/registry.py:228` `create_tree_cache` |
| Your branch | `mem_cache/registry.py:149` `_create_unified_radix_cache` -> `UnifiedRadixCache(params)` at `:190`, `cache.init_hicache(...)` at `:195` |
| Cache constructor | `mem_cache/unified_radix_cache.py:149` |
| Tree constructor | `mem_cache/unified_cache/unified_tree_core.py:387` (class), `:394` (`__init__`) |
| Eviction strategy resolved | `unified_tree_core.py:408` -> `mem_cache/utils.py:70` `get_eviction_strategy` |
| Evictable-leaf set | `unified_tree_core.py:472` |
| HiCache wiring | `mem_cache/unified_radix_cache.py:379` `init_hicache` |
| Host pool + controller built | `mem_cache/hybrid_cache/hybrid_pool_assembler.py:350` |
| L2 sizing math | `mem_cache/pool_host/base.py:150` (`host_size * 1e9 // size_per_token`) |

**`storage_prefetch_threshold` defaults to 256** (set in `init_hicache`).
Prefixes shorter than 256 tokens are never prefetched from L3 — that is the
`reason="below_threshold"` label on `storage_prefetch_unfulfilled_tokens_total`.

## B. The tree

| | |
|---|---|
| Node | `unified_tree_core.py:108` `UnifiedTreeNode` — `component_data` per `ComponentType`, each with `value` (device) / `host_value` (L2) |
| Intrusive LRU list | `unified_tree_core.py:175` `UnifiedLRUList` |
| Key type | `mem_cache/radix_cache.py:59` `RadixKey` |
| Page alignment | `radix_cache.py:150` `page_aligned` — truncates to `len // page_size * page_size` |
| Divergence search | `radix_cache.py:185` `match_at` — galloping + binary search, no per-token Python loop |
| Child dict key | `radix_cache.py:229` `child_key_at` — hashable tuple of `page_size` tokens, namespaced by `extra_key` / `cache_salt` |
| Match entry | `unified_tree_core.py:703` `match_prefix` -> `:730` `_match_prefix_helper` |
| Insert (per prefill chunk) | `mem_cache/unified_radix_cache.py:941` `cache_unfinished_req` |
| Insert (on finish) | `unified_radix_cache.py:852` `cache_finished_req` |
| Edge split | `unified_tree_core.py:1177` `_split_node` |

`_match_prefix_helper` tracks **two** frontiers under HiCache: the deepest match
(may be host-only) and the deepest *device*-resident match. The scheduler locks
and indexes against the device one; the gap between them is exactly what a
load-back or an L3 prefetch has to fill. That distinction is the whole reason
the function is worth reading closely.

## C. Eviction

| | |
|---|---|
| Entry points | `unified_radix_cache.py:564` `evict`, `:567` `evict_for_alloc` |
| Driver | `unified_radix_cache.py:611` `_evict`, `:711` `_evict_components` |
| **Victim selection (device)** | `mem_cache/unified_cache/components/full_component.py:194` `_evict_device_start` |
| Walk | `full_component.py:204` `_evict_device_next_node` |
| **Victim selection (host / L2)** | `full_component.py:234` `drive_host_eviction` |
| Policies | `mem_cache/evict_policy.py`, factory at `mem_cache/utils.py:70` |
| Pinning | `unified_tree_core.py:620` `inc_lock_ref`, `:639` `dec_lock_ref` |

Selection is a **heap** over `evictable_device_leaves`, keyed

```python
(session_ref > 0, session_ref, eviction_strategy.get_priority(node))
```

Pop a leaf; when it goes its parent may become a leaf and is pushed back. Leaf
first, so a node with live descendants is never reclaimed. `--radix-eviction-policy`
lands here: `lru` -> `last_access_time`, `lfu` -> `(hit_count, time)`, plus
`slru` and `priority`.

## D. Retention: demote vs drop — the one to watch

`unified_tree_core.py:1368` `evict_device_leaf(node_id, is_write_back)`:

```
node.backuped (host copy exists) -> _demote                       free device slots,
                                                                  node survives as L2   [RETAINED]
not backuped + write_back        -> BackupKV, D->H copy, then demote
not backuped + write_through     -> _delete_unbacked_device_leaf  (:1386)               [DROPPED]
```

Under `write_through` a node evicted from L1 **before its L1 -> L2 write has
landed** is deleted outright, not demoted. It is counted as `unbacked_tokens`
and surfaces as

```
sglang:hicache_dropped_tokens_total{reason="write_through_unbacked_eviction"}
```

recorded at `unified_radix_cache.py:672` and `:687`.

**Why this matters for the evaluation:** a nonzero value there means L2 hit rate
is being eaten by a *write/evict race*, not by capacity. That looks identical to
"L2 is too small" in the hit-rate numbers and leads to the opposite conclusion.
Check it in every phase before interpreting anything else.

`unified_tree_core.py:1403` `drop_subtree_no_host` -> `reason="host_pressure"`
(`unified_radix_cache.py:706`) is the write-back-only fallback; it cannot fire
under `write_through`.

## E. HiCache data movement

Live controller is `HybridCacheController`
(`mem_cache/hybrid_cache/hybrid_cache_controller.py:92`), subclassing
`HiCacheController` (`managers/cache_controller.py:283`). Four worker threads
are spawned at `cache_controller.py:444-456`.

| direction | trigger | worker |
|---|---|---|
| L1 -> L2 offload | `cache_controller.py:784` `write` | write path |
| **L2 -> L3 backup** | `unified_radix_cache.py:1455` `_finish_write_through_ack` -> `:1627` `write_backup_storage` -> `cache_controller.py:1226` `write_storage` | `:1278` `backup_thread_func` |
| **L3 -> L2 prefetch** | `cache_controller.py:964` `prefetch` | `:1192` `prefetch_thread_func`, `:1126` `prefetch_io_aux_func`, `:1295` `prefetch_sync_thread_func`, rate limit `:1147` |
| L2 -> L1 load-back | `cache_controller.py:832` `load`; `unified_radix_cache.py:1472` `load_back` | — |
| L2 free | `unified_radix_cache.py:1127` `evict_host` | — |

Backend: `mem_cache/hicache_storage.py:373` `HiCacheFile`; its read is at `:484`
— `open(path, "rb", buffering=0)`, which disables only Python's buffer, **not**
the kernel page cache. LRU cap and disk accounting in
`mem_cache/storage/file/lru_file_evictor.py`.

Flush semantics: `unified_radix_cache.py:343` `reset` -> `:348` `_reset_full`
calls `tree_core.reset()` and `cache_controller.mem_pool_host.clear()` — the
tree and the host pool — and never touches the on-disk files. That is what makes
`POST /flush_cache` produce the cold-L1/L2, warm-L3 state phase 2 depends on.

## F. Where the scheduler touches the tree

Two reads and one write per request, in different places:

```
get_next_batch_to_run()                                managers/scheduler.py:3474
 └─ get_new_batch_prefill()                            :3568 -> :3643
     └─ _get_new_batch_prefill_raw()                   :3670
         ├─ policy.calc_priority()                     :3727   [sorting only]
         │   └─ schedule_policy.py:241
         │       ├─ match_prefix_for_req()             schedule_policy.py:256   (cache-agnostic)
         │       └─ _compute_prefix_matches()          :266 -> :339             (lpm / dfs-weight)
         └─ req.init_next_round_input(tree_cache)      :3834   [the real match]
             └─ tree_cache.match_prefix()              schedule_batch.py:1498
```

`calc_priority` only orders the waiting queue. The match that sets
`req.prefix_indices` for the forward pass is `init_next_round_input`
(`schedule_batch.py:1437`). Note the ordering at `scheduler.py:3828-3834`:
`req.storage_hit_length` / `storage_hit_start` are set from the L3 prefetch
result *before* the match runs, so it sees what storage already staged.

With `--schedule-policy fcfs` (the default) you take the `schedule_policy.py:256`
branch, which populates `req.num_matched_prefix_tokens` for the load snapshot
without affecting order. `_determine_active_policy` (`schedule_policy.py:294`)
also silently downgrades `lpm` to `fcfs` once the waiting queue passes 128.

## G. Making it observable

Reading this cold is slow. Force the paths instead: rerun the workload with a
deliberately tiny device pool,

```bash
./hicache-server.sh restart --wipe-l3 --l1 16384
```

which drives constant eviction and will exercise demote, drop, backup and
prefetch within a minute. Then watch each of the four key functions move its
counter (see `README.md` for the metrics one-liner).
