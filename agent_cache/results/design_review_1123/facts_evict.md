# evict

## summary
I traced eviction and the write path in the code (the frozen engine engine_fa3fix_20261122 has the same files as the working tree) and checked them against campaign 32 and 34 logs.

Host memory is freed only from host leaves: nodes off the GPU, with a host copy, no host or device lock, and no children. Because only leaves can go, a locked node also keeps all its host-only ancestors.

One eviction strategy orders both GPU and host eviction. Under write_through, host copies of KV still on the GPU are never evicted; they took a median 0.65 M of the 1.63 M-token host pool in campaign 34. A page whose SSD write is pending is host-locked from submit to ack, so it cannot leave. Every new leaf is copied GPU→host when it is inserted, and written to the SSD as soon as that copy is acked; nothing is written at eviction time. There are three gaps in "on SSD before it leaves":
- The lock is released at the ack even if the write failed.
- A GPU→host copy that finds no room leaves a GPU-only node, which is deleted, not written, when the GPU evicts it.
- Pages the cleaner deletes still look safe to drop in the tree.

When a read finds no room, the engine:
1. evicts every evictable leaf in policy order, including the unpinned prefixes of other queued requests (their tier is last, but they can still be evicted);
2. only then releases staged pins (ec arms only, and only when the asking request is queued);
3. cuts the read to the room left if that is at least 256 tokens;
4. otherwise refuses it, and the request recomputes that history and writes it again.

Step 2 never ran on FA3: there were 0 ec_bump events in campaign 32 and 34, because the ec arms staged only 1 to 53 reads per run.

One write thread (FIFO, merged by fix 2, skipping stored pages with fix 1) and one read thread share /dev/vdc with no I/O priority. The only link between them is fix 3, which holds timed reads while pending writes hold at least 50 K host tokens.

The cleaner ran at 90 % / 80 % in campaigns 32–34 (set in l3_extra.json; the engine default is 80/70). Each pass deleted about 40 K pages (about 2.5–2.7 M tokens, 232–247 GiB), oldest mtime first, every 10–12 minutes after the first pass. A page's mtime changes only when it is written or touched by fix 1, never on a read or lookup. Page keys are chained hashes, and both the lookup and the read stop at the first missing page. So deleting a live session's oldest pages (its head) makes the rest of its SSD chain unreachable, and a session served from SSD ages out first, while a session that recomputed its history is refreshed.

A CPU-only log check supports this, though it does not prove it. In all three campaign 34 arms, the share of non-resident history whose lookup found nothing rose after the first cleaner pass (8.8→14.9 %, 9.6→12.5 %, 3.2→8.0 %), while partial losses stayed flat.

The code seams for the five requested changes are listed with their constraints:
- the rule that an anchor with no host copy blocks SSD lookups;
- assume_stored's assumption that a stored page's prefix is stored too;
- the tree being scheduler-thread state;
- the end signal being read only at commit.

## findings
- [verified] **setting**: Campaigns 28-34 served Qwen/Qwen3-30B-A3B-Instruct-2507 with bf16 KV, context 262,144: 98,304 B a token, a 64-token page = 6 MiB stored as two 3 MiB files (K and V; page_first layout makes NIXL zero-copy), GPU pool 668,096 tokens, host pool 1,627,648 tokens (160 GB). Host room beyond a full GPU mirror = 959,552 tokens; pin budget 488,294; read budget 813,824.
  - evidence: results/campaign28/DECISIONS.md:16; results/20261123_093604_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H160/run.txt (model=Qwen/Qwen3-30B-A3B-Instruct-2507) and server.log ('KV Cache is allocated. dtype: torch.bfloat16, #tokens: 668096', 'host pool: 1627648 tokens, 160.00 GB', 'pin_budget_tokens=488294 budget_tokens=813824'); unified_radix_cache.py:633-639; hicache_nixl.py:345-349, 666-675
- [verified] **Q1 evictable host pages**: Host eviction takes only host leaves: Full KV off the GPU (node.evicted), a host copy present, host_lock_ref 0 and lock_ref 0 on every component, no children, not root. The heap is rebuilt on every call and pushes a parent once it becomes a leaf, so a locked node keeps all its host-only ancestors.
  - evidence: python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py:2138-2158, 160-171; python/sglang/srt/mem_cache/unified_cache/components/full.py:234-261; python/sglang/srt/mem_cache/host_census.py:3-10
- [verified] **Q1 GPU mirror under write_through**: Under write_through the host copy of a node still on the GPU is never host-evictable: the host-leaf test requires node.evicted, and the duplicate reclaim runs only when is_write_back. Campaign 34 census: mirror median 0.65 M of 1.63 M tokens, host-only 0.90-0.92 M (evict_casc median 0.60-0.84 M), free 16-18 K.
  - evidence: unified_tree_core.py:2145-2146, 1739-1765 (gate at 1748-1752), 1809-1820; results/campaign34/DECISIONS.md:98-102
- [verified] **Q1 pages with a pending SSD write**: A node with a pending SSD write cannot leave host memory. write_backup_storage takes a host lock on the node at submit and adds pending_backup_tokens; _drain_backup releases the lock at the write's ack without checking completed_tokens. So a failed write leaves an unlocked host copy that is evicted like any other, as if it were on the SSD.
  - evidence: python/sglang/srt/mem_cache/unified_radix_cache.py:2027-2061 (lock 2055-2059, pending 2060), 3504-3524 (release 3511-3515); full.py:271-278
- [verified] **Q1 order: lru (hicache)**: The heap orders by node.last_access_time, oldest first. match_prefix stamps the whole matched path to the root. A queued request's prefix is restamped at arrival (_prefetch_kvcache → init_next_round_input) and whenever the FCFS admission loop reaches it, but not while it is skipped for an ongoing prefetch. The Python tree core has no fast match, so the waiting queue is not re-matched each pass.
  - evidence: evict_policy.py:16-18; unified_tree_core.py:983-987; managers/scheduler.py:3138-3140, 4006-4013, 4025; managers/schedule_policy.py:271-284; unified_tree_core_interface.py:413-415; arg_groups/fields/schedule.py:82-98 (fcfs)
- [verified] **Q1 order: next_use (jit_p50, jit_exact)**: Order with horizon 300 s and grace 60 s:
- ENDED, LRU;
- OVERDUE (registered time more than 60 s past), oldest first;
- FAR (more than 300 s ahead), farthest first;
- UNKNOWN, LRU;
- SOON, farthest first;
- QUEUED, LRU (back to UNKNOWN after 600 s).

Only the session's tail node id is registered, so body nodes, shared prefixes and landed spans rank UNKNOWN. When an ended tail goes, its parent re-enters as UNKNOWN, after live sessions' OVERDUE and FAR tails. The registered time is predicted arrival plus queue wait estimated at commit, a static T = G + W.
  - evidence: python/sglang/srt/mem_cache/next_use_eviction.py:41-46, 57-60, 139-172; next_turn_prefetch.py:533-546, 1508-1533, 1591-1604; full.py:250-261
- [verified] **Q1 order: return_chance (ec_full, ec_exact)**: Order with horizon 30 s:
- ENDED, LRU;
- UNKNOWN, LRU (branch points, shared prefixes, un-hinted nodes, queued over 600 s);
- SCORED, by P(return within 30 s | idle time) ascending, ties by idle minus median. Exact hints past their point plus 5 s rank 0 (first); within the grace they rank 1.0;
- QUEUED, LRU.

A registration covers the session's single-child chain, and landed spans adopt their session via on_read_landed. There is no size or memory-price term.
  - evidence: next_use_eviction.py:51-60, 175-271 (205-237, 246-271); mem_cache/next_turn_policy.py:94-99; next_turn_prefetch.py:1203-1212
- [verified] **Q1 GPU eviction uses the same order**: One strategy object orders both GPU-leaf and host-leaf eviction. Under write_through, GPU eviction of a node with a host copy only frees the GPU slots (the host copy stays); a node without one is deleted. So the policy also decides which host copies are unevictable mirror and which are evictable host-only.
  - evidence: unified_tree_core.py:436-438, 1618-1651; full.py:57-61, 194-229
- [verified] **Q1 protection of queued prefixes**: A queued request's resident prefix is locked only while it has an SSD lookup or read under way: prefetch_from_storage host-locks the anchor (the deepest resident node) through lookup, parking and dispatch until the read lands or is revoked, and the landed span is then pinned. A queued turn whose history is fully resident (gate 'covered', or a delta under 256 tokens) has no lock and is protected only by its tier. QUEUED is registered only for turns with a plan.
  - evidence: unified_radix_cache.py:2190-2213, 2529-2531, 2953-2987; next_turn_prefetch.py:787-825 (Queued at 811), 1411-1426; next_use_eviction.py:163-166, 253-256
- [verified] **Q1/Q2 staged and landed reads (pins)**: Under the deadline read order, a landed span is host-locked on its inserted node until admission, abort or 600 s, within a budget of 0.3 × host = 488,294 tokens. Over the budget it is not pinned: srs_pin_skipped occurred 17 times in campaign 34 jit_p50, 9 in campaign 32 jit_p50, 0 in the ec arms. The hicache arm (fifo order, pin fraction 0) pins nothing. Under expected-cost timing a staged pin is also released once its gap entry has passed (+5 s). Campaign 34 census: pins 0-25 K; jit_p50's early reads 136 K median (p90 280 K).
  - evidence: unified_radix_cache.py:172, 633-635, 1089-1090, 1285-1286, 2520-2525, 3318-3373, 3869; arg_groups/fields/memory.py:357-389; next_turn_prefetch.py:133-135, 1171-1183, 1289-1296; grep of results/20261123_073130_* and 20261122_123020_*/server.log; results/campaign34/DECISIONS.md:101-102
- [verified] **Q1 ended sessions**: An ended session is registered None and ranks first in GPU and host eviction. A session ends when commit finds no next-turn hint, on an abort, or when a jit plan expires (_forecast_ends_session). Nothing is evicted ahead of need, its last turn is still written to host and SSD, and its SSD pages stay until the cleaner reaches them by age. The hint rides on the request (req.kv_hints) but is read only at commit, after the insert has queued the GPU→host copy.
  - evidence: next_turn_prefetch.py:422-424, 513-532, 581-623, 1495-1506, 1584-1585; unified_radix_cache.py:1150, 1233-1242; next_use_eviction.py:161-162, 250-252
- [verified] **Q2 allocation order at a read and the refusal path**: A read that finds no free host pages:
1. evict_host(hit) over all evictable leaves in policy order, QUEUED-tier prefixes of other arrived requests included;
2. then, in ec arms only, releases staged pins and evicts again;
3. cuts the read to the free room if that is at least 256 tokens;
4. otherwise refuses it: report to the price, census and room retry, finish with reason host_capacity, revoke (anchor lock released), and the request is later admitted without that history.

GPU→host copies run only step 1 and are refused on failure.
  - evidence: unified_radix_cache.py:3036-3077 (3039, 3041, 3043-3047, 3049-3057, 3058-3077), 3118, 1785-1794, 3225-3239
- [verified] **Q2 soft_pin_victims**: soft_pin_victims returns nothing unless the asking turn is in the waiting queue. Victims are staged plans (turn not arrived, a distribution, no read in flight, pinned tokens > 0), sorted ascending by chance of return within their own read time (zero-backlog load time + 1 s), taken until the shortfall is covered. Each bump is charged to the price, and the plan may be staged again (at most NTP_MAX_STAGINGS = 2 stagings in total). It is wired only with soft pins on and the deadline order (ec arms).
  - evidence: next_turn_prefetch.py:139-142, 391-393, 1256-1258, 1298-1343 (1304-1305, 1310-1323); unified_radix_cache.py:3122-3128
- [verified] **Q2 soft pins never acted on FA3**: There were 0 ec_bump events in campaign 32 ec_full and ec_exact (1,955 and 2,200 dispatches logged launched=0) and in campaign 34 ec_full. The expected-cost arms logged only 2, 53 and 1 stage attempts (ntp_stage) in 2 h, so almost nothing was staged to preempt. jit_p50 logged 804 (c32) and 1,208 (c34) stage attempts but runs without soft pins; 241 and 415 of its pins ended by abort (expiry).
  - evidence: grep counts in results/20261122_082020_*, 20261122_102503_*, 20261122_123020_*, 20261123_073130_*, 20261123_093604_*/server.log (arm mapping results/campaign32/manifest.txt); results/campaign32/DECISIONS.md:165-167
- [verified] **Q2 cost of a refusal**: The refused history is recomputed at admission: 0.042-0.044 ms a token on FA3 under load, about 0.86 s for 20 K tokens; requests queued behind wait too. The recomputed KV is then written through to host and SSD again. In campaign 32's loop, 74-93 % of writes rewrote stored pages and pending writes reached 515-675 K tokens. Planned turns get MAX_REFETCHES = 2 lookups; fix 4 adds up to 8 room retries 2 s apart. The engine's price charges refused tokens at RECOMPUTE_S_PER_TOKEN = 1.1e-4 s × queue length.
  - evidence: results/campaign32/DECISIONS.md:171-183, 249-253; results/campaign34/DECISIONS.md:80; OPEN_ITEMS.md:553-556; next_turn_prefetch.py:110-117, 1345-1360; next_turn_policy.py:41-44
- [verified] **Q3 GPU→host copy timing**: Eager. Under write_through, a newly inserted leaf (threshold 1) gets a BackupKV for itself and its unbacked ancestors: the prompt at the end of prefill (cache_unfinished_req, chunked=False) and the output at finish (cache_finished_req). Chunked-prefill pieces wait for the final insert. Copies are queued with flush=False and merged into one transfer per scheduler step.
  - evidence: unified_radix_cache.py:569-571, 1722-1760, 1781-1799, 3966-3970; unified_tree_core.py:1060-1081, 1368-1396, 2371-2392; scheduler_components/batch_result_processor.py:383; managers/scheduler.py:3561, 4810; managers/cache_controller.py:153-170, 846-870
- [verified] **Q3 refused GPU→host copies**: A GPU→host copy that cannot evict enough room is refused (census backup_refused). The node stays GPU-only and, under write_through, is deleted rather than kept on host when the GPU evicts it. It never reaches the SSD unless a later insert below it copies the chain. Campaign 34: 0, 0 and 11 refusals (8,768 tokens) in jit_p50, ec_full and jit_exact.
  - evidence: unified_radix_cache.py:1742-1747, 1785-1794; unified_tree_core.py:1630-1640, 2371-2392; host_census.py:213-215; census sums over results/20261123_{073130,093604,114134}_*/server.log
- [verified] **Q3 host→SSD timing**: Eager, never at eviction. When a GPU→host copy is acked (writing_check, every scheduler pass), each published node is submitted through write_backup_storage with its full host copy and page keys. Host eviction never writes, and every host-only node already had its write submitted. In each pass, write acks drain (releasing locks) before reads are dispatched.
  - evidence: unified_radix_cache.py:3737-3780, 1840-1855, 2027-2061, 3926-3958; unified_tree_core.py:2262-2287; hybrid_cache_controller.py:1155-1171
- [verified] **Q3 write thread and coalescing (fix 2)**: One backup thread consumes a plain FIFO queue. With WRITE_COALESCE_PAGES = 128 (campaign 34), it merges the mergeable ops (NIXL, KV only) queued behind the first, up to 128 pages, into one page stream written in 128-page batches. Ops are acked in queue order as their last page lands; a non-mergeable op ends the group and is written next; after a failed batch the rest are written alone. Campaign 34: 1,011-1,651 groups a window; submit-to-done median 0.3-0.6 s, against 28-42 s in campaign 32's loop.
  - evidence: hybrid_cache_controller.py:1392-1409, 1411-1442, 1444-1518; cache_controller.py:332-334, 471-476; hicache_storage.py:23; environ.py:776-778; results/campaign34/settings.env; results/campaign34/DECISIONS.md:81, 88
- [verified] **Q3 skip stored writes (fix 1)**: batch_set_v1 drops each page whose every file already exists at full size and refreshes those files' mtime (os.utime) instead of writing. The check runs on the write thread at write time, so a skipped page still holds its host lock while queued. Campaign 34 skipped 0-2 % of pages in minutes 30-60 and 11-12 % in 90-110.
  - evidence: storage/nixl/hicache_nixl.py:95-100, 853-879, 899-904; storage/nixl/nixl_utils.py:50-60, 320-333; environ.py:773-775; results/campaign34/DECISIONS.md:85-87
- [verified] **Q3 read hold on the write backlog (fix 3)**: Timed (deadline) reads stop dispatching while pending writes hold at least 50,000 host tokens and resume at 25,000 or less. 'Now' reads (own lookups, queue positions 0-1) are never held, and writes never yield to reads. The hold switched on 30, 30 and 28 times in the three campaign 34 arms. The hicache arm (fifo, no read scheduler) has no hold.
  - evidence: storage_read_scheduler.py:13-16, 74-76, 147-150, 170-181; unified_radix_cache.py:629-632, 3272-3280; next_turn_prefetch.py:1388-1409; grep 'srs_write_hold state=on' in results/20261123_{073130,093604,114134}_*/server.log
- [verified] **Q3 one disk, separate threads, no priority**: Threads on /mnt/ssd/hicache_l3 (/dev/vdc): one lookup thread (batch_exists), one read I/O thread, one write thread, and the cleaner thread on TP rank 0 with 8 unlink workers. They share one NIXL agent (NIXL_THREAD_SYNC_RW), with bounce buffers per direction. There is no ioprio or other I/O priority. Reads and writes together reach about 2.1 GB/s. Once reads take the disk the write thread writes 3.8-7.0 K tokens/s; the read thread reads 17-20 K tokens/s and was 83-96 % busy on FreeInference.
  - evidence: cache_controller.py:462-483, 1188-1214, 1278-1312; hicache_nixl.py:120-122, 157-162; nixl_cleaner.py:107-109, 121-139; scripts/start_server.sh:110, 130-131; results/campaign32/DECISIONS.md:173-178; OPEN_ITEMS.md:523-524, 557-559
- [inferred] **Q3 what drives host eviction**: Host allocation comes from the GPU→host copy of new KV and from landed reads, both through evict_host. In campaign 34 these ran at about 4.3-4.6 K tokens/s of writes and 4-15 K tokens/s of reads, against about 0.9 M host-only tokens. An idle prefix therefore stays in host memory on the order of a minute under load.
  - evidence: results/campaign34/DECISIONS.md:84-85, 98-100 (8.3/7.7/5.2 M submitted, 7.4/18.8/17.4 M read by ec_full in minutes 30-60/60-90/90-110)
- [verified] **Q4 cleaner mechanics**: Every 30 s on TP rank 0: if the filesystem is at or above the high watermark, scan all buckets and group files by page key (K/V and rank suffixes stripped; group mtime = max over its files). Sort groups by mtime ascending and unlink 50 at a time until usage is below the low watermark. The cleaner takes no input from the cache. Engine default 80/70; start_server.sh default 30/20; campaigns 32-34 used 90/80 via l3_extra.json.
  - evidence: storage/nixl/nixl_cleaner.py:19-22, 47-64, 121-139, 159-223, 265-270; nixl_utils.py:62-81; scripts/start_server.sh:47, 133-135; results/20261123_*/l3_extra.json and server.log 'HiCacheL3Cleaner started: ... high=90.0% low=80.0%'
- [verified] **Q4 cleaner passes in campaign 34**: Each pass deleted 39,650-42,200 groups (79,300-84,400 files, 232-247 GiB, about 2.5-2.7 M tokens) in 10.6-12.1 s. After the first pass, passes came every 10-12 minutes:
- jit_p50: 09:03:16, 09:15:28, 09:25:40;
- ec_full: 11:08:25, 11:20:08, 11:30:20, 11:40:31;
- jit_exact: 13:13:59, 13:25:40, 13:36:22, 13:46:33.
  - evidence: 'NIXL L3 cleanup' lines in results/20261123_073130_*, 20261123_093604_*, 20261123_114134_*/server.log
- [verified] **Q4 what refreshes mtime**: Only a write (the first write, or a rewrite when fix 1 is off) or touch_if_stored on a skipped rewrite. Reads register files read-only ('ro', or opened without O_CREAT) and lookups only query existence. The cleaner therefore deletes the least recently written pages, not the least recently used: a session served from SSD keeps its head's first-write time, while a session that recomputed its history gets it rewritten or touched.
  - evidence: storage/nixl/nixl_registry.py:140-181 (152-153, 170); nixl_utils.py:291-333; hicache_nixl.py:233-242, 853-875; nixl_cleaner.py:181-184, 270
- [verified] **Q4 how the storage hit length is computed**: 1. anchor = last_host_node, the deepest GPU- or host-valid node; start = len(GPU prefix) + host_hit_length.
2. No lookup if the anchor has no host copy and is not the root, or if the span is under 256 tokens.
3. Keys k_j = H(k_{j-1}, tokens of page j), seeded with the anchor's last page hash, so each key encodes the whole prefix.
4. Each key maps to the files <k_j>_<model>_<tp>_<tpsize>_k and _v; all files are queried, then scanned.
5. kv_hit_pages = index of the first page with a missing file.
6. hit = 64 × kv_hit_pages, MIN-reduced across ranks; under 256 tokens it is revoked as below_threshold.
7. At launch it may be cut to the host room.
8. The read completes only the leading run of readable pages.

hicache_storage_pass_prefix_keys is false in every run, and NIXL ignores prefix keys anyway. The retry path's assume_stored skips the query for the left extension of a known hit.
  - evidence: unified_tree_core.py:983-994; managers/scheduler.py:3157-3187; unified_radix_cache.py:2150-2197, 3450-3462; hybrid_cache_controller.py:1173-1208, 1181-1186, 350-352, 1411-1413; mem_cache/utils.py:122-128, 148-158, 172-181; hicache_nixl.py:118-119, 191-192, 233-242, 645-675; cache_controller.py:1278-1305, 1103-1147
- [verified] **Q4 deleting a head makes the chain unreachable**: Yes. A missing page j cuts the hit to j pages, and the pages after it are unreachable even though they are on disk (the cleaner deletes them later by age). The lookup starts after the resident prefix, so this bites a session idle long enough to need its head from the SSD. That session recomputes the whole history, and the write-through then rewrites the head and touches (fix 1) or rewrites the rest, making the chain young again.
  - evidence: hicache_nixl.py:658-664; hybrid_cache_controller.py:1205-1208; cache_controller.py:1133-1146; scheduler.py:3157-3171; unified_radix_cache.py:1840-1855; hicache_nixl.py:853-875
- [inferred] **Q4 log check of the cleaner's effect**: I joined each next-turn lookup (ntp_gate read/reread/retry, history = chain − matched ≥ 4,096 tokens) with its s2h_query result, comparing the 30 minutes before the first cleaner pass with the time after it. History whose lookup found nothing rose in all three campaign 34 arms: ec_full 8.8 → 14.9 %, jit_exact 9.6 → 12.5 %, jit_p50 3.2 → 8.0 %. History lost to partial hits barely moved: 5.0 → 4.2 %, 4.4 → 4.5 %, 1.2 → 2.3 %. Whole chains go rather than tails. The pre-pass baseline also counts prompts that diverge from the committed key.
  - evidence: /tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/facts-evict-write/hist_miss.awk and window.awk over results/20261123_{093604,114134,073130}_*/server.log; per-minute outputs ecfull_c34_hist_miss.txt, jit_exact_c34_hist_miss.txt, jit_p50_c34_hist_miss.txt in the same folder
- [inferred] **Q4 assume_stored and the cleaner**: assume_stored skips the existence query on the left extension of a known hit, on the grounds that writes cover prefixes. Head-first cleaning (and any reordering of writes) breaks that assumption; the read then allocates host pages for the whole span and completes only the leading run of pages still present.
  - evidence: unified_radix_cache.py:2150-2184; hybrid_cache_controller.py:1181-1186; cache_controller.py:1113-1146; scheduler.py:3220-3238
- [inferred] **Q1 losses of queued turns after a read**: Even in campaign 34, a window held 26-216 queued turns whose key was found missing again after a read, with no re-read left (ntp_reevict capped=1). Landed spans are pinned within the budget, so the lost pages are other parts of the key: the unpinned resident prefix, or the generated-output tail of plan.key. Not traced.
  - evidence: results/campaign34/DECISIONS.md:175-176; next_turn_prefetch.py:917-934
- [verified] **Q5 seam: clean/dirty-aware eviction**: Dirty (pending write) pages are already outside the heap through the host lock. Missing pieces:
- an 'on SSD' bit set at the ack only when completed_tokens covers the op (_drain_backup; the linker's external_cache_stored flag is a precedent);
- a heap key in FullComponent.drive_host_eviction that prefers clean leaves;
- an evict_host bounded by class, so a caller can stop before e.g. QUEUED;
- a cleaner→cache channel, so a host copy whose SSD copy was deleted counts as the only copy.
  - evidence: unified_radix_cache.py:1493-1507, 3504-3524; unified_tree_core.py:134, 1739-1765, 2542-2558; full.py:242-261; nixl_cleaner.py:168-223
- [verified] **Q5 seam: eviction by time-to-next-use class**: The class order lives in get_priority (one object for GPU and host). By class:
- (1) ended sessions: exists.
- (2) GPU-mirror copies: reclaim exists only for write_back. Under write_through it conflicts with GPU eviction deleting nodes that have no host copy, and with SSD lookups being blocked at an anchor that has no host copy.
- (3) idle sessions by expected T: registrations are fixed at commit (W from _wait_after_arrival); refreshing them in tick would make W live.
- (4) staged reads: they are pins (host locks) outside the heap, released only after evict_host fails, never for GPU→host copies. Moving them before QUEUED needs a class-bounded evict_host.
- (5) queued requests far from the head: Queued carries only `since`; positions are known in tick.
- (6) near the head: reads at positions ≤ 1 are already 'now', but the resident prefix is unlocked until admission; a host lock on the head's last_host_node would make it unevictable.
  - evidence: next_use_eviction.py:77-81, 157-172, 246-271; unified_tree_core.py:1630-1640, 1748-1758, 1777-1843; scheduler.py:3160-3167; next_turn_prefetch.py:533-546, 636-696, 645, 1298-1343, 1405-1406, 1459-1460, 1544-1546, 1591-1604; unified_radix_cache.py:1785-1794, 3039-3047, 3318-3348
- [verified] **Q5 seam: write ordering by eviction likelihood**: backup_queue is a plain FIFO Queue drained by one thread and grouped FIFO. The only point with node context is write_backup_storage (@rank_consensus), called per node at the GPU→host ack. Options:
- carry a priority snapshot (the node's class or T at submit) on StorageOperation into a priority queue;
- or have the scheduler hold submissions and release them in eviction order (the host lock covers held nodes either way).

Constraints: the tree is scheduler-thread state, and assume_stored assumes every stored page's prefix was written first.
  - evidence: cache_controller.py:476; hybrid_cache_controller.py:105-118, 1392-1442, 1181-1186; unified_radix_cache.py:2026-2061, 1851-1855
- [verified] **Q5 seam: skip writes of ended sessions**: The end is carried by req.kv_hints. For exact hints, and for tables under reply_key_source hint, it is in hand at arrival, but it is read only at commit: after the prompt's GPU→host copy was queued at prefill end and the output's at the insert, and before the SSD write, which waits for the GPU→host ack. Options:
- skip write_backup_storage for nodes whose session registration is None (tail id; return_chance's _tail_of for the rest of the session), never for branch points or shared prefixes;
- or skip the GPU→host copy of an ended session's final nodes (GPU eviction then deletes them).
  - evidence: next_turn_prefetch.py:513-532, 581-623; unified_radix_cache.py:1150, 1233-1242, 1840-1855; batch_result_processor.py:383; next_use_eviction.py:201-244; unified_tree_core.py:1630-1640
- [verified] **Q5 seam: session- or chain-aware cleaner**: The cleaner sees only file names and mtimes. Group key = page hash + model, and page keys are chained hashes, so neither session nor chain position can be recovered from the disk. Options:
- touch on read in batch_get_v1 with touch_if_stored, giving least recently used order instead of least recently written;
- a protect/doom list from the cache, which holds node.hash_value and session state, so ended sessions go first and open sessions are spared;
- tail-first deletion, which needs the chain position recorded at write_backup_storage;
- deletion feedback to the tree.
  - evidence: nixl_cleaner.py:47-64, 168-223; hicache_nixl.py:824-851; nixl_utils.py:320-333; unified_tree_core.py:110-187 (hash_value), 2262-2287; unified_radix_cache.py:2027-2061

## discrepancies
- The workflow context says the server is Qwen/Qwen3-32B-FP8, but campaigns 28-34 ran Qwen/Qwen3-30B-A3B-Instruct-2507 with bf16 KV: 98,304 B a token, 6 MiB a page, GPU 668,096 tokens, host 1,627,648 tokens, context 262,144 (results/campaign28/DECISIONS.md:16; run.txt and server.log of results/20261123_093604_*). Qwen3-32B-FP8 is only start_server.sh's default MODEL (line 43).
- Paper design.tex:98-101 assumes storage holds every prefix cache, so a dropped one can always be read back. In the engine:
- the cleaner deletes the oldest pages at 90 % (campaign 34: first pass at about minute 88, about 2.5 M tokens a pass);
- a deleted head makes the rest of the chain unreachable (hicache_nixl.py:658-664);
- a failed write still releases the host lock (unified_radix_cache.py:3511-3515);
- a refused GPU→host copy never reaches the SSD (unified_radix_cache.py:1785-1794; unified_tree_core.py:1630-1640).
- Paper design.tex:148 leaves GPU eviction to the engine. In the engine, the next_use/return_chance strategy orders GPU eviction too (full.py:194-229), and under write_through that choice decides which host copies are unevictable mirror.
- Paper Retain (design.tex:310-338) ranks by the ν index, with a (1+λL) factor and a max over horizons. ReturnChanceStrategy instead ranks by P(return within a fixed 30 s | idle), with no size or price term, puts unregistered nodes (shared prefixes, branch points, un-hinted) before every scored session, and puts queued sessions last (next_use_eviction.py:246-271). next_use registers only the tail node, so an ended session's body ranks as UNKNOWN.
- Paper Preempt (design.tex:266-280) preempts staged reads when no memory is free. The engine first evicts every unlocked host leaf, including the QUEUED-tier prefixes of other arrived requests, and only then releases staged pins (unified_radix_cache.py:3039-3047). It does so only for a queued asker (next_turn_prefetch.py:1304-1305), with r_j = zero-backlog load time + 1 s (1256-1258, 1320). It never fired in campaigns 32 or 34 (0 ec_bump).
- The design note's §3 order puts class 4 (staged reads of sessions not back) before class 5 (queued requests). The engine does the reverse: queued-tier leaves can be evicted, while staged reads are pins outside the eviction heap.
- The design note's 'A page cannot leave host memory until it is on the SSD' holds for pending writes. It does not hold for failed writes (lock released at ack regardless), refused GPU→host copies, or pages the cleaner deleted later (the tree still treats the host copy as clean).
- The design note's 'cleaner ... once the volume is 90 % full, down to 80 %' is the campaigns' l3_extra.json setting. The engine default is 80/70 (nixl_utils.py:64-67; nixl_cleaner.py:21-22), and start_server.sh's default is 30/20 (line 47).
- Three line pointers in the design note's architecture table point at the switches, not the code that acts:
- the merging is in hybrid_cache_controller.py:1411-1518; cache_controller.py:332-334 only reads the switch;
- skipping stored writes acts in hicache_nixl.py:853-904; nixl_utils.py holds the switch and touch_if_stored;
- the read hold acts at unified_radix_cache.py:3272 and storage_read_scheduler.py:147-150; line 631 is its init.
- The engine's price charges a refused token RECOMPUTE_S_PER_TOKEN = 1.1e-4 s (next_turn_policy.py:41-44), measured on Triton in campaign 25. The design note and campaign 32 give 0.042-0.044 ms a token on FA3 (results/campaign32/DECISIONS.md:249-252), so on FA3 the price overstates refusal harm about 2.5 times.
- Paper design.tex:108-109 says a read holds L_i tokens until its request is admitted. That is true only within the 30 % pin budget (488,294 tokens). Over it the landed span is not pinned (unified_radix_cache.py:3324-3333): 17 skips in campaign 34 jit_p50 and 9 in campaign 32 jit_p50.

## open questions
- Is the post-pass rise in fully missed history really head deletion? A per-request trace would settle it: the first missing key of each lookup against the groups each pass deleted (the cleaner logs only counts). The pre-pass baseline (3-10 % of history) mixes in prompts that diverge from the committed key (compaction, recorded vs generated replies).
- Class 2 (dropping host copies of KV still on the GPU) needs a decision between exclusive tiering and write_through. Two parts of the engine assume write-through contiguity: GPU eviction deletes a node without a host copy (unified_tree_core.py:1630-1640), and an anchor without a host copy blocks SSD lookups (scheduler.py:3160-3167; next_turn_prefetch.py:1544-1546).
- Cleaner order: touch on read (least recently used order at the cost of two utime calls per page on the read thread), or a cache-fed order (ended sessions first, open sessions spared, tails before heads)? Tail-first needs a chain position that page-key file names do not carry.
- Write ordering by eviction likelihood: is it acceptable to break assume_stored's assumption that a stored page's prefix is stored too (already broken by head-first cleaning)? And should the priority be a snapshot at submit, or should the scheduler hold submissions and release them in order?
- What share of SSD writes is never read back: ended sessions' last turns, generated outputs that the replay never re-sends, turns served from host? It is not measured, and it bounds what skipping writes saves in disk bandwidth and in host memory held by pending writes.
- Which pages do the 26-216 ntp_reevict events per campaign 34 window lose, given that landed spans are pinned? Likely the unpinned resident prefix or the generated-output tail of plan.key; not traced.
- Does NIXL's POSIX query_memory or READ path ever touch mtime? Not read on the C++ side; POSIX semantics say no. A lookup also stats every file of the requested span, not just up to the first miss (hicache_nixl.py:233-242, 658-664); its cost on long chains is unmeasured.