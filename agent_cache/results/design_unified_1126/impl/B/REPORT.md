# Implementation report, copy B: Retain and the store (2026-11-26)

The implementer's structured report, saved by the main session (a subagent may not write report files).

## Summary

Steps 1 to 3 of the Retain plan are implemented in copy B (/home/wanhr/sglang/agent_cache/frozen/impl_unified_B)
behind four switches, all off by default, with CPU unit tests. With the switches off the engine behaves as before: all
122 unit test files of test/registered/unit/mem_cache give the same outcome for every case, no existing test body was
changed, and the h2s_done line is byte-identical.

Three limits to know first:
- REPORT.md is not on disk. The Write tool refused it ("Subagents should return findings as text, not write report
files") and I did not write it another way. This output is its full content; the caller has to save it to
/home/wanhr/sglang/agent_cache/results/design_unified_1126/impl/B/REPORT.md if the file is wanted.
- Nothing ran on a GPU or with more than one rank. All evidence is CPU unit tests.
- The merge with copy A was not built or tested (merge notes below).

What was built
- R3 (SGLANG_HICACHE_STORAGE_TRACK_STORED): a tree node counts as stored on the SSD only after an ack that covers all
its pages, agreed over ranks. Pages read back from the SSD are stored at birth. A child's write first rewrites the
host-backed ancestors that have no SSD copy and no write queued (the heal). Clear and detach reset the marks.
- X1: NextTurnPrefetcher.ends_session(req, reply_known=...) is true only when the hint states the end: a table entry
marked ended, or an exact-hint turn that the previous turn named and that names no next turn. Under the R1 or the
retire switch the registration follows it: no usable entry, an expired plan and an abort forget the session's tail (it
ranks as un-hinted) instead of registering it as ended.
- R4 refresh (SGLANG_HICACHE_NIXL_TOUCH_INTERVAL_S): a request's match hands the page keys of the stored chain it was
served to a touch thread in HiCacheNixl, head first. The thread stamps the head newest and each later page 1 ms older.
A page is touched at most once per interval. No file system call runs on the scheduler thread.
- R1 (SGLANG_HICACHE_STORAGE_SKIP_ENDED_WRITES): the inserts of a turn whose hint ends its session keep their host
copy and get no SSD write (no operation, no host lock, no pending tokens).
- R4 retire (SGLANG_HICACHE_NIXL_RETIRE_ENDED_PAGES): at an explicit end the session's own stored pages get mtime zero
and stop counting as stored. Own means past the session's first prompt and on no branch another session built on,
including the session's replies that no later prompt repeated. First prompts are never retired. A page whose write is
queued, or whose host copy is still in flight, is zeroed at the ack of its write. The cleaner (nixl_cleaner.py) is
unchanged, as are scheduler.py, tokenizer_manager.py and model_runner.py.

Evidence
- 30 tests added to five existing test files; no test file added.
- 64 single-line mutations of the engine, each run against the test meant to catch it: all killed on the final code.
Two survived when first written and showed real holes, closed by extending their test (the skip set was never emptied;
an exact-hint end was not retired at commit).
- The ten named test files were last run at 5:32 PM ET on 2026-11-26; counts are in test_results.
- ruff format and ruff check (F401, F821, UP037; ruff 0.16.10 from the uv cache) show nothing new in the changed
lines, except the h2s_done call, kept on one line with its EVAL-PATCH mark as before.

Log events (each appears only under its switch)
- h2s_done ... stored=<0|1>, appended at the end, with TRACK_STORED on.
- h2s_short op= node= tokens= completed=, on an ack that did not cover every page.
- h2s_skip node= tokens= reason=ended, at each skipped write.
- ntp_end rid= sess= explicit=<0|1> why=<entry|no_next|no_entry|expired|abort>.
- l3_retire node= pages= at_ack= keep_tokens=, at an explicit end (pages zeroed now, pages left to their ack).
- l3_retire_ack op= node= pages=, when a retire left to the ack is done.
- l3_touch files= gone=, from the touch thread, at most every 10 s and at stop.
- At start-up, when a switch is asked for: "HiCache retention: track_stored= skip_ended_writes= touch_interval_s=
retire_ended_pages=" and "HiCacheNixl: touch_interval_s= retire_ended_pages=". A switch that lacks what it needs logs
"HiCache retention: <name> stays off: it needs ...".

First campaign arm
Set the planner's four: SGLANG_HICACHE_STORAGE_TRACK_STORED=1 SGLANG_HICACHE_NIXL_TOUCH_INTERVAL_S=300
SGLANG_HICACHE_NIXL_RETIRE_ENDED_PAGES=1 SGLANG_HICACHE_STORAGE_SKIP_ENDED_WRITES=1.
- Compare with the same arm with all four unset, and run past the first cleaner pass (minute 87.5 to 88.3 at campaign
34's setting). Keep the baseline's other switches the same in both.
- It needs what the hinted arms already have: the prefetcher on, host memory mode cache, write-through, the NIXL file
backend, TP = 1, a FULL-only model, and hints that state the end.
- The start-up line must read track_stored=1 skip_ended_writes=1 touch_interval_s=300 retire_ended_pages=1, with no
"stays off" warning.
- This arm also gets X1's registration change. An arm with TRACK_STORED and the touch interval alone needs no hints
and isolates the refresh.
- What to read: h2s_skip (the plan counts 4.7 to 4.8 % of inserted tokens in campaign 34), ntp_end explicit=1 (566
sessions in ec_full) against explicit=0 why=no_entry|expired (the false ends of before), l3_retire and l3_retire_ack,
l3_touch files= per second (the utime load), stored=0 and h2s_short (none expected). The effect to look for: after
each "NIXL L3 cleanup", fewer s2h_query storage_hit=0 on returning turns and fewer "failed to retrieve page".

Merge notes
- /home/wanhr/sglang/agent_cache/results/design_unified_1126/impl/B/steps_1_3.patch is the whole change (16 files).
Its base is HEAD b7074d9cae for 15 files and the working-tree version for test_next_use_eviction.py; base plus patch
reproduces the copy exactly. The two edits the copy came with (read_start(stall_weight=...), queued_order and its
test) are untouched.
- The real tree moved during the work, not by me: between 3:37 and 3:43 PM ET someone added
SGLANG_HICACHE_PREFILL_LOAD_HOLD_STEPS to environ.py right after SGLANG_HICACHE_NEXT_TURN_ROOM_RETRY, changed
managers/scheduler.py and added scheduler_components/prefill_load_hold.py with its test. Copy B does not have these. I
placed my four switches before NEXT_TURN_ROOM_RETRY for that reason; at 5:17 PM ET the patch applied to the real tree
with no rejected hunk, no fuzz and no offset.
- Against copy A's files at 5:17 PM ET (A was still being edited): every engine hunk applies, four of them in
next_turn_prefetch.py with fuzz; one test hunk is rejected, the import block of test_next_use_eviction.py, where both
copies add an import.
- To keep the copies apart, the class default `retention` in unified_radix_cache.py sits above _init_storage_retention
instead of at the class header (where A adds attributes), and my three constructor attributes in next_turn_prefetch.py
sit before _session_tail (A appends after _rid_session).

Notes folder /home/wanhr/sglang/agent_cache/results/design_unified_1126/impl/B/: steps_1_3.patch; checks/final_ten.txt
(the ten files' counts); checks/dir_before_after.txt (122 files before and after); checks/mutations.txt (the 64
mutations and the test that kills each); checks/mutation_scripts/mutate.py and mutations.py (exact texts and a runner
that also takes another tree, for use after the merge). I made no git write and no network call, and touched nothing
outside copy B and this folder.


## Files changed

- /home/wanhr/sglang/agent_cache/frozen/impl_unified_B/python/sglang/srt/environ.py (+11): the four switches, placed
  before SGLANG_HICACHE_NEXT_TURN_ROOM_RETRY
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_B/python/sglang/srt/mem_cache/storage_retention.py (new, 101
  lines): RetentionSwitches (frozen struct), resolve_retention_switches (dependencies and refusal lines), stored_votes
  (a rank's verdict per acked write)
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_B/python/sglang/srt/mem_cache/unified_radix_cache.py (+219 -4):
  _init_storage_retention at init_hicache; the match_prefix hook and _refresh_served_chain; the skip flag on inserts
  and the skip set; the heal and _submit_storage_write in write_backup_storage; retire_storage_chain; _agree_stored
  and _commit_storage_write in the ack drain; stored= on h2s_done; clear_storage_backend
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_B/python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py
  (+156 -5): node fields storage_write_op, storage_stored, storage_touched_at, copied at a split; insert_host marks
  read-back nodes stored; skip_storage on BackupKV at both insert sites; mark_storage_write, finish_storage_write,
  unstored_ancestors, take_stale_storage_keys, take_retired_storage_pages, clear_storage_marks; RetiredPages
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_B/python/sglang/srt/mem_cache/next_turn_prefetch.py (+148 -5):
  TurnOutlook, SessionExtent, ends_session and _turn_outlook, the explicit-end branch in on_turn_committed,
  _note_session_turn, _on_explicit_end (retire call), _forget_session, why= on _end_session, the ntp_end line, two
  factory arguments
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_B/python/sglang/srt/mem_cache/storage/nixl/hicache_nixl.py (+75):
  the touch thread (hicache-l3-touch), touch_pages, _page_file_paths, _touch_loop with the l3_touch line, stop in
  close()
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_B/python/sglang/srt/mem_cache/storage/nixl/nixl_utils.py (+53):
  config keys touch_interval_s and retire_ended_pages with their getters, TOUCH_STEP_NS, NixlFileManager.stamp_pages
  (os.utime only, never creates a file)
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_B/python/sglang/srt/mem_cache/hicache_storage.py (+12):
  HiCacheStorage.touch_pages (no-op) and the class defaults touch_interval_s and retire_ended_pages
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_B/python/sglang/srt/mem_cache/unified_cache/storage_attachment.py
  (+8): the switches are resolved again at a runtime attach; a write dropped at detach clears its mark
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_B/python/sglang/srt/mem_cache/base_prefix_cache.py (+4):
  InsertParams.skip_storage_backup
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_B/python/sglang/srt/mem_cache/unified_cache/cache_action.py (+2):
  BackupKV.skip_storage
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_B/test/registered/unit/mem_cache/test_next_use_eviction.py (+368
  -1): 9 tests in the new class TestStoredMarksOnACpuTree (real CPU tree)
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_B/test/registered/unit/mem_cache/test_storage_read_dispatch.py
  (+266): 6 tests in TestWriteBacklogHold, 2 in the new class TestRetentionSwitches
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_B/test/registered/unit/mem_cache/test_next_turn_prefetch.py
  (+297): 6 tests in TestCreate, TestCommitHook, TestPlanRegistration, TestGapTable and TestDeadlineMode
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_B/test/registered/unit/mem_cache/test_hicache_nixl_cleaner.py
  (+98 -1): 4 tests in the new class TestTouchedPages (no NIXL import)
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_B/test/registered/unit/mem_cache/test_hicache_nixl_storage.py
  (+125): 3 tests in the new class TestNixlTouch (real HiCacheNixl in tempfile directories)
- Notes, not engine: /home/wanhr/sglang/agent_cache/results/design_unified_1126/impl/B/steps_1_3.patch and, under
  checks/, final_ten.txt, dir_before_after.txt, mutations.txt, mutation_scripts/mutate.py,
  mutation_scripts/mutations.py. REPORT.md is missing (see not_done).

## Switches

- SGLANG_HICACHE_STORAGE_TRACK_STORED: gates the stored marks on tree nodes (stored only after an ack that covers
  every page, agreed over ranks), the heal (a child's write first rewrites host-backed ancestors with no SSD copy),
  the mark resets at clear and detach, stored= on h2s_done and the h2s_short line. Default off (EnvBool False). Needs
  --hicache-host-memory-mode cache and the Python tree core; otherwise it stays off with one warning.
- SGLANG_HICACHE_NIXL_TOUCH_INTERVAL_S: gates the touch thread in HiCacheNixl and the match_prefix hook that refreshes
  the mtime of the stored chain a request was served, head newest; the value is the least seconds between two
  refreshes of a page. Default 0.0 = off (EnvFloat; the extra-config key touch_interval_s wins over the env). Needs
  TRACK_STORED and a NIXL file backend that writes files on this rank.
- SGLANG_HICACHE_STORAGE_SKIP_ENDED_WRITES: gates the skip of the SSD write for the inserts of a turn whose hint ends
  its session (h2s_skip), and turns on X1's registration (only an explicit end registers a session as ended; ntp_end
  lines). Default off (EnvBool False). Needs TRACK_STORED and the next-turn prefetcher
  (--hicache-storage-next-turn-prefetch observe or on, which itself admits TP = PP = 1 only).
- SGLANG_HICACHE_NIXL_RETIRE_ENDED_PAGES: gates mtime zero for an explicitly ended session's own pages (l3_retire),
  the same at the ack for pages still on their way (l3_retire_ack), the touch thread, and X1's registration. Default
  off (EnvBool False; the extra-config key retire_ended_pages wins over the env). Needs TRACK_STORED, the next-turn
  prefetcher, a FULL-only cache without sidecar pools, and a NIXL file backend.
- How they are resolved: once per cache into the frozen struct cache.retention, at the end of init_hicache and again
  at a runtime attach of a storage backend. The two STORAGE switches are read from the environment by the cache; the
  two NIXL switches by NixlBackendConfig, published on the store and read off it by the cache. The prefetcher gets
  explicit_ends and retire_ended from its factory. With all four off: no mark is asked of the tree, no thread is
  started, no file is touched, no decision changes and no existing HICACHE_EVT line changes.

## Tests added

-
  test_storage_read_dispatch.py::TestWriteBacklogHold::test_a_write_acked_short_unlocks_its_pages_and_leaves_them_unstored:
  red when a short ack marks the node stored, or keeps the host lock or the pending tokens
- test_storage_read_dispatch.py::TestWriteBacklogHold::test_with_the_switch_off_no_node_is_marked: red when the tree
  is asked for a mark with the switch off, or the h2s_done line changes
- test_storage_read_dispatch.py::TestWriteBacklogHold::test_a_write_short_on_any_rank_is_unstored_on_every_rank: red
  when the verdict is not the minimum over ranks, a rank that writes nothing votes no, or the detach path reduces
-
  test_storage_read_dispatch.py::TestWriteBacklogHold::test_a_write_dropped_at_detach_is_neither_stored_nor_left_queued:
  red when detach leaves the dropped operation's id on the node
-
  test_storage_read_dispatch.py::TestWriteBacklogHold::test_a_node_retired_while_its_write_is_queued_is_retired_at_the_ack:
  red when the ack does not zero the pages, or marks them stored
- test_storage_read_dispatch.py::TestWriteBacklogHold::test_a_split_of_a_skipped_node_skips_both_fragments: red when a
  split fragment loses the skip or the retire decided for its node
- test_storage_read_dispatch.py::TestRetentionSwitches::test_a_switch_missing_what_it_needs_stays_off_and_says_so: red
  when a switch is granted without the marks, the prefetcher or the FULL-only cache it needs, or refuses silently
-
  test_storage_read_dispatch.py::TestRetentionSwitches::test_the_cache_reads_two_switches_from_the_envs_and_two_from_its_store:
  red when the cache reads a wrong env name, ignores the attached store's settings, or trusts a test double as a store
  with a cleaner
-
  test_next_use_eviction.py::TestStoredMarksOnACpuTree::test_a_split_while_the_write_is_queued_marks_both_fragments_at_the_ack:
  red when a fragment split off a queued node keeps the operation id or stays unstored after the ack
- test_next_use_eviction.py::TestStoredMarksOnACpuTree::test_a_child_write_first_rewrites_an_unstored_ancestor: red
  when there is no heal, read-back pages are rewritten, or an ancestor with a queued write or a host copy in flight is
  written again
- test_next_use_eviction.py::TestStoredMarksOnACpuTree::test_clearing_the_store_clears_every_mark: red when a mark
  survives a cleared store, or a split does not copy the mark
-
  test_next_use_eviction.py::TestStoredMarksOnACpuTree::test_a_stale_tail_takes_its_fresh_head_along_and_a_second_match_takes_nothing:
  red when a written page is stale at birth, only the stale nodes are refreshed, the order is tail first, or refreshed
  nodes are not stamped
- test_next_use_eviction.py::TestStoredMarksOnACpuTree::test_only_stored_nodes_are_refreshed: red when a node with a
  queued or failed write is refreshed
- test_next_use_eviction.py::TestStoredMarksOnACpuTree::test_a_probe_without_a_request_refreshes_nothing: red when a
  probe, or a match with storage detached, touches files
- test_next_use_eviction.py::TestStoredMarksOnACpuTree::test_a_turn_after_a_skipped_one_writes_the_skipped_node_first:
  red when an ended turn is written, locked or counted pending, the flag is lost at either insert site, the skip is
  not spent, or a later turn does not heal the skipped node
- test_next_use_eviction.py::TestStoredMarksOnACpuTree::test_retire_takes_the_replies_no_later_prompt_repeated: red
  when own replies are left, the first prompt or a queued write is zeroed at once, retired pages still count as
  stored, or a write that will be skipped is queued for an ack
-
  test_next_use_eviction.py::TestStoredMarksOnACpuTree::test_retire_keeps_the_first_prompt_and_stops_at_another_sessions_branch:
  red when another session's leaf, or a reply something was built on, is taken as the session's own
- test_next_turn_prefetch.py::TestGapTable::test_only_an_explicit_end_ends_the_session: red when no entry, an entry
  without a gap or an unreadable table ends a session, or reply mode ends one before the reply exists
- test_next_turn_prefetch.py::TestPlanRegistration::test_an_exact_hint_session_ends_at_the_turn_with_no_next: red when
  an un-hinted request or a named turn that names a next turn counts as an end, or a named turn with no next does not
- test_next_turn_prefetch.py::TestDeadlineMode::test_a_turn_with_no_usable_entry_is_ranked_as_un_hinted_not_as_ended:
  red when no entry, an expired plan or an abort registers the session as ended under the switch, or ntp_end carries a
  wrong reason
-
  test_next_turn_prefetch.py::TestDeadlineMode::test_only_an_explicit_end_retires_and_only_past_the_sessions_first_prompt:
  red when a no-entry turn's reply is not counted as the session's, the first prompt is not the first seen or not
  page-aligned, an expired session is retired, or an exact-hint end is not retired at commit
-
  test_next_turn_prefetch.py::TestCommitHook::test_an_explicit_end_skips_the_last_turns_write_and_retires_the_sessions_pages:
  red when the commit hook does not flag the ended turn, or the retire through the real tree zeroes the wrong pages
- test_next_turn_prefetch.py::TestCreate::test_the_factory_follows_the_end_switches_the_cache_resolved: red when
  create() does not pass the cache's resolved switches to the prefetcher
- test_hicache_nixl_cleaner.py::TestTouchedPages::test_a_refreshed_chain_loses_its_tail_before_its_head: red when the
  tail is stamped newest or every page gets the same stamp, so that the cleaner would delete a head before its tail
- test_hicache_nixl_cleaner.py::TestTouchedPages::test_touching_a_deleted_file_creates_nothing_and_is_counted_gone:
  red when a touch creates a missing file or does not count it
- test_hicache_nixl_cleaner.py::TestTouchedPages::test_a_page_is_retired_only_once_every_file_of_it_is_at_zero: red
  when the cleaner stops ranking a page by its newest file, the fact that makes retire need a FULL-only, TP = 1 cache
- test_hicache_nixl_cleaner.py::TestTouchedPages::test_nixl_config_resolves_the_touch_switches: red when a config key
  does not win over its env, a key reaches the NIXL plugin, or a wrong type is accepted
- test_hicache_nixl_storage.py::TestNixlTouch::test_a_touch_reaches_every_file_the_write_path_made_for_a_page: red
  when the thread names files the write path did not make, misses the V file, loses the head-newest order, or outlives
  close()
- test_hicache_nixl_storage.py::TestNixlTouch::test_with_the_switches_off_no_thread_runs_and_a_touch_moves_no_mtime:
  red when a thread is started or an mtime moves with the switches off
- test_hicache_nixl_storage.py::TestNixlTouch::test_a_retired_page_has_every_file_at_mtime_zero: red when the retire
  switch alone starts no thread, or a retire leaves a file with its mtime

## Test results

- test_next_turn_prefetch.py: 83 passed, 63 subtests passed (before: 77 passed, 55 subtests)
- test_next_turn_estimators.py: 14 passed (unchanged)
- test_next_use_eviction.py: 30 passed, 6 subtests passed (before: 21 passed, 2 subtests)
- test_storage_read_scheduler.py: 9 passed (unchanged)
- test_storage_read_dispatch.py: 36 passed, 6 subtests passed (before: 28 passed, 2 subtests)
- test_host_census.py: 7 passed, 9 subtests passed (unchanged)
- test_hicache_nixl_cleaner.py: 18 passed, 15 subtests passed (before: 14 passed, 12 subtests)
- test_hicache_nixl_storage.py: 24 passed, 1 skipped, 2 subtests passed (before: 21 passed, 1 skipped). The skip is
  the MinIO object-store case (minio binary or boto3 not available), as before; nixl is installed in the container, so
  no NIXL case skips.
- test_evict_policy.py: 23 passed (unchanged)
- test_unified_radix_cache_unittest.py: 2,791 cases, 1,280 ran and 1,511 skipped. Of the 1,280, 46 passed and 1,234
  failed, every one with 'No accelerator (CUDA, ...) is available': they need the GPU, which is hidden. The skips are
  the file's own parameter conditions (for example 'requires SWA'). Identical before and after, case by case.
- All 122 test files of test/registered/unit/mem_cache (bench files left out), one pytest run per file, before any
  edit and on the final engine code: only the five files with added tests differ, each by its added tests; no case
  that passed before fails or is missing. Eleven files have failures or errors on CPU, the same cases before and
  after.
- Mutation checks: 64 single-line breaks of the engine, each run against the test meant to catch it, all killed on the
  final code. Two survived when first written (the skip set never emptied; an exact-hint end not retired at commit)
  and were killed once their test was extended.
- No test file was added; the new engine file storage_retention.py is tested in test_storage_read_dispatch.py.
- All runs CPU only (CUDA_VISIBLE_DEVICES=9, niced); the ten named files were last run at 5:32 PM ET on 2026-11-26,
  after the last edit.

## Deviations from the plan

- R4 refresh, what counts as stale. In the pseudo-diff a node's stamp starts at zero and only a refresh sets it, so
  every written page is stale from birth and each turn's first match touches its session's whole stored chain: up to
  about 12.8 M utime calls in campaign 34's ec_full (410.48 M ideal cached tokens of returning turns / 64 tokens a
  page x 2 files; my upper bound), against the 1.72 M the plan's own count gives at 300 s. Built instead: a full ack
  stamps the node, and one stale stored node refreshes every stored node of the served chain, head first, so each page
  is touched at most once per interval. Cost of the choice: pages written since the last refresh are newer than their
  head until the next refresh.
- Dependencies added to the plan's list. The refresh needs TRACK_STORED (its walk reads the stored mark; without it
  only read-back nodes would be refreshed). TRACK_STORED needs host memory mode cache (in buffer_only an ack carries
  no node) and the Python tree core (the marks are fields of its nodes). Retire needs a FULL-only cache without
  sidecar pools (the cleaner ranks a page by its newest file and only the KV files are touched).
- Retire, whose leaf. The pseudo-diff takes an off-path leaf as the session's own when it is not registered for next
  use, and calls retire before the commit registers the new tail. At that point the session's previous reply is still
  registered, so the walk stops one node above the last turn and retires nothing older; X1 also makes 'not registered'
  unsafe, since it forgets the tails of live sessions. Built instead: the prefetcher records per session its first
  prompt length and the node each turn committed, and a leaf is own only if it is one of those.
- Retire, smaller points. A node whose host copy is still in flight (no write queued yet) is also left to its ack, and
  split fragments inherit that. The own replies hanging off the first-prompt node are taken. The session name comes
  from the table when the rid chain is broken (after a turn with no usable entry or an expired plan). The first prompt
  is the first one committed, with or without a usable entry, and is kept until the explicit end (bounded by
  REGISTRY_MAX_ENTRIES). A pending node whose write R1 will skip is not queued for the ack. at_ack= counts pages.
  l3_retire_ack is an extra log line.
- Retire at the ack. The pseudo-diff calls finish_storage_write a second time with stored=False; the first call has
  already cleared the operation id, so the second finds nothing and the node stays stored. Built: one call with the
  final verdict.
- ends_session. In the pseudo-diff a next-turn action that cannot be read falls into the exact-hint branch and ends
  the session of a named turn. Built: a next-turn action that is present and unreadable ends nothing.
- Rank agreement. The stored verdict is reduced with _all_reduce (TP minimum on PP rank 0, then the PP sync) whenever
  TP > 1 or PP > 1. The pseudo-diff uses _all_reduce_attn_groups over TP only, which would let PP ranks mark, heal and
  count acks differently.
- Heal. An ancestor whose host copy is still in flight is skipped; its own ack writes it. The pseudo-diff would queue
  a write of a copy that is not complete.
- h2s_done. stored= is appended only with TRACK_STORED on (the plan appends it always), so the line stays
  byte-identical for every existing arm, as the switch contract requires.
- Switch state. One frozen struct on the cache (cache.retention, class default all off) in a new module
  storage_retention.py, instead of four attributes: caches that the existing tests build without __init__ then see
  every switch off. It is resolved again at a runtime attach, and the two NIXL switches are read off the attached
  store instead of the environment, so the extra-config keys are honoured.
- finish_storage_write takes stored_at (the time of a full ack, or None) instead of stored: bool, because the ack also
  stamps the node for the refresh.
- environ.py has only the four switches of steps 1 to 3; the other five of 00_environ.diff belong to steps 4 to 6.
  They sit before SGLANG_HICACHE_NEXT_TURN_ROOM_RETRY, because someone else added a switch after that line in the real
  tree during the work.
- Tests against the plan's list. Not added: test_a_touch_after_the_cleaners_scan_does_not_save_the_page (it would
  assert a defect, X3, not guard against one). test_pages_read_back_from_storage_are_stored_at_birth is folded into
  the heal test. R1's two stub-level tests are one end-to-end test on the CPU tree. The stamp tests run on
  NixlFileManager.stamp_pages without a NIXL import, and the thread is tested on a real HiCacheNixl. Added beyond the
  plan: the rank vote, the detach hook, the switch resolver and its wiring, the factory, the commit hook end to end,
  the real backend.
- The host-census fields of the plan (dirty, n_short, skip_tok, touch_pages, retire_pages) are not added: they change
  an existing line for every arm, and the census walk belongs to P2's evict_unstored in step 4.

## Not done

- REPORT.md was not written. The Write tool refused it for a subagent ('Subagents should return findings as text, not
  write report files. Include this content in your final response instead'), and I did not go around the refusal. Its
  full content is this structured output; the target path is
  /home/wanhr/sglang/agent_cache/results/design_unified_1126/impl/B/REPORT.md.
- Steps 4 to 6 (P1, P2, R5, R2, X3, X5) are not started, as assigned. Nothing here blocks them: the marks they read
  are node.storage_stored and node.storage_write_op.
- No run on a GPU, with a server, or with more than one rank. The TP/PP reduce of the stored verdict is covered only
  by a unit test with a stand-in for the other rank.
- The merge with copy A was not built or tested. As of 5:17 PM ET one test hunk is rejected against A's files (the
  import block of test_next_use_eviction.py) and four hunks of next_turn_prefetch.py apply with fuzz; A was still
  being edited.
- The utime cost on cold inodes of the SSD volume, and its journal load, are still unmeasured; l3_touch files= per
  second is the number to read in the first arm.
- Race X3 (a touch that lands after the cleaner's scan does not save the page) is unchanged, and its test is not
  added; l3_touch gone= counts the files not found.
- Open decision: a request with no next-turn action that no earlier turn named (a client that sends no hints) is still
  registered as ended at commit, as today and as the plan leaves it. It is not skipped and not retired.
- R1 limits. A one-request session that ends explicitly does not get its first prompt written (1 of 566 ended sessions
  in campaign 34); the heal writes it once a session that does not end extends it while it is in memory. R1 acts where
  the insert makes the host copy: under write_through_selective it also covers an earlier node whose copy the ending
  turn triggers, and under write_back it does nothing. An aborted last turn whose hint states the end is skipped too,
  while the commit hook forgets its session without retiring it.
- Retire limits. It stops at a leaf it does not know, so a reply left by an aborted turn keeps everything above it
  un-retired. A session seen for one turn only is not retired. The stored verdict covers the KV pages only (exact for
  FULL-only stacks).
- The prefetcher's two end switches are fixed when it is created: a storage backend attached at runtime with another
  setting changes the cache, not the prefetcher.


# Fix round after the three reviews (2026-11-27 01:37Z)

## Summary

All 27 findings were checked against copy B (/home/wanhr/sglang/agent_cache/frozen/impl_unified_B). None is a
default-off break. Every bug and plan mismatch is real and is fixed in the copy, each with a test that goes red when
the defect is put back. The style and test-gap findings are fixed except the points under "left". With the four
switches off the engine behaves as before: 118 of the 122 test files of the folder give the same summary line as
before the fixes, and the other 4 differ only by the tests added.  Read first - REPORT.md is not updated. A subagent
may not write report files in this session (the refusal the engineer met), so the file on disk still describes the
copy before the fixes. This output is its new content. - Everything ran on CPU. Nothing ran on a GPU or with more than
one rank. The two test files that run the real init_hicache still fail for want of a GPU, as before. - The merge with
copy A was only dry-run on scratch copies.  What the fixes change (switches on only) - Retire and the heal no longer
undo each other. The write of a page parked for retire heals nothing above it, so an explicit end without the write
skip queues the last reply alone, not the chain just zeroed. - Retire at the ack goes fragment by fragment. After a
split, only the fragments the retire walk took are zeroed and left unstored; the rest get the normal verdict and no id
stays parked. - The heal walks the whole root path, so a hole above a stored node is rewritten too. A write below
pages parked for retire takes them off the retire: something was built on them after the end. - A cleared store also
drops the queued operation ids and the parked pages. A detach clears every stored mark and both sets. - An ended
session's own leaves are known by the last page key of each turn it committed, so a reply read back from the SSD under
a new node id is still retired. - Only a turn that an exact (v1) hint named ends its session by carrying no next-turn
action. A known session's turn that comes without a hint forgets the session's tail, with a new log reason: ntp_end
explicit=0 why=no_hint. - The write skip decides nothing while the store is detached. I found this while fixing: the
ids of skipped nodes piled up, since no write came to spend them. - The touch thread holds its backend weakly and
survives a failing job; close() warns and drops the jobs left when the thread is still busy after 5 s.  Switches, all
off by default, and what each needs - SGLANG_HICACHE_STORAGE_TRACK_STORED (bool): the stored marks, the heal, the
resets at clear and detach, stored= on h2s_done, h2s_short. Needs --hicache-host-memory-mode cache, the Python tree
core and PP = 1 (new). - SGLANG_HICACHE_NIXL_TOUCH_INTERVAL_S (seconds, 0 = off; the extra-config key touch_interval_s
wins): the touch thread and the refresh at a request's match. Needs TRACK_STORED and a NIXL file backend that writes
files on this rank. - SGLANG_HICACHE_STORAGE_SKIP_ENDED_WRITES (bool): no SSD write for a turn whose hint ends its
session (h2s_skip), and the registration that follows the explicit end (ntp_end). Needs TRACK_STORED and the next-turn
prefetcher. - SGLANG_HICACHE_NIXL_RETIRE_ENDED_PAGES (bool; the key retire_ended_pages wins): mtime zero for an ended
session's own pages (l3_retire, l3_retire_ack), and the same registration. Needs TRACK_STORED, the prefetcher, the
touch interval above 0 (new), a FULL-only cache without sidecar pools (now checked again when a pool is registered
after init) and a NIXL file backend. - A switch that lacks what it needs stays off and logs "HiCache retention: <name>
stays off: it needs ...".  Deviations from the plan added by the fixes - Two dependencies. TRACK_STORED needs PP = 1:
the stored verdict is a minimum over TP ranks only, and the PP sync hands down PP rank 0's. Retire needs the touch
interval: a retired page read back counts as stored, and only the refresh gives its files a recent mtime again. - The
heal covers the whole root path, not the unstored run right above the node, and is skipped for a write parked for
retire. - A detach clears every stored mark, where the plan only unmarks the dropped operations: the next store
attached may be another one. After a detach and re-attach of the same store, the first write under each chain rewrites
that chain (a stat and a utime per file with skip-stored writes on). - Own leaves are told by last page key. The plan
used "not registered", the engineer node ids. - The refresh rule is kept as built and its bound restated. A request's
match refreshes every stored node of its chain once any of them is an interval old or was read back from the SSD. So a
page is touched once per interval per chain through it, plus once for each read that lands below it. The threads
reviewer estimates about 6.3 M utime calls at campaign 34's ec_full volume against the 1.72 M of the plan's count, at
2.8 us a call on a temp directory; the journal load on the SSD volume is unmeasured. - ntp_end has one more reason
value, no_hint. l3_retire_ack pages= counts the retired fragments' pages.  First campaign arm - Set
SGLANG_HICACHE_STORAGE_TRACK_STORED=1 SGLANG_HICACHE_NIXL_TOUCH_INTERVAL_S=300
SGLANG_HICACHE_NIXL_RETIRE_ENDED_PAGES=1 SGLANG_HICACHE_STORAGE_SKIP_ENDED_WRITES=1, against the same arm with all
four unset, and run past the first cleaner pass (minute 87.5 to 88.3 at campaign 34's setting). - It needs what the
hinted arms have: the prefetcher on, host memory mode cache, write-through, the NIXL file backend, TP = PP = 1, a
FULL-only model, and hints that state the end. - The start-up line must read "HiCache retention: track_stored=1
skip_ended_writes=1 touch_interval_s=300 retire_ended_pages=1", with no "stays off" warning. - Retire alone is now
refused. To isolate the refresh, set TRACK_STORED and the touch interval only (no hints needed). To run retire without
the periodic refresh, set a very large interval: only pages read back are then touched. - What to read: h2s_skip
tokens (4.7 to 4.8 % of inserted tokens by the plan; 0.7 % means only replies are skipped); ntp_end explicit=1 (566
sessions in ec_full) against explicit=0 by reason; l3_retire with pages above 0 (count the pages=0 lines);
l3_retire_ack; l3_touch files= per second (expect several times the plan's 239); stored=0 and h2s_short (none
expected). The effect to look for: after each "NIXL L3 cleanup", fewer s2h_query storage_hit=0 on returning turns and
fewer "failed to retrieve page".  Merge notes (dry run at about 8:30 PM ET on 2026-11-26, git merge-file -p on scratch
copies over frozen/impl_unified_base) - Against copy A: no conflict in the four engine files both copies change
(environ.py, next_turn_prefetch.py, unified_tree_core.py, unified_radix_cache.py), nor in test_next_turn_prefetch.py
and test_storage_read_dispatch.py. The merged engine files parse; they were not run. There are two conflicts in the
import block of test_next_use_eviction.py, where both copies add imports and A extends the components import: a union
keeps that import twice. - Against the working tree (the load hold, keep_touched in the cleaner): environ.py and
test_hicache_nixl_cleaner.py merge without conflict.  Notes folder
/home/wanhr/sglang/agent_cache/results/design_unified_1126/impl/B/ - steps_1_3_fixed.patch: the whole change, 16
files; frozen/impl_unified_base plus this patch reproduces the copy. - fix/fix.patch: the fixes alone, 13 files.
fix/before and fix/after: the 16 files before and after. fix/probes: the reviewers' probes with the two renamed
arguments, and a variant of f4. - checks/dir_after_fix.txt (the 122 files), checks/final_ten_after_fix.txt,
checks/final_rerun_after_last_edit.txt, checks/mutations_after_fix*.out, checks/mutation_scripts/mutations.py
(rebuilt: 100 mutations), checks/run_dir.sh. - steps_1_3.patch and the table in checks/mutations.txt describe the copy
before the fixes. I made no git write and no network call, and touched nothing outside copy B and this folder. No file
under /home/wanhr/sglang/python or /home/wanhr/sglang/test changed during the work.


## Fixed

- Correctness 1 and threads 1 (bug), the heal undid the retire when the write skip was off: real. The last reply's
  write, parked for retire, ran the heal and re-queued the chain just zeroed. Fix: write_backup_storage runs no heal
  for a node parked for retire. Test: test_next_use_eviction.py
  TestStoredMarksOnACpuTree::test_the_last_replys_write_does_not_bring_a_retired_chain_back, on the real CPU tree
  through the host-copy ack and the write ack, with the write skip off and on.
- Correctness 2 (bug), retire at the ack hit the whole operation after a split: real. Fix:
  UnifiedTreeCore.finish_storage_write decides per fragment. Only the fragments the retire walk parked are zeroed and
  left unstored, the others get the normal verdict, and every parked id of the operation is dropped. Test:
  test_a_node_retired_while_its_write_is_queued_is_retired_at_the_ack, moved from the stub tree to the real tree, with
  a split with and without another session's branch.
- Correctness 3 (bug), the heal stopped at the first stored ancestor: real. Fix: unstored_ancestors walks the whole
  root path, up to the first ancestor without a host copy. A write below pages parked for retire also takes them off
  the retire. Tests: test_a_child_write_first_rewrites_an_unstored_ancestor (a short ack followed by a full one leaves
  the hole above a stored node) and test_a_write_below_a_retired_chain_brings_it_back.
- Correctness 4 (bug), retire granted without the refresh: real. Fix: resolve_retention_switches grants the retire
  only with a touch interval above 0 and logs the refusal. Test: test_storage_read_dispatch.py
  TestRetentionSwitches::test_a_switch_missing_what_it_needs_stays_off_and_says_so.
- Correctness 5 (bug), a turn a table named counted as an explicit end when it came with no hint: real. Fix: only a
  turn named by an exact (v1) hint ends by carrying no action; the name recorded for the next rid carries that mark. A
  known session's turn without a hint forgets the tail and logs ntp_end explicit=0 why=no_hint. Tests:
  TestPlanRegistration::test_an_exact_hint_session_ends_at_the_turn_with_no_next (table-named case added) and
  TestDeadlineMode::test_a_turn_with_no_usable_entry_is_ranked_as_un_hinted_not_as_ended.
- Correctness 6 (bug), a clear left the queued operation id on its node: real. Fix: clear_storage_marks also resets
  storage_write_op, and the set of pages parked for retire is emptied. Test: the clear case of
  test_clearing_or_detaching_the_store_clears_every_mark (a write queued before the clear and acked after it stays
  unstored).
- Correctness 7 (test gap), the retire tests stopped at the retire call: real. The tests of the six items above run on
  the real CPU tree and ack through the cache's own drain (the host copy, then the write), including a split between
  submit and retire, a turn after the end while a retired write is queued, and a short ack followed by a full one.
- Correctness 8 (nit), a detach kept the stored marks and both sets: real. Fix: a detach clears every stored mark, the
  queued ids and both sets (StorageAttachment._release_pending_storage_ops calls
  UnifiedRadixCache._forget_storage_copies). Test: the detach case of
  test_clearing_or_detaching_the_store_clears_every_mark.
- Correctness 9 and threads 2 (nit, plan mismatch), the claim that a page is touched at most once per interval: real.
  The bound is per chain. The docstring of _refresh_served_chain, a comment in
  test_a_stale_tail_takes_its_fresh_head_along_and_a_second_match_takes_nothing and this report now say so. The rule
  itself is kept (see rejected).
- Threads 3 (test gap), nothing guarded the try/except that keeps the touch thread alive: real. Test:
  test_hicache_nixl_storage.py TestNixlTouch::test_a_touch_that_fails_does_not_end_the_thread (first job raises,
  second is applied, the warning is logged, no thread after close), with a mutation for it.
- Threads 4 and the first half of rules 6 (plan mismatch, test gap), with PP above 1 the stored verdict was PP rank
  0's: real. Fix: TRACK_STORED is refused when PP is above 1, with a refusal line; _agree_stored reduces over TP ranks
  only and its docstring says so. The PP branch is gone, so it needs no test; the refusal is tested in both
  TestRetentionSwitches cases.
- Threads 5 and rules 3 (plan mismatch, bug), a draft model's sidecar pools are registered after the switches were
  resolved: real. Fix: register_sidecar_pool resolves the switches again when the retire had been granted. Test:
  TestRetentionSwitches::test_the_cache_passes_the_resolver_what_the_server_really_has.
- Threads 7 (nit), the touch thread kept its backend alive and close() hid a thread that did not stop: real. Fix: the
  thread runs a module-level loop over a weak reference; close() logs a warning and drops the queued jobs when the
  thread is still busy after 5 s. Test:
  TestNixlTouch::test_a_backend_dropped_without_close_is_collected_with_its_thread.
- Rules 1 (test gap), R1's flag at the prompt insert had no test: real. Test: test_next_turn_prefetch.py
  TestCommitHook::test_the_prompt_insert_of_an_ending_turn_is_flagged_once_its_end_is_known, through the real
  cache_unfinished_req: key source hint with an ending and a non-ending turn, key source reply (stays unflagged), and
  a detached store.
- Rules 2 (style), overrides entered in a bare with: real. Fix: TestRetentionSwitches enters them through an ExitStack
  closed in addCleanup and pins both STORAGE switches off first; TestNixlTouch._backend wraps the constructor in
  patch.dict.
- Rules 4 (test gap), the dependency checks were tested on the pure function only: real. Tests:
  test_the_cache_passes_the_resolver_what_the_server_really_has (prefetcher off, buffer_only, another tree core, PP 2,
  a sidecar pool, each with its refusal line), test_a_runtime_attach_resolves_the_switches_for_the_store_attached, and
  TestNixlTouch::test_a_rank_that_writes_no_files_takes_neither_switch.
- Rules 5 (test gap), two negative branches had no test: real. A retire with the store detached is added to
  test_retire_takes_the_replies_no_later_prompt_repeated, and a request match at a touch interval of 0 to
  test_a_probe_without_a_request_refreshes_nothing.
- Rules 6, second half (test gap), the copy of the refresh stamp at a split: real. A split of a freshly stamped node
  followed by a match that must return nothing is added to
  test_a_stale_tail_takes_its_fresh_head_along_and_a_second_match_takes_nothing.
- Rules 7 (plan mismatch), a reply read back under a new node id counted as another session's leaf: real. Fix: the
  prefetcher keeps the last page key of each committed turn (SessionExtent.leaf_keys) and the retire walk knows own
  leaves by that key. Test: test_a_reply_read_back_as_a_new_node_is_still_the_sessions_own.
- Rules 8 (style), comment style: real. Every new or changed docstring is at most two lines, except
  HiCacheStorage.touch_pages, a base-class surface, which keeps four. The test docstrings state the constraint in the
  present tense. The design is cited without a section number.
- Rules 9 (style), function size: real. HiCacheNixl.__init__ is back to 119 lines (118 at HEAD) with _init_touch.
  NextTurnPrefetcher.__init__ is 98 lines (104) and on_turn_committed 93, with two helpers (_init_end_handling,
  _ends_without_forecast).
- Rules 10 (nit), positional arguments: real. mark_storage_write, finish_storage_write, _on_explicit_end and
  _commit_storage_write take op_id, outlook and operation by keyword only, and the touch job is a msgspec struct.
- Rules 11 (nit), small contracts unpinned: real, four of five pinned. The ntp_end fields at the exact-hint end;
  tokens= of h2s_skip and at_ack= of l3_retire on a node of two pages; the bound of the per-session records
  (test_what_is_kept_per_session_for_its_end_is_bounded_oldest_first).
- Found while fixing, not in the findings: with the store detached, the write skip kept flagging ended turns, and
  their node ids piled up because no write came to spend them. _turn_ends_session now decides nothing while storage is
  off; the prompt-insert test covers it.

## Rejected, with reasons

- Correctness 9, the cut 'touch from the head down to the deepest stale node only (the plan's rule)': not taken. It
  makes more calls, not fewer: each time a deeper node turns stale the whole prefix above it is touched again, whereas
  the present rule stamps the chain together so its next refresh is one interval away.
- Threads 2, the cut 'stamp a stale tail just below its nearest fresh ancestor': not taken here. It needs a wall-clock
  stamp per node and changes the design; it is listed under left as an open decision.
- Correctness 1 and threads 1, the interim 'make the resolver refuse retire without the write skip': not needed once
  the bug is fixed; retire without the write skip is tested.
- Threads 4, the alternative of a minimum over the PP group: not taken. It needs a new collective on the PP group that
  cannot be tested here; the switch is refused with PP above 1 instead.
- Threads 7, 'a stop flag so that close() discards queued jobs': only after the join times out. close() still does the
  jobs queued before it, because the detach path and the tests rely on a retire queued just before close being done.
- Rules 11, the reset of the two id sets in _reset_full: no test added. Node ids are never reused (a class-level
  counter), so a stale id cannot match a node of the rebuilt tree.
- Rules 9, the class default `retention` between two methods: left where it is until the merge, since copy A adds
  attributes at the class header.

## Test results

- test_next_turn_prefetch.py: 85 passed, 64 subtests passed (before the fixes: 83 passed, 63 subtests).
- test_next_turn_estimators.py: 14 passed (unchanged).
- test_next_use_eviction.py: 34 passed, 12 subtests passed (before: 30 passed, 6 subtests).
- test_storage_read_scheduler.py: 9 passed (unchanged).
- test_storage_read_dispatch.py: 36 passed, 11 subtests passed (before: 36 passed, 6 subtests; two stub-tree cases
  moved to the real tree, two added).
- test_host_census.py: 7 passed, 9 subtests passed (unchanged).
- test_hicache_nixl_cleaner.py: 18 passed, 15 subtests passed (unchanged).
- test_hicache_nixl_storage.py: 27 passed, 1 skipped, 2 subtests passed (before: 24 passed, 1 skipped). The skip is
  the MinIO object-store case, as before (the engineer's reading: minio binary or boto3 not available). nixl is
  installed in the container, so no NIXL case skips. TestNixlTouch (6 cases) sits in a file registered for the CUDA
  lane that imports NIXL at module scope: without the nixl wheel that file does not run at all. TestTouchedPages (4
  cases, in test_hicache_nixl_cleaner.py) needs no NIXL.
- test_evict_policy.py: 23 passed (unchanged).
- test_unified_radix_cache_unittest.py: 1,234 failed, 46 passed, 1,511 skipped in 7 min 15 s, the same summary as
  before. The engineer attributes the failures to the hidden GPU ('No accelerator'); I compared the summary line only.
- Whole folder, one pytest run per file (122 files, bench files left out), ended 8:17 PM ET on 2026-11-26: 118 files
  give the same summary line as in the engineer's table, and the 4 that differ are the files with added tests.
  test_decode_retraction_backup.py is 10 failed, 4 passed, as before. After that run only next_turn_prefetch.py
  changed (two helpers extracted); the three test files that import it and the other named files were run again at
  8:27 PM ET with the same counts.
- Tests of steps 1 to 3: 39 now, 30 before. 13 in TestStoredMarksOnACpuTree, 8 in test_storage_read_dispatch.py, 8 in
  test_next_turn_prefetch.py, 4 in TestTouchedPages, 6 in TestNixlTouch. No test file was added.
- Mutation checks: 100 single breaks of the engine, each run against the test meant to catch it, all killed. They are
  the engineer's 64 (20 re-anchored where the fix moved their text) and 36 that put back each defect fixed here. 99
  ran in one pass that ended 7:59 PM ET; the 27 that the later small edits touch, one of them new, were run again on
  the final files.
- Reviewers' probes on the fixed copy. Of the nine correctness probes that failed, eight pass with only two keyword
  names adapted (f1, f2, f3, f5, g1, g2, g3, h1). The ninth, f4, sets the retire without the touch interval by hand, a
  state the resolver now refuses; its variant with the interval passes. The threads reviewer's retire-heal probe
  passes (3 of 3), his touch-thread probes pass (a backend dropped without close() is collected with the switch on),
  and his process-exit probe returns 0.
- Four tests from outside the folder (scheduler attach, scheduler events, invariant checker, decode tree core, the
  threads reviewer's copies of HEAD's) give the same outcome on the fixed copy and on the base copy: 2, 2, 10 and 5
  passed. A fifth, the radix walker of the CUDA lane, errors at collection on both.
- ruff 0.16.10 from the uv cache: check F401, F821, UP037 is clean on the changed files, and format --diff shows the
  same number of hunks per file as before the fixes, none in a changed line.
- All runs were CPU only (CUDA_VISIBLE_DEVICES=9, niced), four pytest processes at most; the GPU run in progress was
  not touched.

## Files changed in this round

- /home/wanhr/sglang/agent_cache/frozen/impl_unified_B/python/sglang/srt/mem_cache/unified_radix_cache.py (+64 -35):
  no heal for a write parked for retire, _heal_storage_chain_above, retire at the ack per fragment,
  _forget_storage_copies at clear and detach, the switches resolved again in register_sidecar_pool, PP passed to the
  resolver, _agree_stored over TP only, the write skip off while storage is detached, own_leaf_keys
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_B/python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py
  (+50 -30): finish_storage_write per fragment with keyword-only op_id, ancestor_ids, unstored_ancestors over the
  whole root path, clear_storage_marks also resets the queued op, own leaves by last page key
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_B/python/sglang/srt/mem_cache/next_turn_prefetch.py (+64 -29):
  _ExactHintSession, SessionExtent.leaf_keys, _init_end_handling and _ends_without_forecast, the no_hint reason,
  keyword-only _on_explicit_end
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_B/python/sglang/srt/mem_cache/storage_retention.py (+9 -2):
  TRACK_STORED needs PP = 1, the retire needs the touch interval, the design cited without a section number
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_B/python/sglang/srt/mem_cache/unified_cache/storage_attachment.py
  (+4 -6): a detach clears every stored mark and both sets
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_B/python/sglang/srt/mem_cache/storage/nixl/hicache_nixl.py (+102
  -34): _init_touch, _TouchJob, the module-level _touch_loop over a weak reference, the warning and the abandon flag
  in close()
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_B/python/sglang/srt/mem_cache/storage/nixl/nixl_utils.py (+8
  -13): docstrings of the two getters and of stamp_pages cut to two lines
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_B/python/sglang/srt/environ.py (+3 -3): the comments of two
  switches state the new dependencies
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_B/test/registered/unit/mem_cache/test_next_use_eviction.py (+319
  -102): the fixtures ack through the cache's own drain; 13 tests in TestStoredMarksOnACpuTree (4 more, one of them
  moved from the stub tree)
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_B/test/registered/unit/mem_cache/test_storage_read_dispatch.py
  (+154 -89): 4 stored-mark cases on the stub tree, 4 in TestRetentionSwitches (wiring through the cache and a runtime
  attach added)
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_B/test/registered/unit/mem_cache/test_next_turn_prefetch.py (+180
  -34): the prompt-insert test, the bound test, the table-named and no_hint cases, leaf keys
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_B/test/registered/unit/mem_cache/test_hicache_nixl_storage.py
  (+70 -12): 3 more TestNixlTouch cases, patch.dict around the constructor
- /home/wanhr/sglang/agent_cache/frozen/impl_unified_B/test/registered/unit/mem_cache/test_hicache_nixl_cleaner.py (+8
  -13): docstrings only
- Unchanged by the fixes, part of steps 1 to 3: base_prefix_cache.py, hicache_storage.py and
  unified_cache/cache_action.py in the copy. The cleaner, scheduler.py, tokenizer_manager.py and model_runner.py equal
  the base.
- Notes, not engine, under /home/wanhr/sglang/agent_cache/results/design_unified_1126/impl/B/: steps_1_3_fixed.patch;
  fix/fix.patch, fix/before/, fix/after/, fix/probes/, fix/mutations_before_fix.py; checks/dir_after_fix.txt and .raw,
  checks/final_ten_after_fix.txt, checks/final_rerun_after_last_edit.txt, checks/mutations_after_fix.out and its three
  rerun files, checks/mutation_scripts/mutations.py (rebuilt), checks/run_dir.sh, and a note at the top of
  checks/mutations.txt. REPORT.md was not written.

## Left

- REPORT.md: not updated (a subagent may not write report files here). The file on disk describes the copy before the
  fixes; this output is its new content.
- GPU and multi-rank: nothing ran on a GPU or with more than one rank. Before the first arm, run
  test_unified_radix_cache_unittest.py and test_decode_retraction_backup.py on the GPU, or start the off arm first and
  compare its start-up lines and HICACHE_EVT formats with campaign 34's (threads 6).
- Merge with copy A: only dry-run. Two conflicts remain in the import block of test_next_use_eviction.py (a union
  keeps the components import twice). The class default `retention` should move to the class header after the merge.
  The merged engine was not built or tested.
- Refresh cost, open decision: each read that lands below a chain touches the whole in-memory chain again. l3_touch
  files= per second in the first arm is the number to read. Two ways to cut it, neither built: stamp read-back pages
  just below their head's last stamp (needs a wall-clock stamp per node), or touch only the stale nodes and accept a
  tail newer than its head for up to one interval, as freshly written pages already are.
- Retire without the write skip writes the last turn and then zeroes it at the ack: correct but wasteful. Whether the
  retire alone should also skip those writes is the author's call.
- A store that fails every write: each write retries its whole unstored root path, so the write queue grows with the
  chain depth. This is the plan's heal; h2s_short counts show it. No bound was added.
- A retire's zeroing still queued on the touch thread when a fork's rewrite of the same pages lands leaves them at
  mtime zero until their next refresh. It needs a fork within the touch backlog, which is milliseconds at campaign
  34's volume; not closed.
- Race X3 (a touch that lands after the cleaner's scan does not save the page) is unchanged, and the utime cost on
  cold inodes of the SSD volume is still unmeasured.
- Retire limits that remain: the walk stops at a reply left by an aborted turn; an own reply under a node where
  another session branches is left (for example the first reply when another session shares the whole first prompt); a
  session seen for one turn is not retired.
- R1 limits as in the engineer's report: a one-request session that ends explicitly does not get its first prompt
  written until a session that does not end extends it; under write_back the skip does nothing.
- Open decision from the plan: a request with no next-turn action that no earlier turn named (a client that sends no
  hints) is still registered as ended at commit.
- The prefetcher's two end switches are fixed when it is created; a storage backend attached at runtime with another
  setting changes the cache, not the prefetcher.
- HiCacheNixl's own condition is tested for an MLA rank that writes no files; the object-store plugin branch needs an
  object store and is untested.
- Steps 4 to 6 (P1, P2, R5, R2, X3, X5) are not started, as assigned. The marks they read are still
  node.storage_stored and node.storage_write_op.
