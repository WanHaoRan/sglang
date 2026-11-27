"""Reviewer probes (correctness lens), not part of the change: each test asserts the
behaviour a reader of the plan would expect and is expected to FAIL where the copy
has the defect named in its docstring.

Run: copy into the copy's test folder as zz_review_correctness_1.py, then the recipe.
"""

from array import array
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from test_next_use_eviction import (
    computed_node,
    read_back_node,
    stored_marks_cache,
    stored_node,
)

from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.hicache_storage import HiCacheStorage
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.test.test_utils import CustomTestCase


def insert_turn(cache, prompt, *, ends=False) -> int:
    """Insert a prompt under write-through WITHOUT acking the host copies."""
    key = RadixKey(array("q", prompt))
    held = cache.match_prefix(MatchPrefixParams(key=key)).device_indices
    fresh = cache.token_to_kv_pool_allocator.alloc(len(prompt) - len(held))
    return cache.insert(
        InsertParams(key=key, value=torch.cat([held, fresh]), skip_storage_backup=ends)
    ).last_device_node


def ack_host_copies(cache) -> None:
    for ack_id in list(cache.ongoing_write_through):
        cache._finish_write_through_ack(ack_id)


def ack_storage_writes(cache, *, only=None) -> list:
    """Ack queued storage writes in full, in submit order, as _drain_backup does."""
    acked = []
    for op_id in sorted(cache.ongoing_backup):
        if only is not None and op_id not in only:
            continue
        entry = cache.ongoing_backup.pop(op_id)
        cache.dec_host_lock_ref(entry.node_id, entry.lock_params)
        cache.pending_backup_tokens -= entry.tokens
        operation = SimpleNamespace(
            id=op_id,
            hash_value=cache.cache_controller.writes[op_id],
            completed_tokens=entry.tokens,
        )
        cache._commit_storage_write(entry, operation, stored=True)
        acked.append(entry.node_id)
    return acked


def zeroed_pages(store) -> list:
    return [
        key
        for call in store.touch_pages.call_args_list
        if call.kwargs["newest_ns"] is None
        for key in call.args[0]
    ]


class ReviewRetireAndHeal(CustomTestCase):
    def cache(self, *, write_through=False, **switches):
        cache = stored_marks_cache(retire_ended_pages=True, **switches)
        if write_through:
            # insert_turn: every new node is copied to host memory inside its insert.
            cache.write_through_threshold = 1
        store = Mock(spec=HiCacheStorage)
        cache.cache_controller.storage_backend = store
        return cache, store

    def stored(self, cache, node_id):
        return cache.tree_core.node_by_id(node_id).storage_stored

    def test_f1_the_last_replys_write_does_not_bring_the_retired_chain_back(self):
        """Retire on, skip-ended-writes off. The commit hook retires the session while
        the last reply's host copy is still in flight (cache_finished_req: insert,
        then on_turn_committed). The ack of that copy then submits the reply's storage
        write, whose heal must not rewrite what was just retired."""
        cache, store = self.cache(write_through=True)
        tree, writes = cache.tree_core, cache.cache_controller.writes
        pages = tree.get_hash_values
        first = insert_turn(cache, [1, 2])
        ack_host_copies(cache)
        second = insert_turn(cache, [1, 2, 3])
        ack_host_copies(cache)
        ack_storage_writes(cache)
        self.assertTrue(self.stored(cache, first) and self.stored(cache, second))

        tail = insert_turn(cache, [1, 2, 3, 4])
        cache.retire_storage_chain(tail, keep_tokens=2, own_leaves=frozenset())
        self.assertEqual(zeroed_pages(store), pages(second))
        self.assertFalse(self.stored(cache, second))
        self.assertEqual(cache._retire_at_ack, {tail})

        written_before = len(writes)
        ack_host_copies(cache)
        queued = writes[written_before:]
        ack_storage_writes(cache)
        # Expected: only the tail's own write is queued (and retired at its ack), and
        # the retired node stays unstored.
        self.assertEqual(
            (queued, self.stored(cache, second)), ([pages(tail)], False)
        )

    def test_f2_a_fragment_retired_with_its_queued_write_is_not_retired_again(self):
        """A node split while its storage write is queued, then retired: both
        fragments are parked for the ack, which names only the lower one. The upper
        fragment's id must not stay parked: a later session that builds on it rewrites
        it (the heal), and that write must leave it stored."""
        cache, store = self.cache()
        tree, writes = cache.tree_core, cache.cache_controller.writes
        pages = tree.get_hash_values
        stored_node(cache, [1, 2])
        lower = computed_node(cache, [1, 2, 3, 4, 5, 6])
        cache.write_backup_storage(lower)
        cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3, 4]))))
        upper = tree.node_by_id(lower).parent.id
        cache.retire_storage_chain(lower, keep_tokens=2, own_leaves=frozenset())
        self.assertEqual(cache._retire_at_ack, {lower, upper})
        ack_storage_writes(cache)
        self.assertEqual(sorted(zeroed_pages(store)), sorted(pages(upper) + pages(lower)))
        left_parked = set(cache._retire_at_ack)

        store.touch_pages.reset_mock()
        other = computed_node(cache, [1, 2, 3, 4, 9])
        cache.write_backup_storage(other)
        self.assertEqual(writes[-2:], [pages(upper), pages(other)])
        ack_storage_writes(cache)
        # Expected: nothing is left parked once the op both fragments shared is acked,
        # and the rewrite a live session asked for leaves the fragment stored.
        self.assertEqual(
            (left_parked, zeroed_pages(store), self.stored(cache, upper)),
            (set(), [], True),
        )

    def test_f3_a_session_that_resumes_while_a_retired_write_is_queued_is_healed(self):
        """An explicit end, then the session sends a turn after all while the write of
        its last node is still queued. The turn's pages are written below nodes that
        are zeroed at their ack and never written again: nothing below a stored node
        triggers the heal."""
        cache, store = self.cache()
        tree, writes = cache.tree_core, cache.cache_controller.writes
        pages = tree.get_hash_values
        stored_node(cache, [1, 2])
        middle = stored_node(cache, [1, 2, 3])
        last = computed_node(cache, [1, 2, 3, 4])
        cache.write_backup_storage(last)
        cache.retire_storage_chain(last, keep_tokens=2, own_leaves=frozenset())
        self.assertEqual(cache._retire_at_ack, {last})

        resumed = computed_node(cache, [1, 2, 3, 4, 5])
        cache.write_backup_storage(resumed)
        ack_storage_writes(cache)
        later = computed_node(cache, [1, 2, 3, 4, 5, 6])
        cache.write_backup_storage(later)
        ack_storage_writes(cache)
        # Expected: the live session's chain has no unstored node above a stored one.
        chain = [middle, last, resumed, later]
        self.assertEqual(
            [self.stored(cache, node_id) for node_id in chain], [True] * 4,
            msg=f"writes after the end: {writes}",
        )

    def test_f4_pages_read_back_at_mtime_zero_are_refreshed_or_not_stored(self):
        """Retire on, refresh off. A session that returns (or a fork) after its
        retired pages left host memory reads them back from the SSD: they are stored
        at birth, so no heal rewrites them, and with no refresh their files stay at
        mtime zero, first in the cleaner's order."""
        cache, store = self.cache()
        tree = cache.tree_core
        first = stored_node(cache, [1, 2])
        back = read_back_node(cache, first, [3, 4])
        child = computed_node(cache, [1, 2, 3, 4, 5])
        request = SimpleNamespace(session=None)
        cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3, 4, 5])), req=request)
        )
        cache.write_backup_storage(child)
        rewritten = tree.get_hash_values(back) in cache.cache_controller.writes
        refreshed = any(
            call.kwargs["newest_ns"] is not None
            for call in store.touch_pages.call_args_list
        )
        self.assertTrue(rewritten or refreshed)

    def test_f5_the_ack_retires_only_what_the_walk_took(self):
        """A node is split by another session's match while its write is queued, and
        that session builds on the upper fragment. The retire walk stops at that
        branch and parks only the lower fragment; the ack must not zero the pages of
        the fragment the other session needs, nor drop its stored mark."""
        cache, store = self.cache()
        tree = cache.tree_core
        pages = tree.get_hash_values
        stored_node(cache, [1, 2])
        lower = computed_node(cache, [1, 2, 3, 4, 5, 6])
        cache.write_backup_storage(lower)
        other = computed_node(cache, [1, 2, 3, 4, 9])
        upper = tree.node_by_id(lower).parent.id
        self.assertEqual(tree.node_by_id(other).parent.id, upper)
        cache.retire_storage_chain(lower, keep_tokens=2, own_leaves=frozenset())
        self.assertEqual(cache._retire_at_ack, {lower})
        ack_storage_writes(cache)
        self.assertEqual(
            (sorted(zeroed_pages(store)), self.stored(cache, upper)),
            (sorted(pages(lower)), True),
        )

    def test_f1b_with_the_write_skip_on_the_last_turn_brings_nothing_back(self):
        """Control for f1 (expected to PASS): with skip-ended-writes on and the end
        known at the insert, the last turn's write returns before the heal."""
        cache, store = self.cache(write_through=True, skip_ended_writes=True)
        tree, writes = cache.tree_core, cache.cache_controller.writes
        pages = tree.get_hash_values
        insert_turn(cache, [1, 2])
        ack_host_copies(cache)
        second = insert_turn(cache, [1, 2, 3])
        ack_host_copies(cache)
        ack_storage_writes(cache)
        tail = insert_turn(cache, [1, 2, 3, 4], ends=True)
        cache.retire_storage_chain(tail, keep_tokens=2, own_leaves=frozenset())
        written_before = len(writes)
        ack_host_copies(cache)
        self.assertEqual(
            (writes[written_before:], self.stored(cache, second), cache._retire_at_ack),
            ([], False, set()),
        )
        self.assertEqual(zeroed_pages(store), pages(second))
