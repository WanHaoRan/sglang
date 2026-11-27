"""Reviewer probes (correctness lens), not part of the change: each test asserts the
behaviour a reader of the plan would expect and is expected to FAIL where the copy
has the defect named in its docstring.

Run: copy into the copy's test folder as zz_review_correctness_2.py, then the recipe.
"""

from array import array
from types import SimpleNamespace
from unittest.mock import Mock

from test_next_turn_prefetch import (
    PAGE,
    DeadlineCache,
    hints,
    key_of,
    make_req,
)
from test_next_use_eviction import (
    computed_node,
    read_back_node,
    stored_marks_cache,
    stored_node,
)

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.hicache_storage import HiCacheStorage
from sglang.srt.mem_cache.next_turn_prefetch import NextTurnPrefetcher
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.test.test_utils import CustomTestCase


def ack(cache, op_id, *, full=True, stored=None):
    """Ack one queued storage write as _drain_backup does."""
    entry = cache.ongoing_backup.pop(op_id)
    cache.dec_host_lock_ref(entry.node_id, entry.lock_params)
    cache.pending_backup_tokens -= entry.tokens
    operation = SimpleNamespace(
        id=op_id,
        hash_value=cache.cache_controller.writes[op_id],
        completed_tokens=entry.tokens if full else 0,
    )
    cache._commit_storage_write(entry, operation, stored=full)


class ReviewMarks(CustomTestCase):
    def stored(self, cache, node_id):
        return cache.tree_core.node_by_id(node_id).storage_stored

    def test_g1_a_write_acked_after_the_store_was_cleared_is_not_stored(self):
        """clear_storage_backend() deletes every file but leaves the ids of the writes
        in flight on their nodes. A write that had finished before the clear and is
        acked after it marks its node stored, with no file behind the mark."""
        cache = stored_marks_cache()
        cache._storage_attachment = SimpleNamespace(clear=lambda: True)
        node = computed_node(cache, [1, 2])
        cache.write_backup_storage(node)
        (op_id,) = cache.ongoing_backup
        self.assertTrue(cache.clear_storage_backend())
        ack(cache, op_id)
        self.assertFalse(self.stored(cache, node))

    def test_g2_a_failed_write_above_a_stored_node_is_rewritten(self):
        """Two writes queued back to back, the first acked short, the second in full
        (the write thread goes on after a failed batch). Every later write hangs under
        the stored node, where the heal stops, so the hole above it stays for good."""
        cache = stored_marks_cache()
        tree, writes = cache.tree_core, cache.cache_controller.writes
        pages = tree.get_hash_values
        stored_node(cache, [1, 2])
        failed = computed_node(cache, [1, 2, 3])
        cache.write_backup_storage(failed)
        below = computed_node(cache, [1, 2, 3, 4])
        cache.write_backup_storage(below)
        first_op, second_op = sorted(cache.ongoing_backup)
        ack(cache, first_op, full=False)
        ack(cache, second_op)
        self.assertEqual(
            (self.stored(cache, failed), self.stored(cache, below)), (False, True)
        )
        later = computed_node(cache, [1, 2, 3, 4, 5])
        cache.write_backup_storage(later)
        self.assertIn(pages(failed), writes[2:])

    def test_g3_pages_read_back_under_a_retired_node_leave_it_healed(self):
        """A fork (or a session that returns) reads the evicted tail of a retired chain
        back from the SSD under retired nodes still in memory. The read-back node is
        stored, so its children's writes never rewrite the zeroed nodes above it."""
        cache = stored_marks_cache(retire_ended_pages=True)
        cache.cache_controller.storage_backend = Mock(spec=HiCacheStorage)
        tree, writes = cache.tree_core, cache.cache_controller.writes
        pages = tree.get_hash_values
        stored_node(cache, [1, 2])
        retired = stored_node(cache, [1, 2, 3])
        cache.retire_storage_chain(retired, keep_tokens=2, own_leaves=frozenset())
        self.assertFalse(self.stored(cache, retired))
        read_back_node(cache, retired, [4])
        child = computed_node(cache, [1, 2, 3, 4, 5])
        cache.write_backup_storage(child)
        self.assertIn(pages(retired), writes)


class ReviewEnds(CustomTestCase):
    def test_h1_a_turn_a_table_named_that_carries_no_hint_does_not_end_its_session(
        self,
    ):
        """A table (v2) session ends by an entry marked ended. A turn of such a session
        that arrives with no next-turn action at all (the client had no table for it)
        states nothing; read as an exact hint's end, its writes are skipped and the
        session's pages are zeroed while it is live."""
        cache = DeadlineCache()
        cache.retired = []
        cache.retire_storage_chain = lambda tail, *, keep_tokens, own_leaves: (
            cache.retired.append((tail, keep_tokens, set(own_leaves)))
        )
        ntp = NextTurnPrefetcher(
            mode="on",
            tree_cache=cache,
            page_size=PAGE,
            reply_key_source="hint",
            retire_ended=True,
        )

        def commit(req, *, tail, prompt_tokens):
            req.origin_input_ids = array("q", range(prompt_tokens))
            ntp.on_turn_committed(
                req=req,
                key=key_of(16 * PAGE),
                rotation_tail_declined=False,
                now=0.0,
                tail_node_id=tail,
            )

        def table(next_rid):
            payload = {
                "session": "s",
                "next_rid": next_rid,
                "gaps": [{"key": "tool", "p50": 5.0}, {"key": "_end", "ended": True}],
                "replay_reply_key": "tool",
            }
            return hints(next_rid, version="2", payload=payload)

        commit(make_req("s-t0", kv_hints=table("s-t1")), tail=7, prompt_tokens=2 * PAGE)
        commit(make_req("s-t1", kv_hints=table("s-t2")), tail=9, prompt_tokens=4 * PAGE)
        unhinted = make_req("s-t2")
        ends_at_insert = ntp.ends_session(unhinted, reply_known=False)
        commit(unhinted, tail=11, prompt_tokens=6 * PAGE)
        self.assertEqual((ends_at_insert, cache.retired), (False, []))
