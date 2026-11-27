"""Review check (threads lens): retire without the write skip, on the real CPU tree.

Throwaway: copied into the copy's test folder under this name, run, deleted.
"""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from test_next_use_eviction import stored_marks_cache, write_through_turn

from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.hicache_storage import HiCacheStorage
from sglang.srt.mem_cache.radix_cache import RadixKey


def ack_storage_writes(cache, *, stored=True):
    """What _drain_backup does for each queued write of the recording controller."""
    controller = cache.cache_controller
    acked = []
    for op_id in list(cache.ongoing_backup):
        entry = cache.ongoing_backup.pop(op_id)
        cache.dec_host_lock_ref(entry.node_id, entry.lock_params)
        cache.pending_backup_tokens -= entry.tokens
        operation = SimpleNamespace(
            id=op_id,
            hash_value=controller.writes[op_id],
            completed_tokens=entry.tokens,
        )
        cache._commit_storage_write(entry, operation, stored=stored)
        acked.append(entry.node_id)
    return acked


def insert_without_ack(cache, prompt, *, ends=False):
    key = RadixKey(array("q", prompt))
    held = cache.match_prefix(MatchPrefixParams(key=key)).device_indices
    fresh = cache.token_to_kv_pool_allocator.alloc(len(prompt) - len(held))
    return cache.insert(
        InsertParams(key=key, value=torch.cat([held, fresh]), skip_storage_backup=ends)
    ).last_device_node


class ReviewRetireHeal(unittest.TestCase):
    def run_session(self, *, skip_ended_writes):
        cache = stored_marks_cache(
            retire_ended_pages=True, skip_ended_writes=skip_ended_writes
        )
        cache.write_through_threshold = 1
        store = Mock(spec=HiCacheStorage)
        cache.cache_controller.storage_backend = store
        tree, controller = cache.tree_core, cache.cache_controller
        pages = tree.get_hash_values

        first = write_through_turn(cache, [1, 2])
        reply_1 = write_through_turn(cache, [1, 2, 3])
        prompt_2 = write_through_turn(cache, [1, 2, 3, 4])
        ack_storage_writes(cache)
        stored = lambda node_id: tree.node_by_id(node_id).storage_stored
        self.assertTrue(all(stored(n) for n in (first, reply_1, prompt_2)))

        # The last turn's reply: its host copy is in flight when the turn commits.
        tail = insert_without_ack(cache, [1, 2, 3, 4, 5], ends=skip_ended_writes)
        cache.retire_storage_chain(tail, keep_tokens=2, own_leaves=set())
        (retire,) = store.touch_pages.call_args_list
        self.assertEqual(retire.kwargs, {"newest_ns": None})
        self.assertEqual(
            sorted(retire.args[0]), sorted(pages(reply_1) + pages(prompt_2))
        )
        self.assertFalse(stored(reply_1) or stored(prompt_2))

        writes_before = len(controller.writes)
        for ack_id in list(cache.ongoing_write_through):
            cache._finish_write_through_ack(ack_id)
        resubmitted = controller.writes[writes_before:]
        acked = ack_storage_writes(cache)
        return SimpleNamespace(
            cache=cache,
            store=store,
            pages=pages,
            first=first,
            reply_1=reply_1,
            prompt_2=prompt_2,
            tail=tail,
            resubmitted=resubmitted,
            acked=acked,
            stored=stored,
        )

    def test_with_the_write_skip_the_retired_chain_stays_retired(self):
        run = self.run_session(skip_ended_writes=True)
        self.assertEqual(run.resubmitted, [])
        self.assertFalse(run.stored(run.reply_1) or run.stored(run.prompt_2))

    def test_the_planned_arm_skips_both_inserts_of_the_last_turn_and_requeues_nothing(
        self,
    ):
        """Reply keys from the hint: the end is known at arrival, so the last turn's
        prompt and reply are both flagged."""
        cache = stored_marks_cache(retire_ended_pages=True, skip_ended_writes=True)
        cache.write_through_threshold = 1
        store = Mock(spec=HiCacheStorage)
        cache.cache_controller.storage_backend = store
        tree, controller = cache.tree_core, cache.cache_controller
        pages = tree.get_hash_values
        first = write_through_turn(cache, [1, 2])
        reply_1 = write_through_turn(cache, [1, 2, 3])
        ack_storage_writes(cache)
        prompt_2 = write_through_turn(cache, [1, 2, 3, 4], ends=True)
        tail = insert_without_ack(cache, [1, 2, 3, 4, 5], ends=True)
        cache.retire_storage_chain(tail, keep_tokens=2, own_leaves=set())
        writes_before = len(controller.writes)
        for ack_id in list(cache.ongoing_write_through):
            cache._finish_write_through_ack(ack_id)
        self.assertEqual(controller.writes[writes_before:], [])
        self.assertEqual((cache._storage_skip_nodes, cache._retire_at_ack), (set(), set()))
        (retire,) = store.touch_pages.call_args_list
        self.assertEqual(retire.args[0], pages(reply_1))
        stored = [tree.node_by_id(n).storage_stored for n in (first, reply_1, prompt_2, tail)]
        self.assertEqual(stored, [True, False, False, False])
        self.assertEqual((cache.pending_backup_tokens, cache.ongoing_backup), (0, {}))

    def test_without_the_write_skip_the_tails_ack_rewrites_the_retired_chain(self):
        run = self.run_session(skip_ended_writes=False)
        print("\nresubmitted after the retire:", run.resubmitted)
        print(
            "stored after the acks: reply_1=%s prompt_2=%s tail=%s"
            % (run.stored(run.reply_1), run.stored(run.prompt_2), run.stored(run.tail))
        )
        print("touch calls:", run.store.touch_pages.call_args_list)
        # What a retire that held would give: only the tail is written, and it is
        # zeroed at its ack; the chain stays unstored.
        self.assertEqual(run.resubmitted, [run.pages(run.tail)])
        self.assertFalse(run.stored(run.reply_1) or run.stored(run.prompt_2))


if __name__ == "__main__":
    unittest.main()
