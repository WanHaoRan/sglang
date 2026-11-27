"""Review checks (threads / default-off lens) on the refresh walk, real CPU tree.

Throwaway: copied into the copy's test folder under this name, run, deleted.
"""

import time
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock

from test_next_use_eviction import (
    computed_node,
    read_back_node,
    stored_marks_cache,
)

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.hicache_storage import HiCacheStorage
from sglang.srt.mem_cache.radix_cache import RadixKey


class ReviewRefresh(unittest.TestCase):
    def test_a_read_back_tail_refreshes_a_head_touched_seconds_ago(self):
        cache = stored_marks_cache()
        tree = cache.tree_core
        pages = tree.get_hash_values
        head = computed_node(cache, [1, 2])
        tree.mark_storage_write(head, 0)
        tree.finish_storage_write(head, 0, stored_at=1000.0)

        def stale(node_id, now):
            return tree.take_stale_storage_keys(node_id, now=now, min_interval_s=300.0)

        # The head alone is fresh for 300 s.
        self.assertEqual(stale(head, 1010.0), [])
        touched = []
        head_pages = pages(head)
        tail = head
        for turn, now in enumerate((1010.0, 1020.0, 1030.0)):
            # Each returning turn reads a new part of its history back from the SSD.
            tail = read_back_node(cache, tail, [3 + turn])
            head_then_tail = stale(tail, now)
            touched.append(len(head_then_tail))
            self.assertEqual(head_then_tail[: len(head_pages)], head_pages)
        print("\npages touched by three turns 10 s apart (interval 300 s):", touched)
        # With one refresh per page per interval the head's two pages would be touched
        # at most once in these 30 s; they are touched at every turn.
        self.assertEqual(touched, [3, 4, 5])

    def test_cost_of_the_walk_on_the_scheduler_thread(self):
        cache = stored_marks_cache(touch_interval_s=300.0)
        store = Mock(spec=HiCacheStorage)
        cache.cache_controller.storage_backend = store
        tree = cache.tree_core
        parent = cache.root_node_handle()
        depth = 60
        for token in range(1, depth + 1):
            parent = read_back_node(cache, parent, [token])
        now = time.monotonic()
        # Stamp them all, then time walks that find nothing stale.
        tree.take_stale_storage_keys(parent, now=now, min_interval_s=300.0)
        rounds = 2000
        started = time.perf_counter()
        for _ in range(rounds):
            tree.take_stale_storage_keys(parent, now=now + 1.0, min_interval_s=300.0)
        per_walk_us = (time.perf_counter() - started) / rounds * 1e6
        key = RadixKey(array("q", range(1, depth + 1)))
        request = SimpleNamespace(session=None)
        started = time.perf_counter()
        for _ in range(200):
            cache.match_prefix(MatchPrefixParams(key=key, req=request))
        with_hook_us = (time.perf_counter() - started) / 200 * 1e6
        started = time.perf_counter()
        for _ in range(200):
            cache.match_prefix(MatchPrefixParams(key=key))
        without_hook_us = (time.perf_counter() - started) / 200 * 1e6
        print(
            "\n%d-node chain: fresh walk %.1f us; match_prefix %.1f us with the hook, "
            "%.1f us without" % (depth, per_walk_us, with_hook_us, without_hook_us)
        )


if __name__ == "__main__":
    unittest.main()
