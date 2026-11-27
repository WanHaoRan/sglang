"""Reviewer probe, adapted by the fixer to the fixed signatures only (own_leaf_keys, operation=); the
assertions are the reviewer's. Run: copy into the copy's test folder under a zz_ name, then the recipe."""
"""Reviewer's throwaway check (lens: repository rules and tests). Not part of the change.

A session's own reply that was evicted from memory and read back from the SSD by a staged
read comes back under a new node id (insert_host -> _new_node), which the prefetcher's
SessionExtent.leaves never saw. Run with -s to see the REVIEW line.
"""

import unittest

from test_next_use_eviction import (
    computed_node,
    read_back_node,
    stored_marks_cache,
    stored_node,
)


class ReviewReread(unittest.TestCase):
    def test_retire_stops_at_the_sessions_own_reply_read_back_under_a_new_id(self):
        def build():
            cache = stored_marks_cache(retire_ended_pages=True)
            first = stored_node(cache, [1, 2])  # the first prompt: two pages
            reply_1 = stored_node(cache, [1, 2, 50])  # turn 1's reply, id known
            # Turn 2 added prompt page [3] and reply page [60]; both left memory. The
            # plan for turn 3 reads the committed key's tail back as one host node.
            read_back = read_back_node(cache, first, [3, 60])
            # Turn 3's prompt repeats [3] and not the reply: its match splits the node.
            third = computed_node(cache, [1, 2, 3, 4])
            tail = computed_node(cache, [1, 2, 3, 4, 70])
            return cache, first, reply_1, read_back, third, tail

        cache, first, reply_1, read_back, third, tail = build()
        tree = cache.tree_core
        upper = tree.node_by_id(third).parent
        lower = tree.node_by_id(read_back)
        self.assertIs(lower.parent, upper)
        self.assertEqual((len(upper.key), len(lower.key)), (1, 1))
        self.assertTrue(upper.storage_stored and lower.storage_stored)

        # What the prefetcher recorded: turn 1's reply, and turn 2's reply under the id
        # it had before it was evicted.
        known = tree.take_retired_storage_pages(
            tail, keep_tokens=2, own_leaf_keys={reply_1, 10**9}
        )
        cache2, first2, reply_12, read_back2, third2, tail2 = build()
        tree2 = cache2.tree_core
        ideal = tree2.take_retired_storage_pages(
            tail2, keep_tokens=2, own_leaf_keys={reply_12, read_back2}
        )
        print(
            "REVIEW re-read reply: pages retired with the ids the prefetcher has = %d, "
            "with the re-read leaf counted as own = %d"
            % (len(known.page_keys), len(ideal.page_keys)),
            flush=True,
        )
        self.assertEqual(known.page_keys, [])
        self.assertEqual(len(ideal.page_keys), 3)


if __name__ == "__main__":
    unittest.main()
