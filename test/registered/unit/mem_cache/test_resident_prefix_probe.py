"""Read-only residency probe and SSD read backlog on UnifiedRadixCache."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from array import array
from types import SimpleNamespace

import torch
from test_session_unified_radix_cache import insert, make_params

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.test_utils import CustomTestCase


def key(token_ids):
    return RadixKey(array("q", token_ids))


def access_times(cache):
    """last_access_time by node id: a split adds an id, a stamp changes a value."""
    times, stack = {}, [cache.tree_core.root_node]
    while stack:
        node = stack.pop()
        times[node.id] = node.last_access_time
        stack.extend(node.children.values())
    return times


class TestResidentPrefixProbe(CustomTestCase):
    def setUp(self):
        # Page size 2: a probe that forgets page_size in child_key_at or
        # match_at still passes at page size 1, where the defaults agree.
        self.cache = UnifiedRadixCache(make_params(enable_session=False, page_size=2))
        self.cache.tree_core.set_hicache_enabled()
        # Stem [1..4] with three branches: [5..8] on device, [9..12] in host
        # memory only, [13..16] dead (on neither).
        insert(self.cache, [1, 2, 3, 4, 5, 6, 7, 8])
        host_only = insert(self.cache, [1, 2, 3, 4, 9, 10, 11, 12])
        dead = insert(self.cache, [1, 2, 3, 4, 13, 14, 15, 16])
        full = host_only.component_data[ComponentType.FULL]
        full.host_value = full.value.clone()
        full.value = None
        dead.component_data[ComponentType.FULL].value = None

    def test_probe_agrees_with_match_prefix(self):
        """l1 and l12 are the device and device + host prefixes match_prefix reports."""
        cases = [
            [1, 2, 3, 4, 5, 6, 7, 8],  # all on device: (8, 8)
            [1, 2, 3, 4, 9, 10, 11, 12],  # host-only tail: (4, 8)
            [1, 2, 3, 4, 13, 14, 15, 16],  # stops at the dead node: (4, 4)
            [1, 2, 3, 4, 5, 6, 7, 8, 21, 22],  # runs past the tree: (8, 8)
            [1, 2, 3, 4, 9, 10, 77, 78],  # ends inside the host-only node: (4, 6)
            [1, 2, 3, 4, 5, 6, 7],  # odd length, the half page is dropped: (6, 6)
        ]
        for tokens in cases:
            with self.subTest(tokens=tokens):
                # Probe first: match_prefix splits the node a key ends inside.
                l1, l12 = self.cache.resident_prefix_len(key(tokens))
                match = self.cache.match_prefix(MatchPrefixParams(key=key(tokens)))
                self.assertEqual(l1, len(match.device_indices))
                self.assertEqual(l12, len(match.device_indices) + match.host_hit_length)

    def test_probe_is_read_only(self):
        """No split and no LRU stamp, even for a key that ends inside a node."""
        before = access_times(self.cache)
        self.cache.resident_prefix_len(key([1, 2, 3, 4, 9, 10, 77, 78]))
        self.assertEqual(access_times(self.cache), before)

    def test_read_backlog_counts_allocated_reads_in_pages(self):
        """Reads holding host pages count, in pages; a read in its lookup does not."""

        def read(pages, allocated):
            return SimpleNamespace(
                host_indices=torch.zeros(2 * pages) if allocated else None,
                operation=SimpleNamespace(hash_value=["h"] * pages),
            )

        self.cache.ongoing_prefetch = {
            "queued": read(3, True),
            "running": read(5, True),
            "lookup": read(7, False),
        }
        self.assertEqual(self.cache.storage_read_backlog_pages(), 8)

    def test_disabled_cache_probes_nothing(self):
        params = make_params(enable_session=False, page_size=2)
        params.disable = True
        cache = UnifiedRadixCache(params)
        self.assertEqual(cache.resident_prefix_len(key([1, 2, 3, 4])), (0, 0))


if __name__ == "__main__":
    unittest.main()
