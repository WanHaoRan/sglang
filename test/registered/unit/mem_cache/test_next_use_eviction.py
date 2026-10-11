"""Next-use eviction: the tier order, the registry contract, the factory wiring
and the eviction order of a CPU tree (agent_cache/ORACLE_V2_PLAN.md 4.5)."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import time
import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.next_use_eviction import (
    TIER_FAR,
    TIER_OVERDUE,
    TIER_SOON,
    NextUseRegistry,
    NextUseStrategy,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.mem_cache.utils import get_eviction_strategy
from sglang.test.test_utils import CustomTestCase


def node(node_id, last_access=0.0):
    return SimpleNamespace(id=node_id, last_access_time=last_access)


class TestNextUseStrategy(CustomTestCase):
    def test_tiers_and_the_order_within_each(self):
        """Ended sessions go first (LRU among them), then sessions whose turn is
        overdue by more than the grace (oldest first), then sessions returning
        beyond the horizon farthest first, then unregistered nodes by LRU, and
        last the sessions returning within the horizon, farthest first. An
        overdue entry that stayed in the last tier would outrank every live
        session for as long as it existed."""
        strategy = NextUseStrategy(
            horizon_s=100.0, overdue_grace_s=60.0, clock=lambda: 1000.0
        )
        registry = strategy.registry
        registry.register(1, None)
        registry.register(9, None)
        registry.register(2, 1500.0)
        registry.register(3, 1200.0)
        registry.register(5, 1050.0)
        registry.register(6, 1010.0)
        registry.register(8, 100.0)
        registry.register(10, 900.0)
        registry.register(11, 950.0)
        nodes = [
            node(6, 1.0),
            node(5, 2.0),
            node(4, 0.5),
            node(3, 3.0),
            node(2, 4.0),
            node(1, 5.0),
            node(9, 0.1),
            node(7, 9.0),
            node(8, 8.0),
            node(10, 8.0),
            node(11, 8.0),
        ]
        order = [n.id for n in sorted(nodes, key=strategy.get_priority)]
        # 11 is 50 s overdue, within the 60 s grace: still a soon return.
        self.assertEqual(order, [9, 1, 8, 10, 2, 3, 4, 7, 5, 6, 11])
        self.assertEqual(strategy.get_priority(node(8))[0], TIER_OVERDUE)

    def test_registry_is_bounded_oldest_registration_first(self):
        registry = NextUseRegistry(max_entries=3)
        for node_id in (1, 2, 3):
            registry.register(node_id, None)
        registry.register(1, 5.0)  # re-registering moves 1 to the newest end
        registry.register(4, None)
        self.assertEqual(
            [registry.lookup(n)[0] for n in (1, 2, 3, 4)], [True, False, True, True]
        )

    def test_horizon_boundary_is_inclusive_on_the_soon_side(self):
        strategy = NextUseStrategy(horizon_s=100.0, clock=lambda: 1000.0)
        strategy.registry.register(1, 1100.0)
        strategy.registry.register(2, 1100.001)
        self.assertEqual(strategy.get_priority(node(1))[0], TIER_SOON)
        self.assertEqual(strategy.get_priority(node(2))[0], TIER_FAR)

    def test_registry_tells_unknown_from_ended(self):
        """An unregistered node is plain LRU; a registered one without a next
        turn is ended. A lookup that conflated the two would evict un-hinted
        traffic first."""
        registry = NextUseRegistry()
        self.assertEqual(registry.lookup(1), (False, None))
        registry.register(1, None)
        self.assertEqual(registry.lookup(1), (True, None))
        registry.forget(1)
        registry.forget(1)
        self.assertEqual((registry.lookup(1), len(registry)), ((False, None), 0))

    def test_factory_builds_the_policy_from_its_config(self):
        strategy = get_eviction_strategy(
            "next_use", {"horizon_s": 60, "overdue_grace_s": 10}
        )
        self.assertIsInstance(strategy, NextUseStrategy)
        self.assertEqual((strategy.horizon_s, strategy.overdue_grace_s), (60.0, 10.0))
        with self.assertRaises(ValueError):
            get_eviction_strategy("next_use", {"horizon_s": -1})


def next_use_params(horizon_s: float) -> CacheInitParams:
    dtype = torch.float16
    kv_pool = MHATokenToKVPool(
        size=64,
        page_size=1,
        dtype=dtype,
        head_num=2,
        head_dim=8,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )
    allocator = TokenToKVPoolAllocator(
        size=64, dtype=dtype, device="cpu", kvcache=kv_pool, need_sort=False
    )
    req_pool = ReqToTokenPool(
        size=8, max_context_len=128, device="cpu", enable_memory_saver=False
    )
    return CacheInitParams(
        disable=False,
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=1,
        eviction_policy="next_use",
        eviction_policy_config={"horizon_s": horizon_s},
        tree_components=(ComponentType.FULL,),
    )


class TestNextUseOnACpuTree(CustomTestCase):
    def test_device_eviction_follows_the_registered_next_use(self):
        """Four two-token sessions; the prefetcher registered three of their tail
        nodes. Eviction frees the ended one, then the far one, then the un-hinted
        one, and keeps the one returning soonest."""
        cache = UnifiedRadixCache(next_use_params(horizon_s=100.0))
        strategy = cache.tree_core.eviction_strategy
        self.assertIsInstance(strategy, NextUseStrategy)
        keys = {
            "soon": RadixKey(array("q", [1, 2])),
            "unhinted": RadixKey(array("q", [3, 4])),
            "far": RadixKey(array("q", [5, 6])),
            "ended": RadixKey(array("q", [7, 8])),
        }
        tails = {}
        for name, key in keys.items():
            value = cache.token_to_kv_pool_allocator.alloc(2)
            tails[name] = cache.insert(InsertParams(key=key, value=value)).last_device_node
        now = time.perf_counter()
        strategy.registry.register(tails["ended"], None)
        strategy.registry.register(tails["far"], now + 1500.0)
        strategy.registry.register(tails["soon"], now + 10.0)

        def resident():
            return {
                name
                for name, key in keys.items()
                if len(cache.match_prefix(MatchPrefixParams(key=key)).device_indices) == 2
            }

        evicted = []
        alive = resident()
        for _ in range(3):
            cache.evict(EvictParams(num_tokens=2))
            still = resident()
            evicted.append((alive - still).pop())
            alive = still
        self.assertEqual(evicted, ["ended", "far", "unhinted"])
        self.assertEqual(alive, {"soon"})


if __name__ == "__main__":
    unittest.main()
