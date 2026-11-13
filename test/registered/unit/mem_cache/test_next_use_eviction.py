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
from sglang.srt.mem_cache.next_turn_policy import GapDistribution
from sglang.srt.mem_cache.next_use_eviction import (
    POINT_GRACE_S,
    QUEUED_MAX_AGE_S,
    RETURN_TIER_ENDED,
    RETURN_TIER_QUEUED,
    RETURN_TIER_SCORED,
    RETURN_TIER_UNKNOWN,
    TIER_FAR,
    TIER_OVERDUE,
    TIER_SOON,
    NextUseRegistry,
    NextUseStrategy,
    ReturnChanceStrategy,
    ReturnForecast,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.mem_cache.utils import get_eviction_strategy
from sglang.test.test_utils import CustomTestCase


def node(node_id, last_access=0.0, parent=None):
    made = SimpleNamespace(
        id=node_id, last_access_time=last_access, parent=parent, children={}
    )
    if parent is not None:
        parent.children[node_id] = made
    return made


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


class TestReturnChanceStrategy(CustomTestCase):
    def test_order_is_ended_unhinted_rising_chance_then_queued(self):
        """A session scored from a distribution is never ended by it: one idle for
        ten times its median only ranks below one still inside its entry. A turn
        that has arrived (registered with a time) goes last."""
        now = 1000.0
        strategy = ReturnChanceStrategy(horizon_s=30.0, clock=lambda: now)
        wide = GapDistribution.from_quantiles(p10=0.85, p50=22.0, p90=58.0)
        registry = strategy.registry
        registry.register(1, None)
        registry.register(2, ReturnForecast(dist=wide, idle_from=now - 300.0))
        registry.register(3, ReturnForecast(dist=wide, idle_from=now - 5.0))
        registry.register(4, now)
        nodes = [node(4), node(3), node(2), node(1), node(5, 2.0), node(6, 1.0)]
        order = [n.id for n in sorted(nodes, key=strategy.get_priority)]
        self.assertEqual(order, [1, 6, 5, 2, 3, 4])
        tier, chance, _ = strategy.get_priority(node(2))
        self.assertEqual(tier, RETURN_TIER_SCORED)
        self.assertGreater(chance, 0.0)

    def test_exact_hints_give_farthest_next_use_with_a_grace_for_late_turns(self):
        """A horizon alone only splits exact hints into due within it or not; the
        tie-break restores Belady's order. A turn a moment late is still due, and
        one long overdue is not coming."""
        now = 1000.0
        strategy = ReturnChanceStrategy(horizon_s=30.0, clock=lambda: now)
        for node_id, gap_s, idle_s in (
            (1, 500.0, 0.0),
            (2, 60.0, 0.0),
            (3, 10.0, 0.0),
            (4, 10.0, 10.0 + POINT_GRACE_S / 2),
            (5, 10.0, 10.0 + 2 * POINT_GRACE_S),
        ):
            strategy.registry.register(
                node_id,
                ReturnForecast(dist=GapDistribution.point(gap_s), idle_from=now - idle_s),
            )
        order = sorted((node(i) for i in range(1, 6)), key=strategy.get_priority)
        self.assertEqual([n.id for n in order], [5, 1, 2, 3, 4])

    def test_a_registration_covers_the_session_not_one_node(self):
        """The prefetcher registers a session's tail. Its body, once the tail is
        ranked and evicted, and a span read back under it are other nodes, and must
        rank as the session does: a queued turn's body goes last, and a branch point
        that several sessions share takes no single session's registration."""
        now = 1000.0
        strategy = ReturnChanceStrategy(horizon_s=30.0, clock=lambda: now)
        entry = GapDistribution.from_quantiles(p10=0.85, p50=22.0, p90=58.0)
        shared = node(1)
        body = node(2, parent=shared)
        tail = node(3, parent=body)
        other = node(4, parent=shared)
        strategy.registry.register(3, ReturnForecast(dist=entry, idle_from=now - 5.0))
        scored = strategy.get_priority(tail)
        self.assertEqual(scored[0], RETURN_TIER_SCORED)
        del body.children[3]
        self.assertEqual(strategy.get_priority(body), scored)
        reread = node(5, parent=body)
        self.assertEqual(strategy.get_priority(reread), scored)
        self.assertEqual(strategy.get_priority(shared)[0], RETURN_TIER_UNKNOWN)
        self.assertEqual(strategy.get_priority(other)[0], RETURN_TIER_UNKNOWN)
        strategy.registry.register(3, now)
        for member in (body, reread):
            self.assertEqual(strategy.get_priority(member)[0], RETURN_TIER_QUEUED)
        strategy.registry.forget(3)
        self.assertEqual(strategy.get_priority(body)[0], RETURN_TIER_UNKNOWN)

    def test_a_sessions_nodes_follow_its_tail_from_turn_to_turn(self):
        """Each commit registers a new tail. A node mapped to the old one must find
        the new one, also when the old tail was split and its parent now has two
        children: that parent ranked as un-hinted while its turn was queued."""
        now = 1000.0
        strategy = ReturnChanceStrategy(horizon_s=30.0, clock=lambda: now)
        entry = GapDistribution.from_quantiles(p10=0.85, p50=22.0, p90=58.0)
        registry = strategy.registry
        body = node(2)
        old_tail = node(3, parent=body)
        registry.register(3, ReturnForecast(dist=entry, idle_from=now - 5.0))
        strategy.get_priority(old_tail)
        node(4, parent=body)
        registry.move(3, 7)
        registry.register(7, ReturnForecast(dist=entry, idle_from=now - 1.0))
        registry.move(7, 9)
        registry.register(9, now)
        self.assertEqual(strategy.get_priority(body)[0], RETURN_TIER_QUEUED)

    def test_a_span_read_back_under_a_shared_prompt_is_adopted_at_landing(self):
        """Its parent has many children, so the tree alone cannot say whose it is; a
        prompt that once had one child must not keep that session's registration."""
        now = 1000.0
        strategy = ReturnChanceStrategy(horizon_s=30.0, clock=lambda: now)
        entry = GapDistribution.from_quantiles(p10=0.85, p50=22.0, p90=58.0)
        prompt = node(1)
        first = node(2, parent=prompt)
        strategy.registry.register(2, None)
        strategy.get_priority(first)
        self.assertEqual(strategy.get_priority(prompt)[0], RETURN_TIER_ENDED)
        span = node(3, parent=prompt)
        self.assertEqual(strategy.get_priority(span)[0], RETURN_TIER_UNKNOWN)
        self.assertEqual(strategy.get_priority(prompt)[0], RETURN_TIER_UNKNOWN)
        strategy.registry.register(8, now)
        strategy.adopt(3, 8)
        self.assertEqual(strategy.get_priority(span)[0], RETURN_TIER_QUEUED)

    def test_a_queued_registration_does_not_outlive_its_turn_for_ever(self):
        """A turn aborted before it commits leaves its session registered as queued;
        past the bound its path ranks as un-hinted instead of last."""
        clock = [1000.0]
        strategy = ReturnChanceStrategy(clock=lambda: clock[0])
        strategy.registry.register(1, 1000.0)
        self.assertEqual(strategy.get_priority(node(1))[0], RETURN_TIER_QUEUED)
        clock[0] += QUEUED_MAX_AGE_S + 1.0
        self.assertEqual(strategy.get_priority(node(1))[0], RETURN_TIER_UNKNOWN)

    def test_factory_builds_the_policy_from_its_config(self):
        strategy = get_eviction_strategy("return_chance", {"horizon_s": 60})
        self.assertIsInstance(strategy, ReturnChanceStrategy)
        self.assertEqual((strategy.horizon_s, strategy.score), (60.0, "entry"))
        for bad in ({"horizon_s": 0}, {"score": "rate"}):
            with self.assertRaises(ValueError):
                get_eviction_strategy("return_chance", bad)

    def test_samples_score_from_the_keys_gaps_then_all_gaps_then_the_entry(self):
        """The entry says a session idle for 60 s is nearly sure to be back within
        30 s; the gaps seen for its key say a quarter are. With too few of the
        key's gaps outlasting the idle time every key's gaps answer, and the entry
        only until enough gaps were seen at all."""
        now = 1000.0
        strategy = ReturnChanceStrategy(horizon_s=30.0, score="samples", clock=lambda: now)
        sharp = GapDistribution.from_quantiles(p10=6.9, p50=7.3, p90=13.0)
        seen = (5.0, 7.0, 8.0, 70.0, 100.0, 200.0, 400.0)
        cases = {
            1: ReturnForecast(dist=sharp, idle_from=now - 60.0, key_gaps=seen, all_gaps=seen),
            2: ReturnForecast(
                dist=sharp, idle_from=now - 150.0, key_gaps=seen, all_gaps=seen + (500.0, 600.0)
            ),
            3: ReturnForecast(dist=sharp, idle_from=now - 60.0, key_gaps=(5.0,), all_gaps=(5.0, 7.0)),
            4: ReturnForecast(dist=sharp, idle_from=now - 900.0, key_gaps=seen, all_gaps=seen),
        }
        for node_id, forecast in cases.items():
            strategy.registry.register(node_id, forecast)
        chance = {i: strategy.get_priority(node(i))[1] for i in cases}
        self.assertAlmostEqual(chance[1], 1 / 4)
        self.assertAlmostEqual(chance[2], 0.0)
        self.assertGreater(chance[3], 0.9)
        self.assertEqual(chance[4], 0.0)

    def test_lru_score_uses_only_the_end_signal_and_the_queue(self):
        """The rival that takes no forecast: a session with a forecast ranks by LRU
        among the un-hinted, after ended sessions and before queued turns."""
        now = 1000.0
        strategy = ReturnChanceStrategy(score="lru", clock=lambda: now)
        entry = GapDistribution.from_quantiles(p10=0.85, p50=22.0, p90=58.0)
        strategy.registry.register(1, None)
        strategy.registry.register(2, ReturnForecast(dist=entry, idle_from=now - 5.0))
        strategy.registry.register(3, ReturnForecast(dist=entry, idle_from=now - 900.0))
        strategy.registry.register(4, now)
        nodes = [node(4, 0.0), node(3, 9.0), node(2, 1.0), node(1, 5.0), node(5, 4.0)]
        order = [n.id for n in sorted(nodes, key=strategy.get_priority)]
        self.assertEqual(order, [1, 2, 5, 3, 4])

    def test_queue_score_ranks_an_ended_session_by_lru_and_keeps_queued_turns(self):
        """What an engine with no hints could do: the end signal moves nothing, and a
        turn that has arrived still goes last."""
        now = 1000.0
        strategy = ReturnChanceStrategy(score="queue", clock=lambda: now)
        entry = GapDistribution.from_quantiles(p10=0.85, p50=22.0, p90=58.0)
        strategy.registry.register(1, None)
        strategy.registry.register(2, ReturnForecast(dist=entry, idle_from=now - 5.0))
        strategy.registry.register(4, now)
        nodes = [node(4, 0.0), node(2, 1.0), node(1, 5.0), node(5, 4.0)]
        order = [n.id for n in sorted(nodes, key=strategy.get_priority)]
        self.assertEqual(order, [2, 5, 1, 4])


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

    def test_device_eviction_follows_the_chance_of_return(self):
        """The same tree under return_chance: the ended session, the un-hinted one,
        then the session long past its entry; the one inside its entry stays."""
        params = next_use_params(horizon_s=30.0)
        params.eviction_policy = "return_chance"
        cache = UnifiedRadixCache(params)
        strategy = cache.tree_core.eviction_strategy
        self.assertIsInstance(strategy, ReturnChanceStrategy)
        keys = {
            "inside": RadixKey(array("q", [1, 2])),
            "unhinted": RadixKey(array("q", [3, 4])),
            "past": RadixKey(array("q", [5, 6])),
            "ended": RadixKey(array("q", [7, 8])),
        }
        tails = {}
        for name, key in keys.items():
            value = cache.token_to_kv_pool_allocator.alloc(2)
            tails[name] = cache.insert(InsertParams(key=key, value=value)).last_device_node
        now = time.perf_counter()
        entry = GapDistribution.from_quantiles(p10=0.85, p50=22.0, p90=58.0)
        strategy.registry.register(tails["ended"], None)
        strategy.registry.register(
            tails["past"], ReturnForecast(dist=entry, idle_from=now - 600.0)
        )
        strategy.registry.register(
            tails["inside"], ReturnForecast(dist=entry, idle_from=now - 5.0)
        )

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
        self.assertEqual(evicted, ["ended", "unhinted", "past"])
        self.assertEqual(alive, {"inside"})

    def test_a_queued_turn_keeps_its_body_after_its_tail_was_evicted(self):
        """Session a is two nodes and long past its entry, so its tail goes first.
        Its turn then arrives: the body it still has must outlast session b, which
        is only likely to return. Scored per node, the body ranked as un-hinted and
        went before b."""
        params = next_use_params(horizon_s=30.0)
        params.eviction_policy = "return_chance"
        cache = UnifiedRadixCache(params)
        strategy = cache.tree_core.eviction_strategy
        alloc = cache.token_to_kv_pool_allocator.alloc
        body = RadixKey(array("q", [1, 2]))
        whole = RadixKey(array("q", [1, 2, 3, 4]))
        other = RadixKey(array("q", [5, 6]))
        cache.insert(InsertParams(key=body, value=alloc(2)))
        held = cache.match_prefix(MatchPrefixParams(key=body)).device_indices
        tail_a = cache.insert(
            InsertParams(key=whole, value=torch.cat([held, alloc(2)]))
        ).last_device_node
        tail_b = cache.insert(InsertParams(key=other, value=alloc(2))).last_device_node
        now = time.perf_counter()
        entry = GapDistribution.from_quantiles(p10=0.85, p50=22.0, p90=58.0)
        strategy.registry.register(tail_a, ReturnForecast(dist=entry, idle_from=now - 600.0))
        strategy.registry.register(tail_b, ReturnForecast(dist=entry, idle_from=now - 5.0))

        def device_tokens(key):
            return len(cache.match_prefix(MatchPrefixParams(key=key)).device_indices)

        cache.evict(EvictParams(num_tokens=2))
        self.assertEqual((device_tokens(whole), device_tokens(other)), (2, 2))
        strategy.registry.register(tail_a, time.perf_counter())
        cache.evict(EvictParams(num_tokens=2))
        self.assertEqual((device_tokens(whole), device_tokens(other)), (2, 0))


if __name__ == "__main__":
    unittest.main()
