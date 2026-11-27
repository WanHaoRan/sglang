"""Next-use eviction: the tier order, the registry contract, the factory wiring
and the eviction order of a CPU tree (agent_cache/ORACLE_V2_PLAN.md 4.5). On the
same CPU tree: which nodes count as stored on the SSD
(agent_cache/results/design_unified_1126/DESIGN.md, section 4)."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import time
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import HiCacheStorage
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
    TIER_QUEUED,
    TIER_SOON,
    TIER_UNKNOWN,
    NextUseRegistry,
    NextUseStrategy,
    Queued,
    ReturnChanceStrategy,
    ReturnForecast,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage_retention import RetentionSwitches
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

    def test_a_turn_queued_past_the_grace_outlasts_every_session_not_yet_back(self):
        """A turn that waited in the queue longer than the grace ranked as overdue (its
        session not coming) and lost its tail while it waited. An arrived turn is
        needed before any turn still to come, however long it has waited."""
        strategy = NextUseStrategy(
            horizon_s=100.0, overdue_grace_s=60.0, clock=lambda: 1000.0
        )
        registry = strategy.registry
        registry.register(1, None)
        registry.register(2, 100.0)
        registry.register(3, 1500.0)
        registry.register(5, 1010.0)
        registry.register(6, Queued(since=880.0))
        registry.register(7, Queued(since=990.0))
        nodes = [node(7, 2.0), node(6, 1.0), node(5), node(4), node(3), node(2), node(1)]
        order = [n.id for n in sorted(nodes, key=strategy.get_priority)]
        self.assertEqual(order, [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(strategy.get_priority(node(6))[0], TIER_QUEUED)

    def test_a_queued_registration_does_not_outlive_its_turn_for_ever(self):
        """A turn aborted before it commits leaves its tail registered as queued;
        past the bound it ranks as un-hinted, as under return_chance."""
        clock = [1000.0]
        strategy = NextUseStrategy(clock=lambda: clock[0])
        strategy.registry.register(1, Queued(since=1000.0))
        self.assertEqual(strategy.get_priority(node(1))[0], TIER_QUEUED)
        clock[0] += QUEUED_MAX_AGE_S + 1.0
        self.assertEqual(strategy.get_priority(node(1, 3.0)), (TIER_UNKNOWN, 3.0))

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
        that has arrived (registered as Queued) goes last."""
        now = 1000.0
        strategy = ReturnChanceStrategy(horizon_s=30.0, clock=lambda: now)
        wide = GapDistribution.from_quantiles(p10=0.85, p50=22.0, p90=58.0)
        registry = strategy.registry
        registry.register(1, None)
        registry.register(2, ReturnForecast(dist=wide, idle_from=now - 300.0))
        registry.register(3, ReturnForecast(dist=wide, idle_from=now - 5.0))
        registry.register(4, Queued(since=now))
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
        strategy.registry.register(3, Queued(since=now))
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
        registry.register(9, Queued(since=now))
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
        strategy.registry.register(8, Queued(since=now))
        strategy.adopt(3, 8)
        self.assertEqual(strategy.get_priority(span)[0], RETURN_TIER_QUEUED)

    def test_a_queued_registration_does_not_outlive_its_turn_for_ever(self):
        """A turn aborted before it commits leaves its session registered as queued;
        past the bound its path ranks as un-hinted instead of last."""
        clock = [1000.0]
        strategy = ReturnChanceStrategy(clock=lambda: clock[0])
        strategy.registry.register(1, Queued(since=1000.0))
        self.assertEqual(strategy.get_priority(node(1))[0], RETURN_TIER_QUEUED)
        clock[0] += QUEUED_MAX_AGE_S + 1.0
        self.assertEqual(strategy.get_priority(node(1))[0], RETURN_TIER_UNKNOWN)

    def test_farthest_order_evicts_the_last_arrival_among_queued_turns(self):
        """Admission is FCFS, so of two queued turns the one that arrived later is
        admitted later and its history has the farther next use. LRU ranks them the
        other way round once the earlier turn's path is the older stamp: it takes the
        history of the turn about to be admitted."""
        early, late = Queued(since=900.0), Queued(since=990.0)
        for name, config in (("return_chance", {}), ("next_use", {})):
            with self.subTest(policy=name):
                lru = get_eviction_strategy(name, config or None)
                far = get_eviction_strategy(name, {**config, "queued_order": "farthest"})
                for strategy in (lru, far):
                    strategy._clock = lambda: 1000.0
                    strategy.registry.register(1, early)
                    strategy.registry.register(2, late)
                # The earlier turn was matched last, as a turn the admission loop reached.
                nodes = [node(1, last_access=995.0), node(2, last_access=990.0)]
                order = lambda s: [n.id for n in sorted(nodes, key=s.get_priority)]
                self.assertEqual(order(lru), [2, 1])
                self.assertEqual(order(far), [2, 1])
                # Stamps the other way round: only the arrival order may decide.
                nodes = [node(1, last_access=900.0), node(2, last_access=990.0)]
                self.assertEqual(order(lru), [1, 2])
                self.assertEqual(order(far), [2, 1])
                with self.assertRaises(ValueError):
                    get_eviction_strategy(name, {"queued_order": "nearest"})

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
        strategy.registry.register(4, Queued(since=now))
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
        strategy.registry.register(4, Queued(since=now))
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
        strategy.registry.register(tail_a, Queued(since=time.perf_counter()))
        cache.evict(EvictParams(num_tokens=2))
        self.assertEqual((device_tokens(whole), device_tokens(other)), (2, 0))


CACHE_LOGGER = "sglang.srt.mem_cache.unified_radix_cache"


class _StorageWrites:
    """A controller that copies to host memory at once, unless told to refuse, and
    records the pages of each storage write in submit order."""

    backup_skip = False

    def __init__(self):
        self.writes = []
        self.refuse_host_copies = False
        self.mem_pool_host = SimpleNamespace(
            anchor_entry=SimpleNamespace(
                host_pool=SimpleNamespace(shared_allocation_domain=None),
                host_evict_fn=None,
            ),
            available_size=lambda: 10**6,
        )

    def write(self, device_value, node_id, extra_pools=None, flush=False):
        return None if self.refuse_host_copies else torch.arange(len(device_value))

    def write_storage(
        self, host_indices, token_ids, hash_value, prefix_keys, extra_pools=None
    ):
        self.writes.append(list(hash_value))
        return len(self.writes) - 1


def stored_marks_cache(**switches) -> UnifiedRadixCache:
    """A CPU cache (one token a page) with storage on, the stored marks tracked, and a
    controller that records the writes queued."""
    cache = UnifiedRadixCache(next_use_params(horizon_s=100.0))
    cache.cache_controller = _StorageWrites()
    # With the host tier on, a match ends at the deepest node memory holds.
    cache.tree_core.set_hicache_enabled()
    cache.enable_storage = True
    cache.retention = RetentionSwitches(track_stored=True, **switches)
    return cache


def read_back_node(cache, parent_id, tokens) -> int:
    """A host-only node under parent_id, as a landed storage read inserts it."""
    return cache.tree_core.insert_host(
        parent_id,
        RadixKey(array("q", tokens)),
        torch.arange(len(tokens)),
        # Page keys are hex digests: a child's key chains from its parent's last one.
        [f"{token:064x}" for token in tokens],
    ).inserted_host_node


def computed_node(cache, prompt) -> int:
    """Compute a prompt on the device and give its new node a host copy, as an acked
    write-through leaves it; returns that node's id."""
    key = RadixKey(array("q", prompt))
    held = cache.match_prefix(MatchPrefixParams(key=key)).device_indices
    fresh = cache.token_to_kv_pool_allocator.alloc(len(prompt) - len(held))
    node_id = cache.insert(
        InsertParams(key=key, value=torch.cat([held, fresh]))
    ).last_device_node
    pages = len(cache.tree_core.get_hash_values(node_id))
    cache.tree_core.commit_backup(node_id, torch.arange(pages), {})
    return node_id


def stored_node(cache, prompt) -> int:
    """computed_node whose storage write was acked in full."""
    node_id = computed_node(cache, prompt)
    cache.tree_core.mark_storage_write(node_id, node_id)
    cache.tree_core.finish_storage_write(node_id, node_id, stored_at=0.0)
    return node_id


def write_through_turn(cache, prompt, *, ends=False) -> int:
    """Insert a prompt as a turn does under write-through: each new node is copied to
    host memory inside the insert. Then ack those copies, where storage writes are
    queued; returns the last node's id."""
    key = RadixKey(array("q", prompt))
    held = cache.match_prefix(MatchPrefixParams(key=key)).device_indices
    fresh = cache.token_to_kv_pool_allocator.alloc(len(prompt) - len(held))
    leaf = cache.insert(
        InsertParams(key=key, value=torch.cat([held, fresh]), skip_storage_backup=ends)
    ).last_device_node
    for ack_id in list(cache.ongoing_write_through):
        cache._finish_write_through_ack(ack_id)
    return leaf


class TestStoredMarksOnACpuTree(CustomTestCase):
    def marks(self, cache, node_id):
        node = cache.tree_core.node_by_id(node_id)
        return node.storage_write_op, node.storage_stored

    def test_a_split_while_the_write_is_queued_marks_both_fragments_at_the_ack(self):
        """A match that ends inside a node splits it while its storage write is queued,
        and the new fragment holds some of the op's pages under a new id. Unmarked at
        the ack, it would count as unwritten and be written again under every child."""
        cache = stored_marks_cache()
        tree = cache.tree_core
        parent = computed_node(cache, [1, 2])
        child = computed_node(cache, [1, 2, 3, 4, 5, 6])
        tree.mark_storage_write(parent, 6)
        tree.mark_storage_write(child, 7)
        cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3, 4]))))
        fragment = tree.node_by_id(child).parent.id
        self.assertNotEqual(fragment, parent)
        tree.finish_storage_write(child, 7, stored_at=100.0)
        self.assertEqual(
            [self.marks(cache, node_id) for node_id in (child, fragment, parent)],
            [(None, True), (None, True), (6, False)],
        )

    def test_a_child_write_first_rewrites_an_unstored_ancestor(self):
        """A lookup stops at the first page missing on the SSD, so a child written under
        an ancestor whose write failed cannot be reached there. Its write queues that
        ancestor first, once. Pages read back from the SSD are stored at birth and not
        written again, and an ancestor whose host copy is still being filled is left to
        its own ack."""
        cache = stored_marks_cache()
        tree, writes = cache.tree_core, cache.cache_controller.writes
        pages = tree.get_hash_values
        read_back_node(cache, cache.root_node_handle(), [1, 2])
        failed = computed_node(cache, [1, 2, 3])
        cache.write_backup_storage(failed)
        self.assertEqual(writes, [pages(failed)])
        tree.finish_storage_write(failed, 0, stored_at=None)

        child = computed_node(cache, [1, 2, 3, 4])
        cache.write_backup_storage(child)
        self.assertEqual(writes[1:], [pages(failed), pages(child)])
        sibling = computed_node(cache, [1, 2, 3, 5])
        cache.write_backup_storage(sibling)
        self.assertEqual(writes[3:], [pages(sibling)])

        copying = computed_node(cache, [7])
        tree.mark_write_through_pending([copying], ack_id=copying)
        below = computed_node(cache, [7, 8])
        cache.write_backup_storage(below)
        self.assertEqual(writes[4:], [pages(below)])

    def test_clearing_the_store_clears_every_mark(self):
        """An emptied store holds no page. A mark left set, on a node or on the fragment
        a split copied it to, would keep those pages from ever being written again."""
        cache = stored_marks_cache()
        cache._storage_attachment = SimpleNamespace(clear=lambda: True)
        node = read_back_node(cache, cache.root_node_handle(), [1, 2, 3, 4])
        cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", [1, 2]))))
        fragment = cache.tree_core.node_by_id(node).parent.id
        self.assertEqual(
            (self.marks(cache, node), self.marks(cache, fragment)),
            ((None, True), (None, True)),
        )
        self.assertTrue(cache.clear_storage_backend())
        self.assertEqual(
            (self.marks(cache, node), self.marks(cache, fragment)),
            ((None, False), (None, False)),
        )

    def test_a_stale_tail_takes_its_fresh_head_along_and_a_second_match_takes_nothing(
        self,
    ):
        """The cleaner deletes the oldest file first and a lookup stops at the first
        missing page, so a tail refreshed under a head left alone would outlive it: one
        stale node refreshes every stored node of the served chain, head first. A page
        just written is as recent as one just touched; stale from birth, it would have
        every turn touch its session's whole chain."""
        cache = stored_marks_cache()
        tree = cache.tree_core
        pages = tree.get_hash_values
        head = computed_node(cache, [1, 2])
        tail = computed_node(cache, [1, 2, 3])
        fork = computed_node(cache, [1, 2, 9])
        for op, (node_id, acked_at) in enumerate(
            ((head, 1000.0), (tail, 1250.0), (fork, 1100.0))
        ):
            tree.mark_storage_write(node_id, op)
            tree.finish_storage_write(node_id, op, stored_at=acked_at)

        def stale(node_id, now):
            return tree.take_stale_storage_keys(node_id, now=now, min_interval_s=300.0)

        self.assertEqual(stale(tail, 1299.0), [])
        self.assertEqual(stale(tail, 1300.0), pages(head) + pages(tail))
        self.assertEqual(stale(tail, 1310.0), [])
        # The head was refreshed at 1300; the fork under it was written at 1100.
        self.assertEqual(stale(fork, 1450.0), pages(head) + pages(fork))
        self.assertEqual(stale(tail, 1599.0), [])
        self.assertEqual(stale(tail, 1600.0), pages(head) + pages(tail))

    def test_only_stored_nodes_are_refreshed(self):
        """A node whose write is queued or failed has no file to keep recent; pages
        read back have files of unknown age and go at the first match."""
        cache = stored_marks_cache()
        tree = cache.tree_core
        pages = tree.get_hash_values
        read_back = read_back_node(cache, cache.root_node_handle(), [1, 2])
        queued = computed_node(cache, [1, 2, 3])
        tree.mark_storage_write(queued, 0)
        failed = computed_node(cache, [1, 2, 3, 4])
        tree.mark_storage_write(failed, 1)
        tree.finish_storage_write(failed, 1, stored_at=None)
        written = computed_node(cache, [1, 2, 3, 4, 5])
        tree.mark_storage_write(written, 2)
        tree.finish_storage_write(written, 2, stored_at=50.0)
        self.assertEqual(
            tree.take_stale_storage_keys(written, now=60.0, min_interval_s=300.0),
            pages(read_back) + pages(written),
        )

    def test_a_probe_without_a_request_refreshes_nothing(self):
        """Only a request's match is memory serving a chain. The prefetcher probes the
        tree for every plan it stages: counted as serves, its probes would keep the
        files of sessions that never return as recent as those of live ones."""
        cache = stored_marks_cache(touch_interval_s=300.0)
        store = Mock(spec=HiCacheStorage)
        cache.cache_controller.storage_backend = store
        node = read_back_node(cache, cache.root_node_handle(), [1, 2])
        key = RadixKey(array("q", [1, 2]))
        request = SimpleNamespace(session=None)
        cache.match_prefix(MatchPrefixParams(key=key))
        cache.enable_storage = False
        cache.match_prefix(MatchPrefixParams(key=key, req=request))
        store.touch_pages.assert_not_called()

        cache.enable_storage = True
        before_ns = time.time_ns()
        for _ in range(2):
            cache.match_prefix(MatchPrefixParams(key=key, req=request))
        (touch,) = store.touch_pages.call_args_list
        self.assertEqual(touch.args, (cache.tree_core.get_hash_values(node),))
        self.assertGreaterEqual(touch.kwargs["newest_ns"], before_ns)

    def test_a_turn_after_a_skipped_one_writes_the_skipped_node_first(self):
        """The last turn of a session that says it ended is copied to host memory and
        not written to the SSD: no write, no host lock, no pending tokens. Should a
        turn extend it after all, that turn's pages would sit below a hole, so the
        skipped node is written first. A node a refused host copy left above the ended
        turn is backed up by it and skipped with it."""
        cache = stored_marks_cache(skip_ended_writes=True)
        cache.write_through_threshold = 1
        tree, controller = cache.tree_core, cache.cache_controller
        pages = tree.get_hash_values
        first = write_through_turn(cache, [1, 2])
        self.assertEqual(controller.writes, [pages(first)])
        with self.assertLogs(CACHE_LOGGER, level="INFO") as logs:
            last = write_through_turn(cache, [1, 2, 3, 4], ends=True)
        self.assertEqual(controller.writes, [pages(first)])
        self.assertEqual(
            (cache.pending_backup_tokens, len(cache.ongoing_backup)), (2, 1)
        )
        last_node = tree.node_by_id(last)
        self.assertTrue(last_node.backuped)
        self.assertEqual(last_node.component_data[ComponentType.FULL].host_lock_ref, 0)
        self.assertEqual(sum("h2s_skip" in line for line in logs.output), 1)

        resumed = write_through_turn(cache, [1, 2, 3, 4, 5])
        self.assertEqual(controller.writes[1:], [pages(last), pages(resumed)])

        controller.refuse_host_copies = True
        unbacked = write_through_turn(cache, [7, 8])
        controller.refuse_host_copies = False
        self.assertFalse(tree.is_backuped(unbacked))
        written = len(controller.writes)
        write_through_turn(cache, [7, 8, 9], ends=True)
        self.assertTrue(tree.is_backuped(unbacked))
        self.assertEqual(len(controller.writes), written)
        # A skip is spent at the node's ack: nothing is left to grow with the run.
        self.assertEqual(cache._storage_skip_nodes, set())

    def session_tree(self, cache):
        """A three-turn session of one-page prompts: a two-page first prompt, and after
        each prompt a reply the next prompt did not repeat."""
        first = stored_node(cache, [1, 2])
        reply_1 = stored_node(cache, [1, 2, 50])
        second = stored_node(cache, [1, 2, 3])
        reply_2 = stored_node(cache, [1, 2, 3, 60])
        third = computed_node(cache, [1, 2, 3, 4])
        tail = computed_node(cache, [1, 2, 3, 4, 70])
        return SimpleNamespace(
            first=first,
            reply_1=reply_1,
            second=second,
            reply_2=reply_2,
            third=third,
            tail=tail,
        )

    def test_retire_takes_the_replies_no_later_prompt_repeated(self):
        """An ended session's own pages are what its turns added past its first prompt:
        the later prompts and the replies left hanging beside them. Their files get
        mtime zero and they stop counting as stored, so that no refresh brings them
        back. A page whose write is queued, or whose host copy is still in flight, is
        left to the ack of its write, unless that write is one the ended turn skips."""
        for last_turn_skipped in (False, True):
            with self.subTest(last_turn_skipped=last_turn_skipped):
                cache = stored_marks_cache(retire_ended_pages=True)
                store = Mock(spec=HiCacheStorage)
                cache.cache_controller.storage_backend = store
                tree = cache.tree_core
                pages = tree.get_hash_values
                made = self.session_tree(cache)
                tree.mark_storage_write(made.third, 90)
                tree.mark_write_through_pending([made.tail], ack_id=made.tail)
                at_ack = {made.third, made.tail}
                if last_turn_skipped:
                    cache._storage_skip_nodes.add(made.tail)
                    at_ack.remove(made.tail)
                with self.assertLogs(CACHE_LOGGER, level="INFO") as logs:
                    cache.retire_storage_chain(
                        made.tail,
                        keep_tokens=2,
                        own_leaves={made.reply_1, made.reply_2},
                    )
                (touch,) = store.touch_pages.call_args_list
                self.assertEqual(touch.kwargs, {"newest_ns": None})
                own_stored = (made.reply_2, made.second, made.reply_1)
                self.assertEqual(
                    sorted(touch.args[0]),
                    sorted(key for node_id in own_stored for key in pages(node_id)),
                )
                self.assertEqual(cache._retire_at_ack, at_ack)
                stored = [
                    name
                    for name, node_id in vars(made).items()
                    if self.marks(cache, node_id)[1]
                ]
                self.assertEqual(stored, ["first"])
                self.assertIn(
                    f"l3_retire node={made.tail} pages=3 at_ack={len(at_ack)} "
                    "keep_tokens=2",
                    logs.output[-1],
                )

    def test_retire_keeps_the_first_prompt_and_stops_at_another_sessions_branch(self):
        """Later sessions reuse an ended session's first prompt, and now and then a
        later turn of it: a leaf the session did not make, or a reply something was
        built on, marks a branch another session needs. Nothing from that node up is
        retired."""
        for name, fork in (
            ("another session's leaf", [1, 2, 3, 9]),
            ("a reply built on", [1, 2, 3, 60, 61]),
        ):
            with self.subTest(name):
                cache = stored_marks_cache(retire_ended_pages=True)
                tree = cache.tree_core
                made = self.session_tree(cache)
                tree.mark_storage_write(made.third, 90)
                tree.finish_storage_write(made.third, 90, stored_at=0.0)
                stored_node(cache, fork)
                retired = tree.take_retired_storage_pages(
                    made.tail, keep_tokens=2, own_leaves={made.reply_1, made.reply_2}
                )
                self.assertEqual(
                    (retired.page_keys, retired.pending_node_ids),
                    (tree.get_hash_values(made.third), []),
                )
                kept = (made.first, made.reply_1, made.second, made.reply_2)
                self.assertTrue(all(self.marks(cache, node_id)[1] for node_id in kept))


if __name__ == "__main__":
    unittest.main()
