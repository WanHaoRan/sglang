"""Reviewer's throwaway check (style and tests lens): does SGLANG_LOG_HICACHE_PRICES_INTERVAL
alone change which host leaf eviction takes?

environ.py says of the switch "Logging only: no decision reads it". With it on, host eviction
calls the meter for every victim just before it goes, and under return_chance eviction the
meter's holding_rate() runs ReturnChanceStrategy._session_registration, which adopts nodes
into sessions as it ranks them. This runs the copy's own FullComponent.drive_host_eviction
(only the tree core is a stub, as in TestHostEvictionObserver) with a real
ReturnChanceStrategy, once with no observer and once with the meter a prefetcher installs
when only the prices log is on, and prints the eviction order of each.

Tree: root -> P -> {A, B}, root -> U. A and B are the tails of two ended sessions; P (their
shared body) and U carry no registration; U was used before P. Twelve host tokens are asked
for, four a node.
"""

import contextlib
import unittest
from types import SimpleNamespace

import test_next_turn_prefetch as T
import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.next_turn_prefetch import NextTurnPrefetcher
from sglang.srt.mem_cache.next_use_eviction import ReturnChanceStrategy
from sglang.srt.mem_cache.unified_cache.components import ComponentType, FullComponent
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeNode

FULL = ComponentType.FULL


def build(observer_of):
    """The tree, a strategy with the two ended tails registered, and a FullComponent on a
    recording tree core. observer_of(strategy) gives the host eviction observer, or None."""
    strategy = ReturnChanceStrategy(horizon_s=30.0, clock=lambda: 1000.0)

    def made(name, parent, last_access):
        node = UnifiedTreeNode((FULL,))
        node.name = name
        node.parent = parent
        node.last_access_time = last_access
        node.component_data[FULL].host_value = torch.zeros(4, dtype=torch.int64)
        if parent is not None:
            parent.children[node.id] = node
        return node

    root = made("root", None, 0.0)
    body = made("P", root, 50.0)
    unhinted = made("U", root, 10.0)
    tail_a = made("A", body, 1.0)
    tail_b = made("B", body, 2.0)
    strategy.registry.register(tail_a.id, None)
    strategy.registry.register(tail_b.id, None)
    evicted = []

    def evict(victim, tracker, device_frees, host_frees):
        data = victim.component_data[FULL]
        tracker[FULL] += len(data.host_value)
        data.host_value = None
        core.evictable_host_leaves.discard(victim)
        del victim.parent.children[victim.id]
        evicted.append(victim.name)
        parent = victim.parent
        # As the tree core does: a host-backed parent left childless is a host leaf.
        if parent is not root and not parent.children:
            core.evictable_host_leaves.add(parent)

    core = SimpleNamespace(
        evictable_host_leaves={tail_a, tail_b, unhinted},
        host_eviction_observer=observer_of(strategy),
        _evict_host_leaf=evict,
    )
    component = FullComponent.__new__(FullComponent)
    component.component_type = FULL
    component.tree_core = core
    # What _ensure_eviction_strategy binds when no session-ref strategy is set.
    component.session_ref_eviction_strategy = strategy.get_priority
    return component, evicted, strategy, body


class LogOnlyEviction(T.PrefetcherTestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)

    def meter_observer(self, strategy):
        """The observer a prefetcher installs with only the prices log on (mode observe)."""
        self.stack.enter_context(envs.SGLANG_LOG_HICACHE_PRICES_INTERVAL.override(5.0))
        cache = T.MeasuredCache()
        cache.tree_core = SimpleNamespace(
            eviction_strategy=strategy, host_eviction_observer=None
        )
        ntp = NextTurnPrefetcher(mode="observe", tree_cache=cache, page_size=T.PAGE)
        self.assertFalse(ntp._measured_prices)
        return cache.tree_core.host_eviction_observer

    def run_eviction(self, observer_of):
        component, evicted, strategy, body = build(observer_of)
        component.drive_host_eviction(12, {FULL: 0}, {}, {})
        return evicted, strategy.get_priority(body)[0] if body.parent.children.get(body.id) else None

    def test_host_eviction_order_with_and_without_the_prices_log(self):
        off, tier_off = self.run_eviction(lambda strategy: None)
        on, tier_on = self.run_eviction(self.meter_observer)
        print("eviction order, switches off:      ", off, "| tier P ranks in afterwards:", tier_off)
        print("eviction order, prices log only on:", on, "| tier P ranks in afterwards:", tier_on)
        self.assertEqual(off, ["A", "B", "U"])
        # The claim under test: the log changes no decision.
        self.assertEqual(on, off)

    def run_two_passes(self, observer_of):
        """root -> P -> {A, B}, with B the tail of a session likely back (a forecast). The
        first pass frees A and B and stops; then an un-hinted leaf V, used after P, joins
        and a second pass frees one node."""
        from sglang.srt.mem_cache.next_turn_policy import GapDistribution
        from sglang.srt.mem_cache.next_use_eviction import ReturnForecast

        strategy = ReturnChanceStrategy(horizon_s=30.0, clock=lambda: 1000.0)

        def made(name, parent, last_access):
            node = UnifiedTreeNode((FULL,))
            node.name = name
            node.parent = parent
            node.last_access_time = last_access
            node.component_data[FULL].host_value = torch.zeros(4, dtype=torch.int64)
            if parent is not None:
                parent.children[node.id] = node
            return node

        root = made("root", None, 0.0)
        body = made("P", root, 50.0)
        tail_a = made("A", body, 1.0)
        tail_b = made("B", body, 2.0)
        strategy.registry.register(tail_a.id, None)
        entry = GapDistribution.from_quantiles(p10=0.85, p50=22.0, p90=58.0)
        strategy.registry.register(tail_b.id, ReturnForecast(dist=entry, idle_from=995.0))
        evicted = []

        def evict(victim, tracker, device_frees, host_frees):
            data = victim.component_data[FULL]
            tracker[FULL] += len(data.host_value)
            data.host_value = None
            core.evictable_host_leaves.discard(victim)
            del victim.parent.children[victim.id]
            evicted.append(victim.name)
            parent = victim.parent
            if parent is not root and not parent.children:
                core.evictable_host_leaves.add(parent)

        core = SimpleNamespace(
            evictable_host_leaves={tail_a, tail_b},
            host_eviction_observer=observer_of(strategy),
            _evict_host_leaf=evict,
        )
        component = FullComponent.__new__(FullComponent)
        component.component_type = FULL
        component.tree_core = core
        component.session_ref_eviction_strategy = strategy.get_priority
        component.drive_host_eviction(8, {FULL: 0}, {}, {})
        first = list(evicted)
        later = made("V", root, 60.0)
        core.evictable_host_leaves.add(later)
        component.drive_host_eviction(4, {FULL: 0}, {}, {})
        return first, evicted[len(first):]

    def test_the_difference_outlasts_the_pass(self):
        off = self.run_two_passes(lambda strategy: None)
        on = self.run_two_passes(self.meter_observer)
        print("two passes, switches off:      ", off)
        print("two passes, prices log only on:", on)
        self.assertEqual(on, off)


if __name__ == "__main__":
    unittest.main()
