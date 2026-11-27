"""Throwaway review checks (logic lens). Not part of the engine's tests."""

import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.mem_cache.next_turn_meter import PriceMeter
from sglang.srt.mem_cache.next_use_eviction import ReturnChanceStrategy
from sglang.srt.mem_cache.unified_cache.components import ComponentType, FullComponent
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeNode

FULL = ComponentType.FULL


def build(with_meter):
    """root -> H -> {X1, Y1}; root -> D. X1 and Y1 are tails of two ended sessions, H the
    prefix they share, D an un-hinted leaf. Host eviction as FullComponent drives it; the
    stub evict mimics UnifiedTreeCore._evict_host_leaf (detach from the parent, the parent
    becomes an evictable host leaf once childless)."""
    strategy = ReturnChanceStrategy(horizon_s=30.0, clock=lambda: 1000.0)

    def make(name, tokens, last_access, parent=None):
        made = UnifiedTreeNode((FULL,))
        made.name = name
        made.component_data[FULL].host_value = torch.zeros(tokens, dtype=torch.int64)
        made.last_access_time = last_access
        made.parent = parent
        if parent is not None:
            parent.children[name] = made
        return made

    root = make("root", 0, 0.0)
    head = make("H", 8, 3.0, root)
    x1 = make("X1", 4, 1.0, head)
    y1 = make("Y1", 4, 2.0, head)
    d = make("D", 4, 2.5, root)
    strategy.registry.register(x1.id, None)
    strategy.registry.register(y1.id, None)
    evicted = []
    core = SimpleNamespace(
        evictable_host_leaves={x1, y1, d},
        host_eviction_observer=None,
        eviction_strategy=strategy,
    )

    def evict(victim, tracker, device_frees, host_frees):
        data = victim.component_data[FULL]
        tracker[FULL] += len(data.host_value)
        data.host_value = None
        core.evictable_host_leaves.discard(victim)
        parent = victim.parent
        del parent.children[victim.name]
        evicted.append(victim.name)
        if parent is not root and not parent.children:
            core.evictable_host_leaves.add(parent)

    core._evict_host_leaf = evict
    component = FullComponent.__new__(FullComponent)
    component.component_type = FULL
    component.tree_core = core
    component.session_ref_eviction_strategy = None
    if with_meter:
        # What SGLANG_LOG_HICACHE_PRICES_INTERVAL alone installs: the meter binds itself.
        PriceMeter(
            tree_cache=SimpleNamespace(tree_core=core, read_scheduler=None),
            page_size=64,
            ready_after_s=0.5,
            hold_margin_s=2.0,
            log_interval_s=5.0,
            read_deadline=lambda handle: None,
        )
        assert core.host_eviction_observer is not None
    return component, evicted


class ReviewObserverSideEffect(CustomTestCase):
    def test_eviction_order_with_and_without_the_prices_log(self):
        orders = {}
        for with_meter in (False, True):
            component, evicted = build(with_meter)
            component.drive_host_eviction(12, {FULL: 0}, {}, {})
            orders[with_meter] = evicted
        print("\nREVIEW observer: victims without the meter:", orders[False])
        print("REVIEW observer: victims with the meter   :", orders[True])
        self.assertEqual(orders[False], orders[True])


if __name__ == "__main__":
    unittest.main()
