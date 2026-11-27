"""Review check (default-off lens), throwaway: does SGLANG_LOG_HICACHE_PRICES_INTERVAL alone
change a decision?

Host eviction of the real FullComponent under the real ReturnChanceStrategy, on a tree core
stub that removes a victim from its parent as UnifiedTreeCore._evict_host_leaf does (a
childless parent that still holds host KV becomes a host leaf). The only thing varied is the
prices log interval of a NextTurnPrefetcher built on that cache.

Tree: root -> p (shared prefix) -> {x, y}; root -> w; root -> u. x, y and w are tails of ended
sessions, u is un-hinted, p carries no registration. Eviction is asked for three leaves.
"""

from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.next_turn_prefetch import NextTurnPrefetcher
from sglang.srt.mem_cache.next_use_eviction import (
    NextUseStrategy,
    ReturnChanceStrategy,
)
from sglang.srt.mem_cache.unified_cache.components import ComponentType, FullComponent
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeNode

FULL = ComponentType.FULL
LEAF_TOKENS = 8


def evict_three_leaves(*, prices_interval_s, strategy_of, mode="on"):
    root = UnifiedTreeNode((FULL,))

    def make(name, last_access, parent):
        node = UnifiedTreeNode((FULL,))
        node.component_data[FULL].host_value = torch.zeros(LEAF_TOKENS, dtype=torch.int64)
        node.last_access_time = last_access
        node.parent = parent
        node.name = name
        parent.children[name] = node
        return node

    p = make("p", 50.0, root)
    y = make("y", 1.0, p)
    x = make("x", 2.0, p)
    w = make("w", 60.0, root)
    u = make("u", 10.0, root)
    strategy = strategy_of()
    for tail in (x, y, w):
        strategy.registry.register(tail.id, None)  # the session ended
    evicted = []
    core = SimpleNamespace(
        eviction_strategy=strategy,
        evictable_host_leaves={x, y, w, u},
        host_eviction_observer=None,
    )

    def evict_host_leaf(victim, tracker, device_frees, host_frees):
        data = victim.component_data[FULL]
        tracker[FULL] += len(data.host_value)
        data.host_value = None
        core.evictable_host_leaves.discard(victim)
        parent = victim.parent
        del parent.children[victim.name]
        if (
            parent is not root
            and not parent.children
            and parent.component_data[FULL].host_value is not None
        ):
            core.evictable_host_leaves.add(parent)
        evicted.append(victim.name)

    core._evict_host_leaf = evict_host_leaf
    cache = SimpleNamespace(read_scheduler=None, tree_core=core)
    with envs.SGLANG_LOG_HICACHE_PRICES_INTERVAL.override(prices_interval_s):
        ntp = NextTurnPrefetcher(mode=mode, tree_cache=cache, page_size=64)
    assert (ntp._meter is not None) == (prices_interval_s > 0)
    assert (core.host_eviction_observer is not None) == (prices_interval_s > 0)
    # Every decision switch is off in both runs.
    assert not (
        ntp._measured_prices
        or ntp._free_empty_lookups
        or ntp._lateness_weight
        or ntp._read_line
        or ntp._hold_quantile
        or ntp._recompute_by_price
        or ntp._recompute_when_starved
    )
    component = FullComponent.__new__(FullComponent)
    component.component_type = FULL
    component.tree_core = core
    component.session_ref_eviction_strategy = None
    component.drive_host_eviction(3 * LEAF_TOKENS, {FULL: 0}, {}, {})
    return evicted


def test_the_prices_log_alone_does_not_change_host_eviction_under_return_chance():
    def strategy_of():
        return ReturnChanceStrategy(horizon_s=30.0, clock=lambda: 1000.0)

    off = evict_three_leaves(prices_interval_s=0.0, strategy_of=strategy_of)
    logging = evict_three_leaves(prices_interval_s=5.0, strategy_of=strategy_of)
    print("return_chance: log off evicts", off, "; log on evicts", logging)
    assert off == logging, (off, logging)


def test_the_prices_log_alone_does_not_change_host_eviction_under_next_use():
    def strategy_of():
        return NextUseStrategy(horizon_s=300.0, clock=lambda: 1000.0)

    off = evict_three_leaves(prices_interval_s=0.0, strategy_of=strategy_of)
    logging = evict_three_leaves(prices_interval_s=5.0, strategy_of=strategy_of)
    print("next_use: log off evicts", off, "; log on evicts", logging)
    assert off == logging, (off, logging)


def test_in_mode_observe_the_prices_log_alone_does_not_change_host_eviction():
    """Mode observe "logs those decisions ... and changes nothing" (module docstring)."""

    def strategy_of():
        return ReturnChanceStrategy(horizon_s=30.0, clock=lambda: 1000.0)

    off = evict_three_leaves(prices_interval_s=0.0, strategy_of=strategy_of, mode="observe")
    logging = evict_three_leaves(prices_interval_s=5.0, strategy_of=strategy_of, mode="observe")
    print("return_chance, observe: log off evicts", off, "; log on evicts", logging)
    assert off == logging, (off, logging)


def evict_on_the_real_tree(*, prices_interval_s):
    """The same tree on a real CPU UnifiedRadixCache (page size 2, four-token leaves): the
    real UnifiedTreeCore.drive_host_eviction, _evict_host_leaf and leaf sets."""
    from test_session_unified_radix_cache import insert, make_params

    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    params = make_params(enable_session=False, page_size=2)
    params.eviction_policy = "return_chance"
    cache = UnifiedRadixCache(params)
    core = cache.tree_core
    core.set_hicache_enabled()
    x = insert(cache, [1, 2, 3, 4, 5, 6, 7, 8])
    y = insert(cache, [1, 2, 3, 4, 9, 10, 11, 12])
    w = insert(cache, [21, 22, 23, 24])
    u = insert(cache, [31, 32, 33, 34])
    p = y.parent
    assert x.parent is p and p is not core.root_node and len(p.children) == 2
    names = {p.id: "p", x.id: "x", y.id: "y", w.id: "w", u.id: "u"}
    # Host-only, as after device eviction with write-through backups.
    for node, last_access in ((p, 50.0), (y, 1.0), (x, 2.0), (w, 60.0), (u, 10.0)):
        full = node.component_data[FULL]
        full.host_value = full.value.clone()
        full.value = None
        node.last_access_time = last_access
    for node in (x, y, w, u, p):
        core._update_evictable_leaf_sets(node)
    assert core.evictable_host_leaves == {x, y, w, u}, core.evictable_host_leaves
    strategy = core.eviction_strategy
    assert isinstance(strategy, ReturnChanceStrategy)
    for tail in (x, y, w):
        strategy.registry.register(tail.id, None)  # the session ended
    with envs.SGLANG_LOG_HICACHE_PRICES_INTERVAL.override(prices_interval_s):
        ntp = NextTurnPrefetcher(mode="on", tree_cache=cache, page_size=2)
    assert (core.host_eviction_observer is not None) == (prices_interval_s > 0)
    assert not ntp._measured_prices
    core.drive_host_eviction(FULL, 3 * 4)
    left, stack = set(), [core.root_node]
    while stack:
        node = stack.pop()
        if node is not core.root_node:
            left.add(names[node.id])
        stack.extend(node.children.values())
    return sorted(left)


def test_on_the_real_tree_the_prices_log_alone_does_not_change_what_host_eviction_keeps():
    off = evict_on_the_real_tree(prices_interval_s=0.0)
    logging = evict_on_the_real_tree(prices_interval_s=5.0)
    print("real tree, return_chance: log off keeps", off, "; log on keeps", logging)
    assert off == logging, (off, logging)


def read_only_registration(strategy, node):
    """ReturnChanceStrategy._session_registration without its writes to _tail_of."""
    registry = strategy.registry
    parent = node.parent
    only_child = parent is not None and len(parent.children) == 1
    tail_id = node.id
    known, forecast = registry.lookup(tail_id)
    if not known:
        tail_id = strategy._tail_of.get(node.id)
        if tail_id is None and only_child:
            tail_id = (
                parent.id
                if registry.lookup(parent.id)[0]
                else strategy._tail_of.get(parent.id)
            )
        if tail_id is None:
            return False, None
        known, forecast = registry.lookup(registry.current(tail_id))
        if not known:
            return False, None
    return True, forecast


def test_the_proposed_fix_a_read_only_holding_rate_restores_the_eviction_order():
    """Not a test of the copy: holding_rate is patched here to resolve the session
    without adopting, to check that this is what removes the difference."""
    from unittest import mock

    original = ReturnChanceStrategy._session_registration
    in_holding_rate = []

    def registration(self, node):
        if in_holding_rate:
            return read_only_registration(self, node)
        return original(self, node)

    original_rate = ReturnChanceStrategy.holding_rate

    def holding_rate(self, node, *, queued=None):
        in_holding_rate.append(True)
        try:
            return original_rate(self, node, queued=queued)
        finally:
            in_holding_rate.pop()

    with mock.patch.object(
        ReturnChanceStrategy, "_session_registration", registration
    ), mock.patch.object(ReturnChanceStrategy, "holding_rate", holding_rate):
        off = evict_on_the_real_tree(prices_interval_s=0.0)
        logging = evict_on_the_real_tree(prices_interval_s=5.0)
    print("real tree, read-only holding_rate: log off keeps", off, "; log on keeps", logging)
    assert off == logging == ["p", "u"], (off, logging)
