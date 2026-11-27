"""Review check (default-off lens), throwaway.

1. On the real CPU UnifiedRadixCache, a prefetcher built with every new switch off leaves
   the cache with exactly the attributes the base module's prefetcher leaves it with.
2. Which switches act in mode observe (next_turn_switches.py says none does).
"""

import importlib.util
import sys

import test_next_turn_prefetch as fixtures

from sglang.srt.environ import envs
from sglang.srt.mem_cache import next_turn_prefetch as changed

BASE_MODULE = (
    "/sgl-workspace/sglang/agent_cache/frozen/impl_unified_base/python/sglang/srt/"
    "mem_cache/next_turn_prefetch.py"
)
NEW_SWITCHES = (
    "SGLANG_LOG_HICACHE_PRICES_INTERVAL",
    "SGLANG_HICACHE_NEXT_TURN_MEASURED_PRICES",
    "SGLANG_HICACHE_NEXT_TURN_FREE_EMPTY_LOOKUPS",
    "SGLANG_HICACHE_NEXT_TURN_LATENESS_WEIGHT",
    "SGLANG_HICACHE_NEXT_TURN_READ_LINE",
    "SGLANG_HICACHE_NEXT_TURN_HOLD_QUANTILE",
    "SGLANG_HICACHE_NEXT_TURN_RECOMPUTE_BY_PRICE",
    "SGLANG_HICACHE_NEXT_TURN_RECOMPUTE_WHEN_STARVED",
)


def load_base():
    spec = importlib.util.spec_from_file_location("review_ntp_base_hooks", BASE_MODULE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["review_ntp_base_hooks"] = module
    spec.loader.exec_module(module)
    return module


def describe(value):
    if callable(value):
        return ("fn", getattr(value, "__name__", type(value).__name__))
    return (type(value).__name__,)


def left_on_the_cache(module, **kwargs):
    """What building a prefetcher changes on a real CPU cache and its tree core."""
    cache = fixtures.cpu_unified_cache()
    before_cache = dict(vars(cache))
    before_core = dict(vars(cache.tree_core))
    module.NextTurnPrefetcher(mode="on", tree_cache=cache, page_size=2, **kwargs)

    def delta(before, after):
        return {
            name: describe(value)
            for name, value in after.items()
            if name not in before or before[name] is not value
        }

    return delta(before_cache, vars(cache)), delta(before_core, vars(cache.tree_core)), cache


def test_with_the_switches_off_the_real_cache_is_left_as_the_base_module_leaves_it():
    for name in NEW_SWITCHES:
        assert not getattr(envs, name).get(), name
    base = load_base()
    for kwargs in (
        {},
        {"arrival_gate": "on"},
        {"arrival_gate": "on", "arrival_hold": "on", "soft_pins": "on"},
        {"read_timing": "expected_cost", "price_learning": "harm"},
        {"read_timing": "just_in_time", "arrival_gate": "on"},
    ):
        base_cache, base_core, _ = left_on_the_cache(base, **kwargs)
        new_cache, new_core, cache = left_on_the_cache(changed, **kwargs)
        print(kwargs, "->", sorted(new_cache), sorted(new_core))
        assert base_cache == new_cache, (kwargs, base_cache, new_cache)
        assert base_core == new_core, (kwargs, base_core, new_core)
        assert cache.prefetch_gate_prompt_fn is None
        assert cache.prefetch_lookup_fn is None
        assert cache.read_lags is None
        assert cache.tree_core.host_eviction_observer is None


def test_which_switches_act_in_mode_observe():
    """next_turn_switches.py: "A switch with nothing to act on in a setup is off there, as
    every one is in mode observe"."""
    acting = {}
    for switch, kwargs, flag in (
        ("SGLANG_HICACHE_NEXT_TURN_MEASURED_PRICES", {}, "_measured_prices"),
        (
            "SGLANG_HICACHE_NEXT_TURN_FREE_EMPTY_LOOKUPS",
            {"arrival_gate": "on"},
            "_free_empty_lookups",
        ),
        (
            "SGLANG_HICACHE_NEXT_TURN_LATENESS_WEIGHT",
            {"read_timing": "expected_cost"},
            "_lateness_weight",
        ),
        ("SGLANG_HICACHE_NEXT_TURN_READ_LINE", {}, "_read_line"),
    ):
        cache = fixtures.MeasuredCache()
        with getattr(envs, switch).override(True):
            ntp = changed.NextTurnPrefetcher(
                mode="observe", tree_cache=cache, page_size=fixtures.PAGE, **kwargs
            )
        acting[switch] = (getattr(ntp, flag), ntp._meter is not None)
    print("in mode observe (switch acts, meter built):", acting)
    assert not any(acts for acts, _ in acting.values()), acting
