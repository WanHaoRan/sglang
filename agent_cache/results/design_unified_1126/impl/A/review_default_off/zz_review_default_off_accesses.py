"""Review check (default-off lens), throwaway: with every new switch off, does the changed
prefetcher read, write and call the cache exactly as the base module does, and does it end
every step in the same internal state? With only the prices log on, what does it ask of the
cache beyond the base module?

Reuses the engineer's scenario generator (impl/A/checks/test_switches_off_equivalence.py) and
adds what that replay does not compare: every attribute read, attribute write and call made
through the cache object (and its tree_core, cache_controller, read_scheduler and
storage_prefetch_retries), in order, and the prefetcher's own state after every event.

The base module is frozen/impl_unified_base's next_turn_prefetch.py, loaded by path.
"""

import collections
import enum
import importlib.util
import os
import random
import sys
from types import SimpleNamespace

import msgspec

from sglang.srt.environ import envs
from sglang.srt.mem_cache import next_turn_prefetch as changed
from sglang.srt.mem_cache.next_use_eviction import (
    NextUseStrategy,
    ReturnChanceStrategy,
)

ROOT = "/sgl-workspace/sglang/agent_cache"
CHECKS = os.path.join(ROOT, "results/design_unified_1126/impl/A/checks")
BASE_MODULE = os.path.join(
    ROOT, "frozen/impl_unified_base/python/sglang/srt/mem_cache/next_turn_prefetch.py"
)


def load(name, file_path):
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


replay = load("review_replay", os.path.join(CHECKS, "test_switches_off_equivalence.py"))
base = load("review_ntp_base", BASE_MODULE)
fixtures = replay.fixtures
PAGE = fixtures.PAGE

NESTED = ("tree_core", "cache_controller", "read_scheduler", "storage_prefetch_retries")


def brief(value):
    """A value as it compares across the two sides: no object identity."""
    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    if isinstance(value, enum.Enum):
        return str(value)
    rid = getattr(value, "rid", None)
    if isinstance(rid, str):
        return ("rid", rid)
    if isinstance(value, (list, tuple, range)):
        return ("seq", len(value))
    if isinstance(value, dict):
        return ("dict", tuple(sorted(value)))
    if callable(value):
        return ("fn", getattr(value, "__name__", type(value).__name__))
    try:
        return (type(value).__name__, len(value))
    except TypeError:
        return (type(value).__name__,)


class Spy:
    """Forwards to a target, recording every read, write and call made through it."""

    def __init__(self, target, accesses, owner):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_accesses", accesses)
        object.__setattr__(self, "_owner", owner)

    @property
    def __class__(self):
        return type(self._target)

    def __getattr__(self, name):
        value = getattr(self._target, name)
        accesses, owner = self._accesses, self._owner
        accesses.append(("get", owner, name))
        if value is None:
            return None
        if name in NESTED:
            return Spy(value, accesses, f"{owner}.{name}")
        if callable(value):

            def call(*args, **kwargs):
                accesses.append(
                    (
                        "call",
                        owner,
                        name,
                        tuple(brief(arg) for arg in args),
                        tuple(sorted((key, brief(arg)) for key, arg in kwargs.items())),
                    )
                )
                return value(*args, **kwargs)

            return call
        return value

    def __setattr__(self, name, value):
        self._accesses.append(("set", self._owner, name, brief(value)))
        setattr(self._target, name, value)


class SpySide(replay.Side):
    """The engineer's Side with the cache behind a Spy, for the prefetcher and the events."""

    def __init__(self, module, config, clock, *, prices_interval_s=0.0):
        self.module = module
        self.messages = replay.Messages()
        self.logger = replay.logging.getLogger(module.__name__)
        self.logger.setLevel(replay.logging.INFO)
        self.logger.addHandler(self.messages)
        cache_class = {
            "fifo": fixtures.ActingCache,
            "deadline": fixtures.DeadlineCache,
            "priced": fixtures.PricedCache,
        }[config["cache"]]
        cache = cache_class()
        if config["cache"] != "fifo":
            strategy = (
                ReturnChanceStrategy(horizon_s=30.0, score=config["score"], clock=clock)
                if config["policy"] == "return_chance"
                else NextUseStrategy(horizon_s=300.0, clock=clock)
            )
            cache.tree_core = SimpleNamespace(eviction_strategy=strategy)
        else:
            cache.tree_core = SimpleNamespace(eviction_strategy=None)
        cache.pending_backup_tokens = 0
        self.accesses = []
        self.cache = Spy(cache, self.accesses, "cache")
        with envs.SGLANG_LOG_HICACHE_PRICES_INTERVAL.override(prices_interval_s):
            self.ntp = module.NextTurnPrefetcher(
                mode=config["mode"],
                tree_cache=self.cache,
                page_size=PAGE,
                reply_key_source="hint",
                read_timing=config["timing"],
                arrival_gate=config["gate"],
                arrival_hold=config["hold"],
                landing_quantile=config["landing_quantile"],
                memory_price=config["price"],
                price_learning=config["learning"],
                soft_pins=config["soft_pins"],
            )
        self.reqs = {}
        self.queue = []
        self.answers = []


BASE_PLAN_FIELDS = base.TurnPlan.__struct_fields__
# Every attribute the base prefetcher keeps, except the cache and the plans (compared below).
BASE_STATE = (
    "_mode", "_page_size", "_reply_key_source", "_just_in_time", "_expected_cost",
    "_stages_reads", "_landing_quantile", "_price", "_staged_pin_tokens_seen", "_offset",
    "_admissions", "_load", "_pending", "_admitted", "_deadline_mode", "_last_now",
    "_last_rate", "_positions", "_hold_arrived", "_room_retry", "_ready_since",
    "_hold_logged", "_scores_return_chance", "_scores_samples", "_gap_samples",
    "_forecast_ends_session", "_session_tail", "_rid_session",
)


def flat(value):
    if isinstance(value, msgspec.Struct):
        return (type(value).__name__,) + tuple(
            flat(getattr(value, name)) for name in value.__struct_fields__
        )
    if isinstance(value, (list, tuple, collections.deque)):
        return tuple(flat(item) for item in value)
    if isinstance(value, dict):
        return tuple((key, flat(item)) for key, item in value.items())
    if isinstance(value, set):
        return tuple(sorted(value))
    return value


def internal_state(ntp):
    plans = {}
    for rid, plan in ntp._plans.items():
        fields = {}
        for name in BASE_PLAN_FIELDS:
            value = getattr(plan, name)
            fields[name] = len(value) if name == "key" else flat(value)
        plans[rid] = fields
    state = {name: flat(getattr(ntp, name)) for name in BASE_STATE}
    state["_plans"] = plans
    state["_queued"] = tuple(ntp._queued)
    return state


def test_the_base_state_list_covers_every_attribute_the_base_prefetcher_keeps():
    cache = fixtures.PricedCache()
    ntp = base.NextTurnPrefetcher(mode="on", tree_cache=cache, page_size=PAGE)
    kept = set(vars(ntp)) - {"_tree_cache", "_plans", "_queued", "_seq", "_next_use", "_return_chance"}
    assert kept == set(BASE_STATE), kept ^ set(BASE_STATE)


def test_with_the_switches_off_the_cache_is_asked_the_same_and_the_state_is_the_same():
    for name in replay.NEW_SWITCHES:
        assert not getattr(envs, name).get(), name
    rng = random.Random(20261126)
    compared = 0
    kinds = collections.Counter()
    for scenario in range(240):
        config = replay.random_config(rng)
        events = replay.random_events(rng, 600)
        clock = SimpleNamespace(now=0.0)
        with envs.SGLANG_HICACHE_NEXT_TURN_ROOM_RETRY.override(config["room_retry"]):
            before = SpySide(base, config, lambda: clock.now)
            after = SpySide(changed, config, lambda: clock.now)
        try:
            assert before.accesses == after.accesses, (scenario, config, "construction")
            seen = len(before.accesses)
            for step, (dt, event) in enumerate(events):
                clock.now += dt
                before.apply(event, clock.now)
                after.apply(event, clock.now)
                where = (scenario, config, step, event)
                assert before.accesses[seen:] == after.accesses[seen:], (
                    where,
                    before.accesses[seen:],
                    after.accesses[seen:],
                )
                seen = len(before.accesses)
                assert before.messages.lines == after.messages.lines, where
                assert before.answers == after.answers, where
                assert internal_state(before.ntp) == internal_state(after.ntp), where
            compared += len(before.accesses)
            for access in before.accesses:
                kinds[access[:3]] += 1
        finally:
            before.close()
            after.close()
    print("cache accesses compared:", compared)
    print("distinct (kind, owner, name):", len(kinds))
    for access, count in sorted(kinds.items()):
        print("   ", access, count)
    assert compared > 100_000


def test_with_only_the_prices_log_on_what_is_asked_of_the_cache_beyond_the_base():
    rng = random.Random(20261127)
    extras = collections.Counter()
    for scenario in range(240):
        config = replay.random_config(rng)
        events = replay.random_events(rng, 600)
        clock = SimpleNamespace(now=0.0)
        with envs.SGLANG_HICACHE_NEXT_TURN_ROOM_RETRY.override(config["room_retry"]):
            before = SpySide(base, config, lambda: clock.now)
            after = SpySide(changed, config, lambda: clock.now, prices_interval_s=1.0)
        try:
            for dt, event in events:
                clock.now += dt
                before.apply(event, clock.now)
                after.apply(event, clock.now)
            # The base's accesses must be a subsequence of the logging side's.
            position = 0
            wanted = before.accesses
            for access in after.accesses:
                if position < len(wanted) and access == wanted[position]:
                    position += 1
                else:
                    extras[access[:3]] += 1
            assert position == len(wanted), (scenario, config, wanted[position])
            assert internal_state(before.ntp) == internal_state(after.ntp), (scenario, config)
        finally:
            before.close()
            after.close()
    print("with only the prices log on, asked of the cache beyond the base:")
    for access, count in sorted(extras.items()):
        print("   ", access, count)
    writes = {access for access in extras if access[0] == "set"}
    assert writes <= {
        ("set", "cache.tree_core", "host_eviction_observer"),
        ("set", "cache", "prefetch_landed_fn"),
        ("set", "cache", "read_lags"),
    }, writes
