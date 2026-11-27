"""Throwaway review check (logic lens): random scenarios with the new switches ON, looking
for exceptions and broken invariants. Reuses the engineer's scenario generator."""

import collections
import contextlib
import importlib.util
import random
import sys
import unittest
from array import array
from types import SimpleNamespace

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import test_next_turn_prefetch as fixtures

from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.next_turn_prefetch import (
    MAX_REFETCHES,
    NTP_MAX_FREE_LOOKUPS,
    NextTurnPrefetcher,
)
from sglang.srt.mem_cache.next_use_eviction import (
    NextUseStrategy,
    Queued,
    ReturnChanceStrategy,
)

CHECKS = (
    "/sgl-workspace/sglang/agent_cache/results/design_unified_1126/impl/A/checks/"
    "test_switches_off_equivalence.py"
)
spec = importlib.util.spec_from_file_location("eq_checks", CHECKS)
eq = importlib.util.module_from_spec(spec)
sys.modules["eq_checks"] = eq
spec.loader.exec_module(eq)

PAGE = fixtures.PAGE
SWITCHES = (
    envs.SGLANG_HICACHE_NEXT_TURN_MEASURED_PRICES,
    envs.SGLANG_HICACHE_NEXT_TURN_FREE_EMPTY_LOOKUPS,
    envs.SGLANG_HICACHE_NEXT_TURN_LATENESS_WEIGHT,
    envs.SGLANG_HICACHE_NEXT_TURN_READ_LINE,
    envs.SGLANG_HICACHE_NEXT_TURN_HOLD_QUANTILE,
    envs.SGLANG_HICACHE_NEXT_TURN_RECOMPUTE_BY_PRICE,
    envs.SGLANG_HICACHE_NEXT_TURN_RECOMPUTE_WHEN_STARVED,
    envs.SGLANG_HICACHE_NIXL_SKIP_STORED_WRITES,
)


class ReviewSwitchesOnFuzz(CustomTestCase):
    def test_fuzz(self):
        rng = random.Random(1126)
        built = refused = 0
        events_run = collections.Counter()
        lines = collections.Counter()
        for scenario in range(260):
            config = eq.random_config(rng)
            # The switches read the learned price off; keep most scenarios buildable.
            if rng.random() < 0.8:
                config["learning"] = "off"
            events = eq.random_events(rng, 500)
            clock = SimpleNamespace(now=0.0)
            on = [switch for switch in SWITCHES if rng.random() < 0.8]
            with contextlib.ExitStack() as stack:
                for switch in SWITCHES:
                    stack.enter_context(switch.override(switch in on))
                stack.enter_context(
                    envs.SGLANG_LOG_HICACHE_PRICES_INTERVAL.override(
                        rng.choice((0.0, 1.0))
                    )
                )
                stack.enter_context(
                    envs.SGLANG_HICACHE_NEXT_TURN_ROOM_RETRY.override(config["room_retry"])
                )
                if config["cache"] == "fifo":
                    cache = fixtures.ActingCache()
                    cache.tree_core = SimpleNamespace(
                        eviction_strategy=None, host_eviction_observer=None
                    )
                    cache.prefetch_gate_prompt_fn = None
                    cache.prefetch_lookup_fn = None
                else:
                    cache = fixtures.MeasuredCache()
                    strategy = (
                        ReturnChanceStrategy(
                            horizon_s=30.0, score=config["score"], clock=lambda: clock.now
                        )
                        if config["policy"] == "return_chance"
                        else NextUseStrategy(horizon_s=300.0, clock=lambda: clock.now)
                    )
                    cache.tree_core = SimpleNamespace(
                        eviction_strategy=strategy, host_eviction_observer=None
                    )
                cache.pending_backup_tokens = 0
                try:
                    ntp = NextTurnPrefetcher(
                        mode=config["mode"],
                        tree_cache=cache,
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
                except ValueError:
                    refused += 1
                    continue
            built += 1
            side = eq.Side.__new__(eq.Side)
            side.module = None
            side.messages = eq.Messages()
            side.cache, side.ntp = cache, ntp
            side.reqs, side.queue, side.answers = {}, [], []
            import logging

            logger = logging.getLogger("sglang.srt.mem_cache.next_turn_prefetch")
            logger.setLevel(logging.INFO)
            logger.addHandler(side.messages)
            make = side.req

            def req(rid, **kwargs):
                made = make(rid, **kwargs)
                made.cached_tokens = 100
                made.origin_input_ids = array("q", range(2000))
                made.time_stats.prefill_finished_time = (
                    made.time_stats.forward_entry_time + 0.2
                    if made.time_stats.forward_entry_time > 0
                    else 0.0
                )
                return made

            side.req = req
            node_ids = iter(range(10**6, 2 * 10**6))
            try:
                for step, (dt, event) in enumerate(events):
                    clock.now += dt
                    context = (scenario, config, [s.name for s in on], step, event)
                    try:
                        side.apply(event, clock.now)
                        events_run[event[0]] += 1
                        extra = rng.random()
                        handle = (
                            CacheRequestHandle(event[1], 0)
                            if len(event) > 1 and isinstance(event[1], str)
                            else None
                        )
                        registry = (
                            None
                            if cache.tree_core.eviction_strategy is None
                            else cache.tree_core.eviction_strategy.registry
                        )
                        if extra < 0.10 and cache.tree_core.host_eviction_observer:
                            node_id = next(node_ids)
                            kind = rng.random()
                            if registry is None:
                                pass
                            elif kind < 0.3:
                                registry.register(node_id, None)
                            elif kind < 0.5:
                                registry.register(node_id, Queued(since=clock.now))
                            elif kind < 0.8 and isinstance(
                                cache.tree_core.eviction_strategy, NextUseStrategy
                            ):
                                registry.register(node_id, clock.now + rng.choice((0.0, 1.0, 50.0)))
                            parent = SimpleNamespace(id=next(node_ids), parent=None, children={})
                            victim = SimpleNamespace(id=node_id, parent=parent, children={})
                            parent.children[0] = victim
                            cache.tree_core.host_eviction_observer(victim, rng.choice((0, 64, 6400)))
                            if hasattr(cache, "victim"):
                                cache.victim = rng.choice((None, victim))
                            events_run["evict"] += 1
                        elif extra < 0.20 and handle is not None and cache.prefetch_lookup_fn:
                            cache.ongoing.discard(handle)
                            cache.prefetch_lookup_fn(
                                handle,
                                storage_start=rng.choice((0, 4, 16)) * PAGE,
                                requested_tokens=16 * PAGE,
                                hit_tokens=rng.choice((0, 0, 64, 256, 640)),
                            )
                            events_run["lookup"] += 1
                        elif extra < 0.32 and handle is not None and cache.prefetch_gate_prompt_fn:
                            matched = rng.choice((0, 4, 8, 16, 40)) * PAGE
                            tail = array(
                                "q",
                                [*range(matched, rng.choice((8, 16, 40, 1000)) * PAGE), *range(9 * 10**6, 9 * 10**6 + 300)],
                            )
                            if cache.prefetch_gate_prompt_fn(handle, matched, tail):
                                cache.ongoing.add(handle)
                            events_run["gate_prompt"] += 1
                        elif extra < 0.40 and getattr(cache, "read_lags", None) is not None:
                            cache.read_lags.lookup.observe(rng.uniform(0.0, 0.05))
                            cache.read_lags.fixed.observe(rng.uniform(-0.05, 0.05))
                            cache.read_lags.note_write_hold(held=rng.random() < 0.5, now=clock.now)
                            events_run["lags"] += 1
                        elif extra < 0.46:
                            cache.pending_backup_tokens = rng.choice((0, 2000, 60_000))
                        elif extra < 0.52 and handle is not None and hasattr(cache, "park"):
                            if not cache.read_scheduler.is_parked(handle):
                                cache.park(handle, start=rng.choice((0, 640)), tokens=rng.choice((640, 6400)))
                    except Exception as error:
                        raise AssertionError(f"{type(error).__name__}: {error} at {context}") from error
                    for plan in ntp._plans.values():
                        self.assertGreaterEqual(plan.lookups, 0, context)
                        self.assertGreaterEqual(plan.refetches, 0, context)
                        self.assertLessEqual(plan.free_lookups, NTP_MAX_FREE_LOOKUPS, context)
                    self.assertTrue(set(ntp._pred_wait) <= set(ntp._queued), context)
                    # Landing stamps are kept only for planned or queued requests.
                    known = set(ntp._plans) | set(ntp._queued)
                    self.assertTrue(set(ntp._landed_at) <= known, context)
                    if ntp._prices is not None:
                        board = ntp._prices
                        for value in (
                            board.c(clock.now),
                            board.mu(clock.now),
                            board.lam(clock.now),
                            board.kappa(clock.now),
                        ):
                            self.assertTrue(value == value and value >= 0.0, context)
                        self.assertLessEqual(board.c(clock.now), 1.0 + 1e-9, context)
                for line in side.messages.lines:
                    lines[line.split()[1]] += 1
                    for field in line.split()[2:]:
                        self.assertRegex(field, r"\A\w+=[\w./-]+\Z", line)
            finally:
                logger.removeHandler(side.messages)
        print(f"\nREVIEW switches-on fuzz: built {built}, refused at construction {refused}")
        print("REVIEW events:", dict(sorted(events_run.items())))
        print("REVIEW lines:", dict(sorted(lines.items())))


if __name__ == "__main__":
    unittest.main()
