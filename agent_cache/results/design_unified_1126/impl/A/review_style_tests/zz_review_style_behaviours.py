"""Reviewer's throwaway check (style and tests lens): branches of the FETCH and PRICE change
that no test of the copy runs, exercised once to see whether they hold.

Nothing here edits the copy. Run with the test recipe on a copy of this file placed in the
copy's test folder (see zz_review_style_mutations.py).
"""

import contextlib
import unittest
from array import array

import test_next_turn_prefetch as T

from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.next_turn_prefetch import NextTurnPrefetcher
from sglang.srt.mem_cache.next_use_eviction import NextUseStrategy

PAGE = T.PAGE


class FifoMeasuredCache(T.MeasuredCache):
    """MeasuredCache under the fifo read order: no read scheduler."""

    def __init__(self):
        super().__init__()
        self.read_scheduler = None


class ReviewBehaviours(T.PrefetcherTestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)

    def hold(self, switch, value):
        self.stack.enter_context(switch.override(value))

    def prices_lines(self, fn):
        return [fields for kind, fields in self.events(fn) if kind == "prices"]

    def test_fifo_read_order_with_only_the_prices_log(self):
        """No read scheduler (the hicache baseline's order): the meter's srs-is-None
        branches, in both modes."""
        self.hold(envs.SGLANG_LOG_HICACHE_PRICES_INTERVAL, 5.0)
        for mode in ("observe", "on"):
            with self.subTest(mode=mode):
                cache = FifoMeasuredCache()
                ntp = NextTurnPrefetcher(mode=mode, tree_cache=cache, page_size=PAGE)
                # The fake has no read_lags attribute at all under fifo: nothing may read it.
                self.assertNotIn("read_lags", vars(cache))
                queue = [T.make_req("u-t1", entered=0.0)]
                (line,) = self.prices_lines(
                    lambda: [ntp.tick(waiting_queue=queue, now=now) for now in (0.0, 1.0)]
                )
                print("fifo prices line:", mode, line)
                self.assertEqual(
                    [line[name] for name in ("parked", "win_holds", "lookup_ms", "late")],
                    ["0", "na", "na", "0"],
                )

    def test_real_cpu_cache_with_only_the_prices_log(self):
        """The real cache, not a fake: every attribute the meter reads must exist on it."""
        self.hold(envs.SGLANG_LOG_HICACHE_PRICES_INTERVAL, 5.0)
        for mode in ("observe", "on"):
            with self.subTest(mode=mode):
                cache = T.cpu_unified_cache()
                ntp = NextTurnPrefetcher(mode=mode, tree_cache=cache, page_size=2)
                self.assertIsNotNone(cache.tree_core.host_eviction_observer)
                queue = [T.make_req("u-t1", entered=0.0)]
                (line,) = self.prices_lines(
                    lambda: [ntp.tick(waiting_queue=queue, now=now) for now in (0.0, 1.0)]
                )
                print("real cache prices line:", mode, line)

    def test_real_cpu_cache_helpers_added_for_the_switches(self):
        """The new cache methods on a real cache with nothing parked or evictable."""
        cache = T.cpu_unified_cache()
        handle = CacheRequestHandle("nobody", 0)
        self.assertIsNone(cache.next_host_victim())
        self.assertEqual(cache.storage_read_line_pages(None), 0)
        self.assertIsNone(cache.parked_read_span(handle))
        self.assertFalse(cache.drop_parked_read(handle))
        self.assertIsNone(cache.prefetch_gate_prompt_fn)
        self.assertIsNone(cache.prefetch_lookup_fn)
        self.assertIsNone(cache.read_lags)
        # Class-level defaults: an instance assignment must not reach the class.
        cache.read_lags = object()
        self.assertIsNone(type(cache).read_lags)

    def test_a_landing_under_next_use_eviction_with_the_log_on(self):
        """Measuring binds the landing hook under every eviction policy. Under next_use the
        session's tail is known but there is no return_chance strategy to adopt into."""
        self.hold(envs.SGLANG_LOG_HICACHE_PRICES_INTERVAL, 5.0)
        cache = T.MeasuredCache()
        self.assertIsInstance(cache.tree_core.eviction_strategy, NextUseStrategy)
        ntp = NextTurnPrefetcher(mode="on", tree_cache=cache, page_size=PAGE)
        ntp.on_turn_committed(
            req=T.make_req("s-t0", kv_hints=T.hints("s-t1")),
            key=T.key_of(T.KEY_TOKENS),
            rotation_tail_declined=False,
            now=0.0,
            tail_node_id=5,
        )
        self.assertEqual(ntp._session_tail, {"s": 5})
        self.assertIsNone(ntp._return_chance)
        ntp.on_read_landed(CacheRequestHandle("s-t1", 0), 9)
        self.assertIn("s-t1", ntp._landed_at)

    def test_measured_prices_with_price_learning_on_is_refused_too(self):
        """Both learning rules ("on" = room, "harm") move a price the switch replaces."""
        self.hold(envs.SGLANG_HICACHE_NEXT_TURN_MEASURED_PRICES, True)
        for learning in ("on", "harm"):
            with self.subTest(learning=learning):
                with self.assertRaises(ValueError):
                    NextTurnPrefetcher(
                        mode="on",
                        tree_cache=T.MeasuredCache(),
                        page_size=PAGE,
                        read_timing="expected_cost",
                        price_learning=learning,
                    )

    def test_every_switch_is_inert_in_observe_whatever_else_is_set(self):
        """Mode observe with every switch on and none of the prerequisites: nothing raises,
        nothing is bound, only the log would measure."""
        for switch in (
            envs.SGLANG_HICACHE_NEXT_TURN_MEASURED_PRICES,
            envs.SGLANG_HICACHE_NEXT_TURN_FREE_EMPTY_LOOKUPS,
            envs.SGLANG_HICACHE_NEXT_TURN_HOLD_QUANTILE,
            envs.SGLANG_HICACHE_NEXT_TURN_RECOMPUTE_BY_PRICE,
            envs.SGLANG_HICACHE_NEXT_TURN_RECOMPUTE_WHEN_STARVED,
        ):
            self.hold(switch, True)
        cache = T.MeasuredCache()
        ntp = NextTurnPrefetcher(
            mode="observe",
            tree_cache=cache,
            page_size=PAGE,
            arrival_gate="on",
            arrival_hold="on",
            price_learning="harm",
            read_timing="expected_cost",
        )
        self.assertIsNone(ntp._meter)
        self.assertIsNone(cache.prefetch_gate_prompt_fn)
        self.assertIsNone(cache.prefetch_lookup_fn)
        self.assertIsNone(cache.tree_core.host_eviction_observer)

    def test_lateness_weight_and_read_line_in_observe(self):
        """LATENESS_WEIGHT and READ_LINE are gated by the setup (timing, read order), not by
        the mode: what do they do in observe?"""
        self.hold(envs.SGLANG_HICACHE_NEXT_TURN_LATENESS_WEIGHT, True)
        self.hold(envs.SGLANG_HICACHE_NEXT_TURN_READ_LINE, True)
        cache = T.MeasuredCache()
        ntp = NextTurnPrefetcher(
            mode="observe", tree_cache=cache, page_size=PAGE, read_timing="expected_cost"
        )
        print(
            "observe:",
            dict(
                lateness_weight=ntp._lateness_weight,
                read_line=ntp._read_line,
                meter=ntp._meter is not None,
                observer=cache.tree_core.host_eviction_observer is not None,
                landed_fn=cache.prefetch_landed_fn is not None,
                read_lags=cache.read_lags is not None,
            ),
        )

    def test_the_prompt_gate_on_the_real_cache(self):
        """The planner asked for TestGateOnTheRealCache with the three-argument gate: the
        real prefetch_from_storage handing the prompt to a real prefetcher's gate."""
        self.hold(envs.SGLANG_HICACHE_NEXT_TURN_FREE_EMPTY_LOOKUPS, True)
        cache = T.cpu_unified_cache()
        cache.enable_storage = True
        cache.prefetch_threshold = 2
        cache.cache_controller = T.SimpleNamespace(
            get_prefetch_submission=lambda rid: None,
            prefetch_rate_limited=lambda: False,
        )
        ntp = NextTurnPrefetcher(mode="on", tree_cache=cache, page_size=2, arrival_gate="on")
        self.assertIsNotNone(cache.prefetch_gate_prompt_fn)
        ntp.on_turn_committed(
            req=T.make_req("s-t0", kv_hints=T.hints("s-t1")),
            key=T.key_of(8),
            rotation_tail_declined=False,
            now=0.0,
        )
        handle = CacheRequestHandle("s-t1", 0)
        # The prompt repeats 7 of the key's 8 tokens (the reply's last token differs), all
        # 7 matched in memory: covered, where the whole key would leave 1 token missing.
        found = self.events(
            lambda: cache.prefetch_from_storage(
                handle,
                cache.root_node_handle(),
                array("q", [107, 108, 109, 110]),
                matched_prefix_tokens=list(range(7)),
            )
        )
        print("real cache, prompt gate:", [f for kind, f in found if kind == "ntp_gate"])
        self.assertEqual(cache._prefetch_outcome_stats["declined_gate"], 1)
        self.assertEqual(ntp._plans["s-t1"].shared_tokens, 7)
        self.assertFalse(cache.has_ongoing_prefetch(handle))

    def test_a_match_past_the_key_at_the_prompt_gate(self):
        """The first gate call of a turn whose match is longer than its plan's key (another
        session shares more of the prompt): match_at refuses an offset past the key."""
        self.hold(envs.SGLANG_HICACHE_NEXT_TURN_FREE_EMPTY_LOOKUPS, True)
        cache = T.MeasuredCache()
        ntp = NextTurnPrefetcher(mode="on", tree_cache=cache, page_size=PAGE, arrival_gate="on")
        ntp.on_turn_committed(
            req=T.make_req("s-t0", kv_hints=T.hints("s-t1")),
            key=T.key_of(T.KEY_TOKENS),
            rotation_tail_declined=False,
            now=0.0,
        )
        handle = CacheRequestHandle("s-t1", 0)
        tail = array("q", range(9000, 9400))
        found = self.events(
            lambda: cache.prefetch_gate_prompt_fn(handle, T.KEY_TOKENS + 500, tail)
        )
        print("match past the key:", [f for kind, f in found if kind == "ntp_gate"])
        self.assertIsNone(ntp._plans["s-t1"].shared_tokens)


if __name__ == "__main__":
    unittest.main()
