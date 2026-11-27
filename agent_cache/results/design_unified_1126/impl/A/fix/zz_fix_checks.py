"""Fixer's check (notes folder, not an engine test): the logic reviewer's three estimator
scenarios (review_logic/zz_review_logic_b.py) on the fixed copy, adapted to the present API
(PriceMeter.follow takes rate=, and the prefetcher binds the eviction observer).

Prints: the frontier price under pending writes at the reviewer's four loop gains; the hold
margin at a predicted wait of 1.5 s when most turns arrive to an empty queue; the absent
mark when the match grows past it.

Run as zz_fix_mutations.py is run (same recipe, this file's name).
"""

import random
import unittest
from array import array
from types import SimpleNamespace

from test_next_turn_prefetch import (
    KEY_TOKENS,
    PAGE,
    MeasuredCache,
    PrefetcherTestCase,
    gate_decisions,
    hints,
    hold_switches,
    key_of,
    make_req,
)

from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.next_turn_estimators import L3LoadModel
from sglang.srt.mem_cache.next_turn_meter import PriceMeter
from sglang.srt.mem_cache.next_turn_prefetch import NextTurnPrefetcher
from sglang.srt.mem_cache.next_use_eviction import NextUseStrategy


class FixLamLoop(PrefetcherTestCase):
    """lam -> kappa (pending writes at lam) -> lam: no longer a loop."""

    def setUp(self):
        pass

    def test_loop(self):
        load = L3LoadModel()
        for rate_per_s, pending in ((1.0, 50_000), (0.5, 50_000), (0.2, 50_000), (1.0, 10_000)):
            clock = [0.0]
            strategy = NextUseStrategy(horizon_s=300.0, clock=lambda: clock[0])
            cache = SimpleNamespace(
                tree_core=SimpleNamespace(eviction_strategy=strategy),
                read_scheduler=None,
                pending_backup_tokens=pending,
            )
            meter = PriceMeter(
                tree_cache=cache,
                page_size=PAGE,
                ready_after_s=0.5,
                hold_margin_s=2.0,
                log_interval_s=0.0,
                read_deadline=lambda handle: None,
            )
            seen = {}
            for second in range(1, 901):
                clock[0] = float(second)
                # One victim a second, due back in 1 / rate seconds.
                strategy.registry.register(second, clock[0] + 1.0 / rate_per_s)
                meter.on_host_evicted(SimpleNamespace(id=second), 640)
                meter.follow(
                    [],
                    admitted=0,
                    plans=0,
                    sessions_waiting=0,
                    rate=None,
                    load=load,
                    now=clock[0],
                )
                if second in (1, 60, 300, 900):
                    seen[second] = meter.board.lam(clock[0])
            price_free = rate_per_s * load.read_s_per_page / PAGE
            gain = pending * load.read_s_per_page / PAGE * rate_per_s
            print(
                f"\nFIX lam: victim rate {rate_per_s}/s, pending writes {pending} (old loop "
                f"gain {gain:.2f}): price-free value {price_free:.3g}; lam at 1/60/300/900 s = "
                + ", ".join(f"{seen[s]:.3g}" for s in (1, 60, 300, 900))
            )
            for value in seen.values():
                self.assertAlmostEqual(value, price_free)


class FixErrorBins(PrefetcherTestCase):
    """The hold's margin at a predicted wait of 1.5 s, through the meter."""

    def setUp(self):
        pass

    def test_bin_zero(self):
        rng = random.Random(3)
        cache = SimpleNamespace(
            tree_core=SimpleNamespace(eviction_strategy=None), read_scheduler=None
        )
        meter = PriceMeter(
            tree_cache=cache,
            page_size=PAGE,
            ready_after_s=0.5,
            hold_margin_s=2.0,
            log_interval_s=0.0,
            read_deadline=lambda handle: None,
        )
        # Light load: most turns arrive to an empty queue (predicted 0, admitted in ms).
        for _ in range(216):
            meter.observe_admission(
                predicted=0.0, wait=rng.uniform(0.001, 0.02), reader=False, landed_after=None
            )
        # A few arrive behind 2 to 4 turns: predicted 1 to 1.9 s, error -1.2 .. +1.5 s.
        for _ in range(40):
            predicted = rng.uniform(1.0, 1.9)
            wait = max(0.0, predicted + rng.uniform(-1.2, 1.5))
            meter.observe_admission(
                predicted=predicted, wait=wait, reader=False, landed_after=None
            )
        tracker = meter.admission_error
        self.assertEqual(len(tracker.errors[0]), 40)
        for q in (0.05, 0.1, 0.25, 0.5):
            print(
                f"\nFIX bins: q={q}: margin for a held read predicted 1.5 s = "
                f"{tracker.margin(q, predicted_wait_s=1.5, fallback_s=2.0):+.2f} s "
                f"(40 queued arrivals in the bin, 216 empty-queue arrivals left out)"
            )


class FixAbsentMarkAndGrowth(PrefetcherTestCase):
    """The absent mark when the match grows past it, against the switch off."""

    mode = "on"
    SHARED = KEY_TOKENS - 256

    def setUp(self):
        pass

    def build(self, switch):
        stack = hold_switches(
            self, (envs.SGLANG_HICACHE_NEXT_TURN_FREE_EMPTY_LOOKUPS, switch)
        )
        cache = MeasuredCache()
        cache.resident = cache.matched_tokens = 0
        ntp = NextTurnPrefetcher(
            mode="on", tree_cache=cache, page_size=PAGE, arrival_gate="on"
        )
        stack.close()
        ntp.on_turn_committed(
            req=make_req("s-t0", kv_hints=hints("s-t1", 5.0)),
            key=key_of(KEY_TOKENS),
            rotation_tail_declined=False,
            now=0.0,
        )
        return cache, ntp

    def test_growth(self):
        handle = CacheRequestHandle(rid="s-t1", attempt_id=0)
        asked_again = {}
        for switch in (False, True):
            cache, ntp = self.build(switch)
            self.ntp = ntp
            plan = ntp._plans["s-t1"]
            req = make_req("s-t1", entered=5.1)

            def gate(matched):
                tail = array("q", [*range(matched, self.SHARED), *range(9000, 9400)])
                if cache.prefetch_gate_prompt_fn is not None:
                    fn = lambda: cache.prefetch_gate_prompt_fn(handle, matched, tail)
                else:
                    fn = lambda: cache.prefetch_gate_fn(handle, matched)
                return gate_decisions(self.events(fn))

            steps = [("arrival gate(0)", gate(0))]
            if cache.prefetch_lookup_fn is not None:
                cache.prefetch_lookup_fn(
                    handle, storage_start=0, requested_tokens=KEY_TOKENS, hit_tokens=0
                )
            steps.append(("absent_from", plan.absent_from))
            # Another session computes the 256-token head: the turn's match grows to 256.
            cache.resident = 256
            ntp.tick(waiting_queue=[req], now=5.2)
            refetch = ntp.tick(waiting_queue=[req], now=5.8)
            asked_again[switch] = [r.rid for r in refetch]
            steps.append(("queue check asks for a refetch", asked_again[switch]))
            steps.append(("gate(256)", gate(256)))
            print(f"\nFIX growth FREE_EMPTY_LOOKUPS={switch}:")
            for step in steps:
                print("   ", step)
        self.assertEqual(asked_again[True], asked_again[False])


if __name__ == "__main__":
    unittest.main()
