"""Throwaway review checks (logic lens). Not part of the engine's tests."""

import random
import time
import unittest
from array import array
from types import SimpleNamespace

from test_next_turn_prefetch import (
    ANCHOR,
    KEY_TOKENS,
    PAGE,
    MeasuredCache,
    PrefetcherTestCase,
    gate_decisions,
    hints,
    hold_switches,
    key_of,
    make_req,
    measure_lags,
)

from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.next_turn_estimators import L3LoadModel
from sglang.srt.mem_cache.next_turn_meter import PriceMeter
from sglang.srt.mem_cache.next_turn_prefetch import (
    MAX_REFETCHES,
    NTP_MAX_FREE_LOOKUPS,
    NextTurnPrefetcher,
)
from sglang.srt.mem_cache.next_turn_prices import AdmissionErrorTracker
from sglang.srt.mem_cache.next_use_eviction import NextUseStrategy


class ReviewAbsentMarkAndGrowth(PrefetcherTestCase):
    """F5: the absent mark when the match GROWS past it (another session computed the
    shared head the SSD had lost), against the same sequence with the switch off."""

    mode = "on"
    SHARED = KEY_TOKENS - 256

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

    def setUp(self):
        pass

    def test_growth(self):
        handle = CacheRequestHandle(rid="s-t1", attempt_id=0)
        out = {}
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

            steps = []
            # Arrival: nothing of the prompt in memory, the lookup from 0 finds nothing.
            steps.append(("arrival gate(0)", gate(0)))
            if cache.prefetch_lookup_fn is not None:
                cache.prefetch_lookup_fn(
                    handle, storage_start=0, requested_tokens=KEY_TOKENS, hit_tokens=0
                )
            steps.append(("absent_from", plan.absent_from))
            # Another session computes the 256-token head: the turn's match grows to 256.
            cache.resident = 256
            ntp.tick(waiting_queue=[req], now=5.2)
            refetch = ntp.tick(waiting_queue=[req], now=5.8)
            steps.append(("queue check asks for a refetch", [r.rid for r in refetch]))
            steps.append(("gate(256)", gate(256)))
            out[switch] = steps
        for switch, steps in out.items():
            print(f"\nREVIEW growth FREE_EMPTY_LOOKUPS={switch}:")
            for step in steps:
                print("   ", step)


class ReviewRefundFuzz(PrefetcherTestCase):
    """F5 accounting under random sequences: counts never negative, lookups bounded."""

    mode = "on"

    def setUp(self):
        pass

    def test_fuzz(self):
        hold_switches(
            self,
            (envs.SGLANG_HICACHE_NEXT_TURN_FREE_EMPTY_LOOKUPS, True),
            (envs.SGLANG_HICACHE_NEXT_TURN_ROOM_RETRY, True),
        )
        rng = random.Random(7)
        worst_lookups = 0
        worst_issued = 0
        for case in range(1500):
            cache = MeasuredCache()
            cache.resident = cache.matched_tokens = KEY_TOKENS
            ntp = NextTurnPrefetcher(
                mode="on", tree_cache=cache, page_size=PAGE, arrival_gate="on"
            )
            ntp.on_turn_committed(
                req=make_req("s-t0", kv_hints=hints("s-t1", 5.0)),
                key=key_of(KEY_TOKENS),
                rotation_tail_declined=False,
                now=0.0,
            )
            plan = ntp._plans["s-t1"]
            handle = CacheRequestHandle(rid="s-t1", attempt_id=0)
            req = make_req("s-t1", entered=5.0)
            shared = KEY_TOKENS - 256
            now = 5.0
            under_way = None  # storage_start of the lookup under way
            issued = 0
            ntp.tick(waiting_queue=[req], now=now)
            for step in range(120):
                now += rng.choice((0.05, 0.3, 0.6, 2.5))
                op = rng.random()
                matched = rng.choice((0, 128, 256, 384, 512, 640, shared, KEY_TOKENS))
                tail = array("q", [*range(matched, shared), *range(9000, 9400)])
                if op < 0.35 and under_way is None:
                    # The request's own trigger.
                    if len(tail) >= 256 and cache.prefetch_gate_prompt_fn(
                        handle, matched, tail
                    ):
                        cache.ongoing.add(handle)
                        under_way = matched
                        issued += 1
                elif op < 0.7:
                    cache.resident = matched
                    for back in ntp.tick(waiting_queue=[req], now=now):
                        if under_way is None and cache.prefetch_gate_prompt_fn(
                            handle, matched, tail
                        ):
                            cache.ongoing.add(handle)
                            under_way = matched
                            issued += 1
                        ntp.on_refetch_result(
                            req=back, issued=cache.has_ongoing_prefetch(handle)
                        )
                elif op < 0.95 and under_way is not None:
                    cache.ongoing.discard(handle)
                    kind = rng.random()
                    if kind < 0.15:
                        pass  # terminated: not reported
                    else:
                        hit = 0 if kind < 0.7 else rng.choice((64, 256, 512))
                        cache.prefetch_lookup_fn(
                            handle,
                            storage_start=under_way,
                            requested_tokens=KEY_TOKENS,
                            hit_tokens=hit,
                        )
                        if hit >= 256 and rng.random() < 0.5:
                            ntp.on_read_refused(handle, tokens=hit, cut=False, now=now)
                    under_way = None
                self.assertGreaterEqual(plan.lookups, 0, (case, step))
                self.assertGreaterEqual(plan.refetches, 0, (case, step))
                self.assertLessEqual(plan.free_lookups, NTP_MAX_FREE_LOOKUPS)
                self.assertLessEqual(plan.lookups, MAX_REFETCHES + 1)
            worst_lookups = max(worst_lookups, plan.lookups)
            worst_issued = max(worst_issued, issued)
        print(
            f"\nREVIEW fuzz: 1500 cases ok; max charged lookups {worst_lookups}, "
            f"max lookups issued for one turn {worst_issued}"
        )


class ReviewLamLoop(PrefetcherTestCase):
    """lam -> kappa (pending writes at lam) -> lam: where the loop settles."""

    mode = "on"

    def setUp(self):
        pass

    def test_loop(self):
        for rate_per_s, pending in ((1.0, 50_000), (0.5, 50_000), (0.2, 50_000), (1.0, 10_000)):
            clock = [0.0]
            strategy = NextUseStrategy(horizon_s=300.0, clock=lambda: clock[0])
            cache = SimpleNamespace(
                tree_core=SimpleNamespace(
                    eviction_strategy=strategy, host_eviction_observer=None
                ),
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
            load = L3LoadModel()
            seen = {}
            for second in range(1, 901):
                clock[0] = float(second)
                # One victim a second, due back in 1 / rate seconds.
                strategy.registry.register(second, clock[0] + 1.0 / rate_per_s)
                cache.tree_core.host_eviction_observer(SimpleNamespace(id=second), 640)
                meter.follow(
                    [],
                    admitted=0,
                    plans=0,
                    sessions_waiting=0,
                    arrival_rate=0.0,
                    load=load,
                    now=clock[0],
                )
                if second in (1, 60, 300, 900):
                    seen[second] = meter.board.lam(clock[0])
            gain = pending * load.read_s_per_page / PAGE * rate_per_s
            print(
                f"\nREVIEW lam loop: victim rate {rate_per_s}/s, pending writes {pending}: "
                f"loop gain {gain:.2f}; lam at 1/60/300/900 s = "
                + ", ".join(f"{seen[s]:.3g}" for s in (1, 60, 300, 900))
            )


class ReviewErrorBins(PrefetcherTestCase):
    mode = "on"

    def setUp(self):
        pass

    def test_bin_zero(self):
        rng = random.Random(3)
        tracker = AdmissionErrorTracker()
        only_queued = AdmissionErrorTracker()
        # Light load: most turns arrive to an empty queue (predicted 0, admitted in ms).
        for _ in range(216):
            tracker.observe(predicted_wait_s=0.0, wait_s=rng.uniform(0.001, 0.02))
        # A few arrive behind 2 to 4 turns: predicted 1 to 2 s, error -1.2 .. +1.5 s.
        for _ in range(40):
            predicted = rng.uniform(1.0, 1.9)
            wait = max(0.0, predicted + rng.uniform(-1.2, 1.5))
            tracker.observe(predicted_wait_s=predicted, wait_s=wait)
            only_queued.observe(predicted_wait_s=predicted, wait_s=wait)
        for q in (0.05, 0.1, 0.25, 0.5):
            print(
                f"\nREVIEW bins: q={q}: margin for a held read predicted 1.5 s = "
                f"{tracker.margin(q, predicted_wait_s=1.5, fallback_s=2.0):+.2f} s; "
                f"from queued arrivals only "
                f"{only_queued.margin(q, predicted_wait_s=1.5, fallback_s=2.0):+.2f} s"
            )


class ReviewTickCost(PrefetcherTestCase):
    """Time of one tick with 60 queued planned turns and 40 parked reads."""

    mode = "on"

    def setUp(self):
        pass

    def one(self, *, read_line, log):
        stack = hold_switches(
            self,
            (envs.SGLANG_HICACHE_NEXT_TURN_READ_LINE, read_line),
            (envs.SGLANG_LOG_HICACHE_PRICES_INTERVAL, 5.0 if log else 0.0),
        )
        cache = MeasuredCache()
        cache.resident = cache.matched_tokens = KEY_TOKENS
        ntp = NextTurnPrefetcher(
            mode="on",
            tree_cache=cache,
            page_size=PAGE,
            arrival_gate="on",
            arrival_hold="on",
        )
        stack.close()
        self.ntp = ntp
        # 700 admissions over the last five minutes: the rate the queue drains at.
        now = 0.0
        for i in range(700):
            req = make_req(f"warm-{i}", entered=now)
            ntp.tick(waiting_queue=[req], now=now)
            req.time_stats.forward_entry_time = now + 0.01
            now += 0.4
        queue = []
        for i in range(60):
            ntp.on_turn_committed(
                req=make_req(f"s{i}-t0", kv_hints=hints(f"s{i}-t1", 1.0)),
                key=key_of(KEY_TOKENS),
                rotation_tail_declined=False,
                now=now,
            )
            queue.append(make_req(f"s{i}-t1", entered=now))
        ntp.tick(waiting_queue=queue, now=now)
        for i in range(20, 60):
            cache.park(queue[i].cache_request_handle, start=0, tokens=KEY_TOKENS)
        if read_line:
            measure_lags(cache)
        ticks = 300
        start = time.perf_counter()
        for _ in range(ticks):
            now += 0.025
            ntp.tick(waiting_queue=queue, now=now)
        return (time.perf_counter() - start) / ticks * 1e6

    def test_cost(self):
        base = self.one(read_line=False, log=False)
        log = self.one(read_line=False, log=True)
        line = self.one(read_line=True, log=False)
        print(
            f"\nREVIEW tick cost (60 queued planned turns, 40 parked reads): "
            f"switches off {base:.0f} us; prices log only {log:.0f} us; READ_LINE {line:.0f} us"
        )


if __name__ == "__main__":
    unittest.main()
