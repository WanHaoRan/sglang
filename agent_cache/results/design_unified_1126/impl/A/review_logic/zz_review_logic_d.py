"""Throwaway review check (logic lens): what one prefetcher tick costs with the switches on,
at campaign 34's saturated queue (60 queued planned turns, 27 to 43 parked reads)."""

import cProfile
import io
import pstats
import time
import unittest

from test_next_turn_prefetch import (
    KEY_TOKENS,
    PAGE,
    MeasuredCache,
    PrefetcherTestCase,
    hints,
    hold_switches,
    key_of,
    make_req,
    measure_lags,
)

from sglang.srt.environ import envs
from sglang.srt.mem_cache.next_turn_prefetch import NextTurnPrefetcher

E = envs


class ReviewTickCost(PrefetcherTestCase):
    mode = "on"

    def setUp(self):
        pass

    def build(self, switches, *, queued, parked, timing="horizon"):
        stack = hold_switches(self, *switches)
        cache = MeasuredCache()
        cache.resident = cache.matched_tokens = KEY_TOKENS
        ntp = NextTurnPrefetcher(
            mode="on",
            tree_cache=cache,
            page_size=PAGE,
            arrival_gate="on",
            arrival_hold="on",
            read_timing=timing,
        )
        stack.close()
        self.ntp = ntp
        now = 0.0
        for i in range(700):
            req = make_req(f"warm-{i}", entered=now)
            ntp.tick(waiting_queue=[req], now=now)
            req.time_stats.forward_entry_time = now + 0.01
            now += 0.4
        queue = []
        for i in range(queued):
            ntp.on_turn_committed(
                req=make_req(f"s{i}-t0", kv_hints=hints(f"s{i}-t1", 1.0)),
                key=key_of(KEY_TOKENS),
                rotation_tail_declined=False,
                now=now,
            )
            queue.append(make_req(f"s{i}-t1", entered=now))
        ntp.tick(waiting_queue=queue, now=now)
        for i in range(queued - parked, queued):
            cache.park(queue[i].cache_request_handle, start=0, tokens=KEY_TOKENS)
        if getattr(cache, "read_lags", None) is not None:
            measure_lags(cache)
        return ntp, queue, now

    def cost(self, switches, **shape):
        ntp, queue, now = self.build(switches, **shape)
        ticks = 400
        best = float("inf")
        for _ in range(3):
            start = time.perf_counter()
            for _ in range(ticks):
                now += 0.025
                ntp.tick(waiting_queue=queue, now=now)
            best = min(best, (time.perf_counter() - start) / ticks * 1e6)
        return best

    def test_cost(self):
        line = ((E.SGLANG_HICACHE_NEXT_TURN_READ_LINE, True),)
        log = ((E.SGLANG_LOG_HICACHE_PRICES_INTERVAL, 5.0),)
        unified = (
            (E.SGLANG_HICACHE_NEXT_TURN_MEASURED_PRICES, True),
            (E.SGLANG_HICACHE_NEXT_TURN_FREE_EMPTY_LOOKUPS, True),
            (E.SGLANG_HICACHE_NEXT_TURN_READ_LINE, True),
            (E.SGLANG_HICACHE_NEXT_TURN_HOLD_QUANTILE, True),
            (E.SGLANG_LOG_HICACHE_PRICES_INTERVAL, 5.0),
        )
        for queued, parked in ((60, 30), (60, 43), (120, 100)):
            shape = dict(queued=queued, parked=parked)
            print(
                f"\nREVIEW tick us, {queued} queued planned turns, {parked} parked reads: "
                f"off {self.cost((), **shape):.0f}; prices log {self.cost(log, **shape):.0f}; "
                f"READ_LINE {self.cost(line, **shape):.0f}; "
                f"unified set {self.cost(unified, **shape):.0f}"
            )

    def test_profile(self):
        ntp, queue, now = self.build(
            ((E.SGLANG_HICACHE_NEXT_TURN_READ_LINE, True),), queued=60, parked=43
        )
        profile = cProfile.Profile()
        profile.enable()
        for _ in range(300):
            now += 0.025
            ntp.tick(waiting_queue=queue, now=now)
        profile.disable()
        out = io.StringIO()
        pstats.Stats(profile, stream=out).sort_stats("cumulative").print_stats(12)
        print("\nREVIEW profile (300 ticks, READ_LINE, 60 queued, 43 parked):")
        for row in out.getvalue().splitlines():
            if "function calls" in row or "ncalls" in row or ".py" in row:
                print("   ", row.strip()[:150])


if __name__ == "__main__":
    unittest.main()
