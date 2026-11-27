"""Throwaway review check (logic lens): the starved-GPU rule in the tick a turn arrives."""

import unittest

from test_next_turn_prefetch import (
    PAGE,
    SHORT_HISTORY,
    RecomputeTestCase,
    hints,
    key_of,
    make_req,
)

from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle


class ReviewStarvedAtArrival(RecomputeTestCase):
    switched = (envs.SGLANG_HICACHE_NEXT_TURN_RECOMPUTE_WHEN_STARVED,)

    def plan(self, session, gap_s):
        payload = {"session": session, "next_rid": f"{session}-t1", "next_gap_s": gap_s}
        self.ntp.on_turn_committed(
            req=make_req(f"{session}-t0", kv_hints=hints("-", payload=payload)),
            key=key_of(SHORT_HISTORY),
            rotation_tail_declined=False,
            now=0.0,
        )
        return CacheRequestHandle(f"{session}-t1", 0)

    def run_case(self, arrive_at_tick):
        a = self.plan("a", 0.5)
        x = self.plan("x", 5.0)
        staged = [self.plan(f"p{i}", 2.0) for i in range(3)]
        # a queued since 1.0 with its read on the read thread: the queue is starved.
        self.cache.ongoing.add(a)
        self.cache.spans[a] = (0, SHORT_HISTORY)
        # Three staged reads due at 2.1 s wait for the window, 300 pages each.
        for handle in staged:
            self.cache.park(handle, start=0, tokens=300 * PAGE)
        # x's staged read: 126 pages behind 40K tokens (607 ms to recompute, 345 ms to read).
        self.cache.park(x, start=40_000, tokens=SHORT_HISTORY)
        queue = [make_req("a-t1", entered=1.0)]
        out = []

        def tick(now):
            found = self.events(lambda: self.ntp.tick(waiting_queue=list(queue), now=now))
            out.append(
                (now, [(f["rid"], f["land_ms"], f["rec_ms"]) for kind, f in found if kind == "ntp_starve"])
            )

        tick(1.0)
        tick(1.4)
        if arrive_at_tick == "first":
            # x arrives 3.5 s before its predicted arrival and joins its staged read.
            queue.append(make_req("x-t1", entered=1.6))
            tick(1.6)
            tick(1.625)
        else:
            # The same, but x was already registered as arrived one tick before the rule can fire.
            queue.append(make_req("x-t1", entered=1.45))
            tick(1.45)
            tick(1.6)
            tick(1.625)
        return out, self.cache.read_scheduler.is_parked(x)

    def test_arrival_tick(self):
        out, still_parked = self.run_case("first")
        print("\nREVIEW starved at arrival: x arrives in the tick the rule fires:")
        for row in out:
            print("   ", row)
        print("    x's read still parked:", still_parked)

    def test_registered_before(self):
        out, still_parked = self.run_case("earlier")
        print("\nREVIEW starved at arrival: x registered one tick earlier:")
        for row in out:
            print("   ", row)
        print("    x's read still parked:", still_parked)


if __name__ == "__main__":
    unittest.main()
