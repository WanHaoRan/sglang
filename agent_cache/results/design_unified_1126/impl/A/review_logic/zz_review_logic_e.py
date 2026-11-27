"""Throwaway review check (logic lens): the starved-GPU rule after its first drop, when
admission takes the dropped turn before the next tick."""

import unittest

from test_next_turn_prefetch import (
    PAGE,
    SHORT_HISTORY,
    RecomputeTestCase,
    make_req,
)

from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle


class ReviewStarvedBurst(RecomputeTestCase):
    switched = (envs.SGLANG_HICACHE_NEXT_TURN_RECOMPUTE_WHEN_STARVED,)

    def test_consecutive_passes(self):
        names = ["a", "b", "c", "d", "e", "f"]
        handles = {name: CacheRequestHandle(f"{name}-t1", 0) for name in names}
        for name in names:
            self.plan_turn(name)
        # a's read is on the read thread; the others' reads are parked behind 5 s of reads.
        self.cache.ongoing.add(handles["a"])
        self.cache.spans[handles["a"]] = (0, SHORT_HISTORY)
        for name in names[1:]:
            self.cache.park(handles[name], start=0, tokens=SHORT_HISTORY)
        self.cache.backlog_pages = 2000
        queue = [make_req(f"{name}-t1", entered=1.0) for name in names]

        def tick(now):
            found = self.events(lambda: self.ntp.tick(waiting_queue=list(queue), now=now))
            return [fields["rid"] for kind, fields in found if kind == "ntp_starve"]

        log = []
        log.append((1.0, tick(1.0)))
        log.append((1.4, tick(1.4)))
        now = 1.6
        for _ in range(6):
            dropped = tick(now)
            log.append((round(now, 3), dropped))
            # Admission takes a turn with nothing to wait on within the same pass.
            for rid in dropped:
                req = next(r for r in queue if r.rid == rid)
                req.time_stats.forward_entry_time = now + 0.001
                queue.remove(req)
            now += 0.025
        print("\nREVIEW starved rule, (tick time, turns whose read was dropped):")
        for row in log:
            print("   ", row)


if __name__ == "__main__":
    unittest.main()
