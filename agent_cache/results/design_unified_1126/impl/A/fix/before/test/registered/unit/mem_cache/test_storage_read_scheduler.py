"""Deadline-ordered dispatch of parked storage reads (agent_cache/ORACLE_V2_PLAN.md 4.2)."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

import unittest

from sglang.srt.mem_cache.storage_read_scheduler import StorageReadScheduler
from sglang.test.test_utils import CustomTestCase

PAGE = 64


def scheduler(window_pages=100, budget_tokens=10**9, write_hold_tokens=0):
    return StorageReadScheduler(
        window_pages=window_pages,
        budget_tokens=budget_tokens,
        page_size=PAGE,
        write_hold_tokens=write_hold_tokens,
    )


class TestStorageReadScheduler(CustomTestCase):
    def dispatch(
        self, srs, deadlines, *, now=100.0, in_flight=0, held=0, write_backlog=0
    ):
        """One pass as the cache runs it: the write hold's update, then dispatch."""
        srs.update_write_hold(write_backlog)
        return [
            d.read.handle
            for d in srs.dispatch(
                now=now,
                deadline_of=lambda h: deadlines.get(h),
                in_flight_pages=in_flight,
                held_tokens=held,
            )
        ]

    def test_now_reads_go_first_then_earliest_deadline_then_park_order(self):
        srs = scheduler()
        for handle in ("late", "early", "now", "early-twin"):
            srs.park(handle=handle, operation=handle, pages=10, now=0.0)
        order = self.dispatch(
            srs, {"late": 300.0, "early": 120.0, "early-twin": 120.0, "now": None}
        )
        self.assertEqual(order, ["now", "early", "early-twin", "late"])
        self.assertEqual(srs.parked_count(), 0)

    def test_window_holds_timed_reads_but_not_now_reads(self):
        srs = scheduler(window_pages=100)
        srs.park(handle="a", operation=None, pages=60, now=0.0)
        srs.park(handle="b", operation=None, pages=60, now=0.0)
        srs.park(handle="c", operation=None, pages=60, now=0.0)
        srs.park(handle="urgent", operation=None, pages=60, now=0.0)
        deadlines = {"a": 10.0, "b": 20.0, "c": 30.0, "urgent": None}
        # The "now" read goes first and still counts against the 100-page
        # window (0 -> 60); one timed read fits (-> 120); the next two wait.
        self.assertEqual(self.dispatch(srs, deadlines), ["urgent", "a"])
        self.assertEqual(srs.parked_handles(), ["b", "c"])
        self.assertEqual(srs.stats.window_holds, 1)
        # With the thread still full nothing timed moves, while a new "now" read does.
        srs.park(handle="urgent2", operation=None, pages=60, now=5.0)
        deadlines["urgent2"] = None
        self.assertEqual(self.dispatch(srs, deadlines, in_flight=120), ["urgent2"])
        self.assertEqual(self.dispatch(srs, deadlines, in_flight=0), ["b", "c"])

    def test_budget_charges_each_read_once_on_top_of_the_held_tokens(self):
        """held_tokens already contains the in-flight reads (the cache charges
        them at dispatch), so in_flight_pages must not be added to the budget
        again; it only bounds the window."""
        srs = scheduler(window_pages=10**6, budget_tokens=3 * 10 * PAGE)
        for handle in ("a", "b", "c"):
            srs.park(handle=handle, operation=None, pages=10, now=0.0)
        deadlines = {"a": 1.0, "b": 2.0, "c": 3.0}
        # One read's worth held (and in flight) leaves room for two more.
        self.assertEqual(
            self.dispatch(srs, deadlines, in_flight=10, held=10 * PAGE), ["a", "b"]
        )
        self.assertEqual(srs.stats.budget_holds, 1)
        # Freed budget lets the last one go.
        self.assertEqual(self.dispatch(srs, deadlines, held=0), ["c"])

    def test_a_held_read_is_passed_over_and_its_pages_stay_ahead_of_the_next(self):
        """Unlike a full window, a hold must not block the due read behind it. The held
        read still goes first once it starts, so the next estimate has to count it."""
        srs = scheduler(window_pages=100)
        for handle in ("held", "due", "last"):
            srs.park(handle=handle, operation=None, pages=10, now=0.0)
        deadlines = {"held": 110.0, "due": 120.0, "last": 130.0}
        hold = {"held": 150.0}
        ahead = {}

        def not_before(handle, read, ahead_pages):
            ahead[handle] = ahead_pages
            return hold.get(handle)

        def dispatch(now, in_flight=5):
            return [
                d.read.handle
                for d in srs.dispatch(
                    now=now,
                    deadline_of=deadlines.get,
                    in_flight_pages=in_flight,
                    held_tokens=0,
                    not_before_of=not_before,
                )
            ]

        self.assertEqual(dispatch(100.0), ["due", "last"])
        self.assertEqual(ahead, {"held": 5, "due": 15, "last": 25})
        self.assertEqual(srs.parked_handles(), ["held"])
        self.assertEqual(srs.stats.hold_skips, 1)
        # Past its start time the read still respects the window, like any timed read.
        self.assertEqual(dispatch(150.0, in_flight=100), [])
        self.assertEqual(srs.stats.window_holds, 1)
        self.assertEqual(dispatch(150.0), ["held"])

    def test_the_line_ahead_of_a_read_is_the_now_reads_and_the_timed_ones_due_no_later(self):
        """Dispatch serves "now" reads before any timed one and timed ones by deadline, so
        that is the line a read waits in. Counting every parked read makes an early read
        wait for late ones; counting none is how a planned 8 s read took 30 to 52 s."""
        srs = scheduler()
        for handle, pages in (("now", 1), ("early", 2), ("mid", 4), ("late", 8)):
            srs.park(handle=handle, operation=None, pages=pages, now=0.0)
        deadlines = {"now": None, "early": 110.0, "mid": 120.0, "late": 130.0}
        self.assertEqual(
            [
                srs.pages_due_by(deadline, deadline_of=deadlines.get)
                for deadline in (None, 100.0, 120.0, 500.0)
            ],
            [1, 1, 1 + 2 + 4, 15],
        )

    def test_ahead_of_a_parked_read_are_those_dispatch_takes_first(self):
        """A read already parked waits for the reads ordered before it, not for one of
        the same deadline parked after it: counted, that one makes the read look later
        than it is and its turn give it up for a recompute."""
        srs = scheduler()
        for handle, pages in (("now", 1), ("mid", 2), ("mid-twin", 4), ("early", 8)):
            srs.park(handle=handle, operation=None, pages=pages, now=0.0)
        deadlines = {"now": None, "mid": 120.0, "mid-twin": 120.0, "early": 110.0}
        ahead = {
            handle: srs.pages_ahead_of(handle, deadline_of=deadlines.get)
            for handle in deadlines
        }
        self.assertEqual(ahead, {"now": 0, "early": 1, "mid": 1 + 8, "mid-twin": 1 + 8 + 2})
        order = [
            d.read.handle
            for d in srs.dispatch(
                now=100.0, deadline_of=deadlines.get, in_flight_pages=0, held_tokens=0
            )
        ]
        self.assertEqual(order, sorted(ahead, key=ahead.get))

    def test_write_hold_stops_timed_reads_from_the_threshold_until_half_of_it(self):
        """Timed reads let through while SSD writes still lock host pages find no room.
        The hold has memory both ways (60 neither starts nor ends it); a "now" read,
        which an admission waits on, is never held."""
        srs = scheduler(write_hold_tokens=100)
        srs.park(handle="first", operation=None, pages=1, now=0.0)
        self.assertEqual(
            self.dispatch(srs, {"first": 110.0}, write_backlog=60), ["first"]
        )
        for handle in ("late", "early", "now"):
            srs.park(handle=handle, operation=None, pages=1, now=0.0)
        deadlines = {"late": 300.0, "early": 120.0, "now": None}
        self.assertEqual(self.dispatch(srs, deadlines, write_backlog=100), ["now"])
        self.assertEqual(srs.stats.write_holds, 1)
        self.assertEqual(self.dispatch(srs, deadlines, write_backlog=60), [])
        self.assertEqual(
            self.dispatch(srs, deadlines, write_backlog=50), ["early", "late"]
        )

    def test_write_hold_off_ignores_any_write_backlog(self):
        """0 is the default and must leave dispatch as it was: off, not a threshold
        that every pending write reaches."""
        srs = scheduler()
        srs.park(handle="a", operation=None, pages=1, now=0.0)
        self.assertEqual(self.dispatch(srs, {"a": 110.0}, write_backlog=10**9), ["a"])

    def test_discard_clear_and_duplicate_park(self):
        srs = scheduler()
        srs.park(handle="a", operation=None, pages=1, now=0.0)
        self.assertTrue(srs.is_parked("a"))
        with self.assertRaises(ValueError):
            srs.park(handle="a", operation=None, pages=1, now=0.0)
        self.assertTrue(srs.discard("a"))
        self.assertFalse(srs.discard("a"))
        srs.park(handle="b", operation=None, pages=4, now=0.0)
        self.assertEqual(srs.parked_pages(), 4)
        srs.clear()
        self.assertEqual((srs.parked_count(), srs.parked_pages()), (0, 0))

    def test_wait_is_measured_from_parking(self):
        srs = scheduler()
        srs.park(handle="a", operation=None, pages=1, now=10.0)
        decisions = srs.dispatch(
            now=13.5, deadline_of=lambda h: None, in_flight_pages=0, held_tokens=0
        )
        self.assertAlmostEqual(decisions[0].waited_s, 3.5)
        self.assertAlmostEqual(srs.stats.max_wait_s, 3.5)
        self.assertEqual(srs.stats.dispatched_now, 1)

    def test_rejects_bad_limits(self):
        with self.assertRaises(ValueError):
            StorageReadScheduler(window_pages=0, budget_tokens=1, page_size=PAGE)
        # A negative write hold would otherwise pass silently as off.
        with self.assertRaises(ValueError):
            scheduler(write_hold_tokens=-1)


if __name__ == "__main__":
    unittest.main()
