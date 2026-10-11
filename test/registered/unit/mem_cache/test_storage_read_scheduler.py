"""Deadline-ordered dispatch of parked storage reads (agent_cache/ORACLE_V2_PLAN.md 4.2)."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

import unittest

from sglang.srt.mem_cache.storage_read_scheduler import StorageReadScheduler
from sglang.test.test_utils import CustomTestCase

PAGE = 64


def scheduler(window_pages=100, budget_tokens=10**9):
    return StorageReadScheduler(
        window_pages=window_pages, budget_tokens=budget_tokens, page_size=PAGE
    )


class TestStorageReadScheduler(CustomTestCase):
    def dispatch(self, srs, deadlines, *, now=100.0, in_flight=0, held=0):
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


if __name__ == "__main__":
    unittest.main()
