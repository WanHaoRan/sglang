"""Oracle next-turn prefetch estimators: arrival offset, admission rate, queue wait."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.mem_cache.next_turn_estimators import (
    RATE_FALLBACK_WINDOW_S,
    RATE_WINDOW_S,
    AdmissionRate,
    ArrivalOffset,
    Rate,
    residual_wait,
    wait_at_arrival,
)
from sglang.test.test_utils import CustomTestCase


def admit(rate: AdmissionRate, *stamps: float, wait_s=1.0, storage_hit=False):
    for stamp in stamps:
        rate.record_admission(now=stamp, wait_s=wait_s, storage_hit=storage_hit)


class TestArrivalOffset(CustomTestCase):
    def test_outliers_are_dropped_not_clipped(self):
        """A client stall or an arrival before commit + gap leaves the estimate as it
        was; an in-range sample still moves it."""
        offset = ArrivalOffset()
        seed = offset.seconds
        self.assertFalse(offset.observe(138.0))
        self.assertFalse(offset.observe(-0.05))
        self.assertEqual((offset.seconds, offset.discarded), (seed, 2))
        self.assertTrue(offset.observe(0.9))
        self.assertGreater(offset.seconds, seed)


class TestAdmissionRate(CustomTestCase):
    def test_quiet_minute_falls_back_to_five_minutes_then_to_none(self):
        rate = AdmissionRate()
        admit(rate, 0.0, 10.0, 20.0, 30.0, 40.0)
        self.assertEqual(rate.rate(50.0), Rate(per_s=5 / 60, window_s=RATE_WINDOW_S))
        # Only 3 admissions in the last minute: count all 5 over five minutes.
        self.assertEqual(
            rate.rate(75.0), Rate(per_s=5 / 300, window_s=RATE_FALLBACK_WINDOW_S)
        )
        # The first admission has left the five-minute window: 4 are too few.
        self.assertIsNone(rate.rate(300.5))
        # Recording prunes what no window can count, so the log stays bounded.
        admit(rate, 400.0)
        self.assertEqual(list(rate.stamps), [400.0])

    def test_no_admissions_gives_no_estimate_instead_of_dividing_by_zero(self):
        self.assertIsNone(AdmissionRate().rate(10.0))
        self.assertIsNone(residual_wait(position=3, rate=None))
        self.assertEqual(
            wait_at_arrival(queue_len=4, rate=None, wait_ewma=None), (None, None)
        )
        # An empty queue needs no rate: the turn is admitted on the next pass.
        self.assertEqual(
            wait_at_arrival(queue_len=0, rate=None, wait_ewma=None), (0.0, 0.0)
        )

    def test_storage_hits_count_as_admissions_but_not_as_waits(self):
        """A request credited with its own SSD read waited for that read; its wait
        would teach the estimator the very delay the oracle removes."""
        rate = AdmissionRate()
        admit(rate, 1.0, 2.0, 3.0, 4.0, 5.0, wait_s=100.0, storage_hit=True)
        self.assertIsNone(rate.wait_ewma)
        self.assertIsNotNone(rate.rate(6.0))
        admit(rate, 6.0, wait_s=2.0)
        self.assertEqual(rate.wait_ewma, 2.0)


if __name__ == "__main__":
    unittest.main()
