"""Oracle next-turn prefetch estimators: arrival offset, admission rate, queue wait."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import random

from sglang.srt.mem_cache.next_turn_estimators import (
    RATE_FALLBACK_WINDOW_S,
    RATE_WINDOW_S,
    AdmissionRate,
    ArrivalOffset,
    Rate,
    residual_wait,
    wait_at_arrival,
)
from sglang.srt.mem_cache.next_turn_policy import (
    MIN_REMAINING_POINTS,
    GapDistribution,
    PriceController,
    read_start,
    remaining_gaps,
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


def expected_cost(remaining, *, start, read_s, wait_s, rent_per_s):
    """E[stall] + rent * E[seconds pinned] of a read started `start` seconds from now,
    written out per support point."""
    total = 0.0
    for gap in remaining:
        if gap <= start:
            total += max(0.0, read_s - wait_s)
            continue
        land, due = start + read_s, gap + wait_s
        total += max(0.0, land - due) + rent_per_s * max(0.0, due - land)
    return total / len(remaining)


def random_entry(rng):
    p50 = rng.uniform(0.5, 120.0)
    return GapDistribution.from_quantiles(
        p10=p50 / rng.uniform(1.02, 20.0), p50=p50, p90=p50 * rng.uniform(1.02, 20.0)
    )


class TestGapDistribution(CustomTestCase):
    def test_two_piece_fit_passes_through_the_three_quantiles(self):
        """Each side of the median has its own sigma, so an entry as skewed as pip's
        (p10 0.85 s, p50 22 s) still puts 10 % below its p10 and 10 % above its p90."""
        dist = GapDistribution.from_quantiles(p10=0.85, p50=22.0, p90=58.0)
        for gap, above in ((0.85, 0.9), (22.0, 0.5), (58.0, 0.1)):
            self.assertAlmostEqual(dist.survival(gap), above, places=6)
        for q, gap in ((0.1, 0.85), (0.5, 22.0), (0.9, 58.0)):
            self.assertAlmostEqual(dist.quantile(q), gap, places=6)

    def test_far_tail_keeps_a_chance_of_return(self):
        """Eviction ranks sessions idle far past their p90; a survival that rounded to
        zero there would score them all alike."""
        dist = GapDistribution.from_quantiles(p10=6.9, p50=7.3, p90=13.0)
        self.assertGreater(dist.survival(300.0), 0.0)
        self.assertGreater(dist.chance_within(idle_s=300.0, horizon_s=30.0), 0.0)

    def test_remaining_gaps_drops_an_entry_down_to_its_last_points(self):
        """The last support points alone read as a near-certain return; a point hint
        has one point by construction and keeps it."""
        support = GapDistribution.from_quantiles(p10=2.0, p50=7.0, p90=17.0).support()
        idle = support[-MIN_REMAINING_POINTS]
        self.assertEqual(
            len(remaining_gaps(support=support, idle_s=idle - 1e-6)),
            MIN_REMAINING_POINTS,
        )
        self.assertEqual(remaining_gaps(support=support, idle_s=idle), ())
        point = GapDistribution.point(27.7).support()
        self.assertEqual(remaining_gaps(support=point, idle_s=20.0), (27.7 - 20.0,))
        self.assertEqual(remaining_gaps(support=point, idle_s=27.7), ())


class TestReadStart(CustomTestCase):
    def test_start_is_the_cheapest_on_a_fine_grid(self):
        """The search looks only at the starts where a support point comes due as the
        read lands; the cost is piecewise linear, so no other start may be cheaper."""
        rng = random.Random(0)
        for _ in range(200):
            dist = random_entry(rng)
            remaining = remaining_gaps(
                support=dist.support(),
                idle_s=rng.choice((0.0, rng.uniform(0.0, 2.0 * dist.median_s))),
            )
            args = dict(
                read_s=rng.uniform(0.3, 8.0),
                wait_s=rng.choice((0.0, rng.uniform(0.0, 6.0))),
                rent_per_s=rng.choice((0.04, 0.4, 3.2, 12.8)),
            )
            start = read_start(remaining=remaining, **args)
            at_arrival = max(0.0, args["read_s"] - args["wait_s"])
            if not remaining or at_arrival == 0.0:
                self.assertIsNone(start)
                continue
            grid = [k * remaining[-1] / 500 for k in range(501)]
            cheapest = min(expected_cost(remaining, start=s, **args) for s in grid)
            if start is None:
                self.assertLessEqual(at_arrival, cheapest + 1e-9)
            else:
                cost = expected_cost(remaining, start=start, **args)
                self.assertLessEqual(cost, cheapest + 1e-9)
                self.assertLess(cost, at_arrival)

    def test_exact_hint_starts_one_read_before_the_turn_is_due(self):
        """With a point hint any positive price gives the just-in-time start; free
        room reads the history back at once."""
        remaining = remaining_gaps(support=GapDistribution.point(27.7).support(), idle_s=5.0)
        for rent in (0.004, 0.4, 12.8):
            self.assertAlmostEqual(
                read_start(remaining=remaining, read_s=2.0, wait_s=0.5, rent_per_s=rent),
                27.7 - 5.0 + 0.5 - 2.0,
            )
        self.assertEqual(
            read_start(remaining=remaining, read_s=2.0, wait_s=0.5, rent_per_s=0.0), 0.0
        )

    def test_sharp_entry_accepts_the_newsvendor_stall_risk(self):
        """With no queue the read starts where the chance of the turn being due before
        it lands is q = rent / (1 + rent): 0.29 for 8K tokens and 0.87 for 128K at a
        price of 5e-5, so larger reads land later in the entry."""
        dist = GapDistribution.from_quantiles(p10=17.5, p50=20.0, p90=22.6)
        for tokens in (8000, 32000, 64000, 128000):
            rent = 5e-5 * tokens
            read_s = tokens * 98304 / 2.11e9
            start = read_start(
                remaining=dist.support(), read_s=read_s, wait_s=0.0, rent_per_s=rent
            )
            risk = dist.chance_within(idle_s=start, horizon_s=read_s)
            self.assertAlmostEqual(risk, rent / (1.0 + rent), delta=0.06)

    def test_wide_entry_and_covering_queue_read_at_arrival(self):
        """An early read for pip (p10 0.85 s, p50 22 s) would sit pinned through the
        long mode, and a queue wait longer than the read hides the read anyway."""
        wide = GapDistribution.from_quantiles(p10=0.85, p50=22.0, p90=58.0).support()
        start = read_start(remaining=wide, read_s=3.05, wait_s=0.0, rent_per_s=3.2)
        # Nothing starts inside the entry's bulk; a start among its last points is
        # dropped by remaining_gaps before the session has been idle that long.
        self.assertGreater(start, wide[-MIN_REMAINING_POINTS])
        sharp = GapDistribution.from_quantiles(p10=17.5, p50=20.0, p90=22.6).support()
        self.assertIsNone(
            read_start(remaining=sharp, read_s=3.05, wait_s=3.05, rent_per_s=3.2)
        )


class TestPriceController(CustomTestCase):
    def test_price_follows_refused_against_idle_room_within_its_bounds(self):
        """Refused reads raise the price and idle room lowers it, by the same factor
        for the same excess; nothing moves before a full interval was observed."""
        price = PriceController(price=5e-5, rule="room")
        price.observe(now=0.0, idle_tokens=0)
        price.refuse(22000)
        price.observe(now=4.0, idle_tokens=0)
        self.assertIsNone(price.step(room_tokens=44000))
        price.observe(now=5.0, idle_tokens=0)
        up = price.step(room_tokens=44000)
        self.assertAlmostEqual(up.excess, 0.5)
        self.assertGreater(up.price, 5e-5)
        price.observe(now=10.0, idle_tokens=22000)
        down = price.step(room_tokens=44000)
        self.assertAlmostEqual(down.excess, -0.5)
        self.assertAlmostEqual(down.price, 5e-5)
        for stamp in range(15, 2000, 5):
            price.observe(now=float(stamp), idle_tokens=44000)
            price.step(room_tokens=44000)
        self.assertEqual(price.price, price.floor)

    def test_harm_rule_prices_staged_pins_at_what_they_cost_others(self):
        """The price is the harm charged per staged token-second, both fading: with
        nothing staged it keeps its start, pins that hurt no one pull it down over
        minutes rather than seconds, and harm lifts it again."""
        price = PriceController(price=5e-5, rule="harm")
        price.observe(now=0.0, idle_tokens=0)
        for stamp in range(5, 3000, 5):
            price.observe(now=float(stamp), idle_tokens=0)
            self.assertAlmostEqual(price.step(room_tokens=44000).price, 5e-5)
        price = PriceController(price=5e-5, rule="harm")
        price.observe(now=0.0, idle_tokens=0)
        trace = {}
        for stamp in range(5, 905, 5):
            price.observe(now=float(stamp), idle_tokens=0, staged_tokens=100_000)
            trace[stamp] = price.step(room_tokens=44000).price
        self.assertGreater(trace[10], 0.8 * 5e-5)
        self.assertLess(trace[900], 0.02 * 5e-5)
        price.charge(600.0)
        price.observe(now=905.0, idle_tokens=0, staged_tokens=100_000)
        self.assertGreater(price.step(room_tokens=44000).price, 10 * trace[900])

    def test_a_fixed_price_is_reported_as_given(self):
        """Only a learned price is held to the controller's range; a fixed one is
        what the operator set, zero included."""
        for given in (0.0, 5e-3):
            price = PriceController(price=given)
            price.observe(now=0.0, idle_tokens=0)
            price.refuse(10**6)
            price.observe(now=6.0, idle_tokens=0)
            self.assertEqual(price.step(room_tokens=44000).price, given)


if __name__ == "__main__":
    unittest.main()
