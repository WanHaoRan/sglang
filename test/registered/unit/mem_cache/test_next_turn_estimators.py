"""Oracle next-turn prefetch estimators: arrival offset, admission rate, queue wait."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import random

from sglang.srt.mem_cache.next_turn_estimators import (
    LAG_MIN_SAMPLES,
    RATE_FALLBACK_WINDOW_S,
    RATE_WINDOW_S,
    AdmissionRate,
    ArrivalOffset,
    Rate,
    ReadPathLags,
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
from sglang.srt.mem_cache.next_turn_prices import (
    FRONTIER_STALE_S,
    LAM_CEILING,
    MIN_ERROR_SAMPLES,
    Q_MAX,
    Q_MIN,
    AdmissionErrorTracker,
    DecayedMean,
    PrefillCostModel,
    PriceBoard,
    landing_quantile,
    quote_recompute,
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


class TestReadPathLags(CustomTestCase):
    def test_a_lag_is_published_only_after_the_minimum_samples(self):
        """Until then the load model keeps its seed: one early sample, taken while the
        read thread warms up, must not set every read-time estimate."""
        lags = ReadPathLags()
        for _ in range(LAG_MIN_SAMPLES - 1):
            lags.lookup.observe(0.02)
        self.assertIsNone(lags.lookup.seconds)
        lags.lookup.observe(0.02)
        self.assertAlmostEqual(lags.lookup.seconds, 0.02)
        self.assertIsNone(lags.fixed.seconds)

    def test_a_timed_read_waits_what_is_left_of_a_write_hold_of_the_usual_length(self):
        """A read staged while timed reads are write-held waits out the hold before its
        own line; nothing is added outside a hold or once it has outlasted the usual."""
        lags = ReadPathLags()
        for hold in range(LAG_MIN_SAMPLES):
            self.assertEqual(lags.write_hold_left_s(100.0 * hold + 1.0), 0.0)
            lags.note_write_hold(held=True, now=100.0 * hold)
            lags.note_write_hold(held=False, now=100.0 * hold + 4.0)
        self.assertEqual(lags.write_hold_left_s(900.0), 0.0)
        lags.note_write_hold(held=True, now=1000.0)
        self.assertAlmostEqual(lags.write_hold_left_s(1001.0), 3.0)
        self.assertEqual(lags.write_hold_left_s(1009.0), 0.0)


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


class TestStallWeight(CustomTestCase):
    """read_start with a second of stall weighed by c instead of 1."""

    def test_weighted_start_is_the_cheapest_on_a_fine_grid(self):
        """c x stall + rent x pinned is c times the unweighted cost at rent / c, so the
        unweighted helper prices every start; a weight applied to one of the two stall
        terms only (arrival, or late landing) would pick a start the grid beats."""
        rng = random.Random(4)
        for _ in range(200):
            dist = random_entry(rng)
            remaining = remaining_gaps(support=dist.support(), idle_s=0.0)
            weight = rng.choice((0.05, 0.2, 0.7))
            read_s, wait_s = (
                rng.uniform(0.3, 8.0),
                rng.choice((0.0, rng.uniform(0.0, 6.0))),
            )
            rent = rng.choice((0.04, 0.4, 3.2))
            start = read_start(
                remaining=remaining,
                read_s=read_s,
                wait_s=wait_s,
                rent_per_s=rent,
                stall_weight=weight,
            )

            def cost(at):
                return weight * expected_cost(
                    remaining,
                    start=at,
                    read_s=read_s,
                    wait_s=wait_s,
                    rent_per_s=rent / weight,
                )

            at_arrival = weight * max(0.0, read_s - wait_s)
            if at_arrival == 0.0:
                self.assertIsNone(start)
                continue
            grid = [k * remaining[-1] / 500 for k in range(501)]
            cheapest = min(cost(at) for at in grid)
            if start is None:
                self.assertLessEqual(at_arrival, cheapest + 1e-9)
            else:
                self.assertLessEqual(cost(start), cheapest + 1e-9)
                self.assertLess(cost(start), at_arrival)

    def test_free_lateness_never_reads_early(self):
        """With ready turns always queued a late read delays nobody, so no rent is worth
        paying, however much longer than the queue wait the read line makes the read: with
        the line counted and the weight left at 1, 349 of 1,525 decisions of campaign 34's
        saturated phase turned into reads started at once."""
        for p10, p50, p90 in (
            (0.414, 1.692, 16.795),
            (3.084, 96.228, 1040.492),
            (18.0, 20.0, 22.0),
        ):
            support = GapDistribution.from_quantiles(
                p10=p10, p50=p50, p90=p90
            ).support()
            self.assertIsNone(
                read_start(
                    remaining=remaining_gaps(support=support, idle_s=0.0),
                    read_s=45.0,
                    wait_s=7.0,
                    rent_per_s=1e-6,
                    stall_weight=0.0,
                )
            )

    def test_a_smaller_weight_never_starts_a_read_earlier(self):
        """As the GPU saturates c falls, and reads must move later or to arrival; a weight
        on the rent instead of the stall would start them earlier."""
        rng = random.Random(1)
        for _ in range(200):
            remaining = remaining_gaps(support=random_entry(rng).support(), idle_s=0.0)
            args = dict(
                remaining=remaining,
                read_s=rng.uniform(2.0, 30.0),
                wait_s=0.0,
                rent_per_s=rng.uniform(0.01, 1.0),
            )
            starts = [
                read_start(stall_weight=weight, **args) for weight in (1.0, 0.5, 0.1)
            ]
            as_time = [float("inf") if start is None else start for start in starts]
            self.assertEqual(as_time, sorted(as_time))


class TestPriceBoard(CustomTestCase):
    def queue(self, board, *, ready, now, sessions=0, rate=0.0):
        board.observe_queue(
            ready_waiting=ready, sessions_waiting=sessions, arrival_rate=rate, now=now
        )

    def test_c_is_the_share_of_admissions_that_left_no_ready_turn_waiting(self):
        """Every admission of a pass counts: a mean whose weight follows the time since the
        last sample would keep the first admission of a pass and drop the rest."""
        board = PriceBoard()
        self.assertEqual(board.c(now=0.0), 1.0)
        board.observe_admissions(count=30, ready_left=False, now=1.0)
        board.observe_admissions(count=90, ready_left=True, now=1.0)
        self.assertAlmostEqual(board.c(now=1.0), 31 / 121)
        board.observe_admissions(count=0, ready_left=True, now=1.0)
        self.assertAlmostEqual(board.c(now=1.0), 31 / 121)

    def test_c_forgets_a_saturated_episode(self):
        """Once the queue has drained, what was seen while it stood must fade within
        minutes, or early reads stay off for the rest of the run."""
        board = PriceBoard()
        board.observe_admissions(count=200, ready_left=True, now=0.0)
        self.assertLess(board.c(now=1.0), 0.02)
        for step in range(1, 10):
            board.observe_admissions(count=2, ready_left=False, now=30.0 * step)
        self.assertGreater(board.c(now=300.0), 0.9)

    def test_mu_at_light_load_is_one_plus_the_time_average_of_ready_turns(self):
        """Passes are 20 ms to seconds apart, so each count stands for the time until the
        next pass; an average over passes would weigh a burst of short passes like minutes."""
        board = PriceBoard(ready=DecayedMean(half_life_s=1e9))
        self.queue(board, ready=10, now=0.0)
        self.queue(board, ready=0, now=9.0)  # 10 turns waited for 9 s
        self.queue(board, ready=0, now=10.0)  # none for 1 s
        self.assertAlmostEqual(board.mu(now=10.0), 1.0 + 9.0, places=2)

    def test_at_saturation_a_gpu_second_delays_every_session_not_running(self):
        """Measured: 91-109 TTFT-seconds per GPU-second where 31 ready turns waited
        (campaigns 32 and 34), the users in session minus the turns running. Read off the
        ready queue there, mu prices a recompute three times too low; at light load a job
        also delays the turns that arrive while it runs."""
        board = PriceBoard(ready=DecayedMean(half_life_s=1e9))
        self.queue(board, ready=30, sessions=109, rate=2.6, now=0.0)
        self.queue(board, ready=30, sessions=109, rate=2.6, now=10.0)
        self.assertAlmostEqual(board.mu(now=10.0), 31.0, places=2)
        self.assertAlmostEqual(
            board.mu_for(job_s=0.5, now=10.0), 31.0 + 2.6 * 0.5 / 2, places=2
        )
        board.observe_admissions(count=200, ready_left=True, now=10.0)
        self.assertEqual(board.mu(now=10.0), 109.0)
        self.assertEqual(board.mu_for(job_s=0.5, now=10.0), 109.0)
        # The sessions seen fall before the smoothed c recovers: not below the ready turns.
        self.queue(board, ready=0, sessions=2, rate=2.6, now=10.0)
        self.assertAlmostEqual(board.mu(now=10.0), 31.0, places=2)

    def test_room_is_free_until_something_is_evicted_and_again_once_evictions_stop(
        self,
    ):
        """The price is the index of the holders given up lately. Victims with no index
        (a queued turn's history, un-hinted prefixes) are tokens evicted, not a price of
        zero; and the price stays within the learned price's ceiling."""
        board = PriceBoard()
        self.assertEqual(board.lam(now=5.0), 0.0)
        board.observe_eviction(marginal_index=2e-7, tokens=4096, now=10.0)
        board.observe_eviction(marginal_index=4e-7, tokens=4096, now=10.0)
        board.observe_eviction(marginal_index=None, tokens=640, now=10.0)
        self.assertAlmostEqual(board.lam(now=10.0), 3e-7)
        self.assertEqual(board.take_evicted_tokens(), 2 * 4096 + 640)
        self.assertEqual(board.take_evicted_tokens(), 0)
        self.assertEqual(board.lam(now=10.0 + FRONTIER_STALE_S + 0.1), 0.0)
        board.observe_eviction(marginal_index=5.0, tokens=64, now=50.0)
        self.assertEqual(board.lam(now=50.0), LAM_CEILING)

    def test_kappa_prices_late_reads_at_c_and_pending_writes_at_lambda(self):
        """A second of device time delays every transfer behind it: a late read's turn at
        c, the room a pending write locks at lambda. With c near 0, as in campaign 34's
        saturated phase, late reads alone give no price."""
        board = PriceBoard()
        board.observe_eviction(marginal_index=1e-6, tokens=64, now=0.0)
        board.observe_device(late_reads=12, pending_write_tokens=50_000)
        self.assertAlmostEqual(board.kappa(now=0.0), 12.0 + 0.05)
        board.observe_admissions(count=10_000, ready_left=True, now=0.0)
        self.assertAlmostEqual(board.kappa(now=0.0), 12.0 / 10_001 + 0.05)


class TestLandingQuantile(CustomTestCase):
    def test_cheap_room_lands_early_and_a_saturated_gpu_late_within_the_clamp(self):
        """q = rent / (c + rent). At saturation both prices are near zero and their ratio
        is noise: unclamped, q near 1 releases held reads seconds after the point estimate.
        The hold must never plan to land past the median of the admission time."""
        q = landing_quantile
        self.assertEqual(q(price=0.0, tokens=20_000, lateness_weight=1.0), Q_MIN)
        self.assertAlmostEqual(
            q(price=1e-5, tokens=20_000, lateness_weight=1.0), 0.2 / 1.2
        )
        self.assertEqual(q(price=5e-5, tokens=20_000, lateness_weight=0.0), Q_MAX)
        self.assertEqual(q(price=1e-9, tokens=64, lateness_weight=0.0), Q_MIN)
        rising = [
            landing_quantile(price=price, tokens=20_000, lateness_weight=1.0)
            for price in (1e-7, 1e-6, 1e-5, 1e-4)
        ]
        self.assertEqual(rising, sorted(rising))
        falling = [
            landing_quantile(price=1e-5, tokens=20_000, lateness_weight=c)
            for c in (0.05, 0.3, 1.0)
        ]
        self.assertEqual(falling, sorted(falling, reverse=True))


class TestAdmissionErrorTracker(CustomTestCase):
    def test_margin_is_minus_the_quantile_of_the_bin_the_prediction_falls_in(self):
        """A read released at admission - load - margin lands before its turn is due with
        chance 1 - q only if the margin is minus the q-quantile of measured - predicted:
        the other sign, another bin's quantile, or a sort kept past the bin's next sample
        lands reads late where the error is wide."""
        tracker = AdmissionErrorTracker()
        for k in range(100):
            tracker.observe(predicted_wait_s=3.0, wait_s=3.0 + (k - 50) / 25.0)
            tracker.observe(predicted_wait_s=15.0, wait_s=15.0 + (k - 50) / 5.0)
        self.assertAlmostEqual(
            tracker.margin(0.1, predicted_wait_s=4.0, fallback_s=2.0), 1.6
        )
        self.assertAlmostEqual(
            tracker.margin(0.1, predicted_wait_s=12.0, fallback_s=2.0), 8.0
        )
        self.assertAlmostEqual(
            tracker.margin(0.5, predicted_wait_s=4.0, fallback_s=2.0), 0.0
        )
        for _ in range(100):
            tracker.observe(predicted_wait_s=3.0, wait_s=3.0 - 6.0)
        self.assertAlmostEqual(
            tracker.margin(0.1, predicted_wait_s=4.0, fallback_s=2.0), 6.0
        )

    def test_an_empty_bin_keeps_the_fixed_margin(self):
        """Before 30 admissions were seen at a predicted wait nothing is known of the error
        there; the errors of another bin must not stand in."""
        tracker = AdmissionErrorTracker()
        for _ in range(MIN_ERROR_SAMPLES - 1):
            tracker.observe(predicted_wait_s=3.0, wait_s=9.0)
        self.assertEqual(tracker.margin(0.1, predicted_wait_s=3.0, fallback_s=2.0), 2.0)
        tracker.observe(predicted_wait_s=3.0, wait_s=9.0)
        self.assertEqual(
            tracker.margin(0.1, predicted_wait_s=3.0, fallback_s=2.0), -6.0
        )
        self.assertEqual(
            tracker.margin(0.1, predicted_wait_s=30.0, fallback_s=2.0), 2.0
        )

    def test_readers_do_not_feed_the_estimate_of_the_gpu_turn(self):
        """A reader waits for its own landing: in campaign 34 its wait ran a median 8 to
        27 s over the queue's prediction. Counted as an error of that prediction it would
        push every hold's start earlier."""
        tracker = AdmissionErrorTracker()
        for _ in range(60):
            tracker.observe(predicted_wait_s=12.0, wait_s=12.5)
            tracker.observe(
                predicted_wait_s=12.0, wait_s=40.0, landed_after_arrival_s=39.6
            )
        self.assertAlmostEqual(tracker.quantile(0.9, predicted_wait_s=12.0), 0.5)
        self.assertAlmostEqual(tracker.reader_lag(), 0.4)
        self.assertEqual(len(tracker.pooled()), 60)


class TestPrefillCostModel(CustomTestCase):
    # FA3, cold prefills at an empty queue (campaign 33): seconds per token, per token squared.
    A, B = 13.4e-6, 0.69e-9

    def test_fit_recovers_both_terms_from_warm_and_cold_prefills(self):
        """Most admitted turns prefill a few hundred new tokens on a long cached prefix, so
        the attention term is identified through total^2 - cached^2, not through the cold
        prefills alone (4 to 7 % of the requests)."""
        rng = random.Random(2)
        model = PrefillCostModel(seed_a=1.1e-4)
        for _ in range(400):
            total = rng.randint(2_000, 60_000)
            new = total if rng.random() < 0.06 else rng.randint(64, min(total, 3_000))
            cached = total - new
            seconds = self.A * new + self.B * (total * total - cached * cached)
            model.observe(
                new_tokens=new,
                total_tokens=total,
                seconds=seconds * rng.uniform(0.95, 1.05),
            )
        a, b = model.coefficients()
        self.assertAlmostEqual(a / self.A, 1.0, delta=0.1)
        self.assertAlmostEqual(b / self.B, 1.0, delta=0.1)
        self.assertAlmostEqual(
            model.seconds(new_tokens=24_000, total_tokens=24_000),
            self.A * 24_000 + self.B * 24_000**2,
            delta=0.05,
        )

    def test_the_seed_holds_until_enough_turns_were_seen(self):
        model = PrefillCostModel(seed_a=1.1e-4)
        for _ in range(model.min_samples - 1):
            model.observe(new_tokens=1_000, total_tokens=10_000, seconds=0.03)
        self.assertEqual(model.coefficients(), (1.1e-4, 0.0))

    def test_the_fit_follows_a_change_of_cost(self):
        """A sample weighs half after 200 more: 400 prefills at half the cost bring the estimate
        to 0.30 s, where the plain mean, 0.375 s, would price the rest of the run."""
        model = PrefillCostModel(seed_a=1.1e-4)
        for seconds in (0.5, 0.25):
            for _ in range(400):
                model.observe(new_tokens=10_000, total_tokens=10_000, seconds=seconds)
        self.assertAlmostEqual(
            model.seconds(new_tokens=10_000, total_tokens=10_000), 0.3, delta=0.005
        )

    def test_one_shape_of_request_gives_a_per_token_cost_not_a_negative_term(self):
        """Samples that all look alike cannot tell the two terms apart; the fit must fall
        back to one rate, not return a negative coefficient that prices a long history
        below zero."""
        model = PrefillCostModel(seed_a=1.1e-4)
        for _ in range(50):
            model.observe(new_tokens=10_000, total_tokens=10_000, seconds=0.25)
        a, b = model.coefficients()
        self.assertGreaterEqual(b, 0.0)
        self.assertAlmostEqual(
            model.seconds(new_tokens=10_000, total_tokens=10_000), 0.25, delta=0.01
        )


class TestRecomputeQuote(CustomTestCase):
    """Read-or-recompute with the costs measured on FA3 (campaigns 33 and 34): cold prefill
    12.8 us x n + 0.71 ns x n^2; a read issued at arrival with an idle read thread keeps
    its turn 0.1 s + 45 us a token."""

    FA3 = PrefillCostModel(seed_a=12.8e-6, seed_b=0.71e-9)

    def wins(
        self, tokens, *, cost=FA3, matched=0, arrivals_per_s=0.0, admitted_behind=0
    ):
        board = PriceBoard()
        board.observe_queue(
            ready_waiting=0, sessions_waiting=0, arrival_rate=arrivals_per_s, now=0.0
        )
        board.observe_admissions(count=admitted_behind, ready_left=True, now=0.0)
        return quote_recompute(
            board=board,
            cost=cost,
            matched=matched,
            missing=tokens,
            total=matched + tokens + 500,
            read_s=0.1 + 45e-6 * tokens,
            now=0.0,
        ).wins

    def test_a_recompute_must_also_beat_the_turns_that_arrive_while_it_runs(self):
        """With nothing arriving, FA3 recomputes 24K tokens faster than it reads them. At
        2.6 arrivals a second the 0.7 s recompute delays two more turns, and only reads
        under about 16K tokens gained on the recorded runs; priced at mu = 1 the rule was
        wrong on a quarter to half of its recomputes."""
        self.assertTrue(self.wins(24_000))
        self.assertFalse(self.wins(60_000))
        self.assertTrue(self.wins(8_000, arrivals_per_s=2.6))
        self.assertFalse(self.wins(24_000, arrivals_per_s=2.6))

    def test_history_deep_in_a_long_prompt_costs_its_position_too(self):
        """Prefill is a x new + b x (total^2 - cached^2): the same 8K tokens take 148 ms
        at the start of a prompt and 600 ms behind 40K cached ones. A constant per token
        recomputes them at any depth."""
        self.assertTrue(self.wins(8_000))
        self.assertFalse(self.wins(8_000, matched=40_000))

    def test_once_ready_turns_were_left_waiting_the_read_is_kept(self):
        """Behind a queue the read is hidden by the wait, whatever the sizes say; a c that
        fell below 0.8 is the sign a queue formed in the last half minute."""
        self.assertTrue(self.wins(2_000))
        self.assertFalse(self.wins(2_000, admitted_behind=1))

    def test_at_triton_cost_the_read_wins_at_every_size(self):
        """110 us a token to recompute against 45 us to read: always reading, as the engine
        does without the switch, is the priced answer there."""
        triton = PrefillCostModel(seed_a=1.1e-4)
        for tokens in (2_000, 16_000, 60_000):
            self.assertFalse(self.wins(tokens, cost=triton))


class TestHoldingRate(CustomTestCase):
    # The two FreeInference entries (tool waits, human waits) and a sharp one.
    TOOL = GapDistribution.from_quantiles(p10=0.414, p50=1.692, p90=16.795)
    USER = GapDistribution.from_quantiles(p10=3.084, p50=96.228, p90=1040.492)
    SHARP = GapDistribution.from_quantiles(p10=18.0, p50=20.0, p90=22.0)

    def test_falling_hazard_entries_lose_value_with_idle_time(self):
        """For both trace entries the rate is the hazard and falls as the session stays
        idle, so the longest idle goes first and the eviction frontier's price falls as
        room grows. Over the 64 support points the rate jumps instead (1.72 against 0.46
        per second for the tool entry at 1 s idle)."""
        for dist in (self.TOOL, self.USER):
            rates = [
                dist.holding_rate(idle_s=idle, min_horizon_s=0.5)
                for idle in (1.0, 5.0, 30.0, 120.0, 600.0)
            ]
            self.assertEqual(rates, sorted(rates, reverse=True))

    def test_a_sharp_entry_is_worth_one_over_its_mean_until_it_is_due(self):
        """Rising hazard: the best horizon is the whole gap, so the rate is 1 / E[gap] and
        climbs as the return nears."""
        fresh = self.SHARP.holding_rate(idle_s=0.0, min_horizon_s=0.5)
        self.assertAlmostEqual(fresh, 1.0 / 20.0, delta=0.003)
        self.assertGreater(
            self.SHARP.holding_rate(idle_s=15.0, min_horizon_s=0.5), 3 * fresh
        )

    def test_no_mass_left_means_no_value(self):
        self.assertEqual(
            self.SHARP.holding_rate(idle_s=5_000.0, min_horizon_s=0.5), 0.0
        )


if __name__ == "__main__":
    unittest.main()
