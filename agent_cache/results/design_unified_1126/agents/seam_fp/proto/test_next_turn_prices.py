"""CPU tests of the seam_fp prototype (next_turn_prices.py), written the way the engine's
test/registered/unit/mem_cache/test_next_turn_estimators.py is: one case per failure mode, the docstring says
which regression turns it red. Run: <venv python> -I -B test_next_turn_prices.py  (needs msgspec only)."""

import importlib.util
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


policy = _load("ntp_policy", "/home/wanhr/sglang/python/sglang/srt/mem_cache/next_turn_policy.py")
prices = _load("next_turn_prices", os.path.join(HERE, "next_turn_prices.py"))

GapDistribution = policy.GapDistribution
TOOL = GapDistribution.from_quantiles(p10=0.414, p50=1.692, p90=16.795)
USER = GapDistribution.from_quantiles(p10=3.084, p50=96.228, p90=1040.492)
SHARP = GapDistribution.from_quantiles(p10=18.0, p50=20.0, p90=22.0)


def random_entry(rng):
    p50 = rng.uniform(0.5, 300.0)
    return GapDistribution.from_quantiles(
        p10=p50 * rng.uniform(0.05, 1.0), p50=p50, p90=p50 * rng.uniform(1.0, 20.0)
    )


class TestReadStartWeighted(unittest.TestCase):
    def test_weight_one_is_the_engines_rule(self):
        """c = 1 must leave every decision of today's read_start unchanged: the switch-off path and the
        light-load path are this one."""
        rng = random.Random(0)
        for _ in range(300):
            dist = random_entry(rng)
            remaining = policy.remaining_gaps(support=dist.support(), idle_s=rng.uniform(0.0, 60.0))
            kwargs = dict(
                remaining=remaining,
                read_s=rng.uniform(0.5, 40.0),
                wait_s=rng.uniform(0.0, 20.0),
                rent_per_s=rng.uniform(1e-3, 3.0),
            )
            self.assertEqual(prices.read_start_weighted(stall_weight=1.0, **kwargs), policy.read_start(**kwargs))

    def test_the_working_trees_read_start_is_the_same_rule_when_it_takes_the_weight(self):
        """On 2026-11-26 18:23Z the working tree's read_start gained stall_weight (uncommitted, not by this
        check). Where it is present it must equal this prototype for every weight; skipped at a tree without it."""
        import inspect

        if "stall_weight" not in inspect.signature(policy.read_start).parameters:
            self.skipTest("read_start has no stall_weight in this tree")
        rng = random.Random(3)
        for _ in range(300):
            dist = random_entry(rng)
            remaining = policy.remaining_gaps(support=dist.support(), idle_s=rng.uniform(0.0, 60.0))
            kwargs = dict(
                remaining=remaining,
                read_s=rng.uniform(0.5, 40.0),
                wait_s=rng.uniform(0.0, 20.0),
                rent_per_s=rng.uniform(1e-3, 3.0),
                stall_weight=rng.choice((0.0, 0.01, 0.2, 0.7, 1.0)),
            )
            self.assertEqual(prices.read_start_weighted(**kwargs), policy.read_start(**kwargs))

    def test_free_lateness_never_reads_early(self):
        """With ready work always queued a late read costs the sum nothing, so no rent is worth paying: any
        start must come out as 'arrival', also when the read line makes r far larger than W (the case that
        turned 349 of 1,525 campaign-34 decisions into 'now' with the line counted and c left at 1)."""
        for dist in (TOOL, USER, SHARP):
            remaining = policy.remaining_gaps(support=dist.support(), idle_s=0.0)
            self.assertIsNone(
                prices.read_start_weighted(
                    remaining=remaining, read_s=45.0, wait_s=7.0, rent_per_s=1e-6, stall_weight=0.0
                )
            )

    def test_a_smaller_weight_never_starts_earlier(self):
        """Lowering c can only move a start later or to arrival; a sign slip in the weighted cost would start
        reads earlier as the GPU saturates."""
        rng = random.Random(1)
        for _ in range(200):
            dist = random_entry(rng)
            remaining = policy.remaining_gaps(support=dist.support(), idle_s=0.0)
            kwargs = dict(remaining=remaining, read_s=rng.uniform(2.0, 30.0), wait_s=0.0, rent_per_s=rng.uniform(0.01, 1.0))
            starts = [prices.read_start_weighted(stall_weight=c, **kwargs) for c in (1.0, 0.5, 0.1)]
            as_time = [float("inf") if s is None else s for s in starts]
            self.assertEqual(as_time, sorted(as_time))


class TestLandingQuantile(unittest.TestCase):
    def test_cheap_room_lands_early_and_a_saturated_gpu_late_within_the_clamp(self):
        """q = rent / (c + rent). The clamp is the guard for the saturated case, where both prices are near
        zero and their ratio is noise: the hold must never plan to land past the median of the admission."""
        q = prices.landing_quantile
        self.assertEqual(q(price=0.0, tokens=20_000, lateness_weight=1.0), prices.Q_MIN)
        self.assertAlmostEqual(q(price=1e-5, tokens=20_000, lateness_weight=1.0), 0.2 / 1.2)
        self.assertEqual(q(price=5e-5, tokens=20_000, lateness_weight=0.0), prices.Q_MAX)
        self.assertEqual(q(price=1e-9, tokens=64, lateness_weight=0.0), prices.Q_MIN)
        rising = [q(price=p, tokens=20_000, lateness_weight=1.0) for p in (1e-7, 1e-6, 1e-5, 1e-4)]
        self.assertEqual(rising, sorted(rising))
        falling = [q(price=1e-5, tokens=20_000, lateness_weight=c) for c in (0.05, 0.3, 1.0)]
        self.assertEqual(falling, sorted(falling, reverse=True))


class TestAdmissionErrorTracker(unittest.TestCase):
    def test_margin_is_minus_the_quantile_of_the_bin_the_prediction_falls_in(self):
        """A read released at admission - load - margin lands before the turn is due with chance 1 - q only if
        the margin is minus the q-quantile of (measured - predicted); taking the other sign, or the quantile of
        another bin, lands late where the error is wide."""
        tracker = prices.AdmissionErrorTracker()
        for k in range(100):
            tracker.observe(predicted_wait_s=3.0, wait_s=3.0 + (k - 50) / 25.0)  # errors -2.0 .. +1.96
            tracker.observe(predicted_wait_s=15.0, wait_s=15.0 + (k - 50) / 5.0)  # errors -10 .. +9.8
        self.assertAlmostEqual(tracker.margin(0.1, predicted_wait_s=4.0, fallback_s=2.0), 1.6)
        self.assertAlmostEqual(tracker.margin(0.1, predicted_wait_s=12.0, fallback_s=2.0), 8.0)
        self.assertAlmostEqual(tracker.margin(0.5, predicted_wait_s=4.0, fallback_s=2.0), 0.0)

    def test_an_empty_bin_keeps_the_fixed_margin(self):
        """Before 30 admissions were seen at a predicted wait nothing is known about the error there."""
        tracker = prices.AdmissionErrorTracker()
        for _ in range(prices.MIN_ERROR_SAMPLES - 1):
            tracker.observe(predicted_wait_s=3.0, wait_s=9.0)
        self.assertEqual(tracker.margin(0.1, predicted_wait_s=3.0, fallback_s=2.0), 2.0)
        tracker.observe(predicted_wait_s=3.0, wait_s=9.0)
        self.assertEqual(tracker.margin(0.1, predicted_wait_s=3.0, fallback_s=2.0), -6.0)
        self.assertEqual(tracker.margin(0.1, predicted_wait_s=30.0, fallback_s=2.0), 2.0)

    def test_readers_do_not_feed_the_estimate_of_the_gpu_turn(self):
        """A reader waits for its own landing: in campaign 34 its wait ran a median 8 to 27 s over the queue's
        prediction. Counted as an error of that prediction it would push every hold's start earlier."""
        tracker = prices.AdmissionErrorTracker()
        for _ in range(60):
            tracker.observe(predicted_wait_s=12.0, wait_s=12.5)
            tracker.observe(predicted_wait_s=12.0, wait_s=40.0, landed_after_arrival_s=39.6)
        self.assertAlmostEqual(tracker.quantile(0.9, predicted_wait_s=12.0), 0.5)
        self.assertAlmostEqual(tracker.reader_lag(), 0.4)


class TestPriceBoard(unittest.TestCase):
    def test_c_is_the_share_of_admissions_that_left_no_ready_turn_waiting(self):
        """Every admission of a pass counts: a mean whose weight depends on the time since the last sample
        would count the first admission of a pass and drop the rest."""
        board = prices.PriceBoard()
        self.assertEqual(board.c(now=0.0), 1.0)
        board.observe_admissions(count=30, ready_left=False, now=1.0)
        board.observe_admissions(count=90, ready_left=True, now=1.0)
        self.assertAlmostEqual(board.c(now=1.0), 31 / 121)
        board.observe_admissions(count=0, ready_left=True, now=1.0)
        self.assertAlmostEqual(board.c(now=1.0), 31 / 121)

    def test_c_forgets_a_saturated_episode(self):
        """After the queue drained, what was seen while it stood must fade within minutes, or early reads stay
        off for the rest of the run."""
        board = prices.PriceBoard()
        board.observe_admissions(count=200, ready_left=True, now=0.0)
        self.assertLess(board.c(now=1.0), 0.02)
        for step in range(1, 10):
            board.observe_admissions(count=2, ready_left=False, now=30.0 * step)
        self.assertGreater(board.c(now=300.0), 0.9)

    def test_mu_is_one_plus_the_time_average_of_ready_turns_waiting(self):
        """Passes are irregular (20 ms to seconds): each sample stands for the time until the next one."""
        board = prices.PriceBoard(ready=prices.DecayedMean(half_life_s=1e9))
        board.observe_queue(ready_waiting=10, now=0.0)
        board.observe_queue(ready_waiting=0, now=9.0)   # 10 turns waited for 9 s
        board.observe_queue(ready_waiting=0, now=10.0)  # none for 1 s
        self.assertAlmostEqual(board.mu(now=10.0), 1.0 + 9.0, places=2)

    def test_room_is_free_until_something_is_evicted_and_again_once_evictions_stop(self):
        board = prices.PriceBoard()
        self.assertEqual(board.lam(now=5.0), 0.0)
        board.observe_eviction(marginal_index=2e-7, tokens=4096, now=10.0)
        board.observe_eviction(marginal_index=4e-7, tokens=4096, now=10.0)
        self.assertAlmostEqual(board.lam(now=10.0), 3e-7)
        self.assertEqual(board.lam(now=10.0 + prices.FRONTIER_STALE_S + 0.1), 0.0)

    def test_kappa_prices_late_reads_at_c_and_pending_writes_at_lambda(self):
        """A second of device time delays every transfer behind it: a late read's turn at c, a pending write's
        room at lambda. With c = 0 late reads alone give no price, as in campaign 34's saturated phase."""
        board = prices.PriceBoard()
        board.observe_eviction(marginal_index=1e-6, tokens=64, now=0.0)
        board.observe_device(late_reads=12, pending_write_tokens=50_000)
        self.assertAlmostEqual(board.kappa(now=0.0), 12.0 + 0.05)
        board.observe_admissions(count=10_000, ready_left=True, now=0.0)
        self.assertAlmostEqual(board.kappa(now=0.0), 12.0 / 10_001 + 0.05)

    def test_a_snapshot_reports_the_tokens_evicted_since_the_previous_one(self):
        board = prices.PriceBoard()
        board.observe_eviction(marginal_index=1e-6, tokens=640, now=0.0)
        self.assertEqual(board.snapshot(now=0.0).evicted_tokens, 640)
        self.assertEqual(board.snapshot(now=0.0).evicted_tokens, 0)


class TestPrefillCostModel(unittest.TestCase):
    A, B = 13.4e-6, 0.69e-9  # FA3, campaign 33's cold prefills

    def test_fit_recovers_the_per_token_and_the_attention_term_from_warm_and_cold_prefills(self):
        """Most admitted turns prefill a few hundred new tokens on a long cached prefix; the attention term is
        identified through total^2 - cached^2, not through cold prefills alone (4 to 7 % of the requests)."""
        rng = random.Random(2)
        model = prices.PrefillCostModel(seed_a=1.1e-4)
        for _ in range(400):
            total = rng.randint(2_000, 60_000)
            new = total if rng.random() < 0.06 else rng.randint(64, min(total, 3_000))
            cached = total - new
            seconds = self.A * new + self.B * (total * total - cached * cached)
            model.observe(new_tokens=new, total_tokens=total, seconds=seconds * rng.uniform(0.95, 1.05))
        a, b = model.coefficients()
        self.assertAlmostEqual(a / self.A, 1.0, delta=0.1)
        self.assertAlmostEqual(b / self.B, 1.0, delta=0.1)
        cold_24k = model.seconds(new_tokens=24_000, total_tokens=24_000)
        self.assertAlmostEqual(cold_24k, self.A * 24_000 + self.B * 24_000**2, delta=0.05)

    def test_the_seed_holds_until_enough_turns_were_seen(self):
        model = prices.PrefillCostModel(seed_a=1.1e-4)
        for _ in range(model.min_samples - 1):
            model.observe(new_tokens=1_000, total_tokens=10_000, seconds=0.03)
        self.assertEqual(model.coefficients(), (1.1e-4, 0.0))

    def test_one_shape_of_request_gives_a_per_token_cost_not_a_negative_term(self):
        """Samples that all look alike cannot separate the two terms; the fit must fall back to one rate
        instead of returning a negative coefficient that prices long prefixes below zero."""
        model = prices.PrefillCostModel(seed_a=1.1e-4)
        for _ in range(50):
            model.observe(new_tokens=10_000, total_tokens=10_000, seconds=0.25)
        a, b = model.coefficients()
        self.assertGreaterEqual(b, 0.0)
        self.assertAlmostEqual(model.seconds(new_tokens=10_000, total_tokens=10_000), 0.25, delta=0.01)


class TestHoldingRate(unittest.TestCase):
    def test_falling_hazard_entries_lose_value_with_idle_time(self):
        """Both FreeInference entries: the index is the hazard and falls as the session stays idle, so the
        longest idle goes first (LRU) and the eviction frontier's price falls as room grows."""
        for dist in (TOOL, USER):
            rates = [prices.holding_rate(dist, idle_s=a, min_horizon_s=0.5) for a in (1.0, 5.0, 30.0, 120.0, 600.0)]
            self.assertEqual(rates, sorted(rates, reverse=True))

    def test_a_sharp_entry_is_worth_one_over_its_mean_until_it_is_due(self):
        """Rising hazard: the best horizon is the whole gap, the index is 1 / E[gap], and it rises as the
        return nears."""
        fresh = prices.holding_rate(SHARP, idle_s=0.0, min_horizon_s=0.5)
        self.assertAlmostEqual(fresh, 1.0 / 20.0, delta=0.003)
        self.assertGreater(prices.holding_rate(SHARP, idle_s=15.0, min_horizon_s=0.5), 3 * fresh)

    def test_no_mass_left_means_no_value(self):
        self.assertEqual(prices.holding_rate(SHARP, idle_s=5_000.0, min_horizon_s=0.5), 0.0)


class TestRecomputeWins(unittest.TestCase):
    # Measured on FA3 (campaigns 33, 34): cold prefill 0.05 s + 12.8 us n + 0.71 ns n^2; a read issued at
    # arrival with an idle read thread keeps its turn 0.1 s + 45 us a token in the queue.
    @staticmethod
    def recompute_s(tokens):
        return 12.8e-6 * tokens + 0.71e-9 * tokens * tokens

    @staticmethod
    def read_s(tokens):
        return 0.1 + 45e-6 * tokens

    def decide(self, tokens, **kw):
        base = dict(queue_wait_s=0.0, mu=1.0, lateness_weight=1.0, kappa=0.0)
        base.update(kw)
        return prices.recompute_wins(
            recompute_s=self.recompute_s(tokens),
            read_s=self.read_s(tokens),
            device_s=45e-6 * tokens,
            **base,
        )

    def test_on_an_idle_fa3_gpu_recompute_wins_up_to_long_prefixes_and_one_waiter_flips_long_ones(self):
        """The three regimes of DESIGN.md section 2 with the measured FA3 costs: idle (mu 1) recompute wins
        for 8K and 24K tokens; with turns waiting behind (mu 3) the read wins for 24K; saturated (c ~ 0 is
        not this case: here mu alone at 30 with c 1) the read wins for every size."""
        self.assertTrue(self.decide(8_000))
        self.assertTrue(self.decide(24_000))
        self.assertFalse(self.decide(24_000, mu=3.0))
        self.assertTrue(self.decide(2_000, mu=3.0))
        self.assertFalse(self.decide(2_000, mu=30.0))

    def test_a_queue_that_hides_the_read_keeps_the_read(self):
        """Once the turn waits longer for the GPU than its read takes, the read costs it nothing: recompute
        must not be chosen whatever the sizes, or queued turns recompute behind others."""
        self.assertFalse(self.decide(8_000, queue_wait_s=5.0))

    def test_on_triton_the_read_wins_at_every_size(self):
        """110 us a token to recompute against about 45 us to read: today's always-read is the priced answer."""
        for tokens in (2_000, 16_000, 60_000):
            self.assertFalse(
                prices.recompute_wins(
                    recompute_s=1.1e-4 * tokens,
                    read_s=self.read_s(tokens),
                    queue_wait_s=0.0,
                    mu=1.0,
                    lateness_weight=1.0,
                    device_s=45e-6 * tokens,
                    kappa=0.0,
                )
            )


if __name__ == "__main__":
    unittest.main(verbosity=1)
