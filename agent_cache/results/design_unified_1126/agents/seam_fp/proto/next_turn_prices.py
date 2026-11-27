"""Prototype (seam_fp, not engine code): the pure estimators the F1-F5 / PR1 plan adds, in the style of
python/sglang/srt/mem_cache/next_turn_policy.py and next_turn_estimators.py: no clock, no cache; every time is
passed in, in seconds. Intended home: a new python/sglang/srt/mem_cache/next_turn_prices.py, plus two functions
(read_start's stall_weight, GapDistribution.holding_rate) that belong in next_turn_policy.py.

Units: c is dimensionless (seconds of summed TTFT per second of one turn's stall); mu is seconds of summed TTFT per
GPU-second; kappa is seconds of TTFT per second of device time; lam is seconds of TTFT per token-second of host
memory (the unit of --hicache-storage-next-turn-memory-price).
"""

from __future__ import annotations

import bisect
import math
from collections import deque
from typing import Optional

import msgspec

# Arbitrary: the weight of what was seen this long ago is half. Long against a pass (tens of ms) and against a
# burst of admissions, short against a load episode (minutes).
PRICE_HALF_LIFE_S = 30.0
# Arbitrary: below this lateness weight the landing quantile is decided by noise in two near-zero prices.
C_FLOOR = 0.02
# Arbitrary: a hold never plans to land later than the median of the admission time, nor earlier than its 5 %.
Q_MIN, Q_MAX = 0.05, 0.5
# Predicted-wait bin edges of the admission error, seconds. Measured: the error's spread grows with the wait
# predicted (campaign 34, minutes 30-110: q10/q90 -1.4/+2.2 s at 2-5 s, -5.2/+4.6 s at 10-20 s).
ERROR_BIN_EDGES_S = (2.0, 5.0, 10.0, 20.0)
ERROR_WINDOW = 256
MIN_ERROR_SAMPLES = 30
# A host eviction this long ago no longer says room is short: the price falls back to zero.
FRONTIER_STALE_S = 30.0


class DecayedMean(msgspec.Struct):
    """Mean of samples whose weight halves every half_life_s. Samples of one instant all count (a per-sample
    EWMA with a time-based alpha would drop every sample after the first of a pass)."""

    half_life_s: float = PRICE_HALF_LIFE_S
    num: float = 0.0
    den: float = 0.0
    last: Optional[float] = None

    def _factor(self, now: float) -> float:
        if self.last is None or now <= self.last:
            return 1.0
        return 0.5 ** ((now - self.last) / self.half_life_s)

    def observe(self, value: float, *, now: float, weight: float = 1.0) -> None:
        factor = self._factor(now)
        self.num = self.num * factor + value * weight
        self.den = self.den * factor + weight
        self.last = now if self.last is None else max(self.last, now)

    def value(self, *, now: float, prior: float, prior_weight: float = 1.0) -> float:
        factor = self._factor(now)
        return (self.num * factor + prior * prior_weight) / (self.den * factor + prior_weight)


class PriceSnapshot(msgspec.Struct, frozen=True):
    """One HICACHE_EVT prices line."""

    c: float
    mu: float
    kappa: float
    lam: float
    ready_waiting: int
    late_reads: int
    pending_write_tokens: int
    evicted_tokens: int


class PriceBoard(msgspec.Struct):
    """The four prices, each read off one queue (DESIGN.md section 6)."""

    admissions: DecayedMean = msgspec.field(default_factory=DecayedMean)
    ready: DecayedMean = msgspec.field(default_factory=DecayedMean)
    frontier: DecayedMean = msgspec.field(default_factory=DecayedMean)
    last_eviction: Optional[float] = None
    last_queue_t: Optional[float] = None
    ready_waiting: int = 0
    late_reads: int = 0
    pending_write_tokens: int = 0
    evicted_tokens: int = 0

    def observe_admissions(self, *, count: int, ready_left: bool, now: float) -> None:
        """`count` turns were admitted in one pass; ready_left says whether a turn with nothing to read was
        left waiting for the GPU after it."""
        if count > 0:
            self.admissions.observe(0.0 if ready_left else 1.0, now=now, weight=float(count))

    def observe_queue(self, *, ready_waiting: int, now: float) -> None:
        """Turns waiting for the GPU with no read of their own under way, as of this pass."""
        elapsed = 0.0 if self.last_queue_t is None else max(0.0, now - self.last_queue_t)
        # Time-weighted: a sample stands for the interval since the previous pass.
        self.ready.observe(float(self.ready_waiting), now=now, weight=elapsed)
        self.ready_waiting = ready_waiting
        self.last_queue_t = now

    def observe_device(self, *, late_reads: int, pending_write_tokens: int) -> None:
        self.late_reads = late_reads
        self.pending_write_tokens = pending_write_tokens

    def observe_eviction(self, *, marginal_index: float, tokens: int, now: float) -> None:
        """One host eviction pass gave up `tokens`; marginal_index is the highest holding index among them,
        in seconds of TTFT per token-second."""
        self.frontier.observe(marginal_index, now=now)
        self.last_eviction = now
        self.evicted_tokens += tokens

    def c(self, now: float) -> float:
        return self.admissions.value(now=now, prior=1.0)

    def mu(self, now: float) -> float:
        return 1.0 + self.ready.value(now=now, prior=float(self.ready_waiting), prior_weight=1e-3)

    def lam(self, now: float) -> float:
        if self.last_eviction is None or now - self.last_eviction > FRONTIER_STALE_S:
            return 0.0
        return self.frontier.value(now=now, prior=0.0, prior_weight=0.0)

    def kappa(self, now: float) -> float:
        """Late reads cost c each second each; a pending write holds its tokens of room at lam."""
        return self.c(now) * self.late_reads + self.lam(now) * self.pending_write_tokens

    def snapshot(self, now: float) -> PriceSnapshot:
        snap = PriceSnapshot(
            c=self.c(now),
            mu=self.mu(now),
            kappa=self.kappa(now),
            lam=self.lam(now),
            ready_waiting=self.ready_waiting,
            late_reads=self.late_reads,
            pending_write_tokens=self.pending_write_tokens,
            evicted_tokens=self.evicted_tokens,
        )
        self.evicted_tokens = 0
        return snap


def landing_quantile(*, price: float, tokens: int, lateness_weight: float) -> float:
    """q of DESIGN.md section 3.2: the read lands at the q-quantile of its turn's admission time, where a second
    late costs lateness_weight and a second early costs price x tokens."""
    rent = max(0.0, price) * tokens
    q = rent / (max(lateness_weight, C_FLOOR) + rent)
    return min(Q_MAX, max(Q_MIN, q))


class AdmissionErrorTracker(msgspec.Struct):
    """Measured minus predicted queue wait of admitted turns that had no storage read of their own, by the wait
    predicted: the error of the estimate of D. Readers are kept apart: their admission is the later of the GPU
    turn and their own landing, so their error against the same prediction is the read line, and what is tracked
    for them is the lag from that later time to admission."""

    errors: list[deque[float]] = msgspec.field(
        default_factory=lambda: [deque(maxlen=ERROR_WINDOW) for _ in range(len(ERROR_BIN_EDGES_S) + 1)]
    )
    reader_lags: deque[float] = msgspec.field(default_factory=lambda: deque(maxlen=ERROR_WINDOW))

    @staticmethod
    def _bin(predicted_wait_s: float) -> int:
        return bisect.bisect_right(ERROR_BIN_EDGES_S, predicted_wait_s)

    def observe(
        self,
        *,
        predicted_wait_s: float,
        wait_s: float,
        landed_after_arrival_s: Optional[float] = None,
    ) -> None:
        """landed_after_arrival_s: seconds from the turn's arrival to its last read's landing; None = no read."""
        if landed_after_arrival_s is None:
            self.errors[self._bin(predicted_wait_s)].append(wait_s - predicted_wait_s)
        else:
            self.reader_lags.append(wait_s - max(predicted_wait_s, landed_after_arrival_s))

    def quantile(self, q: float, *, predicted_wait_s: float) -> Optional[float]:
        """The q-quantile of (measured - predicted) at this predicted wait; None until the bin has samples."""
        seen = self.errors[self._bin(predicted_wait_s)]
        if len(seen) < MIN_ERROR_SAMPLES:
            return None
        ordered = sorted(seen)
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    def reader_lag(self) -> Optional[float]:
        if len(self.reader_lags) < MIN_ERROR_SAMPLES:
            return None
        ordered = sorted(self.reader_lags)
        return ordered[len(ordered) // 2]

    def margin(self, q: float, *, predicted_wait_s: float, fallback_s: float) -> float:
        """How long before the load estimate requires a held read starts: minus the q-quantile of the error."""
        error = self.quantile(q, predicted_wait_s=predicted_wait_s)
        return fallback_s if error is None else -error


class PrefillCostModel(msgspec.Struct):
    """Seconds of prefill = a x new tokens + b x (total^2 - cached^2), fitted on admitted turns: flash attention
    costs each new token its own work plus attention over everything before it. Measured on FA3 (campaign 33,
    503 cold prefills at an empty queue): a 13.4 us, b 0.69 ns, where Triton's constant was 110 us a token."""

    seed_a: float
    seed_b: float = 0.0
    # Arbitrary: a sample's weight halves every 200 samples.
    forget: float = 0.5 ** (1 / 200)
    min_samples: int = 20
    samples: int = 0
    # Features are scaled to order one: x1 = new / 1e4, x2 = (total^2 - cached^2) / 1e9.
    s11: float = 0.0
    s12: float = 0.0
    s22: float = 0.0
    s1y: float = 0.0
    s2y: float = 0.0

    def observe(self, *, new_tokens: int, total_tokens: int, seconds: float) -> None:
        if new_tokens <= 0 or seconds <= 0.0:
            return
        cached = total_tokens - new_tokens
        x1 = new_tokens / 1e4
        x2 = (total_tokens * total_tokens - cached * cached) / 1e9
        f = self.forget
        self.s11 = self.s11 * f + x1 * x1
        self.s12 = self.s12 * f + x1 * x2
        self.s22 = self.s22 * f + x2 * x2
        self.s1y = self.s1y * f + x1 * seconds
        self.s2y = self.s2y * f + x2 * seconds
        self.samples += 1

    def coefficients(self) -> tuple[float, float]:
        """(a, b) in seconds per token and per token squared; the seed until the fit is usable."""
        if self.samples < self.min_samples or self.s11 <= 0.0:
            return self.seed_a, self.seed_b
        det = self.s11 * self.s22 - self.s12 * self.s12
        if det > 1e-9 * self.s11 * self.s22:
            a = (self.s1y * self.s22 - self.s2y * self.s12) / det / 1e4
            b = (self.s2y * self.s11 - self.s1y * self.s12) / det / 1e9
            if a > 0.0 and b >= 0.0:
                return a, b
        # The samples cannot tell the two terms apart (one shape of request): one rate per new token.
        return self.s1y / self.s11 / 1e4, 0.0

    def seconds(self, *, new_tokens: int, total_tokens: int) -> float:
        a, b = self.coefficients()
        cached = total_tokens - new_tokens
        return a * new_tokens + b * (total_tokens * total_tokens - cached * cached)


def read_start_weighted(
    *,
    remaining: tuple[float, ...],
    read_s: float,
    wait_s: float,
    rent_per_s: float,
    stall_weight: float = 1.0,
) -> Optional[float]:
    """next_turn_policy.read_start with each second the turn waits on the read costing stall_weight (c) instead
    of 1. The body is read_start's; only the two marked lines differ."""
    stall_at_arrival = max(0.0, read_s - wait_s)
    count = len(remaining)
    if count == 0 or stall_at_arrival == 0.0:
        return None
    prefix = [0.0]
    for gap in remaining:
        prefix.append(prefix[-1] + gap)
    total = prefix[-1]
    lead = read_s - wait_s
    best_cost, best_start = stall_weight * stall_at_arrival, None  # changed
    for start in sorted({0.0, *(max(0.0, gap - lead) for gap in remaining)}):
        arrived = bisect.bisect_right(remaining, start)
        late = bisect.bisect_right(remaining, start + lead)
        stall = arrived * stall_at_arrival + (
            (late - arrived) * (start + lead) - (prefix[late] - prefix[arrived])
        )
        pinned = (total - prefix[late]) - (count - late) * (start + lead)
        cost = (stall_weight * stall + rent_per_s * pinned) / count  # changed
        if cost < best_cost:
            best_cost, best_start = cost, start
    return best_start


def holding_rate(dist, *, idle_s: float, min_horizon_s: float, max_horizon_s: float = 1200.0, steps: int = 48) -> float:
    """sup over horizons h >= min_horizon_s of P(back within h | idle) / E[min(time left, h) | idle], per second:
    the session's holding index over its restore cost per token (DESIGN.md section 4.1). On the continuous
    survival, not the 64-point support: the sup over support points jumps with the distance to the next point.
    `dist` is a next_turn_policy.GapDistribution."""
    alive = dist.survival(idle_s)
    if alive <= 0.0:
        return 0.0
    ratio = (max_horizon_s / min_horizon_s) ** (1.0 / steps)
    best = 0.0
    # tau up to the first horizon: the survival is taken as linear over it.
    h_prev, s_prev = 0.0, 1.0
    tau = 0.0
    h = min_horizon_s
    for _ in range(steps + 1):
        s = dist.survival(idle_s + h) / alive
        tau += 0.5 * (s_prev + s) * (h - h_prev)
        if tau > 0.0:
            best = max(best, (1.0 - s) / tau)
        h_prev, s_prev = h, s
        h *= ratio
    return best


def recompute_wins(
    *,
    recompute_s: float,
    read_s: float,
    queue_wait_s: float,
    mu: float,
    lateness_weight: float,
    device_s: float,
    kappa: float,
) -> bool:
    """Read-or-recompute at arrival (DESIGN.md section 2) for a prefix with a complete storage copy: recompute
    when its GPU time, at the price of GPU time, costs less than the stall the read adds beyond the queue wait
    plus the device time it takes. The room the read holds until admission is left out: at light load it is free."""
    k_read = lateness_weight * max(0.0, read_s - queue_wait_s) + kappa * device_s
    return mu * recompute_s < k_read
