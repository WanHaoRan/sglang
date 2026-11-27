"""Prices of GPU time, host room and the SSD, measured from their queues, and the estimators
the priced decisions of the next-turn prefetcher read
(agent_cache/results/design_unified_1126/DESIGN.md, sections 2, 3.2 and 6).

Units: c is seconds of summed TTFT per second one turn waits on its read; mu is seconds of
summed TTFT per GPU-second; kappa is seconds of TTFT per second of device time; lam is seconds
of TTFT per token-second of host memory, the unit of --hicache-storage-next-turn-memory-price.

Pure policy: no clock, no cache; every time is passed in, in seconds.
"""

from __future__ import annotations

import bisect
import itertools
from collections import deque
from typing import Optional

import msgspec

# Arbitrary: what was seen this long ago weighs half. Long against a pass (tens of ms) and a
# burst of admissions, short against a load episode (minutes).
PRICE_HALF_LIFE_S = 30.0
# Below this c the ready queue stays non-empty. Measured: c is 0.93-0.98 at light load and at
# most 0.06 from the minute the GPU saturates (campaign 34).
C_SATURATED = 0.1
# Arbitrary: below this c the landing quantile is the ratio of two prices that are both noise.
C_FLOOR = 0.02
# Arbitrary: a hold plans to land between the 5 % point and the median of the admission time.
Q_MIN, Q_MAX = 0.05, 0.5
# Predicted-wait bin edges of the admission error, seconds. Measured: the error widens with
# the wait predicted (campaign 34: q10/q90 -1.4/+2.2 s at 2-5 s, -5.2/+4.6 s at 10-20 s).
ERROR_BIN_EDGES_S = (2.0, 5.0, 10.0, 20.0)
# Arbitrary, as GapSamples.window: the error drifts with the load, so recent admissions count.
ERROR_WINDOW = 256
# Arbitrary: fewest errors seen in a bin for its quantiles to replace the fixed margin.
MIN_ERROR_SAMPLES = 30
# A host eviction this long ago no longer says room is short. Arbitrary; one half-life.
FRONTIER_STALE_S = 30.0
# As PriceController.ceiling: bounds the loop lam -> kappa (pending writes at lam) -> lam.
LAM_CEILING = 1e-3
# A recompute is priced against a read only while c is at least this. Measured: wrong on
# 4-20 % of its recomputes with the condition, 25-50 % without (campaigns 33 and 34).
RECOMPUTE_MIN_C = 0.8


class DecayedMean(msgspec.Struct):
    """Mean of weighted samples whose weight halves every half_life_s. Samples of one instant
    all count: a per-sample EWMA with a time-based alpha would keep the first of a pass only."""

    half_life_s: float = PRICE_HALF_LIFE_S
    num: float = 0.0
    den: float = 0.0
    last: Optional[float] = None

    def _fade(self, now: float) -> float:
        if self.last is None or now <= self.last:
            return 1.0
        return 0.5 ** ((now - self.last) / self.half_life_s)

    def observe(self, value: float, *, now: float, weight: float = 1.0) -> None:
        fade = self._fade(now)
        self.num = self.num * fade + value * weight
        self.den = self.den * fade + weight
        self.last = now if self.last is None else max(self.last, now)

    def value(self, *, now: float, prior: float, prior_weight: float = 1.0) -> float:
        """The mean with `prior` counted as a sample of prior_weight, which must be positive
        unless a sample was observed."""
        fade = self._fade(now)
        return (self.num * fade + prior * prior_weight) / (self.den * fade + prior_weight)


class PriceBoard(msgspec.Struct):
    """c, mu, kappa and lam, each read off one queue (DESIGN.md section 6)."""

    admissions: DecayedMean = msgspec.field(default_factory=DecayedMean)
    ready: DecayedMean = msgspec.field(default_factory=DecayedMean)
    frontier: DecayedMean = msgspec.field(default_factory=DecayedMean)
    last_eviction: Optional[float] = None
    last_queue_t: Optional[float] = None
    ready_waiting: int = 0
    sessions_waiting: int = 0
    # Admissions per second: in the closed loop, the rate turns arrive at.
    arrival_rate: float = 0.0
    late_reads: int = 0
    pending_write_tokens: int = 0
    evicted_tokens: int = 0

    def observe_admissions(self, *, count: int, ready_left: bool, now: float) -> None:
        """One pass admitted `count` turns; ready_left says whether a turn with nothing to
        read was left waiting for the GPU after it."""
        if count > 0:
            self.admissions.observe(0.0 if ready_left else 1.0, now=now, weight=float(count))

    def observe_queue(
        self, *, ready_waiting: int, sessions_waiting: int, arrival_rate: float, now: float
    ) -> None:
        """As of this pass: the queued turns waiting for the GPU with no read of their own
        under way, and the sessions whose next turn is not running (users in session minus
        turns running: those between turns and those queued)."""
        elapsed = 0.0 if self.last_queue_t is None else max(0.0, now - self.last_queue_t)
        # Time-weighted: the previous pass's count stood for the interval that ends now.
        self.ready.observe(float(self.ready_waiting), now=now, weight=elapsed)
        self.ready_waiting = ready_waiting
        self.sessions_waiting = sessions_waiting
        self.arrival_rate = arrival_rate
        self.last_queue_t = now

    def observe_device(self, *, late_reads: int, pending_write_tokens: int) -> None:
        """Parked reads past their latest start, and host tokens pending SSD writes lock."""
        self.late_reads = late_reads
        self.pending_write_tokens = pending_write_tokens

    def observe_eviction(
        self, *, marginal_index: Optional[float], tokens: int, now: float
    ) -> None:
        """One pass of host eviction gave up `tokens`. marginal_index is the highest holding
        index among its victims, in seconds of TTFT per token-second; None when none had one."""
        self.evicted_tokens += tokens
        if marginal_index is not None:
            self.frontier.observe(marginal_index, now=now)
            self.last_eviction = now

    def take_evicted_tokens(self) -> int:
        """Host tokens evicted since the previous call."""
        tokens, self.evicted_tokens = self.evicted_tokens, 0
        return tokens

    def c(self, now: float) -> float:
        """Share of recent admissions that left no ready turn waiting; 1 before any."""
        return self.admissions.value(now=now, prior=1.0)

    def _mu_light(self, now: float) -> float:
        # The current count stands in until a pass interval was observed.
        return 1.0 + self.ready.value(
            now=now, prior=float(self.ready_waiting), prior_weight=1e-3
        )

    def mu(self, now: float) -> float:
        """One plus the turns a GPU-second delays: at light load the ready turns waiting;
        once ready turns always wait, every session whose next turn is not running (measured
        91-109 TTFT-seconds per GPU-second against 31 ready turns, campaigns 32 and 34).
        Never below the light-load value, which lags a queue that just drained."""
        light = self._mu_light(now)
        if self.c(now) >= C_SATURATED:
            return light
        return max(light, float(self.sessions_waiting))

    def mu_for(self, job_s: float, now: float) -> float:
        """mu for a job of job_s GPU-seconds. At light load the job also delays the turns
        that arrive while it runs, half of it each on average; at saturation they are in mu."""
        if self.c(now) < C_SATURATED:
            return self.mu(now)
        return self._mu_light(now) + self.arrival_rate * job_s / 2.0

    def lam(self, now: float) -> float:
        """The holding index of the marginal host holder; 0 while nothing is evicted."""
        if self.last_eviction is None or now - self.last_eviction > FRONTIER_STALE_S:
            return 0.0
        return min(LAM_CEILING, self.frontier.value(now=now, prior=0.0, prior_weight=0.0))

    def kappa(self, now: float) -> float:
        """A second of device time delays every transfer behind it: a late read's turn at c,
        the room a pending write locks at lam."""
        return self.c(now) * self.late_reads + self.lam(now) * self.pending_write_tokens


def landing_quantile(*, price: float, tokens: int, lateness_weight: float) -> float:
    """q of DESIGN.md section 3.2: a read should land at the q-quantile of its turn's
    admission time when a second late costs lateness_weight and a second early costs
    price x tokens."""
    rent = max(0.0, price) * tokens
    q = rent / (max(lateness_weight, C_FLOOR) + rent)
    return min(Q_MAX, max(Q_MIN, q))


def quantile_of(ascending: list[float], q: float) -> float:
    return ascending[min(len(ascending) - 1, int(q * len(ascending)))]


class AdmissionErrorTracker(msgspec.Struct):
    """Measured minus predicted queue wait of admitted turns that read nothing from storage,
    by the wait predicted: the error of the admission estimate. A turn that read is admitted
    at the later of its GPU turn and its landing, so it is kept apart: against the same
    prediction its error would be the read line. Its lag from that later time to admission
    is what is kept."""

    errors: list[deque[float]] = msgspec.field(
        default_factory=lambda: [
            deque(maxlen=ERROR_WINDOW) for _ in range(len(ERROR_BIN_EDGES_S) + 1)
        ]
    )
    # Ascending copy of each bin, shared until the bin's next sample.
    ordered: list[Optional[list[float]]] = msgspec.field(
        default_factory=lambda: [None] * (len(ERROR_BIN_EDGES_S) + 1)
    )
    reader_lags: deque[float] = msgspec.field(
        default_factory=lambda: deque(maxlen=ERROR_WINDOW)
    )

    def observe(
        self,
        *,
        predicted_wait_s: float,
        wait_s: float,
        landed_after_arrival_s: Optional[float] = None,
    ) -> None:
        """landed_after_arrival_s: from the turn's arrival to the landing of its last
        storage read, negative for a read that landed first; None for a turn that read none."""
        if landed_after_arrival_s is not None:
            self.reader_lags.append(wait_s - max(predicted_wait_s, landed_after_arrival_s))
            return
        index = bisect.bisect_right(ERROR_BIN_EDGES_S, predicted_wait_s)
        self.errors[index].append(wait_s - predicted_wait_s)
        self.ordered[index] = None

    def quantile(self, q: float, *, predicted_wait_s: float) -> Optional[float]:
        """The q-quantile of measured minus predicted wait at this predicted wait; None
        until its bin holds MIN_ERROR_SAMPLES."""
        index = bisect.bisect_right(ERROR_BIN_EDGES_S, predicted_wait_s)
        if len(self.errors[index]) < MIN_ERROR_SAMPLES:
            return None
        ordered = self.ordered[index]
        if ordered is None:
            ordered = self.ordered[index] = sorted(self.errors[index])
        return quantile_of(ordered, q)

    def margin(self, q: float, *, predicted_wait_s: float, fallback_s: float) -> float:
        """Seconds before its load estimate requires that a held read starts, so that it
        lands by admission with chance 1 - q: minus the q-quantile of the error."""
        error = self.quantile(q, predicted_wait_s=predicted_wait_s)
        return fallback_s if error is None else -error

    def pooled(self) -> list[float]:
        """The errors of every bin, ascending."""
        return sorted(itertools.chain.from_iterable(self.errors))

    def reader_lag(self) -> Optional[float]:
        """Median lag of the turns that read; None until MIN_ERROR_SAMPLES were seen."""
        if len(self.reader_lags) < MIN_ERROR_SAMPLES:
            return None
        return quantile_of(sorted(self.reader_lags), 0.5)


class PrefillCostModel(msgspec.Struct):
    """Seconds of prefill = a x new tokens + b x (total^2 - cached^2), fitted on admitted
    turns: each new token costs its own work plus attention over all before it. Measured on
    FA3 at an empty queue: a 12.3-13.4 us, b 0.69-0.72 ns (campaigns 33 and 34); the constant
    per token it replaces holds at one length only."""

    seed_a: float
    seed_b: float = 0.0
    # Arbitrary: a sample's weight halves every 200 samples.
    forget: float = 0.5 ** (1 / 200)
    # Arbitrary: fewest samples for the fit to replace the seed.
    min_samples: int = 20
    samples: int = 0
    # Normal equations of the two features, scaled to order one: new / 1e4 and
    # (total^2 - cached^2) / 1e9.
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
        fade = self.forget
        self.s11 = self.s11 * fade + x1 * x1
        self.s12 = self.s12 * fade + x1 * x2
        self.s22 = self.s22 * fade + x2 * x2
        self.s1y = self.s1y * fade + x1 * seconds
        self.s2y = self.s2y * fade + x2 * seconds
        self.samples += 1

    def coefficients(self) -> tuple[float, float]:
        """(a, b) in seconds per token and per token squared; the seed until the fit holds."""
        if self.samples < self.min_samples or self.s11 <= 0.0:
            return self.seed_a, self.seed_b
        det = self.s11 * self.s22 - self.s12 * self.s12
        if det > 1e-9 * self.s11 * self.s22:
            a = (self.s1y * self.s22 - self.s2y * self.s12) / det / 1e4
            b = (self.s2y * self.s11 - self.s1y * self.s12) / det / 1e9
            if a > 0.0 and b >= 0.0:
                return a, b
        # One shape of request cannot tell the two terms apart: one rate per new token.
        return self.s1y / self.s11 / 1e4, 0.0

    def seconds(self, *, new_tokens: int, total_tokens: int) -> float:
        """Prefill of new_tokens on top of total_tokens - new_tokens cached ones."""
        a, b = self.coefficients()
        cached = total_tokens - new_tokens
        return a * new_tokens + b * (total_tokens * total_tokens - cached * cached)


class RecomputeQuote(msgspec.Struct, frozen=True):
    """Read-or-recompute for a turn's missing history, as priced."""

    recompute_s: float
    read_s: float
    mu: float
    c: float
    wins: bool


def quote_recompute(
    *,
    board: PriceBoard,
    cost: PrefillCostModel,
    matched: int,
    missing: int,
    total: int,
    read_s: float,
    now: float,
) -> RecomputeQuote:
    """DESIGN.md section 2 for a turn that arrives with no ready turn waiting, `matched` of
    its `total` prompt tokens in memory and `missing` more in storage: recompute them when
    their prefill, at the price of GPU time for a job that long, takes less than the read
    keeps the turn waiting. Not once c has fallen: a queue then hides the read."""
    with_read = total - matched - missing
    recompute_s = cost.seconds(
        new_tokens=total - matched, total_tokens=total
    ) - cost.seconds(new_tokens=with_read, total_tokens=total)
    mu = board.mu_for(recompute_s, now)
    c = board.c(now)
    return RecomputeQuote(
        recompute_s=recompute_s,
        read_s=read_s,
        mu=mu,
        c=c,
        wins=c >= RECOMPUTE_MIN_C and mu * recompute_s < read_s,
    )
