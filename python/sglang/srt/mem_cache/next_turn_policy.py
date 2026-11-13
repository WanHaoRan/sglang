"""Expected-cost decisions on a gap distribution (agent_cache/PAPER_PLAN.md, "Formulation").

A finished turn's hint is a distribution of the gap before its session's next turn. Host
memory has a price, in seconds of TTFT per token-second, and each decision weighs the
stall it risks against the rent it pays:

- read_start: when to start reading a history back from storage (rule 1);
- GapDistribution.chance_within: the chance of return within a horizon given the idle
  time so far, which ranks staged reads (rule 2) and eviction (rule 3);
- PriceController: the price itself, moved by how much room reads were refused (rule 4).

Pure policy: no clock, no cache; every time is passed in, in seconds.
"""

from __future__ import annotations

import bisect
import itertools
import math
from statistics import NormalDist
from typing import Optional

import msgspec

# Standard normal quantile of 0.9: an entry's p10 and p90 sit this many sigmas from its median.
Z90 = 1.2815515655446004
# ln units. Arbitrary; keeps an entry whose quantiles coincide from becoming a point.
SIGMA_FLOOR = 0.05
# ln units: a p90 400 times the median. Arbitrary; keeps exp() finite for any entry.
SIGMA_CEILING = 6.0
# Seconds. A zero gap has no logarithm; the client floors its gaps here too (gap_hints.py).
GAP_FLOOR_S = 0.01
# Support points of a distribution in read_start. Arbitrary; the room model
# (agent_cache/scripts/room_model.py) used 100 and its costs moved under 1 % from 50 on.
SUPPORT_POINTS = 64
# Longest gap a plan counts on, seconds. It bounds the rent a wide entry's tail adds to an
# early read; 1200 s is the room model's cap, which the starting price was calibrated with.
MAX_PLANNED_GAP_S = 1200.0
# Fewest support points a session's remaining gap is planned on (the top 6 % of 64). Arbitrary.
MIN_REMAINING_POINTS = 4
# Seconds of GPU prefill to recompute one token of history. Measured: first-turn prefill of
# Qwen3-30B-A3B on an H200, 0.108-0.113 ms per token (agent_cache campaign 25); re-measure
# for another model or GPU.
RECOMPUTE_S_PER_TOKEN = 1.1e-4
# Fewest gaps seen for a key, and fewest of them longer than the idle time, for the key's
# own gaps to give a chance of return. Arbitrary; as the eviction replay used
# (agent_cache/frozen/ec_policy_20261111/evict_replay).
MIN_GAP_SAMPLES = 5
MIN_LONGER_SAMPLES = 3

_STD_NORMAL = NormalDist()


class GapDistribution(msgspec.Struct, frozen=True):
    """ln(gap) is normal on each side of the median, with its own sigma: the two-piece
    lognormal through an entry's p10, p50 and p90. Both sigmas zero is a point (exact hint)."""

    median_s: float
    sigma_low: float
    sigma_high: float

    @classmethod
    def from_quantiles(cls, *, p10: float, p50: float, p90: float) -> GapDistribution:
        median = max(p50, GAP_FLOOR_S)
        low = max(min(p10, median), GAP_FLOOR_S)
        high = max(p90, median)
        return cls(
            median_s=median,
            sigma_low=min(SIGMA_CEILING, max(SIGMA_FLOOR, math.log(median / low) / Z90)),
            sigma_high=min(
                SIGMA_CEILING, max(SIGMA_FLOOR, math.log(high / median) / Z90)
            ),
        )

    @classmethod
    def point(cls, gap_s: float) -> GapDistribution:
        return cls(median_s=max(gap_s, GAP_FLOOR_S), sigma_low=0.0, sigma_high=0.0)

    @property
    def is_point(self) -> bool:
        return self.sigma_low == 0.0 and self.sigma_high == 0.0

    def survival(self, gap_s: float) -> float:
        """P(gap > gap_s)."""
        if gap_s <= 0.0:
            return 1.0
        if self.is_point:
            return 1.0 if gap_s < self.median_s else 0.0
        sigma = self.sigma_low if gap_s < self.median_s else self.sigma_high
        z = math.log(gap_s / self.median_s) / sigma
        # erfc keeps the far tail from rounding to zero long before it matters.
        return 0.5 * math.erfc(z / math.sqrt(2.0))

    def chance_within(self, *, idle_s: float, horizon_s: float) -> float:
        """P(gap <= idle_s + horizon_s | gap > idle_s); 0 once no mass is left."""
        alive = self.survival(idle_s)
        if alive <= 0.0:
            return 0.0
        return 1.0 - self.survival(idle_s + horizon_s) / alive

    def quantile(self, q: float) -> float:
        if self.is_point:
            return self.median_s
        z = _STD_NORMAL.inv_cdf(q)
        sigma = self.sigma_low if z < 0.0 else self.sigma_high
        return self.median_s * math.exp(sigma * z)

    def support(self, points: int = SUPPORT_POINTS) -> tuple[float, ...]:
        """Equal-weight gaps, ascending: the mid-quantiles, capped at MAX_PLANNED_GAP_S."""
        if self.is_point:
            return (self.median_s,)
        return tuple(
            min(MAX_PLANNED_GAP_S, self.quantile((k + 0.5) / points))
            for k in range(points)
        )


def sampled_chance(
    *, gaps: tuple[float, ...], idle_s: float, horizon_s: float
) -> Optional[float]:
    """P(gap <= idle_s + horizon_s | gap > idle_s) over gaps seen, ascending; None when
    too few were seen or too few outlast the idle time to say."""
    if len(gaps) < MIN_GAP_SAMPLES:
        return None
    first = bisect.bisect_right(gaps, idle_s)
    longer = len(gaps) - first
    if longer < MIN_LONGER_SAMPLES:
        return None
    return (bisect.bisect_right(gaps, idle_s + horizon_s) - first) / longer


class GapSamples(msgspec.Struct):
    """The latest gaps seen per reply key, and over all keys under ANY_KEY."""

    # Arbitrary; a key's distribution drifts with the workload, so only recent gaps count.
    window: int = 256
    gaps: dict[str, list[float]] = {}
    snapshots: dict[str, tuple[float, ...]] = {}

    def add(self, *, key: str, gap_s: float) -> None:
        seen = self.gaps.setdefault(key, [])
        seen.append(gap_s)
        if len(seen) > self.window:
            del seen[0]
        self.snapshots.pop(key, None)

    def ascending(self, key: str) -> tuple[float, ...]:
        """An immutable sorted copy, shared until the key's next gap arrives."""
        snapshot = self.snapshots.get(key)
        if snapshot is None:
            snapshot = self.snapshots[key] = tuple(sorted(self.gaps.get(key, ())))
        return snapshot


def remaining_gaps(*, support: tuple[float, ...], idle_s: float) -> tuple[float, ...]:
    """The support given the session has been idle for idle_s: seconds left, ascending.
    Empty once an entry's last few points are all that is left: they would read as a
    near-certain return, where the entry in fact says little."""
    first = bisect.bisect_right(support, idle_s)
    if len(support) > 1 and len(support) - first < MIN_REMAINING_POINTS:
        return ()
    return tuple(gap - idle_s for gap in support[first:])


def read_start(
    *,
    remaining: tuple[float, ...],
    read_s: float,
    wait_s: float,
    rent_per_s: float,
) -> Optional[float]:
    """Rule 1: seconds from now at which a staged read should start, 0 for now; None when
    reading at the turn's arrival costs no more. `remaining` is remaining_gaps(), `read_s`
    the time from issuing the read to its pages being usable, `wait_s` the queue wait a
    turn arriving now would see, `rent_per_s` the price times the tokens the read holds.

    A read started at s lands at s + read_s and the turn is due at its arrival plus
    wait_s. Each second the turn waits on the read costs 1, each second the read waits
    for the turn costs rent_per_s, and a turn that arrives before s starts its own read.
    """
    stall_at_arrival = max(0.0, read_s - wait_s)
    count = len(remaining)
    if count == 0 or stall_at_arrival == 0.0:
        return None
    prefix = list(itertools.accumulate(remaining, initial=0.0))
    total = prefix[-1]
    lead = read_s - wait_s
    best_cost, best_start = stall_at_arrival, None
    # The cost is piecewise linear in s with a kink where a support point becomes due as
    # the read lands, so a minimum is at one of those starts or at now.
    for start in sorted({0.0, *(max(0.0, gap - lead) for gap in remaining)}):
        arrived = bisect.bisect_right(remaining, start)
        late = bisect.bisect_right(remaining, start + lead)
        stall = arrived * stall_at_arrival + (
            (late - arrived) * (start + lead) - (prefix[late] - prefix[arrived])
        )
        pinned = (total - prefix[late]) - (count - late) * (start + lead)
        cost = (stall + rent_per_s * pinned) / count
        if cost < best_cost:
            best_cost, best_start = cost, start
    return best_start


class PriceUpdate(msgspec.Struct, frozen=True):
    """One interval of the price controller, as logged."""

    price: float
    refused_tokens: int
    idle_tokens: float
    excess: float
    staged_tokens: float
    harm_s: float


PRICE_RULES = ("fixed", "room", "harm")


class PriceController(msgspec.Struct):
    """Rule 4: the price of host memory, seconds of TTFT per token-second, closed
    every interval_s from what the room showed in it.

    rule "room" is dual ascent on the room: the price is multiplied by
    exp(eta * excess), excess being the tokens refused room in the interval minus the
    room left idle on average, over the room's size. rule "harm" sets the price to
    what staged reads cost others per token-second they held: the harm charged to
    them (seconds) over the token-seconds they were pinned, both fading with
    half_life_s and both starting from prior_token_s token-seconds at the starting
    price; with too little staged left to divide by, the price stays. rule "fixed"
    keeps the price and only reports."""

    price: float
    rule: str = "fixed"
    # Arbitrary; about a minute per tenfold change when the room is fully over or under used.
    eta: float = 0.2
    interval_s: float = 5.0
    floor: float = 1e-7
    ceiling: float = 1e-3
    # Arbitrary; long against a burst of refusals, short against a load episode (minutes).
    half_life_s: float = 120.0
    # Arbitrary; what about 50K staged tokens add up to over a few half-lives, so the
    # starting price holds through the first minutes and is gone after ten.
    prior_token_s: float = 1e7
    # Arbitrary; below this much faded staging the ratio is noise.
    min_staged_token_s: float = 1e5
    faded_harm_s: float = 0.0
    faded_staged_token_s: float = 0.0
    refused_tokens: int = 0
    harm_s: float = 0.0
    idle_token_s: float = 0.0
    staged_token_s: float = 0.0
    observed_s: float = 0.0
    last_observed: Optional[float] = None

    def __post_init__(self):
        if self.rule not in PRICE_RULES:
            raise ValueError(f"rule {self.rule!r} must be one of {PRICE_RULES}")
        if self.rule == "harm":
            self.faded_harm_s = self.price * self.prior_token_s
            self.faded_staged_token_s = self.prior_token_s

    def refuse(self, tokens: int) -> None:
        self.refused_tokens += tokens

    def charge(self, harm_s: float) -> None:
        """Harm a staged read's pin did to another turn, in seconds of TTFT."""
        self.harm_s += harm_s

    def observe(self, *, now: float, idle_tokens: int, staged_tokens: int = 0) -> None:
        """Fold in the idle room and the staged pins seen since the previous call."""
        if self.last_observed is not None and now > self.last_observed:
            elapsed = now - self.last_observed
            self.idle_token_s += idle_tokens * elapsed
            self.staged_token_s += staged_tokens * elapsed
            self.observed_s += elapsed
        self.last_observed = now

    def step(self, *, room_tokens: int) -> Optional[PriceUpdate]:
        """Close an interval once it has been observed, moving the price by the rule;
        None until then."""
        if self.observed_s < self.interval_s or room_tokens <= 0:
            return None
        idle_tokens = self.idle_token_s / self.observed_s
        excess = (self.refused_tokens - idle_tokens) / room_tokens
        fade = 0.5 ** (self.observed_s / self.half_life_s)
        self.faded_harm_s = self.faded_harm_s * fade + self.harm_s
        self.faded_staged_token_s = self.faded_staged_token_s * fade + self.staged_token_s
        if self.rule == "room":
            # The clamp below bounds the price; this one keeps exp() finite in a tiny room.
            self.price *= math.exp(max(-20.0, min(20.0, self.eta * excess)))
        elif self.rule == "harm" and self.faded_staged_token_s >= self.min_staged_token_s:
            self.price = self.faded_harm_s / self.faded_staged_token_s
        if self.rule != "fixed":
            self.price = min(self.ceiling, max(self.floor, self.price))
        update = PriceUpdate(
            price=self.price,
            refused_tokens=self.refused_tokens,
            idle_tokens=idle_tokens,
            excess=excess,
            staged_tokens=self.staged_token_s / self.observed_s,
            harm_s=self.harm_s,
        )
        self.refused_tokens = 0
        self.harm_s = 0.0
        self.idle_token_s = 0.0
        self.staged_token_s = 0.0
        self.observed_s = 0.0
        return update
