"""Policies for kvsim: what the engine decides, with the information the engine has.

Existing arms, written from the engine's code (python/sglang/srt/mem_cache/next_turn_prefetch.py, next_turn_policy.py,
next_turn_estimators.py, next_use_eviction.py, storage_read_scheduler.py, storage_prefetch.py, managers/scheduler.py):
  hicache    no hints; lookup at arrival and paced miss retries (8 scheduler steps apart, at most 8), a re-lookup at
             admission when the match shrank while that budget lasts; reads dispatched in arrival order, not pinned;
             LRU on GPU and host
  jit_p50    hints = the gap table's entry; reads before arrival timed to land at the entry's p10 before the predicted
             admission; arrival gate (3 lookups a turn), earliest-deadline read queue (512-page window, half the host
             pool), the hold of queued turns' reads until admission is near, pins until admission (0.3 of the pool),
             next-use eviction
  jit_exact  the same on the exact gap
  ec_full    the four rules: expected-cost read start on the entry's 64-point support, soft pins, return-chance
             eviction (30 s), the 'harm' price; plus the gate, the hold, pins
  ec_exact   the same on the exact gap
  nohint     no hints, but the queue-aware engine: gate, hold, pins, queued turns evicted last (campaign 26's
             read timing 'arrival'); never run on FA3, so it has no anchor
  unified    ec_full (or ec_exact with --hints exact) plus the switches of DESIGN.md section 9, each one small
             function below: F1 F2 F3 F4 F5 R1 R2 R3 R4 R5 P1 P2 PR1 ('all' turns every one on); CALIBRATION.md
             section 6 says how each was read from the draft

The write-path fixes of campaign 34 (cfg.fixes): skip stored pages, merged writes, the 50 K / 25 K write hold of timed
reads, room retry.
"""
from __future__ import annotations

import bisect
import itertools
import math
import random
from collections import deque
from statistics import NormalDist

PAGE = 64
THRESH = 256
Z90 = 1.2815515655446004
SIGMA_FLOOR, SIGMA_CEILING, GAP_FLOOR_S = 0.05, 6.0, 0.01
SUPPORT_POINTS = 64
MAX_PLANNED_GAP_S = 1200.0
MIN_REMAINING_POINTS = 4
RECOMPUTE_S_PER_TOKEN_TRITON = 1.1e-4
_N = NormalDist()
_QUANT_Z = [_N.inv_cdf((k + 0.5) / SUPPORT_POINTS) for k in range(SUPPORT_POINTS)]

# next_turn_prefetch.py
PROBE_INTERVAL_S = 0.5
MIN_EXPIRY_S = 60.0
MAX_REFETCHES = 2
NTP_ROOM_RETRY_S = 2.0
NTP_MAX_ROOM_RETRIES = 8
NTP_LEAD_HORIZON_S = 120.0
NTP_MAX_PARKED = 256
NTP_JIT_MARGIN_S = 1.0
NTP_HOLD_MARGIN_S = 2.0
NTP_PLAN_MAX_AGE_S = 1800.0
NTP_RELEASE_GRACE_S = 5.0
NTP_GPU_BOUND_S = 0.5
NTP_MAX_STAGINGS = 2
PIN_MAX_AGE_S = 600.0
# next_turn_estimators.py: the engine's read-time model (seeds; the per-page term follows the read thread's EWMA)
ALLOC_LAG_S, READ_FIXED_S, DRAIN_LAG_S, SEED_S_PER_PAGE = 0.5, 0.044, 0.36, 0.0025
ARRIVAL_OFFSET_SEED_S = 0.10
# next_use_eviction.py
NU_ENDED, NU_OVERDUE, NU_FAR, NU_UNKNOWN, NU_SOON, NU_QUEUED = 0, 1, 2, 3, 4, 5
RC_ENDED, RC_UNKNOWN, RC_SCORED, RC_QUEUED = 0, 1, 2, 3
POINT_GRACE_S = 5.0
QUEUED_MAX_AGE_S = 600.0
# hicache: paced miss retries (storage_prefetch.py; hicache_storage_prefetch_retry_*)
HICACHE_RETRY_STEPS = 8
HICACHE_MAX_ATTEMPTS = 8

SWITCHES = ("F1", "F2", "F3", "F4", "F5", "R1", "R2", "R3", "R4", "R5", "P1", "P2", "PR1")
# Added after the calibration, not part of 'all' (which stays draft v1's set): rules as revised in DESIGN_v2.md.
#   F4B: read or recompute as in DESIGN_v2 4.4 (see recompute_beats_read_v2).
LATER_SWITCHES = ("F4B",)
# Ceiling of the host-room price, seconds of TTFT per token-second (the engine's PriceController ceiling). The index
# of DESIGN.md 4.1 contains lambda through K_read, so the frontier price feeds itself and grows without bound once
# (read time) x (return hazard of the marginal holder) reaches 1; the ceiling keeps it finite.
LAMBDA_CEILING = 1e-3
_MISSING = object()


# ===== gap distributions (next_turn_policy.py) =====

class GapDist:
    """ln(gap) normal on each side of the median (the two-piece lognormal through p10/p50/p90); both sigmas zero
    is a point (an exact hint)."""

    __slots__ = ("median", "lo", "hi", "_support")

    def __init__(self, median: float, lo: float, hi: float):
        self.median, self.lo, self.hi = median, lo, hi
        self._support = None

    @classmethod
    def from_quantiles(cls, p10: float, p50: float, p90: float) -> "GapDist":
        median = max(p50, GAP_FLOOR_S)
        low = max(min(p10, median), GAP_FLOOR_S)
        high = max(p90, median)
        return cls(median, min(SIGMA_CEILING, max(SIGMA_FLOOR, math.log(median / low) / Z90)),
                   min(SIGMA_CEILING, max(SIGMA_FLOOR, math.log(high / median) / Z90)))

    @classmethod
    def point(cls, gap_s: float) -> "GapDist":
        return cls(max(gap_s, GAP_FLOOR_S), 0.0, 0.0)

    @property
    def is_point(self) -> bool:
        return self.lo == 0.0 and self.hi == 0.0

    def survival(self, gap_s: float) -> float:
        if gap_s <= 0.0:
            return 1.0
        if self.is_point:
            return 1.0 if gap_s < self.median else 0.0
        sigma = self.lo if gap_s < self.median else self.hi
        return 0.5 * math.erfc(math.log(gap_s / self.median) / sigma / 1.4142135623730951)

    def chance_within(self, idle_s: float, horizon_s: float) -> float:
        alive = self.survival(idle_s)
        if alive <= 0.0:
            return 0.0
        return 1.0 - self.survival(idle_s + horizon_s) / alive

    def quantile(self, q: float) -> float:
        if self.is_point:
            return self.median
        z = _N.inv_cdf(q)
        return self.median * math.exp((self.lo if z < 0.0 else self.hi) * z)

    def support(self) -> tuple:
        if self._support is None:
            if self.is_point:
                self._support = (self.median,)
            else:
                self._support = tuple(min(MAX_PLANNED_GAP_S, self.median * math.exp((self.lo if z < 0.0 else self.hi) * z))
                                      for z in _QUANT_Z)
        return self._support


def remaining_gaps(support: tuple, idle_s: float) -> tuple:
    first = bisect.bisect_right(support, idle_s)
    if len(support) > 1 and len(support) - first < MIN_REMAINING_POINTS:
        return ()
    return tuple(g - idle_s for g in support[first:])


def read_start(remaining: tuple, read_s: float, wait_s: float, rent_per_s: float, stall_weight: float = 1.0):
    """next_turn_policy.read_start (rule 1): seconds from now at which a staged read should start, 0 for now, None
    when reading at the turn's arrival costs no more. stall_weight is 1 in the engine; F2 passes c(t)."""
    stall_at_arrival = max(0.0, read_s - wait_s)
    count = len(remaining)
    if count == 0 or stall_at_arrival == 0.0:
        return None
    prefix = list(itertools.accumulate(remaining, initial=0.0))
    total = prefix[-1]
    lead = read_s - wait_s
    best_cost, best_start = stall_weight * stall_at_arrival, None
    for start in sorted({0.0, *(max(0.0, g - lead) for g in remaining)}):
        arrived = bisect.bisect_right(remaining, start)
        late = bisect.bisect_right(remaining, start + lead)
        stall = arrived * stall_at_arrival + ((late - arrived) * (start + lead) - (prefix[late] - prefix[arrived]))
        pinned = (total - prefix[late]) - (count - late) * (start + lead)
        cost = (stall_weight * stall + rent_per_s * pinned) / count
        if cost < best_cost:
            best_cost, best_start = cost, start
    return best_start


def gittins_index(dist: GapDist, age: float) -> float:
    """DESIGN.md 4.1: sup over h of F(h | age) / E[min(R, h)], per second; multiply by K / L for nu. The sup is
    taken over a geometric grid of horizons from 0.25 s to 1,200 s (the h -> 0 end is the hazard)."""
    if dist.is_point:
        left = dist.median - age
        if left <= -POINT_GRACE_S:
            return 0.0
        return 1.0 / max(left, 0.25)
    s0 = dist.survival(age)
    if s0 <= 1e-12:
        return 0.0
    best = 0.0
    tau = 0.0
    prev_h, prev_s = 0.0, 1.0
    h = 0.25
    while h <= 1200.0:
        s = dist.survival(age + h) / s0
        tau += 0.5 * (prev_s + s) * (h - prev_h)
        if tau > 0:
            v = (1.0 - s) / tau
            if v > best:
                best = v
        prev_h, prev_s = h, s
        h *= 2.0
    return best


class PriceController:
    """next_turn_policy.PriceController, rule 'harm' or 'fixed'."""

    def __init__(self, price: float, rule: str):
        self.price, self.rule = price, rule
        self.interval_s, self.floor, self.ceiling, self.half_life_s = 5.0, 1e-7, 1e-3, 120.0
        self.prior_token_s, self.min_staged_token_s = 1e7, 1e5
        self.faded_harm_s = price * self.prior_token_s if rule == "harm" else 0.0
        self.faded_staged = self.prior_token_s if rule == "harm" else 0.0
        self.refused_tokens = 0
        self.harm_s = self.staged_token_s = self.observed_s = 0.0
        self.last = None
        self.updates = 0
        self.moved = 0

    def refuse(self, tokens: int) -> None:
        self.refused_tokens += tokens

    def charge(self, harm_s: float) -> None:
        self.harm_s += harm_s

    def observe(self, now: float, staged_tokens: int) -> None:
        if self.last is not None and now > self.last:
            self.staged_token_s += staged_tokens * (now - self.last)
            self.observed_s += now - self.last
        self.last = now

    def step(self) -> None:
        if self.observed_s < self.interval_s:
            return
        fade = 0.5 ** (self.observed_s / self.half_life_s)
        self.faded_harm_s = self.faded_harm_s * fade + self.harm_s
        self.faded_staged = self.faded_staged * fade + self.staged_token_s
        old = self.price
        if self.rule == "harm" and self.faded_staged >= self.min_staged_token_s:
            self.price = min(self.ceiling, max(self.floor, self.faded_harm_s / self.faded_staged))
        self.updates += 1
        if abs(self.price - old) > 1e-12:
            self.moved += 1
        self.refused_tokens = 0
        self.harm_s = self.staged_token_s = self.observed_s = 0.0


class Plan:
    """next_turn_prefetch.TurnPlan: what the engine keeps about a session's next turn after a turn commits."""

    __slots__ = ("rn", "chain", "chain_tokens", "t_commit", "gap_s", "predicted_arrival", "expires_at", "next_check",
                 "early_s", "dist", "support", "staged", "stagings", "lookups", "refetches", "capped",
                 "room_refused", "room_refused_at", "room_retries", "last_probe", "last_refetch", "missing",
                 "room_tokens", "arrival_t", "decision", "sess", "pending_room", "counted", "room_wait_since")

    def __init__(self):
        self.staged = False
        self.stagings = self.lookups = self.refetches = self.room_retries = 0
        self.capped = self.room_refused = False
        self.room_refused_at = 0.0
        self.last_probe = self.last_refetch = -1e9
        self.missing = self.room_tokens = 0
        self.arrival_t = None
        self.decision = ""
        self.pending_room = False
        self.counted = False
        self.room_wait_since = None


# ===== the policy =====

class Policy:
    """One class; an arm is a set of attributes (make_policy). Every rule sits in one small method."""

    def __init__(self, cfg, *, name, hint_mode, order, prefetcher, timing, eviction, pin_fraction, soft_pins,
                 price_rule, switches=()):
        self.cfg, self.name = cfg, name
        self.hint_mode = cfg.hints or hint_mode
        self.order, self.prefetcher, self.timing, self.eviction = order, prefetcher, timing, eviction
        self.pin_fraction, self.soft_pins = pin_fraction, soft_pins
        self.sw = frozenset(switches)
        fixes = bool(cfg.fixes)
        self.skip_stored = fixes                     # fix 1
        self.coalesce = fixes                        # fix 2: merged writes (no effect of its own in this model)
        self.write_hold_tokens = 50000 if fixes and order == "deadline" else 0     # fix 3
        self.room_retry = fixes and prefetcher       # fix 4
        self.gate = self.hold = prefetcher
        self.dynamic = eviction != "lru"
        self.evict_stop = self._p1_stop if "P1" in self.sw else None
        self.price = PriceController(5e-5, price_rule) if timing == "ec" else None
        self.rng = random.Random(f"{cfg.seed}:policy")

    def describe(self) -> dict:
        return {"name": self.name, "hints": self.hint_mode, "order": self.order, "timing": self.timing,
                "eviction": self.eviction, "pin_fraction": self.pin_fraction, "soft_pins": self.soft_pins,
                "switches": sorted(self.sw), "fixes": bool(self.cfg.fixes), "mirror": bool(self.cfg.mirror),
                "price_updates": None if self.price is None else self.price.updates,
                "price_moved": None if self.price is None else self.price.moved,
                "price": None if self.price is None else self.price.price}

    def attach(self, sim) -> None:
        self.sim = sim
        cost = sim.cost
        self.gamma_a, self.gamma_b = cost["a"], cost["b"]
        self.recompute_s_per_token = (cost["a"] + cost["b"] * 25000.0) if "PR1" in self.sw else RECOMPUTE_S_PER_TOKEN_TRITON
        self.wqueue: deque = deque()
        self.parked: dict = {}
        self.pending: dict = {}        # plans whose turn has not arrived (a dict: iteration order must not depend on addresses)
        self.positions: dict = {}
        self.last_rate = None
        self.write_held = False
        self.offset_s = ARRIVAL_OFFSET_SEED_S
        self.wait_ewma = None
        self.read_seq = 0
        self.ready_since: dict = {}
        self.staged_pin_seen = 0
        self.room_refusals: list = []
        # eviction registrations
        self.reg: dict = {}            # tail node -> None (ended) | float | ('F', dist, idle_from) | ('Q', since)
        self.moved: dict = {}
        self.tail_of: dict = {}
        self.sess_tail: dict = {}
        n = sim.wl.n_nodes
        self.node_sess = [None] * n    # the last session that committed over a node (unified index, lambda)
        self.sess_fc: dict = {}        # session -> ('F', dist, idle_from) | None (ended) | ('Q', since, request)
        # measured prices (DESIGN.md section 6)
        self.c_ewma = 1.0
        self.lam_frontier: deque = deque()
        self.sigma = 0.0
        self.wait_ratio: deque = deque(maxlen=256)   # F1: measured admission wait over the estimate at arrival
        self.mu_now = 1.0
        self.kappa_now = 0.0
        self.p1_phase = 0
        self.p2_release = None

    # ----- small helpers -----

    def load_seconds(self, tokens: int, backlog_pages: int) -> float:
        """The engine's read-time model: fixed lags plus the read thread's ms per page for own and backlog pages."""
        spp = self.sim.read_ewma / 1000.0 if self.sim.read_samples >= 8 else SEED_S_PER_PAGE
        return ALLOC_LAG_S + READ_FIXED_S + spp * math.ceil(tokens / PAGE) + spp * backlog_pages + DRAIN_LAG_S

    def residual_wait(self, position: int):
        return None if self.last_rate is None else (position + 0.5) / self.last_rate

    def wait_after_arrival(self, queue_len: int):
        if queue_len == 0:
            return 0.0
        w = self.residual_wait(queue_len)
        return w if w is not None else self.wait_ewma

    def backlog_pages(self) -> int:
        """Pages the planner counts ahead of a read: those already dispatched; F3 adds the parked line and the rest
        of a write hold."""
        pages = self.sim.inflight_pages
        if "F3" in self.sw:
            pages += sum(rd.pages for rd in self.parked)
            if self.write_held:
                spp = self.sim.read_ewma / 1000.0 if self.sim.read_samples >= 8 else SEED_S_PER_PAGE
                pages += int(max(0, self.sim.pending_write - self.write_hold_tokens // 2) / self.cfg.write_alone / spp)
        return pages

    def missing_tokens(self, chain) -> int:
        """Tokens of a chain that are not in memory (walked from the tail: the resident set is prefix-closed)."""
        sim = self.sim
        in_gpu, in_host, tok = sim.in_gpu, sim.in_host, sim.tok
        missing = 0
        for i in range(len(chain) - 1, -1, -1):
            n = chain[i]
            if in_gpu[n] or in_host[n]:
                break
            missing += tok[n]
        return missing

    def measured_c(self) -> float:
        return self.c_ewma

    def measured_mu(self) -> float:
        return self.mu_now

    def refresh_prices(self) -> None:
        """mu and kappa as of this tick (DESIGN.md section 6)."""
        sim, now = self.sim, self.sim.now
        ready = sum(1 for r in sim.waitq if r.reading is None)
        self.mu_now = float(min(1 + ready, max(1, sim.in_session)))
        late = 0
        for rd in self.parked:
            if rd.r.t_send is not None and (rd.deadline is None or rd.deadline - self.load_seconds(rd.tokens, 0) < now):
                late += 1
        k = self.c_ewma * late
        if self.write_held:
            k += self.measured_lambda() * sim.pending_write
        self.kappa_now = k

    def measured_lambda(self) -> float:
        """Host room's price: the index of the marginal holder (the highest index given up in the last 30 s);
        0 while room is free."""
        now = self.sim.now
        fr = self.lam_frontier
        while fr and fr[0][0] < now - 30.0:
            fr.popleft()
        return min(LAMBDA_CEILING, max((v for _, v in fr), default=0.0))

    def price_lambda(self) -> float:
        if "PR1" in self.sw or self.price is None:
            return self.measured_lambda()
        return self.price.price

    def measured_kappa(self) -> float:
        """Device time's price: the lateness cost rate of the transfers waiting: c for each parked read of an
        arrived turn past its latest start, lambda x tokens for pending writes while reads are held for them."""
        return self.kappa_now

    def prices(self, ready: int) -> dict:
        sim = self.sim
        self.refresh_prices()
        return {"mu": float(min(1 + ready, max(1, sim.in_session))), "c": round(self.c_ewma, 4),
                "kappa": round(self.measured_kappa(), 6), "lambda": self.measured_lambda(), "sigma": self.sigma,
                "engine": None if self.price is None else self.price.price}

    def parked_count(self) -> int:
        return len(self.parked)

    # ----- hints at commit -----

    def resolve_hint(self, r):
        """(gap_s, early_s, dist) of the finished turn's hint, or None: the session ended, or nothing was sent."""
        h = r.hint
        if h is None or h.get("end") or h.get("program"):
            return None
        if "exact" in h:
            return h["exact"], 0.0, GapDist.point(h["exact"])
        return h["p50"], max(0.0, h["p50"] - h["p10"]), GapDist.from_quantiles(h["p10"], h["p50"], h["p90"])

    def session_ended(self, r) -> bool:
        """R1: the agent said this turn ends its session (the table's end key; the last turn under exact hints)."""
        h = r.hint
        if self.hint_mode == "dist":
            return bool(h and h.get("end"))
        if self.hint_mode == "exact":
            return h is None and r.is_last
        return False

    # ----- eviction order -----

    def prio(self, n: int) -> tuple:
        if self.eviction == "lru":
            return (self.sim.stamp[n],)
        if self.eviction == "next_use":
            return self._prio_next_use(n)
        if self.eviction == "index":
            return self._prio_index(n)
        return self._prio_return_chance(n)

    def _prio_next_use(self, n: int) -> tuple:
        v = self.reg.get(n, _MISSING)
        stamp = self.sim.stamp[n]
        if v is _MISSING:
            return (NU_UNKNOWN, stamp)
        if v is None:
            return (NU_ENDED, stamp)
        now = self.sim.now
        if type(v) is tuple:
            if now - v[1] > QUEUED_MAX_AGE_S:
                return (NU_UNKNOWN, stamp)
            return (NU_QUEUED, -self._queued_wait(v[2])) if "P1" in self.sw else (NU_QUEUED, stamp)
        delta = v - now
        if delta < -60.0:
            return (NU_OVERDUE, v)
        if delta > 300.0:
            return (NU_FAR, -v)
        return (NU_SOON, -v)

    def _session_registration(self, n: int):
        """next_use_eviction.ReturnChanceStrategy._session_registration: a node takes the registration of the tail
        it was adopted by, or of the parent it is the only child of."""
        sim, reg, tail_of = self.sim, self.reg, self.tail_of
        par = sim.par[n]
        only_child = par >= 0 and sim.nct[par] == 1
        if par >= 0 and not only_child:
            tail_of.pop(par, None)
        if n in reg:
            tail = n
        else:
            tail = tail_of.get(n)
            if tail is None and only_child:
                tail = par if par in reg else tail_of.get(par)
            if tail is None:
                return _MISSING
            moved = self.moved
            for _ in range(4096):
                m = moved.get(tail)
                if m is None:
                    break
                tail = m
            if tail not in reg:
                return _MISSING
            tail_of[n] = tail
        if only_child and par not in reg:
            tail_of[par] = tail
        return reg[tail]

    def _prio_return_chance(self, n: int) -> tuple:
        v = self._session_registration(n)
        stamp = self.sim.stamp[n]
        if v is _MISSING:
            return (RC_UNKNOWN, stamp, 0.0)
        if v is None:
            return (RC_ENDED, stamp, 0.0)
        now = self.sim.now
        if v[0] == "Q":
            if now - v[1] > QUEUED_MAX_AGE_S:
                return (RC_UNKNOWN, stamp, 0.0)
            return (RC_QUEUED, -self._queued_wait(v[2]), stamp) if "P1" in self.sw else (RC_QUEUED, stamp, 0.0)
        dist, idle_from = v[1], v[2]
        idle = now - idle_from
        if dist.is_point and idle >= dist.median:
            if idle < dist.median + POINT_GRACE_S:
                return (RC_SCORED, 1.0, 0.0)
            return (RC_SCORED, 0.0, -math.inf)
        return (RC_SCORED, dist.chance_within(idle, 30.0), idle - dist.median)

    def restore_cost_per_token(self, n: int) -> float:
        """K / L of DESIGN.md section 2 for node n: a read if the SSD holds it (lags over length, device time at
        kappa, memory at lambda), a recompute at mu x gamma if this is the only copy (R5)."""
        sim = self.sim
        L = sim.tok[n]
        if "R5" in self.sw and (sim.unwritten[n] or not sim.ssd[n]) and not sim.dirty[n]:
            return self.measured_mu() * (self.gamma_a + self.gamma_b * 25000.0)
        r = self.load_seconds(L, sim.inflight_pages)
        return self.c_ewma * 0.9 / max(L, PAGE) + self.measured_kappa() / self.cfg.beta + max(self.price_lambda(), 1e-9) * r

    def _prio_index(self, n: int) -> tuple:
        """DESIGN.md 4.1: evict in increasing nu = (K / L) x sup_h F(h | a) / tau(h | a). Ended sessions first;
        nodes of no single session by LRU next; queued turns last, the farthest from admission first."""
        sim = self.sim
        stamp = sim.stamp[n]
        sess = self.node_sess[n]
        fc = self.sess_fc.get(sess, _MISSING) if sess is not None else _MISSING
        if fc is _MISSING:
            return (1, stamp, 0.0)
        if fc is None:
            return (0, stamp, 0.0)
        k = self.restore_cost_per_token(n)
        now = sim.now
        if fc[0] == "Q":
            return (3, -self._queued_wait(fc[2]), stamp)
        return (2, k * gittins_index(fc[1], now - fc[2]), stamp)

    def _queued_wait(self, r) -> float:
        """Seconds until the queued request r is admitted, by its position (0 when unknown or at the head)."""
        pos = self.positions.get(r)
        w = self.residual_wait(pos) if pos is not None else None
        return w if w is not None else 0.0

    def _p1_stop(self, pr: tuple, n: int, asker) -> bool:
        """P1 (DESIGN.md 5.1): a read (asker is not None) short of room does not take a queued turn's history
        until staged reads have yielded (phase 1), and then only from turns far enough from admission to read it
        again: slack W_k - r > 0 (phase 2). Copies of new pages (asker None) follow the engine."""
        queued_tier = {"index": 3, "return_chance": RC_QUEUED, "next_use": NU_QUEUED}.get(self.eviction)
        if asker is None or queued_tier is None or pr[0] < queued_tier:
            return False
        if self.p1_phase < 2:
            return True
        wait = -pr[1]
        return wait < self.load_seconds(self.sim.tok[n], self.sim.inflight_pages) + NTP_HOLD_MARGIN_S

    def dynamic_candidates(self):
        """Candidates whose priority moves with the clock (registered or adopted nodes)."""
        sim = self.sim
        if self.eviction == "index":
            ns = self.node_sess
            return [n for n in itertools.chain(sim.hcand, sim.gcand) if ns[n] is not None]
        reg, tail_of = self.reg, self.tail_of
        if self.eviction == "next_use":
            return [n for n in itertools.chain(sim.hcand, sim.gcand) if n in reg]
        return list(itertools.chain(sim.hcand, sim.gcand))

    def on_evict(self, n: int, pr: tuple) -> None:
        """A host eviction: remember the index of what was given up (lambda is read off this frontier)."""
        sess = self.node_sess[n]
        if sess is None:
            return
        fc = self.sess_fc.get(sess)
        if fc is None or fc[0] != "F":
            return
        v = self.restore_cost_per_token(n) * gittins_index(fc[1], self.sim.now - fc[2])
        self.lam_frontier.append((self.sim.now, min(v, LAMBDA_CEILING)))

    def on_node_gone(self, n: int) -> None:
        pass

    def demote_on_gpu_evict(self, n: int) -> bool:
        """Mirror off: when the GPU gives a page up, keep it in host memory unless its session ended."""
        sess = self.node_sess[n]
        return not (sess is not None and self.sess_fc.get(sess, _MISSING) is None)

    def on_served(self, nodes, upto: int) -> None:
        """R4: memory served these pages, so their SSD copies are touched (heads stay younger than tails)."""
        if "R4" in self.sw:
            sim = self.sim
            ssd = sim.ssd
            for i in range(upto):
                if ssd[nodes[i]]:
                    sim.ssd_touch(nodes[i])

    # ----- registrations -----

    def _register(self, sess, tail, value) -> None:
        if tail is None or self.eviction == "lru":
            return
        prev = self.sess_tail.get(sess)
        if prev is not None and prev != tail:
            self.reg.pop(prev, None)
            self.moved[prev] = tail
            self.sim.touch(prev)
        self.sess_tail[sess] = tail
        self.reg[tail] = value
        self.sim.touch(tail)

    # ----- arrival -----

    def on_arrival(self, r) -> None:
        sim = self.sim
        ig, it = sim.match(r.path)
        cum = sim.cum(r)
        sim.stamp_path(r.path, it)
        r.last_match = cum[it]
        r.attempts = 0
        if not self.prefetcher:
            self._hicache_prefetch(r, it)
            return
        plan = r.plan
        if plan is not None:
            plan.arrival_t = sim.now
            self.pending.pop(plan, None)
            offset = sim.now - plan.t_commit - plan.gap_s
            if plan.dist is not None and 0.0 <= offset <= 5.0:      # out-of-range offsets are dropped, not clipped
                self.offset_s += (offset - self.offset_s) / 8
            sess = plan.sess
            tail = self.sess_tail.get(sess)
            self._register(sess, tail, ("Q", sim.now, r))
            self.sess_fc[sess] = ("Q", sim.now, r)
        r.pred_wait = self.wait_after_arrival(len(sim.waitq) - 1)
        if r.reading is not None:
            return                      # joins its staged read
        self._prefetch(r, it)

    def _prefetch(self, r, it: int) -> bool:
        """Scheduler._prefetch_kvcache plus the arrival gate: may this request look up the SSD now?"""
        sim = self.sim
        matched = sim.cum(r)[it]
        r.last_match = matched
        if r.pages * PAGE - matched < THRESH:
            return False
        plan = r.plan
        if plan is not None and self.gate:
            if plan.chain_tokens - matched < THRESH:
                plan.room_refused = False              # covered: the history is resident, the rest is new
                return False
            if self._room_retry_due(plan):
                self._take_room_retry(plan)
                plan.counted = False
            elif plan.lookups > MAX_REFETCHES:
                if not plan.capped:
                    sim.cnt["capped"] += 1
                sim.cnt["capped_gate"] += 1
                plan.capped = True
                return False
            else:
                plan.lookups += 1
                plan.room_refused = False
                plan.counted = True
        r.lookups += 1
        return sim.lookup(r, r.path, it)

    def on_lookup_miss(self, r, rd) -> None:
        if not self.prefetcher:
            self._hicache_miss(r)
            return
        plan = r.plan
        if plan is not None and plan.counted and "F5" in self.sw:
            plan.lookups -= 1                           # F5: a lookup that finds nothing is free
            plan.counted = False

    def on_lookup_hit(self, r, rd) -> None:
        self.read_seq += 1
        rd.seq = self.read_seq
        if self.order == "fifo":
            self.sim.launch(rd)
            return
        if r.t_send is not None and (
            ("F4" in self.sw and self.recompute_beats_read(r, rd))
            or ("F4B" in self.sw and self.recompute_beats_read_v2(r, rd))
        ):
            self.sim.cnt["recompute_by_choice"] += 1
            self.sim._revoke(rd)
            self.sim._ready(r)
            return
        self.parked[rd] = True
        self.dispatch()

    def recompute_beats_read(self, r, rd) -> bool:
        """F4 (DESIGN.md section 2): restore by the cheaper of K_read = c (r - W)+ + kappa L / beta + lambda L r and
        K_rec = mu x (GPU seconds to compute L tokens at their position)."""
        sim = self.sim
        L = rd.tokens
        pos = self.positions.get(r)
        if pos is None:
            pos = len(sim.waitq) - 1
        w = self.residual_wait(pos) or 0.0
        r_est = self.load_seconds(L, sim.inflight_pages + sum(x.pages for x in self.parked))
        k_read = self.c_ewma * max(0.0, r_est - w) + self.measured_kappa() * L / self.cfg.beta + self.price_lambda() * L * r_est
        x0 = r.last_match
        k_rec = self.measured_mu() * (self.gamma_a * L + self.gamma_b * (x0 * L + 0.5 * L * L))
        return k_rec < k_read

    def recompute_beats_read_v2(self, r, rd) -> bool:
        """F4 as revised (DESIGN_v2.md 4.4, engine amendment A2). Recompute only where a GPU slot would stay
        empty while the read runs: no ready turn waiting and the measured c at 0.8 or more. The job then delays
        itself and the turns that arrive while it runs (mu = 1 + rate x job / 2), and the read's cost is the
        time it would take, with no rent for the time it waits in the device's line."""
        sim = self.sim
        if self.c_ewma < 0.8 or any(x.reading is None for x in sim.waitq if x is not r):
            return False
        L = rd.tokens
        r_est = self.load_seconds(L, sim.inflight_pages + sum(x.pages for x in self.parked))
        x0 = r.last_match
        rec = self.gamma_a * L + self.gamma_b * (x0 * L + 0.5 * L * L)
        rate = self.last_rate or 0.0
        return (1.0 + rate * rec / 2.0) * rec < r_est

    # ----- hicache's own lookup budget -----

    def _hicache_prefetch(self, r, it: int) -> None:
        sim = self.sim
        matched = sim.cum(r)[it]
        r.last_match = matched
        if r.pages * PAGE - matched < THRESH:
            return
        r.lookups += 1
        sim.lookup(r, r.path, it)

    def _hicache_miss(self, r) -> None:
        """A lookup that found less than the threshold arms a retry 8 scheduler steps later, at most 8 a request;
        the queue head's is cancelled (storage_prefetch.StoragePrefetchRetries)."""
        sim = self.sim
        if r.t_admit is not None or r.attempts >= HICACHE_MAX_ATTEMPTS:
            return
        if "F5" in self.sw:
            r.attempts = max(0, r.attempts - 1)          # F5: the lookup found nothing and is not charged
            return
        step = sim.step_s if sim.n_run > 0 else 0.002
        sim.at(sim.now + HICACHE_RETRY_STEPS * step, self._hicache_retry, r)

    def _hicache_retry(self, r) -> None:
        sim = self.sim
        if r.t_admit is not None or r.reading is not None or r.attempts >= HICACHE_MAX_ATTEMPTS:
            return
        if sim.waitq and sim.waitq[0] is r:
            return
        r.attempts += 1
        if r.attempts >= HICACHE_MAX_ATTEMPTS:
            sim.cnt["storm_cap"] += 1
        ig, it = sim.match(r.path)
        sim.stamp_path(r.path, it)
        self._hicache_prefetch(r, it)

    def before_admit(self, r, resident: int) -> bool:
        """The admission loop reached r: if the match shrank since the last lookup and re-issues are left (8 a
        request), issue the lookup again and skip r this pass (Scheduler._prefetch_after_device_hit_loss; every
        arm). hicache's paced miss retries spend the same budget; the hinted arms' lookups also pass the gate."""
        if resident >= r.last_match:
            return False
        if r.attempts >= HICACHE_MAX_ATTEMPTS:
            r.last_match = resident
            return False
        r.attempts += 1
        sim = self.sim
        sim.cnt["admit_requery"] += 1
        if r.attempts >= HICACHE_MAX_ATTEMPTS:
            sim.cnt["storm_cap"] += 1
        ig, it = sim.match(r.path)
        sim.stamp_path(r.path, it)
        if self.prefetcher:
            self._prefetch(r, it)                       # through the gate, as every read trigger
        else:
            self._hicache_prefetch(r, it)
        return r.reading is not None

    # ----- room -----

    def make_room(self, rd) -> int:
        """Host room for a read being dispatched: tokens granted (all, or a cut), 0 = refused, -1 = wait (P1).
        Engine order (unified_radix_cache._launch_storage_hit): evict every evictable leaf in policy order; then,
        with soft pins and a queued asker, release staged pins and evict again; then cut to the room left if that
        is at least the threshold; otherwise refuse."""
        sim = self.sim
        need = rd.tokens
        self.p1_phase = 1
        if sim.evict_host(need, asker=rd):
            return need
        r = rd.r
        queued = r.t_send is not None and r.t_admit is None
        if (self.soft_pins or "P1" in self.sw) and queued:
            free = sim.C_H - sim.host_used
            for victim in self.soft_pin_victims(rd, need - free):
                sim.unpin(victim)
            if sim.evict_host(need, asker=rd):
                return need
        if "P1" in self.sw:
            self.p1_phase = 2
            if queued and sim.evict_host(need, asker=rd):
                return need
            if rd.t_disp is None:
                rd.t_disp = -sim.now - 1e-9             # negative: the time it began to wait for room
                sim.cnt["room_waits"] += 1
            if sim.now + rd.t_disp < 60.0:
                return -1                               # delay, not refusal (DESIGN.md 5.1); 60 s is a safety valve
        if "P2" in self.sw and sim.pending_write > 0:
            # writes first: the read keeps its place until the pending writes have released what it needs
            if self.p2_release is None:
                self.p2_release = max(0, sim.pending_write - need)
                sim.cnt["write_hold_on"] += 1
            self.write_held = True
            return -1
        free = sim.C_H - sim.host_used
        return free if free >= THRESH else 0

    def soft_pin_victims(self, rd, tokens_needed: int) -> list:
        """Rule 2: staged reads of turns not yet back, least likely back within their own read time first."""
        sim, now = self.sim, self.sim.now
        staged = []
        for plan in self.pending:
            rn = plan.rn
            if plan.dist is None or rn.reading is not None:
                continue
            tokens = sim.pinned_of(rn)
            if tokens <= 0:
                continue
            read_s = self.load_seconds(tokens, 0) + NTP_JIT_MARGIN_S
            chance = plan.dist.chance_within(now - plan.t_commit - self.offset_s, read_s)
            staged.append((chance, tokens, chance, read_s, plan))
        staged.sort(key=lambda e: e[0])
        victims, freed = [], 0
        for score, tokens, chance, read_s, plan in staged:
            if freed >= tokens_needed:
                break
            freed += tokens
            plan.staged = False
            plan.decision = ""
            victims.append(plan.rn)
            sim.cnt["bumps"] += 1
            if self.price is not None:
                self.price.refuse(tokens)
                self.price.charge(chance * read_s)
        return victims

    def on_refused(self, r, rd, cut: bool) -> None:
        r.refused += 1
        plan = r.plan
        self.room_refusals.append((r, rd.tokens))
        if plan is not None:
            plan.room_refused = True
            plan.room_refused_at = self.sim.now

    def _room_retry_open(self, plan) -> bool:
        return self.room_retry and plan.room_refused and plan.room_retries < NTP_MAX_ROOM_RETRIES

    def _room_retry_due(self, plan) -> bool:
        return self._room_retry_open(plan) and self.sim.now - plan.room_refused_at >= NTP_ROOM_RETRY_S

    @staticmethod
    def _take_room_retry(plan) -> None:
        if plan.room_refused:
            plan.room_refused = False
            plan.room_retries += 1

    # ----- landing and pins -----

    def on_landed(self, r, rd) -> None:
        sim = self.sim
        if self.eviction == "return_chance" and r.plan is not None:
            tail = self.sess_tail.get(r.plan.sess)
            if tail is not None:
                for n in rd.nodes:
                    self.tail_of[n] = tail
        if r.plan is not None:
            sess = r.plan.sess
            ns = self.node_sess
            for n in rd.nodes:
                ns[n] = sess
        if self.pin_fraction <= 0 or self.order != "deadline" or not rd.nodes:
            return
        tokens = sum(sim.tok[n] for n in rd.nodes)
        if sim.pinned_tokens + tokens <= self.pin_fraction * sim.C_H:
            sim.pin_read(r, rd)
            sim.at(sim.now + PIN_MAX_AGE_S, self._pin_timeout, (r, rd))
        else:
            sim.cnt["pin_skips"] += 1

    def _pin_timeout(self, arg) -> None:
        r, rd = arg
        if rd.pinned and r.t_admit is None:
            self.sim.unpin(r)

    # ----- the read queue (storage_read_scheduler.py) -----

    def _update_write_hold(self) -> None:
        sim = self.sim
        if "P2" in self.sw:
            self.write_held = self._p2_writes_first()
            return
        if self.write_hold_tokens <= 0:
            return
        was = self.write_held
        if sim.pending_write >= self.write_hold_tokens:
            self.write_held = True
        elif sim.pending_write <= self.write_hold_tokens // 2:
            self.write_held = False
        if self.write_held and not was:
            sim.cnt["write_hold_on"] += 1

    def _p2_writes_first(self) -> bool:
        """P2 (DESIGN.md 5.2): timed reads yield to writes only while dirty pages hold room that due reads need.
        Two triggers, both released by write acknowledgements: a read that found no room while writes were pending
        (the line then waits until the writes have given back that much), and a clean reserve (free room plus the
        evictable leaves, counted only when they are few) under the next due read."""
        sim = self.sim
        if sim.pending_write <= 0:
            self.p2_release = None
            return False
        if self.p2_release is not None:
            if sim.pending_write > self.p2_release:
                return True
            self.p2_release = None
        held = False
        if self.parked and len(sim.hcand) < 32:
            now = sim.now
            due = min((rd.tokens for rd in self.parked
                       if rd.deadline is None or rd.deadline - self.load_seconds(rd.tokens, 0) <= now), default=0)
            reserve = (sim.C_H - sim.host_used) + sum(sim.tok[n] for n in sim.hcand)
            held = 0 < due and reserve < due
        if held and not self.write_held:
            sim.cnt["write_hold_on"] += 1
        return held

    def read_deadline(self, rd):
        """next_turn_prefetch.read_deadline: None = needed now."""
        r = rd.r
        plan = r.plan
        if plan is None and not self.hold:
            return None
        now = self.sim.now
        if plan is not None and plan.arrival_t is None:
            wait = self.wait_after_arrival(len(self.sim.waitq))
            deadline = plan.predicted_arrival + (wait or 0.0)
            if self.timing == "ec" or plan.early_s < 0.0:
                return max(deadline, now + NTP_JIT_MARGIN_S)
            return deadline if deadline > now else None
        pos = self.positions.get(r)
        if pos is None or pos <= 1:
            return None
        wait = self.residual_wait(pos)
        return None if wait is None else now + wait

    def read_not_before(self, rd, ahead_pages: int):
        """next_turn_prefetch.read_not_before: the hold of a queued turn's read until admission is near."""
        if not self.hold:
            return None
        r = rd.r
        pos = self.positions.get(r)
        if pos is None or pos <= 1:
            return None
        wait = self.residual_wait(pos)
        if wait is None:
            return None
        spp = self.sim.read_ewma / 1000.0 if self.sim.read_samples >= 8 else SEED_S_PER_PAGE
        load_s = ALLOC_LAG_S + READ_FIXED_S + spp * rd.pages + spp * ahead_pages + DRAIN_LAG_S
        now = self.sim.now
        if "F1" in self.sw:
            return self.landing_quantile_start(rd, wait, load_s)
        return now + wait - load_s - NTP_HOLD_MARGIN_S

    def landing_quantile_start(self, rd, wait: float, load_s: float) -> float:
        """F1 (DESIGN.md 3.2): land at the q-quantile of the admission time, q = lambda L / (c + lambda L); the
        admission time's spread is the measured ratio of waits to their estimates at arrival."""
        lam, c = max(self.price_lambda(), 0.0), self.c_ewma
        q = lam * rd.tokens / (c + lam * rd.tokens) if c + lam * rd.tokens > 0 else 1.0
        ratios = sorted(self.wait_ratio)
        if len(ratios) >= 16:
            factor = ratios[min(len(ratios) - 1, int(q * (len(ratios) - 1) + 0.5))]
        else:
            factor = 1.0
        return self.sim.now + wait * factor - load_s

    def dispatch(self) -> None:
        if self.order != "deadline":
            return
        sim = self.sim
        self._update_write_hold()
        if not self.parked:
            return
        now = sim.now
        self.positions = {r: i for i, r in enumerate(sim.waitq)}
        self.last_rate = sim.adm_rate(now)
        reads = list(self.parked)
        for rd in reads:
            rd.deadline = self.read_deadline(rd)
        reads.sort(key=lambda rd: (rd.deadline is not None, rd.deadline or 0.0, rd.seq))
        held = sim.inflight_tokens + sim.pinned_tokens
        in_flight = sim.inflight_pages
        ahead = in_flight
        budget = 0.5 * sim.C_H
        for rd in reads:
            nb = self.read_not_before(rd, ahead)
            ahead += rd.pages
            if nb is not None and now < nb:
                continue
            if rd.deadline is not None:
                if self.write_held or in_flight >= 512 or held + rd.tokens > budget:
                    break
            res = sim.launch(rd)
            if res is None:                              # P1: waits for room, stays in the line
                continue
            del self.parked[rd]
            if res:
                in_flight += rd.pages
                held += rd.tokens

    # ----- the tick (next_turn_prefetch.tick) -----

    def tick(self) -> None:
        sim = self.sim
        now = sim.now
        self.refresh_prices()
        if not self.prefetcher:
            return
        self.positions = {r: i for i, r in enumerate(sim.waitq)}
        self.last_rate = rate = sim.adm_rate(now)
        queue_ahead = self._queue_ahead(now)
        if self.price is not None:
            self._follow_room(queue_ahead, now)
        backlog = self.backlog_pages()
        refetch = []
        admissible = 0
        for pos, r in enumerate(sim.waitq):
            admissible_ahead = admissible
            if self.room_retry and r.reading is None:
                admissible += 1
            plan = r.plan
            if plan is None:
                continue
            if self._check_queued(plan, r, pos, admissible_ahead):
                refetch.append(r)
        for r in refetch:
            self._refetch(r)
        if self.pending:
            for plan in [p for p in self.pending if p.next_check <= now]:
                nxt = self._check_pending(plan, queue_ahead, backlog)
                if nxt is None:
                    self.pending.pop(plan, None)
                    if plan.rn.plan is plan:
                        plan.rn.plan = None
                else:
                    plan.next_check = nxt
        self.dispatch()

    def _queue_ahead(self, now: float) -> int:
        """Queue a turn arriving now would wait behind. Expected-cost timing counts it only while a queued turn
        with no read of its own has waited 0.5 s (the GPU is the bottleneck)."""
        sim = self.sim
        if self.timing != "ec":
            return len(sim.waitq)
        ready_since = {}
        old = self.ready_since
        for r in sim.waitq:
            if r.reading is None:
                ready_since[r] = old.get(r, now)
        self.ready_since = ready_since
        if any(now - s >= NTP_GPU_BOUND_S for s in ready_since.values()):
            return len(sim.waitq)
        return 0

    def _check_queued(self, plan, r, pos: int, admissible_ahead: int) -> bool:
        """Case 2: the turn waits in the queue; True asks for its prefetch to be issued again."""
        sim, now = self.sim, self.sim.now
        wait = self.residual_wait(pos)
        if wait is None:
            if pos > 1:
                return False
        baseline = max(plan.chain_tokens, r.last_match or 0)
        if now - plan.last_probe >= PROBE_INTERVAL_S:
            plan.last_probe = now
            plan.missing = self.missing_tokens(plan.chain) + (baseline - plan.chain_tokens)
            if plan.missing < THRESH:
                plan.room_refused = False
        if plan.missing < THRESH:
            return False
        room = self._room_retry_due(plan)
        if not room and self._room_retry_open(plan) and admissible_ahead > 1:
            return False
        if r.reading is not None or now - plan.last_refetch < PROBE_INTERVAL_S or (plan.capped and not room):
            return False
        plan.missing = self.missing_tokens(plan.chain) + (baseline - plan.chain_tokens)
        if plan.missing < THRESH:
            plan.room_refused = False
            return False
        if plan.refetches > 0 and not room:
            plan.capped = plan.refetches >= MAX_REFETCHES
            sim.cnt["reevict"] += 1
            if plan.capped:
                sim.cnt["reevict_capped"] += 1
                return False
        plan.last_refetch = now
        plan.pending_room = room
        return True

    def _refetch(self, r) -> None:
        sim = self.sim
        plan = r.plan
        ig, it = sim.match(r.path)
        sim.stamp_path(r.path, it)
        self._prefetch(r, it)
        if r.reading is not None:
            if plan.pending_room:
                self._take_room_retry(plan)
            else:
                plan.refetches += 1
                plan.room_refused = False

    def _check_pending(self, plan, queue_len: int, backlog: int):
        """Case 1: the turn has not arrived. Returns the next check time, or None once the plan is gone."""
        now = self.sim.now
        if now >= plan.expires_at:
            return self._expire(plan)
        recheck = min(now + PROBE_INTERVAL_S, plan.expires_at)
        wait = self.wait_after_arrival(queue_len)
        if wait is None:
            return recheck
        if self.timing == "ec":
            return self._pending_expected_cost(plan, wait, backlog, recheck)
        if self.timing == "jit":
            return self._pending_just_in_time(plan, wait, backlog, recheck)
        return plan.expires_at                          # 'arrival': plans serve the gate and the hold only

    def _pending_just_in_time(self, plan, wait: float, backlog: int, recheck: float) -> float:
        """Stage the read so that it lands early_s + 1 s before the predicted admission (jit_p50: at the p10)."""
        sim, now = self.sim, self.sim.now
        admission_at = plan.predicted_arrival + wait
        if admission_at - now > NTP_LEAD_HORIZON_S:
            return min(admission_at - NTP_LEAD_HORIZON_S, recheck)
        if now - plan.last_probe >= PROBE_INTERVAL_S:
            plan.last_probe = now
            plan.missing = self.missing_tokens(plan.chain)
        enough = plan.missing >= THRESH
        start_at = admission_at - self.load_seconds(plan.missing, backlog) - (plan.early_s + NTP_JIT_MARGIN_S)
        if now < start_at:
            return min(start_at, recheck)
        if plan.staged or not enough:
            return recheck
        if len(self.parked) >= NTP_MAX_PARKED:
            return recheck
        self._stage(plan)
        return recheck

    def _pending_expected_cost(self, plan, wait: float, backlog: int, recheck: float) -> float:
        """Rule 1: start the read where the stall it risks equals the rent it would pay pinned, or leave it to the
        turn's arrival; a landed staged read keeps its pin until its entry has passed."""
        sim, now = self.sim, self.sim.now
        rn = plan.rn
        idle_s = now - plan.t_commit - self.offset_s
        passed = not remaining_gaps(plan.support, idle_s - NTP_RELEASE_GRACE_S)
        if plan.staged and rn.reading is not None:
            if passed and rn.reading.state == "parked":
                self._release_staged(plan)
            return recheck
        remaining = remaining_gaps(plan.support, idle_s)
        pinned = sim.pinned_of(rn)
        if pinned > 0:
            if passed:
                self._release_staged(plan)
            return recheck
        if plan.staged:
            return recheck
        if now - plan.last_probe >= PROBE_INTERVAL_S:
            plan.last_probe = now
            plan.missing = self.missing_tokens(plan.chain)
            ig, _ = sim.match(plan.chain)
            plan.room_tokens = plan.chain_tokens - sum(sim.tok[n] for n in plan.chain[:ig])
        if plan.missing < THRESH or plan.stagings >= NTP_MAX_STAGINGS:
            return recheck
        read_s = self.load_seconds(plan.missing, backlog) + NTP_JIT_MARGIN_S
        weight = self.c_ewma if "F2" in self.sw else 1.0
        start_in = read_start(remaining, read_s, wait, self.price_lambda() * plan.room_tokens, weight)
        if start_in is None:
            return recheck
        if start_in > 0.0:
            return min(now + start_in, recheck)
        if len(self.parked) >= NTP_MAX_PARKED:
            return recheck
        plan.stagings += 1
        self._stage(plan)
        plan.staged = rn.reading is not None
        return recheck

    def _stage(self, plan) -> None:
        """Read the evicted span of the finished turn's key under the next request's handle."""
        sim = self.sim
        rn = plan.rn
        plan.staged = True
        ig, it = sim.match(plan.chain)
        sim.stamp_path(plan.chain, it)
        sim.cnt["staged"] += 1
        if rn.reading is None:
            # every read trigger passes the gate: a staged lookup takes one of the turn's lookups
            if plan.lookups > MAX_REFETCHES:
                plan.capped = True
                return
            plan.lookups += 1
            plan.counted = True
            sim.lookup(rn, plan.chain, it, staged=True)

    def _release_staged(self, plan) -> None:
        sim = self.sim
        rn = plan.rn
        if rn.reading is not None and rn.reading.state == "parked":
            rd = rn.reading
            self.parked.pop(rd, None)
            sim._revoke(rd)
        sim.unpin(rn)
        plan.staged = False
        plan.decision = ""

    def _expire(self, plan):
        """A plan whose turn never arrived. Just-in-time arms end the session (its tail is evicted first)."""
        sim = self.sim
        rn = plan.rn
        if plan.staged and rn.reading is not None:
            return sim.now + PROBE_INTERVAL_S           # mid-I/O: let it land
        sim.unpin(rn)
        if self.timing != "ec" and self.eviction != "return_chance":
            self._end_session(plan.sess)
        return None

    def _end_session(self, sess) -> None:
        tail = self.sess_tail.get(sess)
        if tail is not None:
            self._register(sess, tail, None)
            self.sess_tail.pop(sess, None)
        self.sess_fc[sess] = None

    def _follow_room(self, queue_ahead: int, now: float) -> None:
        """Rule 4 ('harm'): what staged pins cost the turns that had arrived."""
        sim = self.sim
        staged_left = self.staged_pin_seen
        for r, tokens in self.room_refusals:
            self.price.refuse(tokens)
            if r.t_send is not None and r.t_admit is None:
                lost = min(tokens, staged_left)
                staged_left -= lost
                self.price.charge(lost * self.recompute_s_per_token * max(1, queue_ahead))
        self.room_refusals = []
        staged_tokens = sum(sim.pinned_of(p.rn) for p in self.pending)
        self.staged_pin_seen = staged_tokens
        self.price.observe(now, staged_tokens)
        self.price.step()

    # ----- admission and commit -----

    def on_admit(self, r) -> None:
        sim = self.sim
        wait = sim.now - r.t_send
        if not r.tok_ssd:
            self.wait_ewma = wait if self.wait_ewma is None else self.wait_ewma + (wait - self.wait_ewma) / 8
            pw = getattr(r, "pred_wait", None)
            if pw is not None and pw > 1.0:
                self.wait_ratio.append(wait / pw)
        self.c_ewma += ((1.0 if r.alone else 0.0) - self.c_ewma) / 32

    def on_commit(self, r) -> None:
        """A finished turn's KV is in the tree (next_turn_prefetch.on_turn_committed)."""
        sim, now = self.sim, self.sim.now
        sess = (r.play.pid, r.s)
        tail = r.twig if r.twig is not None else (r.path[-1] if r.path else None)
        hint = self.resolve_hint(r) if self.prefetcher else None
        ns = self.node_sess
        for n in r.path:
            ns[n] = sess
        if r.twig is not None:
            ns[r.twig] = sess
        if r.k > 0:
            old = r.play.streams[r.s][r.k - 1].twig
            if old is not None:
                ns[old] = None                           # the previous reply's pages are not on the session's key
        r.plan = None
        if not self.prefetcher:
            return
        if r.hint is not None and r.hint.get("program"):
            self._commit_without_forecast(r, sess, tail)
            return
        if hint is None:
            self._register(sess, tail, None)
            self.sess_tail.pop(sess, None)
            self.sess_fc[sess] = None
            if "R1" in self.sw and self.session_ended(r):
                self.mark_chain_dead(r)
            return
        gap_s, early_s, dist = hint
        predicted = now + gap_s + self.offset_s
        wait = self.wait_after_arrival(len(sim.waitq))
        forecast = ("F", dist, now + self.offset_s)
        self.sess_fc[sess] = forecast
        self._register(sess, tail, forecast if self.eviction in ("return_chance", "index") else predicted + (wait or 0.0))
        st = r.play.streams[r.s]
        if r.k + 1 >= len(st):
            return
        rn = st[r.k + 1]
        plan = Plan()
        plan.rn, plan.sess = rn, sess
        plan.chain = r.path + (r.twig,) if r.twig is not None else r.path
        plan.chain_tokens = (r.pages + r.twig_pages) * PAGE
        plan.t_commit, plan.gap_s, plan.predicted_arrival = now, gap_s, predicted
        plan.early_s, plan.dist = early_s, dist
        plan.support = dist.support() if self.timing == "ec" else ()
        plan.expires_at = now + NTP_PLAN_MAX_AGE_S if self.timing == "ec" else predicted + max(MIN_EXPIRY_S, gap_s)
        plan.next_check = now
        rn.plan = plan
        self.pending[plan] = True

    def _commit_without_forecast(self, r, sess, tail) -> None:
        """Arm nohint: the engine knows which request continues the session, not when or whether it comes. The
        session's tail ranks as un-hinted (LRU) until its next turn is queued; the plan serves the gate, the hold
        and the queued rank only."""
        sim, now = self.sim, self.sim.now
        prev = self.sess_tail.get(sess)
        if prev is not None:
            self.reg.pop(prev, None)
            if prev != tail:
                self.moved[prev] = tail
            sim.touch(prev)
        if tail is not None:
            self.sess_tail[sess] = tail
            self.reg.pop(tail, None)
        self.sess_fc.pop(sess, None)
        st = r.play.streams[r.s]
        if r.k + 1 >= len(st):
            return
        plan = Plan()
        plan.rn, plan.sess = st[r.k + 1], sess
        plan.chain = r.path + (r.twig,) if r.twig is not None else r.path
        plan.chain_tokens = (r.pages + r.twig_pages) * PAGE
        plan.t_commit, plan.gap_s, plan.predicted_arrival = now, 0.0, now
        plan.early_s, plan.dist, plan.support = 0.0, None, ()
        plan.expires_at = plan.next_check = now + NTP_PLAN_MAX_AGE_S
        plan.rn.plan = plan
        self.pending[plan] = True

    # ----- writes -----

    def should_write(self, n: int, r) -> bool:
        """R1: the last turn of a session the agent declared ended is not written."""
        if "R1" in self.sw and r is not None and self.session_ended(r):
            self.sim.cnt["write_skipped_ended"] += 1
            return False
        return True

    def mark_chain_dead(self, r) -> None:
        """R1: an ended session's SSD chain goes first: the nodes only this session committed over."""
        sim = self.sim
        sess = (r.play.pid, r.s)
        for n in reversed(r.path):
            if self.node_sess[n] != sess or sim.nct[n] > (1 if r.twig is not None and sim.par[r.twig] == n else 0):
                break
            sim.ssd_dead[n] = True

    def write_push(self, n: int, r) -> None:
        self.wqueue.append(n)

    def write_pop(self):
        """FIFO (the engine); R2: the pending page the eviction order would take first."""
        q = self.wqueue
        if not q:
            return None
        if "R2" in self.sw and len(q) > 1:
            best = min(range(len(q)), key=lambda i: self.prio(q[i]))
            n = q[best]
            del q[best]
            return n
        return q.popleft()

    def requeue_failed_write(self, n: int) -> bool:
        """R3: a page counts as clean only on a full acknowledgement; a failed write is queued again."""
        return "R3" in self.sw

    def on_write_ack(self, n: int) -> None:
        sim = self.sim
        if self.write_held:
            self._update_write_hold()
            if not self.write_held:
                self.dispatch()

    # ----- the cleaner -----

    def cleaner_order(self, target: float):
        """Oldest written first (the engine). R1 / R4: chains of ended sessions first."""
        sim = self.sim
        if "R1" in self.sw or "R4" in self.sw:
            dead = [n for n in range(sim.wl.n_nodes) if sim.ssd_dead[n] and sim.ssd[n]]
            dead.sort(key=lambda n: sim.ssd_t[n])
            return itertools.chain(dead, sim.ssd_oldest())
        return sim.ssd_oldest()

    def on_ssd_deleted(self, n: int) -> None:
        if "R5" in self.sw:
            self.sim.touch(n)

    def on_cleaner_pass(self, marginal) -> None:
        if marginal is not None:
            sess = self.node_sess[marginal]
            fc = self.sess_fc.get(sess) if sess is not None else None
            if fc is not None and fc[0] == "F":
                self.sigma = self.measured_mu() * (self.gamma_a + self.gamma_b * 25000.0) * gittins_index(
                    fc[1], self.sim.now - fc[2])


ARMS = {
    "hicache": dict(hint_mode="none", order="fifo", prefetcher=False, timing=None, eviction="lru", pin_fraction=0.0,
                    soft_pins=False, price_rule="fixed"),
    "jit_p50": dict(hint_mode="dist", order="deadline", prefetcher=True, timing="jit", eviction="next_use",
                    pin_fraction=0.3, soft_pins=False, price_rule="fixed"),
    "jit_exact": dict(hint_mode="exact", order="deadline", prefetcher=True, timing="jit", eviction="next_use",
                      pin_fraction=0.3, soft_pins=False, price_rule="fixed"),
    "ec_full": dict(hint_mode="dist", order="deadline", prefetcher=True, timing="ec", eviction="return_chance",
                    pin_fraction=0.3, soft_pins=True, price_rule="harm"),
    "ec_exact": dict(hint_mode="exact", order="deadline", prefetcher=True, timing="ec", eviction="return_chance",
                     pin_fraction=0.3, soft_pins=True, price_rule="harm"),
    # the no-hint queue-aware engine (campaign 26's read timing 'arrival', hints 'program'): gate, hold, pins, queued
    # turns evicted last, LRU otherwise; never run on FA3, so it has no anchor here
    "nohint": dict(hint_mode="program", order="deadline", prefetcher=True, timing="arrival", eviction="return_chance",
                   pin_fraction=0.3, soft_pins=False, price_rule="fixed"),
}


def parse_switches(text: str) -> tuple:
    if not text:
        return ()
    if text.strip().lower() == "all":
        return SWITCHES
    out = tuple(s.strip().upper() for s in text.split(",") if s.strip())
    bad = [s for s in out if s not in SWITCHES + LATER_SWITCHES]
    if bad:
        raise ValueError(f"unknown switches {bad}: one of {SWITCHES + LATER_SWITCHES}")
    return out


def make_policy(cfg) -> Policy:
    arm = cfg.arm
    if arm == "unified":
        spec = dict(ARMS["ec_full"])
        switches = parse_switches(cfg.switches)
        if "R5" in switches:
            spec["eviction"] = "index"
        return Policy(cfg, name="unified", switches=switches, **spec)
    if arm not in ARMS:
        raise ValueError(f"arm {arm}: one of {sorted(ARMS)} or unified")
    return Policy(cfg, name=arm, switches=parse_switches(cfg.switches), **ARMS[arm])
