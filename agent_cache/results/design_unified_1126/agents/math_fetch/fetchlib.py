"""Shared helpers for the Fetch checks (math_fetch, design_unified_1126).

Everything here is CPU-only and read-only on the repository.

Notation (seconds unless said otherwise):
  R      remaining gap of the session from "now" (arrival A = now + R)
  S(t)   P(R > t), survival; F = 1 - S; h = f / S the hazard
  M(t)   E[(R - t)^+] = integral_t^inf S(v) dv, the stop-loss function
  r      read time (release to usable), W queue wait, x = (r - W)^+
  c      weight of a second of stall, lamL = lambda * L the rent of the staged read per second
"""

from __future__ import annotations

import importlib.util
import math
import sys

import numpy as np
from scipy import special

Z90 = 1.2815515655446004
SQRT2 = math.sqrt(2.0)


def Phi(z):
    return 0.5 * special.erfc(-np.asarray(z, dtype=float) / SQRT2)


def Phibar(z):
    return 0.5 * special.erfc(np.asarray(z, dtype=float) / SQRT2)


def phi(z):
    z = np.asarray(z, dtype=float)
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


class TwoPieceLognormal:
    """The engine's gap entry: ln(gap) normal on each side of the median, own sigma per side
    (next_turn_policy.GapDistribution.from_quantiles). Optional cap puts the mass beyond it
    at the cap (the replay's 1,200 s cap; the engine's MAX_PLANNED_GAP_S)."""

    def __init__(self, p10, p50, p90, cap=None, sigma_floor=0.05, sigma_ceiling=6.0, gap_floor=0.01):
        m = max(p50, gap_floor)
        lo = max(min(p10, m), gap_floor)
        hi = max(p90, m)
        self.m = m
        self.sl = min(sigma_ceiling, max(sigma_floor, math.log(m / lo) / Z90))
        self.sh = min(sigma_ceiling, max(sigma_floor, math.log(hi / m) / Z90))
        self.cap = cap

    # ---- uncapped pieces -------------------------------------------------
    def _z(self, t):
        t = np.asarray(t, dtype=float)
        sig = np.where(t < self.m, self.sl, self.sh)
        with np.errstate(divide="ignore"):
            return np.log(np.maximum(t, 1e-300) / self.m) / sig, sig

    def sf_raw(self, t):
        t = np.asarray(t, dtype=float)
        z, _ = self._z(t)
        return np.where(t <= 0.0, 1.0, Phibar(z))

    def pdf_raw(self, t):
        t = np.asarray(t, dtype=float)
        z, sig = self._z(t)
        return np.where(t <= 0.0, 0.0, phi(z) / (sig * np.maximum(t, 1e-300)))

    def _M_ln(self, t, sig):
        """E[(G - t)^+] for the full lognormal(ln m, sig)."""
        t = np.asarray(t, dtype=float)
        with np.errstate(divide="ignore"):
            z = np.log(np.maximum(t, 1e-300) / self.m) / sig
        return self.m * math.exp(0.5 * sig * sig) * Phi(sig - z) - t * Phibar(z)

    def M_raw(self, t):
        t = np.asarray(t, dtype=float)
        hi = self._M_ln(np.maximum(t, self.m), self.sh)
        lo = self._M_ln(np.minimum(np.maximum(t, 0.0), self.m), self.sl) - self._M_ln(self.m, self.sl)
        return hi + np.where(t < self.m, lo, 0.0)

    # ---- with the cap ------------------------------------------------------
    def sf(self, t):
        t = np.asarray(t, dtype=float)
        s = self.sf_raw(t)
        if self.cap is not None:
            s = np.where(t >= self.cap, 0.0, s)
        return s

    def cdf(self, t):
        return 1.0 - self.sf(t)

    def pdf(self, t):
        t = np.asarray(t, dtype=float)
        f = self.pdf_raw(t)
        if self.cap is not None:
            f = np.where(t >= self.cap, 0.0, f)
        return f

    def hazard(self, t):
        t = np.asarray(t, dtype=float)
        s = self.sf(t)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(s > 0.0, self.pdf(t) / s, np.inf)

    def M(self, t):
        """Stop-loss of the (capped) law: integral_t^cap S(v) dv."""
        t = np.asarray(t, dtype=float)
        if self.cap is None:
            return self.M_raw(t)
        return np.where(t >= self.cap, 0.0, self.M_raw(np.minimum(t, self.cap)) - self.M_raw(self.cap))

    def ppf(self, p):
        p = np.asarray(p, dtype=float)
        z = special.ndtri(p)
        sig = np.where(z < 0.0, self.sl, self.sh)
        q = self.m * np.exp(sig * z)
        if self.cap is not None:
            q = np.minimum(q, self.cap)
        return q

    def sample(self, rng, n):
        return self.ppf(rng.random(n))

    def mean(self):
        return float(self.M(0.0))


class Conditioned:
    """A law given the session has already been idle for `age` seconds (R = G - age | G > age)."""

    def __init__(self, base, age):
        self.base = base
        self.age = float(age)
        self.s0 = float(base.sf(age))
        self.cap = None if base.cap is None else base.cap - age

    def sf(self, t):
        return self.base.sf(self.age + np.asarray(t, dtype=float)) / self.s0

    def cdf(self, t):
        return 1.0 - self.sf(t)

    def pdf(self, t):
        return self.base.pdf(self.age + np.asarray(t, dtype=float)) / self.s0

    def hazard(self, t):
        return self.base.hazard(self.age + np.asarray(t, dtype=float))

    def M(self, t):
        return self.base.M(self.age + np.asarray(t, dtype=float)) / self.s0


# ---- the Fetch cost ---------------------------------------------------------

def J_F(law, s, x, c, lamL):
    """Priced cost of 'start the read at s unless the turn arrives first' for a proper law,
    staged copy held until admission, deterministic x = r - W > 0, without the device term:
        J_F(s) = c x - c M(s) + (c + lamL) M(s + x)."""
    s = np.asarray(s, dtype=float)
    return c * x - c * law.M(s) + (c + lamL) * law.M(s + x)


def dJ_F(law, s, x, c, lamL):
    """The draft's derivative: c [F(s+x) - F(s)] - lamL S(s+x) = S(s) [c F(x|s) - lamL (1 - F(x|s))]."""
    s = np.asarray(s, dtype=float)
    return c * (law.sf(s) - law.sf(s + x)) - lamL * law.sf(s + x)


def F_cond(law, x, s):
    """F(x | s) = P(R <= s + x | R > s)."""
    s = np.asarray(s, dtype=float)
    a = law.sf(s)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(a > 0.0, 1.0 - law.sf(s + x) / a, 1.0)


def start_grid(hi, n_log=4000, n_lin=4000, lo=1e-4):
    g = np.unique(np.concatenate([[0.0], np.geomspace(lo, hi, n_log), np.linspace(0.0, hi, n_lin)]))
    return g


def best_start(law, x, c, lamL, hi, extra=()):
    """Global minimum of J_F over s in [0, hi] on a dense grid (plus the given candidates);
    returns (s*, J(s*), J(arrival) = c x)."""
    g = start_grid(hi)
    if len(extra):
        g = np.unique(np.concatenate([g, np.asarray(extra, dtype=float)]))
        g = g[(g >= 0.0) & (g <= hi)]
    j = J_F(law, g, x, c, lamL)
    k = int(np.argmin(j))
    return float(g[k]), float(j[k]), c * x


def load_engine_policy(path="/home/wanhr/sglang/python/sglang/srt/mem_cache/next_turn_policy.py"):
    """Load the engine's pure policy module by path (read-only; run python with -B so no
    bytecode is written next to it)."""
    assert sys.dont_write_bytecode, "run with -B"
    spec = importlib.util.spec_from_file_location("ntp_policy_ro", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ntp_policy_ro"] = mod
    spec.loader.exec_module(mod)
    return mod


TOOL = dict(p10=0.41, p50=1.69, p90=16.8)
HUMAN = dict(p10=3.1, p50=96.0, p90=1040.0)
CAP = 1200.0
