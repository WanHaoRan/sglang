"""verify_math: shared helpers, written independently of the three mathematicians' code.

Laws are objects with sf(t) = P(G > t) (vectorised), sample(rng, n) and, where available,
M(t) = E[(G - t)^+] in closed form. A defective law carries a mass p_inf at +infinity.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import integrate, special, stats

Z90 = 1.2815515655446004


def Phi(z):
    return 0.5 * special.erfc(-np.asarray(z, dtype=float) / math.sqrt(2.0))


def Phibar(z):
    return 0.5 * special.erfc(np.asarray(z, dtype=float) / math.sqrt(2.0))


def phi(z):
    z = np.asarray(z, dtype=float)
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def ln_stoploss(t, med, sig):
    """E[(G - t)^+] for ln G ~ N(ln med, sig^2), t > 0."""
    t = np.asarray(t, dtype=float)
    z = np.log(np.maximum(t, 1e-300) / med) / sig
    return med * math.exp(0.5 * sig * sig) * Phi(sig - z) - t * Phibar(z)


class TwoPiece:
    """Two-piece lognormal through p10/p50/p90 (the engine's model), optional cap (atom at cap)."""

    def __init__(self, p10, p50, p90, cap=None, sig_floor=0.05, sig_ceil=6.0):
        self.med = float(p50)
        self.sl = min(sig_ceil, max(sig_floor, math.log(p50 / p10) / Z90))
        self.sh = min(sig_ceil, max(sig_floor, math.log(p90 / p50) / Z90))
        self.cap = cap
        self.p_inf = 0.0

    def sf_raw(self, t):
        t = np.asarray(t, dtype=float)
        out = np.ones_like(t)
        pos = t > 0
        tt = np.where(pos, t, 1.0)
        sig = np.where(tt < self.med, self.sl, self.sh)
        out = np.where(pos, Phibar(np.log(tt / self.med) / sig), 1.0)
        return out

    def sf(self, t):
        t = np.asarray(t, dtype=float)
        s = self.sf_raw(t)
        if self.cap is not None:
            s = np.where(t >= self.cap, 0.0, s)
        return s

    def pdf_raw(self, t):
        t = np.asarray(t, dtype=float)
        sig = np.where(t < self.med, self.sl, self.sh)
        z = np.log(t / self.med) / sig
        return phi(z) / (sig * t)

    def hazard_raw(self, t):
        return self.pdf_raw(t) / self.sf_raw(t)

    def quantile(self, u):
        u = np.asarray(u, dtype=float)
        z = stats.norm.ppf(u)
        sig = np.where(z < 0, self.sl, self.sh)
        q = self.med * np.exp(sig * z)
        if self.cap is not None:
            q = np.minimum(q, self.cap)
        return q

    def sample(self, rng, n):
        return self.quantile(rng.random(n))

    def M_raw(self, t):
        t = np.asarray(t, dtype=float)
        hi = ln_stoploss(np.maximum(t, self.med), self.med, self.sh)
        lo = ln_stoploss(np.minimum(np.maximum(t, 1e-300), self.med), self.med, self.sl) - ln_stoploss(
            self.med, self.med, self.sl
        )
        base = ln_stoploss(self.med, self.med, self.sh)
        out = np.where(t >= self.med, hi, lo + base)
        # t <= 0: M(t) = E[G] - t
        mean = float(
            ln_stoploss(1e-300, self.med, self.sl) - ln_stoploss(self.med, self.med, self.sl) + base
        )
        out = np.where(t <= 0, mean - t, out)
        return out

    def M(self, t):
        t = np.asarray(t, dtype=float)
        if self.cap is None:
            return self.M_raw(t)
        return np.where(t >= self.cap, 0.0, self.M_raw(np.minimum(t, self.cap)) - self.M_raw(self.cap))


def stoploss_proper(d, t):
    """E[(G - t)^+] for a scipy frozen law on [0, inf) with a finite mean: closed forms (own derivation)."""
    t = float(t)
    name = d.dist.name
    if t <= 0:
        return float(d.mean()) - t
    if name == "gamma":
        k = d.args[0]
        th = d.kwds.get("scale", 1.0)
        return k * th * stats.gamma.sf(t, k + 1.0, scale=th) - t * stats.gamma.sf(t, k, scale=th)
    if name == "lomax":
        al = d.args[0]
        sc = d.kwds.get("scale", 1.0)
        if al <= 1.0:
            return float("inf")
        return sc / (al - 1.0) * (1.0 + t / sc) ** (-(al - 1.0))
    if name == "weibull_min":
        k = d.args[0]
        eta = d.kwds.get("scale", 1.0)
        return eta / k * special.gamma(1.0 / k) * special.gammaincc(1.0 / k, (t / eta) ** k)
    if name == "expon":
        m = d.kwds.get("scale", 1.0)
        return m * math.exp(-t / m)
    if name == "uniform":
        a = d.args[0] if d.args else d.kwds.get("loc", 0.0)
        w = d.args[1] if len(d.args) > 1 else d.kwds.get("scale", 1.0)
        b = a + w
        if t <= a:
            return 0.5 * (a + b) - t
        if t >= b:
            return 0.0
        return (b - t) ** 2 / (2.0 * w)
    if name == "lognorm":
        sig = d.args[0]
        med = d.kwds.get("scale", 1.0)
        return float(ln_stoploss(t, med, sig))
    val, _ = integrate.quad(lambda v: float(d.sf(v)), t, np.inf, limit=800)
    return val


def _twopiece_int_sf(self, a, b):
    if b <= a:
        return 0.0
    hi = 0.0 if math.isinf(b) else float(self.M(b))
    return float(self.M(a)) - hi


TwoPiece.int_sf = _twopiece_int_sf


class ScipyLaw:
    """A scipy frozen distribution on [0, inf), optionally mixed with a mass p_inf at infinity."""

    def __init__(self, frozen, p_inf=0.0):
        self.d = frozen
        self.p_inf = p_inf

    def sf(self, t):
        t = np.asarray(t, dtype=float)
        return self.p_inf + (1.0 - self.p_inf) * self.d.sf(t)

    def sample(self, rng, n):
        g = self.d.rvs(size=n, random_state=rng)
        if self.p_inf > 0:
            dead = rng.random(n) < self.p_inf
            g = np.where(dead, np.inf, g)
        return g

    def int_sf(self, a, b):
        """integral_a^b S(v) dv."""
        if b <= a:
            return 0.0
        hi = 0.0 if math.isinf(b) else stoploss_proper(self.d, b)
        proper = stoploss_proper(self.d, a) - hi
        if self.p_inf > 0:
            if math.isinf(b):
                return float("inf")
            return (1.0 - self.p_inf) * proper + self.p_inf * (b - a)
        return proper

    def M(self, t, upper=None):
        return self.int_sf(float(t), float("inf") if upper is None else float(upper))


class Discrete:
    """Atoms at points pts (ascending) with probabilities w; remaining mass 1 - sum(w) sits at infinity."""

    def __init__(self, pts, w):
        self.pts = np.asarray(pts, dtype=float)
        self.w = np.asarray(w, dtype=float)
        self.p_inf = max(0.0, 1.0 - float(self.w.sum()))

    def sf(self, t):
        t = np.asarray(t, dtype=float)
        return self.p_inf + (self.w[None, :] * (self.pts[None, :] > t.reshape(-1, 1))).sum(axis=1).reshape(t.shape)

    def sample(self, rng, n):
        u = rng.random(n)
        cum = np.cumsum(self.w)
        idx = np.searchsorted(cum, u, side="right")
        out = np.where(idx < len(self.pts), self.pts[np.minimum(idx, len(self.pts) - 1)], np.inf)
        return out

    def M(self, t, upper=None):
        t = float(t)
        if upper is None or math.isinf(upper):
            if self.p_inf > 0:
                return np.inf
            return float((self.w * np.maximum(self.pts - t, 0.0)).sum())
        return float((self.w * np.maximum(np.minimum(self.pts, upper) - t, 0.0)).sum()) + self.p_inf * max(upper - t, 0.0)

    def int_sf(self, a, b):
        return self.M(a, upper=b)


def quad_sf(law, a, b):
    """integral_a^b S(v) dv: closed form where the law has one, adaptive quadrature otherwise."""
    if b <= a:
        return 0.0
    if hasattr(law, "int_sf"):
        return law.int_sf(a, b)
    val, _ = integrate.quad(lambda v: float(law.sf(v)), a, b, limit=800)
    return val


def quad_sf_numeric(law, a, b, pieces=40):
    """The same integral by brute-force quadrature (used only to cross-check the closed forms)."""
    if b <= a:
        return 0.0
    edges = np.linspace(a, b, pieces + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        val, _ = integrate.quad(lambda v: float(law.sf(v)), lo, hi, limit=200)
        tot += val
    return tot


def zscore(est, se, truth):
    return (est - truth) / se if se > 0 else (0.0 if abs(est - truth) < 1e-12 else float("inf"))


def stoploss_vec(d, t):
    """Vectorised E[(G - t)^+] for the scipy laws used in the checks (t >= 0 array)."""
    t = np.asarray(t, dtype=float)
    name = d.dist.name
    if name == "gamma":
        k = d.args[0]
        th = d.kwds.get("scale", 1.0)
        return k * th * stats.gamma.sf(t, k + 1.0, scale=th) - t * stats.gamma.sf(t, k, scale=th)
    if name == "lomax":
        al = d.args[0]
        sc = d.kwds.get("scale", 1.0)
        return sc / (al - 1.0) * (1.0 + t / sc) ** (-(al - 1.0))
    if name == "weibull_min":
        k = d.args[0]
        eta = d.kwds.get("scale", 1.0)
        return eta / k * special.gamma(1.0 / k) * special.gammaincc(1.0 / k, (t / eta) ** k)
    if name == "expon":
        m = d.kwds.get("scale", 1.0)
        return m * np.exp(-t / m)
    if name == "uniform":
        a = d.args[0] if d.args else d.kwds.get("loc", 0.0)
        w = d.args[1] if len(d.args) > 1 else d.kwds.get("scale", 1.0)
        b = a + w
        return np.where(t <= a, 0.5 * (a + b) - t, np.where(t >= b, 0.0, (b - t) ** 2 / (2.0 * w)))
    if name == "lognorm":
        return ln_stoploss(t, d.kwds.get("scale", 1.0), d.args[0])
    return np.array([stoploss_proper(d, u) for u in t.ravel()]).reshape(t.shape)
