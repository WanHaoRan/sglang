"""Retain as an optimal stopping problem: checks of DESIGN.md section 4.1 and result 7.2.

Model (one idle gap of one session). The gap G takes values in (0, inf]; G = inf means the
session never returns. A prefix of L tokens is held in host memory at rent lam*L per second.
If the prefix is dropped at idle age s and the session returns later (s < G < inf), the
restore cost K is paid. The only information while idle is "not yet back", so a policy is a
drop age s in [a, inf].

Discrete setting used for the exact checks: time grid t_k = k*dt, survival S_k = P(G > t_k),
S_0 = 1, non-increasing; the mass S_n left at the last grid point is "never returns".
I_k = sum_{j<k} S_j dt = E[min(G, t_k)] for a law with atoms on the grid.
Cost of drop age t_k seen from age 0:  C_k = lam*L*I_k + K*(S_k - S_n).
Index:  nu_k = (K/L) * max_{j>k} (S_k - S_j) / (I_j - I_k).

Run: python -I -B hold_index.py   (numpy, scipy)
"""

from __future__ import annotations

import math

import numpy as np
from scipy import integrate, optimize, special, stats

RNG = np.random.default_rng(20261126)


# ----------------------------------------------------------------------------- laws on a grid
def grid(tmax: float, n: int) -> np.ndarray:
    return np.linspace(0.0, tmax, n + 1)


def surv_lognormal(t: np.ndarray, median: float, sigma: float) -> np.ndarray:
    out = np.ones_like(t)
    pos = t > 0
    out[pos] = stats.norm.sf(np.log(t[pos] / median) / sigma)
    return out


def surv_two_piece_lognormal(t: np.ndarray, p10: float, p50: float, p90: float) -> np.ndarray:
    """The engine's hint law (next_turn_policy.GapDistribution): ln G normal on each side of the median."""
    z90 = 1.2815515655446004
    s_lo = math.log(p50 / p10) / z90
    s_hi = math.log(p90 / p50) / z90
    out = np.ones_like(t)
    pos = t > 0
    z = np.log(t[pos] / p50)
    out[pos] = np.where(z < 0, stats.norm.sf(z / s_lo), stats.norm.sf(z / s_hi))
    return out


def surv_weibull(t: np.ndarray, theta: float, k: float) -> np.ndarray:
    return np.exp(-((t / theta) ** k))


def surv_lomax(t: np.ndarray, theta: float, alpha: float) -> np.ndarray:
    return (1.0 + t / theta) ** (-alpha)


def surv_point(t: np.ndarray, r0: float) -> np.ndarray:
    return (t < r0).astype(float)


def surv_uniform(t: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((hi - t) / (hi - lo), 0.0, 1.0)


def with_end_mass(surv: np.ndarray, p_end: float) -> np.ndarray:
    """Mix with a session that has ended (G = inf) with probability p_end."""
    return p_end + (1.0 - p_end) * surv


def mixture(parts: list[tuple[float, np.ndarray]]) -> np.ndarray:
    return sum(w * s for w, s in parts)


def held_time(surv: np.ndarray, dt: float) -> np.ndarray:
    """I_k = E[min(G, t_k)] for the step law with atoms on the grid (left Riemann sum, exact)."""
    return np.concatenate(([0.0], np.cumsum(surv[:-1]) * dt))


# ----------------------------------------------------------------------------- index
def index_bruteforce(surv: np.ndarray, held: np.ndarray, k_over_l: float, dt: float) -> np.ndarray:
    """O(n^2); the denominator is summed from k on, so the far tail does not cancel."""
    n = len(surv) - 1
    nu = np.zeros(n + 1)
    for k in range(n):
        if surv[k] <= 0.0:
            continue
        num = surv[k] - surv[k + 1 :]
        den = np.cumsum(surv[k:n]) * dt
        nu[k] = k_over_l * np.max(num / den)
    return nu


def index_hull(surv: np.ndarray, held: np.ndarray, k_over_l: float) -> np.ndarray:
    """Same index in O(n): slope of the lower convex hull of the curve (I_k, S_k) from k on."""
    n = len(surv) - 1
    nu = np.zeros(n + 1)
    stack = [n]
    for k in range(n - 1, -1, -1):
        if surv[k] <= 0.0 or held[stack[-1]] <= held[k]:
            stack = [k]
            continue
        while len(stack) >= 2:
            a, b = stack[-1], stack[-2]
            slope_a = (surv[a] - surv[k]) / (held[a] - held[k])
            slope_b = (surv[b] - surv[k]) / (held[b] - held[k])
            if slope_a >= slope_b:
                stack.pop()
            else:
                break
        a = stack[-1]
        nu[k] = k_over_l * (surv[k] - surv[a]) / (held[a] - held[k])
        stack.append(k)
    return nu


def cost_curve(surv: np.ndarray, held: np.ndarray, lam: float, big_k: float, tokens: float) -> np.ndarray:
    return lam * tokens * held + big_k * (surv - surv[-1])


def suffix_min_strict(c: np.ndarray) -> np.ndarray:
    """m_k = min_{j>k} c_j (inf at the last point)."""
    out = np.full_like(c, np.inf)
    out[:-1] = np.minimum.accumulate(c[::-1])[::-1][1:]
    return out


def bellman(surv: np.ndarray, dt: float, lam: float, big_k: float, tokens: float):
    """Backward induction on the conditional problem: V_k = min(drop now, keep one step)."""
    n = len(surv) - 1
    value = np.zeros(n + 1)
    drop = np.ones(n + 1, dtype=bool)
    for k in range(n - 1, -1, -1):
        if surv[k] <= 0.0:
            continue
        stop = big_k * (surv[k] - surv[n]) / surv[k]
        cont = lam * tokens * dt + (surv[k + 1] / surv[k]) * value[k + 1]
        value[k] = min(stop, cont)
        drop[k] = stop <= cont + 1e-12 * max(1.0, abs(stop))
    return value, drop


# ----------------------------------------------------------------------------- theorem checks
def check_theorem(name: str, surv: np.ndarray, dt: float, big_k: float, tokens: float, lams) -> dict:
    surv = np.where(surv < 1e-12, 0.0, surv)
    held = held_time(surv, dt)
    k_over_l = big_k / tokens
    nu_bf = index_bruteforce(surv, held, k_over_l, dt)
    nu_h = index_hull(surv, held, k_over_l)
    # ages with a survival below 1e-4 are left out: there C and I cancel in double precision
    alive = surv > 1e-4
    hull_err = float(np.max(np.abs(nu_bf[alive] - nu_h[alive]) / np.maximum(nu_bf[alive], 1e-300)))
    res = {"law": name, "hull_vs_bruteforce_relerr": hull_err}
    nested_prev = None
    worst_iff = 0
    worst_first = 0
    worst_bell = 0.0
    worst_bell_set = 0
    for lam in lams:
        c = cost_curve(surv, held, lam, big_k, tokens)
        later = suffix_min_strict(c)
        tol = 1e-9 * max(1.0, float(np.max(np.abs(c))))
        # (ii) drop now is optimal at k  <=>  nu_k <= lam ; compare away from numerical ties
        stop_opt = c <= later + tol
        clear = np.abs(nu_bf - lam) > 1e-7 * lam
        worst_iff += int(np.sum((stop_opt != (nu_bf <= lam)) & clear & alive))
        # (iii) earliest optimal drop age from each start k  ==  first j >= k with nu_j <= lam
        n = len(surv) - 1
        stop_idx = np.where(nu_bf <= lam * (1 + 1e-12), np.arange(n + 1), n + 1)
        first_pass = np.minimum.accumulate(stop_idx[::-1])[::-1]
        suffix_incl = np.minimum.accumulate(c[::-1])[::-1]
        ok_first = np.abs(c[np.minimum(first_pass, n)] - suffix_incl) <= tol
        worst_first += int(np.sum(~ok_first & alive))
        # (v) Bellman recursion gives the same value and the same stopping set
        value, drop = bellman(surv, dt, lam, big_k, tokens)
        v_from_c = np.zeros_like(c)
        v_from_c[alive] = (suffix_incl[alive] - lam * tokens * held[alive]) / surv[alive]
        worst_bell = max(worst_bell, float(np.max(np.abs(value[alive] - v_from_c[alive]))))
        worst_bell_set += int(np.sum((drop != (nu_bf <= lam)) & clear & alive))
        # (iv) indexability: the drop set grows with lam
        dset = nu_bf <= lam
        if nested_prev is not None:
            res.setdefault("nested_violations", 0)
            res["nested_violations"] += int(np.sum(nested_prev & ~dset))
        nested_prev = dset
    res.update(
        iff_violations=worst_iff,
        first_passage_violations=worst_first,
        bellman_value_maxabs=worst_bell,
        bellman_set_violations=worst_bell_set,
    )
    return res


def monte_carlo(surv: np.ndarray, dt: float, lam: float, big_k: float, tokens: float, n_samples: int) -> dict:
    """Simulate gaps; compare the first-passage index rule, the myopic hazard rule and fixed ages."""
    n = len(surv) - 1
    held = held_time(surv, dt)
    nu = index_hull(surv, held, big_k / tokens)
    c = cost_curve(surv, held, lam, big_k, tokens)
    # gap index: G = t_j with prob S_{j-1} - S_j (j = 1..n), inf with prob S_n
    u = RNG.random(n_samples)
    # P(G > t_j) = S_j ; G = t_j  <=>  S_j <= u' < S_{j-1} with u' uniform
    j = np.searchsorted(-surv, -u, side="left")  # first j with S_j <= u
    never = j > n
    gap = np.where(never, np.inf, j * dt)

    def cost_of_drop_index(s_idx: int) -> tuple[float, float]:
        s = s_idx * dt
        rent = lam * tokens * np.minimum(gap, s)
        pay = big_k * ((gap > s) & ~never)
        tot = rent + pay
        return float(np.mean(tot)), float(np.std(tot) / math.sqrt(n_samples))

    first_idx = int(np.argmax(nu <= lam)) if np.any(nu <= lam) else n
    hazard = np.zeros(n + 1)
    hazard[:-1] = (surv[:-1] - surv[1:]) / (np.maximum(surv[:-1], 1e-300) * dt)
    myopic = (big_k / tokens) * hazard
    myo_idx = int(np.argmax(myopic < lam)) if np.any(myopic < lam) else n
    best_idx = int(np.argmin(c))
    out = {
        "index_rule_drop_age": first_idx * dt,
        "index_rule_cost_exact": float(c[first_idx]),
        "index_rule_cost_mc": cost_of_drop_index(first_idx),
        "bruteforce_best_age": best_idx * dt,
        "bruteforce_best_cost": float(c[best_idx]),
        "myopic_hazard_rule_drop_age": myo_idx * dt,
        "myopic_hazard_rule_cost_exact": float(c[myo_idx]),
        "drop_at_once_cost": float(c[0]),
    }
    return out


# ----------------------------------------------------------------------------- continuous closed forms
def nu_quad(surv_fn, a: float, k_over_l: float, hmax: float, p_inf: float = 0.0) -> tuple[float, float]:
    """Index by quadrature and scalar maximisation over the horizon h (continuous law)."""
    sa = surv_fn(a)

    def ratio(h: float) -> float:
        num = sa - surv_fn(a + h)
        den, _ = integrate.quad(surv_fn, a, a + h, limit=400)
        return num / den if den > 0 else 0.0

    hs = np.unique(np.logspace(-6, math.log10(hmax), 400))
    vals = np.array([ratio(h) for h in hs])
    i = int(np.argmax(vals))
    lo, hi = hs[max(i - 1, 0)], hs[min(i + 1, len(hs) - 1)]
    best_h, best = hs[i], vals[i]
    if hi > lo:
        opt = optimize.minimize_scalar(lambda h: -ratio(h), bounds=(lo, hi), method="bounded", options={"xatol": 1e-10})
        if -opt.fun > best:
            best_h, best = opt.x, -opt.fun
    return k_over_l * best, best_h


def lognormal_pieces(median: float, sigma: float):
    m = math.log(median)

    def surv(t):
        return stats.norm.sf((math.log(t) - m) / sigma) if t > 0 else 1.0

    def hazard(t):
        z = (math.log(t) - m) / sigma
        return math.exp(stats.norm.logpdf(z) - stats.norm.logsf(z)) / (sigma * t)

    def held(t):  # E[min(G, t)]
        z = (math.log(t) - m) / sigma
        return t * stats.norm.sf(z) + math.exp(m + sigma * sigma / 2) * stats.norm.cdf(z - sigma)

    zstar = optimize.brentq(lambda z: math.exp(stats.norm.logpdf(z) - stats.norm.logsf(z)) - z - sigma, -35.0, 35.0)
    tstar = math.exp(m + sigma * zstar)
    return surv, hazard, held, tstar


def nu_lognormal_closed(a: float, median: float, sigma: float, k_over_l: float) -> tuple[float, float]:
    """Closed form: hazard(a) past the hazard peak t*; before it, hazard at the tangent point b*."""
    surv, hazard, held, tstar = lognormal_pieces(median, sigma)
    if a >= tstar:
        return k_over_l * hazard(a), 0.0

    def tangent(b: float) -> float:
        return hazard(b) * (held(b) - held(a)) - (surv(a) - surv(b))

    # on (t*, inf) the curve is convex; tangent() is positive just after t* and negative far out
    hi = tstar * 2
    while tangent(hi) > 0:
        hi *= 2
        if hi > 1e12:
            break
    b = optimize.brentq(tangent, tstar, hi, xtol=1e-12, rtol=1e-12)
    return k_over_l * hazard(b), b - a


def nu_weibull_closed(a: float, theta: float, k: float, k_over_l: float) -> float:
    if k <= 1.0:
        return k_over_l * (k / theta) * (a / theta) ** (k - 1.0) if a > 0 else math.inf
    x = (a / theta) ** k
    mrl = math.exp(x) * (theta / k) * special.gamma(1.0 / k) * special.gammaincc(1.0 / k, x)
    return k_over_l / mrl


# ----------------------------------------------------------------------------- main
def main() -> None:
    np.set_printoptions(linewidth=160)
    print("=" * 100)
    print("T1  Theorem on discrete laws (exact arithmetic up to rounding): keep iff nu >= lam, first passage, Bellman, nested sets")
    dt, n = 0.05, 2400  # 120 s horizon
    t = grid(dt * n, n)
    laws = {
        "lognormal(med 1.69, s 1.79) [tool-like, DFR]": surv_lognormal(t, 1.69, 1.79),
        "two-piece LN tool 0.41/1.69/16.8": surv_two_piece_lognormal(t, 0.41, 1.69, 16.8),
        "two-piece LN human 3.1/96/1040": surv_two_piece_lognormal(t, 3.1, 96.0, 1040.0),
        "weibull k=0.5 (DFR)": surv_weibull(t, 5.0, 0.5),
        "weibull k=3 (IFR)": surv_weibull(t, 20.0, 3.0),
        "uniform[15,25] (sharp, IFR)": surv_uniform(t, 15.0, 25.0),
        "point 20 s (exact hint)": surv_point(t, 20.0),
        "bimodal: 0.5 LN(1,0.3) + 0.5 U[40,45]": mixture([(0.5, surv_lognormal(t, 1.0, 0.3)), (0.5, surv_uniform(t, 40.0, 45.0))]),
        "sharp + end: 0.7 U[15,25] + 0.3 ended": with_end_mass(surv_uniform(t, 15.0, 25.0), 0.3),
        "tool LN + 0.05 ended": with_end_mass(surv_lognormal(t, 1.69, 1.79), 0.05),
        "trimodal 0.4 U[1,2]+0.3 U[30,31]+0.3 U[80,90]": mixture(
            [(0.4, surv_uniform(t, 1.0, 2.0)), (0.3, surv_uniform(t, 30.0, 31.0)), (0.3, surv_uniform(t, 80.0, 90.0))]
        ),
        "random monotone law (seeded)": np.concatenate(([1.0], np.sort(RNG.random(n))[::-1] * 0.999)),
    }
    big_k, tokens = 2.0, 25_000.0
    lams = np.geomspace(1e-8, 1e-2, 37)
    for name, surv in laws.items():
        res = check_theorem(name, surv, dt, big_k, tokens, lams)
        print(
            f"  {name:48s} hull~bf {res['hull_vs_bruteforce_relerr']:.1e}  iff_viol {res['iff_violations']}  "
            f"first_passage_viol {res['first_passage_violations']}  bellman |dV| {res['bellman_value_maxabs']:.1e}  "
            f"bellman_set_viol {res['bellman_set_violations']}  nested_viol {res.get('nested_violations', 0)}"
        )

    print("=" * 100)
    print("T1b Monte Carlo (400k gaps): index first-passage rule vs brute-force best age vs myopic hazard rule (K=2 s, L=25k)")
    for name, lam in [
        ("bimodal: 0.5 LN(1,0.3) + 0.5 U[40,45]", 2.0e-6),
        ("bimodal: 0.5 LN(1,0.3) + 0.5 U[40,45]", 6.0e-6),
        ("trimodal 0.4 U[1,2]+0.3 U[30,31]+0.3 U[80,90]", 1.5e-6),
        ("two-piece LN tool 0.41/1.69/16.8", 5.0e-6),
        ("sharp + end: 0.7 U[15,25] + 0.3 ended", 3.0e-6),
        ("weibull k=3 (IFR)", 3.0e-6),
    ]:
        out = monte_carlo(laws[name], dt, lam, big_k, tokens, 400_000)
        mc, se = out["index_rule_cost_mc"]
        print(
            f"  {name:46s} lam {lam:.1e}: index rule drops at {out['index_rule_drop_age']:7.2f} s cost {out['index_rule_cost_exact']:.5f} "
            f"(MC {mc:.5f} +- {se:.5f}) | best age {out['bruteforce_best_age']:7.2f} s cost {out['bruteforce_best_cost']:.5f} | "
            f"myopic rule drops at {out['myopic_hazard_rule_drop_age']:7.2f} s cost {out['myopic_hazard_rule_cost_exact']:.5f} | "
            f"drop at once {out['drop_at_once_cost']:.5f}"
        )

    print("=" * 100)
    print("T2  Special cases, continuous laws by quadrature (K/L = 1)")
    # DFR: sup at h -> 0, nu = hazard(a)
    for a in (0.5, 5.0, 50.0):
        th, k = 5.0, 0.5
        nu, h = nu_quad(lambda x: math.exp(-((x / th) ** k)), a, 1.0, 1e5)
        print(f"  Weibull k=0.5 th=5  a={a:5.1f}: nu_quad {nu:.6f} (best h {h:.2e})  hazard {(k/th)*(a/th)**(k-1):.6f}  closed {nu_weibull_closed(a, th, k, 1.0):.6f}")
    for a in (0.0, 5.0, 50.0):
        th, al = 3.0, 1.5
        nu, h = nu_quad(lambda x: (1 + x / th) ** (-al), a, 1.0, 1e5)
        print(f"  Lomax th=3 al=1.5   a={a:5.1f}: nu_quad {nu:.6f} (best h {h:.2e})  alpha/(theta+a) {al/(th+a):.6f}")
    # IFR: sup at h -> inf, nu = 1/E[R | a]
    for a in (0.0, 10.0, 25.0):
        th, k = 20.0, 3.0
        nu, h = nu_quad(lambda x: math.exp(-((x / th) ** k)), a, 1.0, 400.0)
        print(f"  Weibull k=3 th=20   a={a:5.1f}: nu_quad {nu:.6f} (best h {h:.1f})  1/E[R|a] closed {nu_weibull_closed(a, th, k, 1.0):.6f}")
    # IFR law mixed with an ended session: the draft's P(R<inf)/E[R] is 0; the true index is interior
    for p_end in (0.0, 0.1, 0.3):
        lo, hi = 15.0, 25.0

        def sv(x, p_end=p_end, lo=lo, hi=hi):
            return p_end + (1 - p_end) * min(1.0, max(0.0, (hi - x) / (hi - lo)))

        nu, h = nu_quad(sv, 0.0, 1.0, 1e4)
        # exact: maximise over h in (lo, hi] ; closed form below
        hs = np.linspace(lo + 1e-6, hi, 200001)
        num = (1 - p_end) * (hs - lo) / (hi - lo)
        den = p_end * hs + (1 - p_end) * (hs - (hs - lo) ** 2 / (2 * (hi - lo)))
        exact = float(np.max(num / den))
        draft = (1 - p_end) / (math.inf if p_end > 0 else (lo + hi) / 2)
        approx = (1 - p_end) / (p_end * hi + (1 - p_end) * (lo + hi) / 2)
        print(
            f"  U[15,25] with P(ended)={p_end:.1f}, a=0: nu_quad {nu:.6f} (h* {h:.2f})  grid-exact {exact:.6f}  "
            f"draft P(R<inf)/E[R] {draft:.6f}  P(R<inf)/E[min(R,h_end)] {approx:.6f}"
        )
    # exact hint
    for r0, p_end in ((20.0, 0.0), (20.0, 0.25)):
        print(f"  point R0={r0} P(ended)={p_end}: nu = (1-p)/R0 = {(1-p_end)/r0:.6f}")

    print("=" * 100)
    print("T3  Lognormal closed form vs quadrature (K/L = 1)")
    for median, sigma in ((1.69, 1.79), (20.0, 0.1), (20.0, 0.5), (96.0, 1.86)):
        surv, hazard, held, tstar = lognormal_pieces(median, sigma)
        print(f"  LN(median {median}, sigma {sigma}): hazard peak at t* = {tstar:.4g} s (= median x {tstar/median:.4f})")
        for a in (0.0 + 1e-9, 0.25 * median, median, 4 * median):
            if surv(a) < 1e-12:
                continue
            closed, hstar = nu_lognormal_closed(a, median, sigma, 1.0)
            nq, hq = nu_quad(surv, a, 1.0, 200 * median * math.exp(3 * sigma))
            print(f"     a={a:9.4f}: closed {closed:.6e} (h* {hstar:9.4f})   quad {nq:.6e} (h {hq:9.4f})   hazard(a) {hazard(a) if a>1e-6 else 0.0:.6e}")


if __name__ == "__main__":
    main()
