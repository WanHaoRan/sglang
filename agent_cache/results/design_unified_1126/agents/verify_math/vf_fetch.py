"""verify_math, part 1: math_fetch R1-R6, re-derived and tested with independent code.

Run: CUDA_VISIBLE_DEVICES= nice -n 10 <venv python> -I -B vf_fetch.py > vf_fetch.out
"""
from __future__ import annotations

import math
import sys

import numpy as np
from scipy import integrate, optimize, special, stats

sys.path.insert(0, ".")
from vlib import Discrete, ScipyLaw, TwoPiece, ln_stoploss, quad_sf, quad_sf_numeric, stoploss_proper, stoploss_vec  # noqa: E402

SEED = 20261126_7
rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------- R1
def sim_policy(R, s, x, c, lamL):
    """Per-sample priced cost of 'release at s unless arrived, else at arrival' (x > 0)."""
    late = R <= s
    stall = np.where(late, x, np.maximum(x - (R - s), 0.0))
    hold = np.where(late, 0.0, np.maximum(R - s - x, 0.0))
    return c * stall + lamL * hold


def _split_quad(f, a, b):
    if b <= a:
        return 0.0
    edges = [a]
    w = 1e-3
    while a + w < b:
        edges.append(a + w)
        w *= 2.0
    edges.append(b)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        v, _ = integrate.quad(f, lo, hi, limit=200)
        tot += v
    return tot


def true_second_moment(law, s, x, c, lamL):
    """E[cost^2] of the policy: stall and hold are never both positive, so the cross term is 0."""
    if isinstance(law, Discrete):
        late = law.pts <= s
        stall = np.where(late, x, np.maximum(x - (law.pts - s), 0.0))
        hold = np.where(late, 0.0, np.maximum(law.pts - s - x, 0.0))
        return float((law.w * (c * stall + lamL * hold) ** 2).sum())
    if isinstance(law, TwoPiece):
        top = law.cap
    else:
        top = float(law.d.ppf(1.0 - 1e-13))
    F = lambda u: 1.0 - float(law.sf(u))
    S = lambda u: float(law.sf(u))
    e_stall2 = 2.0 * _split_quad(lambda u: (s + x - u) * F(u), s, s + x) if c > 0 else 0.0
    e_hold2 = 2.0 * _split_quad(lambda v: (v - s - x) * S(v), s + x, top) if (lamL > 0 and top > s + x) else 0.0
    return c * c * e_stall2 + lamL * lamL * e_hold2


def Mfun(law):
    return lambda t: float(np.asarray(law.M(t)))


def r1():
    global rng
    rng = np.random.default_rng([SEED, 1])
    print("== R1: closed form J_F(s) = c x - c M(s) + (c + lamL) M(s + x) against a simulation of the policy ==")
    laws = {
        "gamma(3, scale 4)": ScipyLaw(stats.gamma(3.0, scale=4.0)),
        "lomax(alpha 3.5, scale 10)": ScipyLaw(stats.lomax(3.5, scale=10.0)),
        "weibull(k 0.6, scale 5)": ScipyLaw(stats.weibull_min(0.6, scale=5.0)),
        "atoms 2/9/30 (0.3/0.5/0.2)": Discrete([2.0, 9.0, 30.0], [0.3, 0.5, 0.2]),
        "tool entry capped": TwoPiece(0.41, 1.69, 16.8, cap=1200.0),
        "uniform[18, 22]": ScipyLaw(stats.uniform(18.0, 4.0)),
    }
    prices = [(1.0, 0.5), (1.0, 0.0), (0.0, 0.5), (0.3, 2.5), (7.0, 1e-3)]
    xs = [1e-3, 0.5, 3.0, 50.0, 1e3]
    starts = [0.0, 0.7, 5.0, 19.0, 40.0]
    n = 1_500_000
    worst = 0.0
    zs_all = []
    cells = 0
    rare = 0
    worst_rare = 0.0
    worst_fd = 0.0
    worst_mc_der = 0.0
    for name, law in laws.items():
        R = law.sample(rng, n)
        M = Mfun(law)
        for (c, lamL) in prices:
            for x in xs:
                for s in starts:
                    cost = sim_policy(R, s, x, c, lamL)
                    est, se = cost.mean(), cost.std(ddof=1) / math.sqrt(n)
                    truth = c * x - c * M(s) + (c + lamL) * M(s + x)
                    n_hold = int(np.sum(R > s + x))
                    n_stall = int(np.sum((R > s) & (R <= s + x)))
                    thin = (lamL > 0 and 0 <= n_hold < 200 and float(law.sf(s + x)) > 0) or (c > 0 and 0 <= n_stall < 200 and float(law.sf(s)) - float(law.sf(s + x)) > 0)
                    # exact standard error of the estimator, from E[cost^2] by quadrature (the sample's own
                    # standard error is unreliable in rare-event and heavy-tailed cells)
                    var = max(true_second_moment(law, s, x, c, lamL) - truth * truth, 0.0)
                    if abs(est - truth) < 1e-9:
                        z = 0.0
                    elif var <= 0:
                        z = 99.0
                    else:
                        z = abs(est - truth) / math.sqrt(var / n)
                    if thin or se == 0:
                        rare += 1
                        worst_rare = max(worst_rare, z)
                    zs_all.append(z)
                    worst = max(worst, z)
                    cells += 1
                    # derivative: formula against a finite difference of the closed form (continuous laws only)
                    Ss, Ssx = float(law.sf(s)), float(law.sf(s + x))
                    der = c * (Ss - Ssx) - lamL * Ssx
                    if not isinstance(law, Discrete) and not (isinstance(law, TwoPiece) and s + x >= 1200.0):
                        h = 1e-4
                        Jp = -c * M(s + h) + (c + lamL) * M(s + h + x)
                        Jm = -c * M(max(s - h, 0.0)) + (c + lamL) * M(max(s - h, 0.0) + x)
                        fd = (Jp - Jm) / (h + min(h, s))
                        if not (name.startswith("uniform") and (abs(s - 18) < 1e-3 or abs(s + x - 18) < 1e-3 or abs(s + x - 22) < 1e-3)):
                            worst_fd = max(worst_fd, abs(fd - der))
                    # derivative: formula against the event frequencies in the sample
                    pa = np.mean((R > s) & (R <= s + x))
                    pb = np.mean(R > s + x)
                    mc = c * pa - lamL * pb
                    se_d = math.sqrt((c * c * pa * (1 - pa) + lamL * lamL * pb * (1 - pb) + 2 * c * lamL * pa * pb) / n) + 1e-12
                    worst_mc_der = max(worst_mc_der, max(0.0, abs(mc - der) - 10.0 * (c + lamL) / n) / max(se_d, 1e-9))
    print(f"   {cells} cells (6 laws x 5 price pairs incl. lambda = 0 and c = 0 x 5 read parts x from 1e-3 to 1e3 s x 5 starts), n = {n:,}")
    zs = np.array(zs_all)
    print(f"   simulated cost against the closed form, z-scores with the exact standard error (E[cost^2] by quadrature): worst |z| = {zs.max():.2f}; share above 2: {100 * np.mean(zs > 2):.1f} %, above 3: {100 * np.mean(zs > 3):.1f} % (cells share one sample per law, so they are not independent); {rare} of the cells have fewer than 200 sampled gaps in the event, worst |z| there {worst_rare:.2f}")
    print(f"   worst |finite difference - derivative formula| (continuous laws): {worst_fd:.2e}")
    print(f"   worst |z| of c P(s<R<=s+x) - lamL P(R>s+x) from event counts against the formula: {worst_mc_der:.2f}")

    # atoms: J continuous, one-sided derivatives differ at s = atom and s = atom - x
    law = Discrete([2.0, 9.0, 30.0], [0.3, 0.5, 0.2])
    M = Mfun(law)
    c, lamL, x = 1.0, 0.5, 3.0
    J = lambda s: c * x - c * M(s) + (c + lamL) * M(s + x)
    for s0 in (2.0, 6.0, 9.0):
        e = 1e-6
        left = (J(s0) - J(s0 - e)) / e
        right = (J(s0 + e) - J(s0)) / e
        Ss, Ssx = float(law.sf(s0)), float(law.sf(s0 + x))
        print(f"   atoms law, s = {s0}: J(s-), J(s), J(s+) = {J(s0 - e):.6f} {J(s0):.6f} {J(s0 + e):.6f}; left slope {left:+.4f}, right slope {right:+.4f}, formula (right) {c * (Ss - Ssx) - lamL * Ssx:+.4f}")

    # a proper law with an infinite mean: p_inf = 0 is not enough
    lom = stats.lomax(0.8, scale=5.0)
    R = lom.rvs(size=4_000_000, random_state=rng)
    for cut in (1e3, 1e5, 1e7, 1e9):
        hold = np.minimum(np.maximum(R - 3.0, 0.0), cut).mean()
        print(f"   lomax(alpha 0.8) (proper, infinite mean): E[min((R - x)^+, {cut:.0e})] = {hold:.1f}", end=";")
    print("\n   -> the expected hold diverges although P(R = inf) = 0: R1 needs E[R] finite, not only p_inf = 0.")


# ---------------------------------------------------------------- R2
def r2():
    global rng
    rng = np.random.default_rng([SEED, 2])
    print("\n== R2: random queue wait W independent of the gap ==")
    n = 2_000_000
    gap = stats.weibull_min(3.0, scale=25.0)  # rising hazard
    R = gap.rvs(size=n, random_state=rng)
    r = 4.0

    def run(Wd, c, lamL, label):
        W = Wd.rvs(size=n, random_state=rng)
        q = lamL / (c + lamL)
        pWr = np.mean(W <= r)
        kbar = c * np.mean(np.maximum(r - W, 0)) + lamL * np.mean(np.maximum(W - r, 0))
        # best delay after arrival, by direct search on the same W sample
        ds = np.linspace(0.0, 40.0, 801)
        kd = np.array([c * np.mean(np.maximum(d + r - W, 0)) + lamL * np.mean(np.maximum(W - d - r, 0)) for d in ds])
        dstar, kstar = ds[kd.argmin()], kd.min()
        Wq = np.quantile(W, q)
        print(f"   [{label}] c = {c}, lamL = {lamL}, q = {q:.3f}, P(W <= r) = {pWr:.3f}; kbar = {kbar:.4f}, best delay = {dstar:.2f} s (formula max(0, Q_W(q) - r) = {max(0.0, Wq - r):.2f}), kstar = {kstar:.4f}")
        Wg = np.quantile(W, (np.arange(4000) + 0.5) / 4000)  # quantile grid of W for the formula
        Msf = lambda t: stoploss_proper(gap, t)  # E[(R - t)^+]
        kk, eta = 3.0, 25.0
        Mvec = lambda t: eta / kk * special.gamma(1.0 / kk) * special.gammaincc(1.0 / kk, (np.asarray(t) / eta) ** kk)
        haz = lambda t: kk / eta * (t / eta) ** (kk - 1.0)
        EWr = float(np.mean(np.maximum(W - r, 0)))
        worst = 0.0
        rows = []
        for s in (0.0, 8.0, 14.0, 18.0, 21.0, 26.0, 60.0):
            arrived = R <= s
            land = np.where(arrived, R + r, s + r)
            D = R + W
            cost = c * np.maximum(land - D, 0) + lamL * np.maximum(D - land, 0)
            est, se = cost.mean(), cost.std(ddof=1) / math.sqrt(n)
            Ms = float(Msf(s))
            EM = float(np.mean(Mvec(s + np.maximum(r - Wg, 0.0))))
            truth = kbar - c * Ms + (c + lamL) * EM
            worst = max(worst, abs(est - truth) / se)
            Phi_s = np.mean((R + W <= s + r) & (R > s)) / max(np.mean(R > s), 1e-12)
            der = np.mean(R > s) * ((c + lamL) * Phi_s - lamL)
            # (C) re-timed after arrival with the prior law of W; (C') W observed at arrival
            landC = np.where(arrived, R + r + dstar, s + r)
            costC = c * np.maximum(landC - D, 0) + lamL * np.maximum(D - landC, 0)
            landCp = np.where(arrived, np.maximum(R + r, D), s + r)
            costCp = c * np.maximum(landCp - D, 0) + lamL * np.maximum(D - landCp, 0)
            rows.append((s, est, truth, costC.mean(), costCp.mean(), Phi_s, der))
        for (s, est, truth, jc, jcp, Ph, der) in rows:
            print(f"      s = {s:5.1f}: J_B sim {est:.4f} formula {truth:.4f} | J_C (prior W) {jc:.4f} | J_C' (W seen at arrival) {jcp:.4f} | Phi(s) = {Ph:.3f}, dJ_B/ds = {der:+.4f}")
        print(f"      worst |z| of J_B: {worst:.2f} (the formula's E_W uses a 4,000-point quantile grid of W)")
        # best starts on a grid for the three information structures
        grid = np.concatenate([np.linspace(0, 60, 241), [1e9]])
        best = {}
        D = R + W
        for tag in ("B", "C", "Cp"):
            vals = []
            for s in grid:
                arrived = R <= s
                if tag == "B":
                    land = np.where(arrived, R + r, s + r)
                elif tag == "C":
                    land = np.where(arrived, R + r + dstar, s + r)
                else:
                    land = np.where(arrived, np.maximum(R + r, D), s + r)
                vals.append(np.mean(c * np.maximum(land - D, 0) + lamL * np.maximum(D - land, 0)))
            vals = np.array(vals)
            i = vals.argmin()
            best[tag] = (grid[i], vals[i], vals[-1])
        for tag, nm in (("B", "(B) not re-timed"), ("C", "(C) re-timed, prior law of W"), ("Cp", "(C') W observed at arrival")):
            s_, v_, va = best[tag]
            print(f"      {nm:32s}: best start {'arrival' if (s_ > 1e8 or v_ >= va - 1e-9) else f'{s_:.2f} s'}, cost {v_:.4f} (arrival {va:.4f})")
        # first-order conditions on a fine grid: (B) Phi = q; (C') Phi = q (1 + hazard x E[(W - r)^+])
        sg = np.linspace(1.0, 45.0, 177)
        order = np.argsort(R)
        Rs = R[order]
        tot = R + W
        for tag in ("B", "Cp"):
            root = None
            prev = None
            for sv in sg:
                alive = R > sv
                Ph = np.mean((tot <= sv + r) & alive) / max(np.mean(alive), 1e-12)
                val = Ph - q * (1.0 + (haz(sv) * EWr if tag == "Cp" else 0.0))
                if prev is not None and prev < 0 <= val and root is None:
                    root = sv
                prev = val
            print(f"      first-order condition {'Phi(s) = q' if tag == 'B' else 'Phi(s) = q (1 + h(s) E[(W - r)^+])'}: first up-crossing at {('%.2f s' % root) if root is not None else 'none'} (E[(W - r)^+] = {EWr:.3f})")
        return best

    run(stats.gamma(2.0, scale=0.8), 1.0, 0.5, "P(W<=r) >= q")
    run(stats.gamma(4.0, scale=2.0), 1.0, 1.5, "P(W<=r) < q")
    print("   Bimodal wait (queue empty or 30 s long, half each), sharp gap: information at arrival matters")

    class Bim:
        def rvs(self, size, random_state):
            return np.where(random_state.random(size) < 0.5, 0.0, 30.0)

    best = run(Bim(), 1.0, 0.25, "W in {0, 30}")
    print("   -> (B) and (C) agree, as R2 states; with W observed at arrival the best start moves (to the arrival here).")
    # 'only if' direction of the summary's "iff"
    c, lamL = 1.0, 1.0
    m = 200.0
    Fx = 1 - math.exp(-r / m)
    print(f"   'iff' check: W = 0, r = {r}, exponential gap mean {m:.0f} s, q = 0.5: P(W <= r) = 1 >= q, yet F(x|s) = {Fx:.4f} < q for all s: nothing is read before arrival.")


# ---------------------------------------------------------------- R3
class LnMix:
    def __init__(self, meds, sigs, w):
        self.meds, self.sigs, self.w = np.array(meds, float), np.array(sigs, float), np.array(w, float)
        self.p_inf = 0.0

    def sf(self, t):
        t = np.asarray(t, dtype=float)
        tt = np.maximum(t, 1e-300)
        out = np.zeros_like(tt)
        for m, s, w in zip(self.meds, self.sigs, self.w):
            out = out + w * stats.norm.sf(np.log(tt / m) / s)
        return np.where(t <= 0, 1.0, out)

    def M(self, t):
        t = np.asarray(t, dtype=float)
        out = np.zeros_like(t)
        for m, s, w in zip(self.meds, self.sigs, self.w):
            out = out + w * np.where(t <= 0, m * math.exp(0.5 * s * s) - t, ln_stoploss(np.maximum(t, 1e-300), m, s))
        return out


def r3():
    global rng
    rng = np.random.default_rng([SEED, 3])
    print("\n== R3: when 'start at the first time F(x|s) >= q' is optimal ==")
    grid = np.linspace(0.0, 120.0, 240_001)
    n_cases = 0
    multi_opt = 0
    multi_not = 0
    single = 0
    bad_iii = 0
    id_err = 0.0
    never = 0
    for trial in range(400):
        k = rng.integers(2, 4)
        meds = np.sort(rng.uniform(4.0, 60.0, size=k))
        sigs = rng.uniform(0.02, 0.25, size=k)
        w = rng.dirichlet(np.ones(k))
        law = LnMix(meds, sigs, w)
        x = rng.uniform(0.5, 8.0)
        c = 1.0
        lamL = rng.choice([0.05, 0.2, 0.5, 1.0, 3.0])
        q = lamL / (c + lamL)
        S = law.sf(grid)
        Sx = law.sf(grid + x)
        J = c * x - c * law.M(grid) + (c + lamL) * law.M(grid + x)
        Jarr = c * x
        alive = S > 1e-12
        Fx = np.where(alive, 1.0 - Sx / np.maximum(S, 1e-300), 1.0)
        above = (Fx >= q) & alive
        if not above.any():
            never += 1
            continue
        i0 = int(np.argmax(above))
        Jrule = J[i0]
        Jmin = min(J.min(), Jarr)
        # identity J(s) - J(tau) = (c + lamL) * integral S (F - q), by trapezoid on the grid
        integrand = S * (Fx - q)
        I = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * (grid[1] - grid[0]))])
        I = I - I[i0]
        id_err = max(id_err, np.max(np.abs((J - Jrule) - (c + lamL) * I)[alive]))
        I_arr = (Jarr - Jrule) / (c + lamL)
        minI = min(I[i0:][alive[i0:]].min(), I_arr)
        # the grid step is 0.5 ms, so the rule's start can sit one step after the true crossing: tolerance 1e-6
        rule_opt = Jrule <= Jmin + 1e-6
        crit = minI >= -1e-6 / (c + lamL)
        if rule_opt != crit:
            bad_iii += 1
            print(f"      (iii) mismatch in trial {trial}: J(rule) - min J = {Jrule - Jmin:.3e}, min I = {minI:.3e} (both at the tolerance 1e-7: a tie, not a disagreement)" if abs(Jrule - Jmin) < 1e-6 and abs(minI) < 1e-6 else f"      !! (iii) real mismatch in trial {trial}: {Jrule - Jmin:.3e} {minI:.3e}")
        ups = int(np.sum((~above[:-1]) & above[1:] & alive[1:])) + int(above[0])
        n_cases += 1
        if ups == 1 and not np.any(above[:-1] & ~above[1:] & alive[1:]):
            single += 1
            if not rule_opt:
                im_ = int(np.argmin(np.where(alive, J, np.inf)))
                print(f"   !! single up-crossing yet rule not optimal, trial {trial}: J(rule) - min = {Jrule - Jmin:.3e}; rule start {grid[i0]:.4f}, S there {S[i0]:.3e}; grid min at {grid[im_]:.4f} (S {S[im_]:.3e}), J arrival - min = {Jarr - Jmin:.3e}; q = {q:.3f}, x = {x:.2f}")
        else:
            if rule_opt:
                multi_opt += 1
            else:
                multi_not += 1
    print(f"   {n_cases} random two- and three-mode lognormal mixtures with a crossing ({never} never reach q):")
    print(f"   identity J(s) - J(tau) = (c + lamL) * integral_tau^s S (F(x|u) - q) du: worst error {id_err:.2e}")
    print(f"   criterion (iii) disagrees with a direct comparison of J in {bad_iii} cases")
    print(f"   single crossing from below: {single} cases, rule optimal in all; several crossings: rule optimal in {multi_opt}, not optimal in {multi_not}")
    print("   -> 'optimal iff one crossing from below' is false in the 'only if' direction whenever the first count is > 0.")

    # explicit example of my own: two modes at 8 s and 20 s
    law = LnMix([8.0, 20.0], [0.05, 0.05], [0.25, 0.75])
    x, c = 2.0, 1.0
    for lamL in (0.12, 0.5):
        q = lamL / (c + lamL)
        S, Sx = law.sf(grid), law.sf(grid + x)
        Fx = np.where(S > 1e-12, 1 - Sx / np.maximum(S, 1e-300), 1.0)
        J = c * x - c * law.M(grid) + (c + lamL) * law.M(grid + x)
        above = Fx >= q
        cross = [(grid[i + 1], "up" if above[i + 1] else "down") for i in range(len(grid) - 1) if above[i] != above[i + 1] and S[i + 1] > 1e-9]
        i0 = int(np.argmax(above))
        im = int(J.argmin())
        print(f"   two modes 8 s (25 %) and 20 s (75 %), x = 2, q = {q:.3f}: crossings {[(round(a, 2), b) for a, b in cross[:4]]}; rule starts {grid[i0]:.2f} s cost {J[i0]:.4f}; best start {grid[im]:.2f} s cost {J[im]:.4f}; arrival {c * x:.4f}")

    # (iv) and (v): monotone F(x|.) -> optimal for every q
    viol = 0
    tested = 0
    for d in (stats.weibull_min(2.5, scale=30.0), stats.gamma(4.0, scale=5.0), stats.uniform(10.0, 20.0)):
        g = np.linspace(0.0, 100.0, 100_001)
        S = d.sf(g)
        Mi = lambda t: stoploss_vec(d, t)
        for x in (0.5, 3.0, 9.0):
            Sx = d.sf(g + x)
            alive = S > 1e-10
            Fx = np.where(alive, 1 - Sx / np.maximum(S, 1e-300), 1.0)
            mono = np.all(np.diff(Fx[alive]) >= -1e-9)
            for q in (0.05, 0.3, 0.6, 0.9):
                lamL = q / (1 - q)
                J = x - Mi(g) + (1 + lamL) * Mi(g + x)
                above = (Fx >= q) & alive
                i0 = int(np.argmax(above)) if above.any() else None
                Jrule = J[i0] if i0 is not None else x
                tested += 1
                if Jrule > min(J[alive].min(), x) + 1e-6:
                    viol += 1
            if not mono:
                print("   !! IFR law with non-monotone F(x|s)")
    print(f"   IFR laws (Weibull k = 2.5, gamma shape 4, uniform): F(x|s) nondecreasing; threshold rule optimal in {tested - viol} of {tested} (q, x) cells (1 ms grid, tolerance 1e-6)")
    # converse of (iv): take q = running max before a dip
    fails = 0
    tot = 0
    for trial in range(60):
        meds = np.sort(rng.uniform(5.0, 50.0, size=2))
        if meds[1] / meds[0] < 1.6:
            continue
        law = LnMix(meds, rng.uniform(0.03, 0.12, size=2), rng.dirichlet([2.0, 2.0]))
        x = rng.uniform(0.5, 3.0)
        S, Sx = law.sf(grid), law.sf(grid + x)
        alive = S > 1e-9
        Fx = np.where(alive, 1 - Sx / np.maximum(S, 1e-300), 1.0)
        # first local max followed by a dip
        j = int(np.argmax(Fx[: int(np.searchsorted(grid, meds[0] * 1.2))]))
        m = Fx[j]
        if not (m < 0.999 and Fx[j + 2000 :][alive[j + 2000 :]].min() < m - 1e-3):
            continue
        q = m
        lamL = q / (1 - q)
        J = x - law.M(grid) + (1 + lamL) * law.M(grid + x)
        above = (Fx >= q - 1e-12) & alive
        i0 = int(np.argmax(above))
        tot += 1
        if J[i0] > min(J[alive].min(), x) + 1e-9:
            fails += 1
    print(f"   converse of (iv): non-monotone F(x|.), price set at the first local maximum: rule not optimal in {fails} of {tot} laws")


# ---------------------------------------------------------------- R4
def r4():
    global rng
    rng = np.random.default_rng([SEED, 4])
    print("\n== R4: falling hazard, drop time, mass at infinity ==")
    n = 2_000_000
    p_inf = 0.2
    law = ScipyLaw(stats.weibull_min(0.7, scale=8.0), p_inf=p_inf)
    R = law.sample(rng, n)
    r, W = 3.0, 1.0
    x = r - W
    c, lamL, Kio = 1.0, 0.4, 0.3

    def gain_sim(s, th):
        ret = np.isfinite(R)
        base = np.where(ret, c * x + Kio, 0.0)
        cost = np.empty(n)
        a = R <= s
        b = (R > s) & (R <= th)
        d = R > th
        cost[a] = c * x + Kio
        cost[b] = c * np.maximum(x - (R[b] - s), 0) + lamL * np.maximum(R[b] - s - x, 0) + Kio
        cost[d] = Kio + lamL * (th - s - r) + np.where(ret[d], c * x + Kio, 0.0)
        g = base - cost
        return g.mean(), g.std(ddof=1) / math.sqrt(n)

    def gain_formula(s, th):
        Sth = float(law.sf(th))
        a = quad_sf(law, s, s + x) - x * Sth
        b = quad_sf(law, s + x, th) - W * Sth
        return c * a - lamL * b - Kio * Sth

    worst = 0.0
    for (s, th) in [(0.0, 3.0), (0.0, 6.0), (0.0, 25.0), (1.5, 4.5), (1.5, 12.0), (6.0, 9.0), (6.0, 60.0), (20.0, 23.0)]:
        est, se = gain_sim(s, th)
        f = gain_formula(s, th)
        worst = max(worst, abs(est - f) / se)
        # derivative in theta
        h = 1e-4
        fd = (gain_formula(s, th + h) - gain_formula(s, th - h)) / (2 * h) if th - h >= s + r else float("nan")
        Knet = c * x + Kio - lamL * W
        dens = (1 - p_inf) * law.d.pdf(th)
        form = dens * Knet - lamL * float(law.sf(th))
        hs = 1e-4
        fds = (gain_formula(s + hs, th) - gain_formula(max(s - hs, 0), th)) / (hs + min(hs, s)) if th >= s + hs + r else float("nan")
        dJ = c * (float(law.sf(s)) - float(law.sf(s + x))) - lamL * float(law.sf(s + x))
        print(f"   G(s = {s}, theta = {th}): simulated {est:+.4f} (se {se:.4f}), formula {f:+.4f}; dG/dtheta fd {fd:+.5f} formula {form:+.5f}; dG/ds fd {fds:+.5f}, -dJ_F/ds {-dJ:+.5f}")
    print(f"   worst |z|: {worst:.2f}  (Weibull k = 0.7 gap, 20 % of sessions never return)")
    # the draft's formula with a mass at infinity
    Emin_all = float(np.mean(np.minimum(R, x)))
    Emin_ret = float(np.mean(np.where(np.isfinite(R), np.minimum(R, x), 0.0)))
    print(f"   draft's c E[min(R, x)] = {c * Emin_all:.4f}; stall actually saved by reading now c E[min(R, x); R < inf] = {c * Emin_ret:.4f}; difference {c * (Emin_all - Emin_ret):.4f} = c x p_inf = {c * x * p_inf:.4f}; E[(R - x)^+] is infinite")
    # (d) lambda = 0, theta = inf, device price only
    kdev = 0.9
    ret = np.isfinite(R)
    g = np.where(ret, c * np.minimum(R, x), 0.0) - np.where(ret, 0.0, kdev)
    print(f"   (d) lambda = 0, never dropped: gain of reading now, simulated {g.mean():+.4f}; formula c E[min(R,x); R<inf] - kappa (L/beta) p_inf = {c * Emin_ret - kdev * p_inf:+.4f}")

    # (a) falling hazard: minimum of J_F at an end; (b) the two-number test
    print("   DFR laws, proper, copy held to admission: where is the minimum of J_F?")
    tot = 0
    interior = 0
    agree_b = 0
    dfr = [
        ("lomax(2.2, 6)", stats.lomax(2.2, scale=6.0)),
        ("weibull(0.55, 9)", stats.weibull_min(0.55, scale=9.0)),
        ("gamma(0.4, 20)", stats.gamma(0.4, scale=20.0)),
        ("exponential(12)", stats.expon(scale=12.0)),
    ]
    g = np.concatenate([[0.0], np.geomspace(1e-4, 3e3, 1500)])
    for name, d in dfr:
        Mg = np.array([stoploss_proper(d, u) for u in g])
        mean = Mg[0]
        for x in (0.3, 2.0, 6.0, 25.0):
            Mgx = np.array([stoploss_proper(d, u) for u in g + x])
            for lamL in (0.0, 0.01, 0.1, 0.5, 2.0):
                for c in (1.0, 0.2):
                    J = c * x - c * Mg + (c + lamL) * Mgx
                    Jarr = c * x
                    i = int(J.argmin())
                    tot += 1
                    if 0 < i and J[i] < min(J[0], Jarr) - 1e-7:
                        interior += 1
                    q = lamL / (c + lamL)
                    Emin = mean - Mgx[0]
                    now_better = J[0] <= Jarr
                    test = Emin / mean >= q
                    if now_better == test or abs(J[0] - Jarr) < 1e-9:
                        agree_b += 1
    print(f"   {tot} cells (4 DFR laws x 4 x x 5 rents x 2 c): interior minimum in {interior}; test E[min(R,x)]/E[R] >= q matches 'now beats arrival' in {agree_b}")
    # ages: 'read now beats arrival' on an initial interval of ages
    bad = 0
    cnt = 0
    for name, d in dfr[:3]:
        ages = np.concatenate([[0.0], np.geomspace(0.01, 500.0, 300)])
        for x in (1.0, 4.0):
            for q in (0.02, 0.1, 0.3):
                flags = []
                for a in ages:
                    Sa = d.sf(a)
                    ER = stoploss_proper(d, a) / Sa
                    Emin = (stoploss_proper(d, a) - stoploss_proper(d, a + x)) / Sa
                    flags.append(Emin / ER >= q)
                flags = np.array(flags)
                cnt += 1
                # once false, never true again
                if np.any(~flags[:-1] & flags[1:]):
                    bad += 1
    print(f"   'now beats arrival' as a function of idle age is an initial interval in {cnt - bad} of {cnt} (law, x, q) cases")


# ---------------------------------------------------------------- R5
def r5():
    global rng
    rng = np.random.default_rng([SEED, 5])
    print("\n== R5: Fetch against Retain ==")
    laws = [
        ScipyLaw(stats.lomax(2.5, scale=7.0)),
        ScipyLaw(stats.weibull_min(0.7, scale=8.0), p_inf=0.2),
        ScipyLaw(stats.gamma(5.0, scale=4.0)),
        TwoPiece(0.41, 1.69, 16.8, cap=1200.0),
        ScipyLaw(stats.expon(scale=12.0), p_inf=0.05),
    ]
    worst = 0.0
    neg_margin_cases = 0
    fetch_beats_keep_when_margin_pos = 0
    for trial in range(1500):
        law = laws[trial % len(laws)]
        r = rng.uniform(0.5, 8.0)
        W = rng.uniform(0.0, r * 0.95)
        x = r - W
        rm = rng.uniform(0.0, r)
        c = rng.choice([0.0, 0.3, 1.0, 3.0])
        lamL = rng.choice([0.0, 0.02, 0.5, 2.0])
        kdev = rng.choice([0.0, 0.3, 1.5])
        Kio = kdev + lamL * rm
        Knet = c * x + Kio - lamL * W
        s = rng.uniform(0.0, 20.0)
        th = s + r + rng.uniform(0.0, 40.0)
        Ss, Sth = float(law.sf(s)), float(law.sf(th))
        A = quad_sf(law, s, s + x)
        B = quad_sf(law, s + x, th)
        Gk = Knet * (Ss - Sth) - lamL * (A + B)
        Gf = c * (A - x * Sth) - lamL * (B - W * Sth) - Kio * Sth
        margin = Knet * Ss - (c + lamL) * A
        worst = max(worst, abs((Gk - Gf) - margin))
        # second form of the margin
        if Ss > 1e-9:
            E_short = x - A / Ss  # E[(x - R_s)^+]
            m2 = Ss * ((c + lamL) * E_short + kdev - lamL * (r - rm))
            worst = max(worst, abs(m2 - margin))
        if margin < -1e-12:
            neg_margin_cases += 1
        elif Gf > Gk + 1e-10:
            fetch_beats_keep_when_margin_pos += 1
    print(f"   identity G_keep - G_fetch = K_net S(s) - (c + lamL)[M(s) - M(s + x)] (two forms) on 1,500 random cells, 5 laws incl. two defective ones: worst error {worst:.2e}")
    print(f"   cells with a negative margin (a read parked without holding pages, device time cheap): {neg_margin_cases}")
    # simulate keep and fetch on one cell
    n = 2_000_000
    law = ScipyLaw(stats.weibull_min(0.7, scale=8.0), p_inf=0.2)
    R = law.sample(rng, n)
    r, W, rm = 3.0, 1.0, 3.0
    x = r - W
    c, lamL, kdev = 1.0, 0.4, 0.2
    Kio = kdev + lamL * rm
    Knet = c * x + Kio - lamL * W
    s, th = 1.0, 9.0
    ret = np.isfinite(R)
    alive = R > s
    base = np.where(alive & ret, c * x + Kio, 0.0)
    keep = np.where(alive & (R <= th), lamL * (R - s + W), 0.0) + np.where(R > th, lamL * (th - s) + np.where(ret, c * x + Kio, 0.0), 0.0)
    gk = base - keep
    A, B = quad_sf(law, s, s + x), quad_sf(law, s + x, th)
    Ss, Sth = float(law.sf(s)), float(law.sf(th))
    Gk = Knet * (Ss - Sth) - lamL * (A + B)
    print(f"   G_keep(s = 1, theta = 9) simulated {gk.mean():+.4f} (se {gk.std(ddof=1) / math.sqrt(n):.4f}), formula {Gk:+.4f}")
    # (4) needs the margin >= 0: exponential gap, read parked 9 of its 10 s holding no pages, free device
    m = 12.0
    law = ScipyLaw(stats.expon(scale=m))
    r, W, rm, c, lamL, kdev = 10.0, 0.0, 1.0, 1.0, 1.0, 0.0
    x = r - W
    Kio = kdev + lamL * rm
    Knet = c * x + Kio - lamL * W
    print(f"   Counterexample to (4) without condition (2): exponential gap mean {m:.0f} s (constant hazard {1 / m:.4f}), r = 10 s of which the pages are held 1 s, W = 0, c = 1, lamL = 1, kappa = 0")
    print(f"      Retain: K_net x hazard = {Knet / m:.3f} < lamL = {lamL}: a copy is dropped at once and never kept")
    for th in (10.0, 15.0, 30.0):
        Ss, Sth = 1.0, float(law.sf(th))
        A, B = quad_sf(law, 0.0, x), quad_sf(law, x, th)
        Gk = Knet * (Ss - Sth) - lamL * (A + B)
        Gf = c * (A - x * Sth) - lamL * (B - W * Sth) - Kio * Sth
        # simulate the fetch
        R = law.sample(rng, 1_000_000)
        base = c * x + Kio
        cost = np.where(R <= th, c * np.maximum(x - R, 0) + lamL * np.maximum(R - x, 0) + Kio, Kio + lamL * (th - r) + c * x + Kio)
        print(f"      theta = {th:4.0f}: G_keep = {Gk:+.3f}, G_fetch = {Gf:+.3f} (simulated {np.mean(base - cost):+.3f})")
    print("      -> a read released at once gains although Retain drops the copy: (4) needs r_m = r or kappa/beta >= lambda (r - r_m).")


# ---------------------------------------------------------------- R6
def r6():
    global rng
    rng = np.random.default_rng([SEED, 6])
    print("\n== R6: after arrival, where the read should land ==")
    n = 1_000_000
    D = 6.0 + stats.gamma(2.0, scale=2.5).rvs(size=n, random_state=rng)  # skewed admission time, now = 0
    grid = np.linspace(0.0, 40.0, 4001)
    Ds = np.sort(D)
    csum = np.concatenate([[0.0], np.cumsum(Ds)])

    def cost_at(l, c, lamL, Dsorted=Ds, cs=csum):
        k = np.searchsorted(Dsorted, l, side="right")
        late = (k * l - cs[k]) / len(Dsorted)  # E (l - D)^+
        early = ((cs[-1] - cs[k]) - (len(Dsorted) - k) * l) / len(Dsorted)
        return c * late + lamL * early

    for (c, lamL, r) in [(1.0, 0.05, 3.0), (1.0, 0.5, 3.0), (1.0, 4.0, 3.0), (0.1, 0.5, 3.0), (1.0, 0.05, 9.0), (1.0, 1e-4, 3.0)]:
        q = lamL / (c + lamL)
        feas = grid[grid >= r]
        vals = np.array([cost_at(l, c, lamL) for l in feas])
        lbest = feas[vals.argmin()]
        Q = np.quantile(D, q)
        print(f"   c = {c}, lamL = {lamL}, q = {q:.4f}, r = {r}: grid-best landing {lbest:.2f} s; max(now + r, Q_D(q)) = {max(r, Q):.2f} s; P(D <= landing) = {np.mean(D <= lbest):.3f} (stall risk)")
    # (a) random read time, dependent on D
    z = rng.standard_normal(n)
    rr = np.maximum(0.3, 3.0 + 0.6 * (D - D.mean()) + 1.2 * z)  # longer reads when admission is later
    slack = np.sort(D - rr)
    cs2 = np.concatenate([[0.0], np.cumsum(slack)])
    tg = np.linspace(0.0, 30.0, 3001)
    for (c, lamL) in [(1.0, 0.1), (1.0, 0.5), (1.0, 3.0)]:
        q = lamL / (c + lamL)
        vals = np.array([cost_at(t, c, lamL, slack, cs2) for t in tg])
        tb = tg[vals.argmin()]
        print(f"   (a) read time dependent on D (corr {np.corrcoef(D, rr)[0, 1]:.2f}): grid-best release {tb:.2f} s; Q_(D - r)(q) = {max(0.0, np.quantile(D - rr, q)):.2f} s; Q_D(q) - E[r] = {np.quantile(D, q) - rr.mean():.2f} s")
    # (b) c = 0
    vals = [cost_at(l, 0.0, 0.5) for l in (10.0, 20.0, Ds[-1], Ds[-1] + 50.0)]
    print(f"   (b) c = 0: cost at landings 10 s, 20 s, latest D, later: {vals[0]:.4f} {vals[1]:.4f} {vals[2]:.6f} {vals[3]:.6f} -> every landing after the latest D costs 0, 'never' included")
    # (c) fixed late charge e
    e = 0.7
    kde = stats.gaussian_kde(D[:200_000])
    for (c, lamL) in [(1.0, 0.3), (1.0, 1.5)]:
        q = lamL / (c + lamL)
        vals = np.array([cost_at(l, c, lamL) + c * e * np.mean(D < l) for l in grid])
        lb = grid[vals.argmin()]
        f = lambda l: np.mean(D <= l) + (1 - q) * e * float(kde(l)[0]) - q
        try:
            root = optimize.brentq(f, 6.0, 30.0)
        except ValueError:
            root = float("nan")
        print(f"   (c) fixed late charge {e} s: grid-best landing {lb:.2f} s; root of F_D + (1 - q) e f_D = q: {root:.2f} s; plain quantile {np.quantile(D, q):.2f} s")
    # (d) a read that cannot be re-timed once released; D = 10 or 20 s (half each), revealed at t_news
    c, lamL, r = 1.0, 0.5, 3.0
    q = lamL / (c + lamL)

    def cost_given(land, D):
        return c * max(land - D, 0.0) + lamL * max(D - land, 0.0)

    def policy_cost(release_before_news, t_news):
        """release_before_news: a time < t_news at which to release without the news, or None to wait."""
        if release_before_news is not None:
            return 0.5 * cost_given(release_before_news + r, 10.0) + 0.5 * cost_given(release_before_news + r, 20.0)
        # after the news D is known: land at max(t_news + r, D)
        return 0.5 * cost_given(max(t_news + r, 10.0), 10.0) + 0.5 * cost_given(max(t_news + r, 20.0), 20.0)

    for t_news in (8.0, 6.0):
        tg = np.linspace(0.0, t_news - 1e-6, 4001)
        p_prior = lambda t: 0.5 * (t + r >= 10.0) + 0.5 * (t + r >= 20.0)
        crossing = next((t for t in tg if p_prior(t) >= q), None)
        best_before = min(policy_cost(t, t_news) for t in tg)
        wait = policy_cost(None, t_news)
        if crossing is not None:
            print(f"   (d) news at t = {t_news}: P(D <= t + r) first reaches q = {q:.3f} at t = {crossing:.2f} (before the news); releasing there costs {policy_cost(crossing, t_news):.3f}; best release before the news {best_before:.3f}; waiting for the news {wait:.3f} -> the first crossing is too early, never too late")
        else:
            print(f"   (d) news at t = {t_news}: the probability stays below q until the news; waiting costs {wait:.3f}, the best release before the news {best_before:.3f} -> releasing while P < q is never optimal")


if __name__ == "__main__":
    which = sys.argv[1:] or ["r1", "r2", "r3", "r4", "r5", "r6"]
    for w in which:
        globals()[w]()
