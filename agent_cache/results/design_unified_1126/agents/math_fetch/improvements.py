"""Check 4: results that would strengthen the Fetch theory, each verified numerically.

I1  Weibull gap: closed form of the optimal start, (s* + x)^k - s*^k = eta^k ln(1 + lamL / c).
I2  Unimodal hazard (lognormal): the optimum is the first up-crossing of F(x|s) = q or the arrival, nothing else.
I3  Value of a hint against its sharpness: V = c x - min_s J_F(s); for a sharp hint V ~ c x - (c + lamL) d phi(z_q);
    exact test of "any read before arrival pays": sup_s [1 - M(s + x) / M(s)] > q.
I4  Point forecast against a distribution: the regret of landing at a fixed quantile.
I5  Landing quantiles on a shared device: reads that queue behind each other land as a block with
    sum_j (c_j + lamL_j) (F_j(C_j) - q_j) = 0.
I6  Life cycle of a sharp hint: drop at the turn's end, read back at the threshold.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script's folder

import numpy as np
from scipy import optimize, special, stats

from fetchlib import TwoPieceLognormal, J_F, F_cond, start_grid, TOOL, Z90


class Weibull:
    def __init__(self, k, eta):
        self.k, self.eta, self.cap = k, eta, None

    def sf(self, t):
        t = np.maximum(np.asarray(t, dtype=float), 0.0)
        return np.exp(-((t / self.eta) ** self.k))

    def M(self, t):
        t = np.maximum(np.asarray(t, dtype=float), 0.0)
        a = 1.0 / self.k
        return self.eta / self.k * special.gamma(a) * special.gammaincc(a, (t / self.eta) ** self.k)


def lognormal(m, sig):
    return TwoPieceLognormal(m * math.exp(-sig * Z90), m, m * math.exp(sig * Z90), sigma_floor=1e-6)


def argmin_J(law, x, c, lamL, hi):
    g = start_grid(hi, n_log=20000, n_lin=20000)
    j = J_F(law, g, x, c, lamL)
    k = int(np.argmin(j))
    return float(g[k]), float(j[k])


def I1():
    print("== I1 Weibull: closed-form start against a grid search ==")
    print("   k     eta    x    c    lamL   closed-form s*   grid s*    J(s*)   J(arrival)")
    worst = 0.0
    for k in [1.5, 2.0, 3.0, 5.0]:
        for eta in [10.0, 30.0]:
            for (x, c, lamL) in [(2.0, 1.0, 0.5), (4.0, 1.0, 0.025), (3.0, 0.3, 2.5)]:
                law = Weibull(k, eta)
                Lam = math.log(1.0 + lamL / c)
                f = lambda s: (s + x) ** k - s ** k - eta ** k * Lam
                s_cf = 0.0 if f(0.0) >= 0 else optimize.brentq(f, 0.0, 1e6)
                s_g, j_g = argmin_J(law, x, c, lamL, 6 * eta)
                alive = float(law.sf(s_cf))
                note = ""
                if alive > 1e-6:
                    worst = max(worst, abs(s_cf - s_g))
                else:
                    note = f"   (P(G > s*) = {alive:.0e}: the same as reading at arrival)"
                print(f"   {k:3.1f}  {eta:5.1f}  {x:3.1f}  {c:3.1f}  {lamL:5.3f}   {s_cf:10.4f}     {s_g:10.4f}  {j_g:7.4f}  {c * x:7.4f}{note}")
    print(f"   max |closed form - grid| = {worst:.4f} s over the cases with P(G > s*) > 1e-6 (grid spacing about 0.01 s)")
    eta, x, c, lamL = 20.0, 3.0, 1.0, 0.5
    print(f"   Rayleigh (k = 2): s* = eta^2 ln(1 + lamL/c) / (2 x) - x / 2 = {eta**2 * math.log(1 + lamL / c) / (2 * x) - x / 2:.4f} s"
          f"; grid {argmin_J(Weibull(2.0, eta), x, c, lamL, 120.0)[0]:.4f} s")
    # exponential: all or nothing
    for mean in [2.0, 8.0, 30.0]:
        law = Weibull(1.0, mean)
        Lam = math.log(1 + lamL / c)
        s_g, j_g = argmin_J(law, x, c, lamL, 400.0)
        print(f"   exponential mean {mean}: rule 'read now iff x >= mean ln(1 + lamL/c) = {mean * Lam:.2f}' -> {'now' if x >= mean * Lam else 'arrival'};"
              f" grid minimum at s = {s_g:.2f} (J = {j_g:.4f}, arrival {c * x:.4f})")


def I2():
    print("== I2 lognormal (unimodal hazard): optimum is the first up-crossing of F(x|s) = q, or the arrival ==")
    worst = 0.0
    n = 0
    kinds = {"crossing": 0, "now": 0, "arrival": 0}
    for sig in [0.1, 0.2, 0.35, 0.5, 0.8, 1.2, 1.8]:
        for m in [5.0, 20.0, 60.0]:
            for x in [1.0, 3.0, 6.0]:
                for lamL in [0.025, 0.5, 2.5]:
                    c = 1.0
                    q = lamL / (c + lamL)
                    law = lognormal(m, sig)
                    hi = m * math.exp(6 * sig) + 50
                    g = start_grid(hi, n_log=20000, n_lin=20000)
                    j = J_F(law, g, x, c, lamL)
                    jmin = float(j.min())
                    fx = F_cond(law, x, g)
                    ok = np.flatnonzero(fx >= q)
                    cand = [c * x]
                    if len(ok):
                        cand.append(float(j[ok[0]]))
                    two = min(cand)
                    worst = max(worst, two - jmin)
                    n += 1
                    if two == c * x:
                        kinds["arrival"] += 1
                    elif ok[0] == 0:
                        kinds["now"] += 1
                    else:
                        kinds["crossing"] += 1
    print(f"   {n} cases: min(J(first up-crossing), J(arrival)) - global minimum <= {worst:.2e}; outcomes {kinds}")


def I3():
    print("== I3 value of a hint against its sharpness (lognormal gap, median 30 s, x = 3 s, c = 1) ==")
    m, x, c = 30.0, 3.0, 1.0
    sigs = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.8]
    for lamL in [0.025, 0.5, 2.5]:
        q = lamL / (c + lamL)
        zq = stats.norm.ppf(q)
        print(f"   lamL = {lamL} (q = {q:.3f}): price of spread (c + lamL) phi(z_q) = {(c + lamL) * stats.norm.pdf(zq):.3f} s per second of standard deviation;"
              f" rule of thumb: worthless once sd > x (1 - q)/phi(z_q) = {x * (1 - q) / stats.norm.pdf(zq):.1f} s")
        print("      sigma   sd d (s)  d/x     V = c x - min J   V/(c x)   sharp-hint formula c x - (c+lamL) d phi(z_q)   best start (s)   sup_s F_e(x|s)")
        for sig in sigs:
            law = lognormal(m, sig)
            d = m * math.sqrt((math.exp(sig * sig) - 1.0) * math.exp(sig * sig))
            hi = m * math.exp(6 * sig) + 50
            s_g, j_g = argmin_J(law, x, c, lamL, hi)
            V = max(c * x - j_g, 0.0)
            approx = c * x - (c + lamL) * d * stats.norm.pdf(zq)
            g = start_grid(hi, n_log=20000, n_lin=20000)
            Ms, Msx = law.M(g), law.M(g + x)
            ok = Ms > 1e-12
            Fe = float(np.max(1.0 - Msx[ok] / Ms[ok]))
            print(f"      {sig:5.3f}   {d:7.2f}  {d / x:6.2f}   {V:8.4f}         {V / (c * x):6.3f}    {approx:8.3f}"
                  f"                                   {'arrival' if V <= 1e-9 else f'{s_g:8.2f}'}       {Fe:6.3f} {'>' if Fe > q else '<='} q")
    # the exact test on the tool entry
    law = TwoPieceLognormal(**TOOL)
    g = start_grid(4000.0, n_log=20000, n_lin=20000)
    print("   tool entry (uncapped), exact test sup_s [1 - M(s+x)/M(s)] > q for 'some read before arrival, held to admission, pays':")
    for x in [1.0, 2.0, 3.0, 4.0, 6.0]:
        Fe = 1.0 - law.M(g + x) / law.M(g)
        k = int(np.argmax(Fe))
        lam_max = float(Fe[k]) / (1 - float(Fe[k])) / 25_000       # q = lamL/(c+lamL) < Fe  <=> lamL/c < Fe/(1-Fe)
        print(f"      x = {x}: sup = {float(Fe[k]):.4f} at s = {g[k]:.3f} s -> pays iff lambda L / c < {float(Fe[k]) / (1 - float(Fe[k])):.4f},"
              f" i.e. lambda < {lam_max:.2e} per token-second at L = 25 K, c = 1")


def I4():
    print("== I4 a point forecast against the distribution (lognormal hint, median 30 s, sigma 0.3; x = 3 s, c = 1) ==")
    m, sig, x, c = 30.0, 0.3, 3.0, 1.0
    law = lognormal(m, sig)
    print("   lamL    q      optimal landing (s) = quantile   cost     | landing at p10   p50     p90     arrival")
    rows = []
    for lamL in [0.025, 0.1, 0.5, 2.5, 10.0]:
        q = lamL / (c + lamL)
        s_g, j_g = argmin_J(law, x, c, lamL, 400.0)
        land = s_g + x
        costs = [float(J_F(law, max(float(law.ppf(p)) - x, 0.0), x, c, lamL)) for p in (0.1, 0.5, 0.9)]
        rows.append((lamL, q, j_g, costs))
        print(f"   {lamL:5.3f}  {q:5.3f}   {land:7.2f} = p{100 * float(law.cdf(land)):4.1f}            {j_g:6.3f}   |   {costs[0]:6.3f}      {costs[1]:6.3f}  {costs[2]:6.3f}  {c * x:6.3f}")
    for i, p in enumerate((0.1, 0.5, 0.9)):
        reg = [r[3][i] - r[2] for r in rows]
        print(f"   regret of always landing at p{int(100 * p)}: " + ", ".join(f"{v:.3f}" for v in reg) + " s over the five prices")
    # the p50 rule on the tool entry: mean-residual test
    tool = TwoPieceLognormal(**TOOL)
    print("   tool entry: a rule that starts at s helps iff E[min(R_s, x)] / E[R_s] > q (R_s = remaining gap at s):")
    for x in [2.0, 4.0]:
        for name, s in [("now", 0.0), ("land at p50", max(1.69 - x, 0.0)), ("land at p90", max(16.8 - x, 0.0))]:
            Sa = float(tool.sf(s))
            ratio = float(tool.M(s) - tool.M(s + x)) / float(tool.M(s))
            print(f"      x = {x}, start {name:12s} (s = {s:5.2f}): ratio {ratio:.4f} -> helps iff lambda < {ratio / (1 - ratio) / 25_000:.2e} (L = 25 K, c = 1)")


def I5():
    print("== I5 landing quantiles on one device: three reads in a fixed order ==")
    # read j: device time p_j, due time D_j ~ N(d_j, sd_j^2), stall weight c_j, rent b_j = lambda L_j
    p = np.array([1.2, 2.0, 1.5])
    d = np.array([10.0, 11.0, 11.5])
    sd = np.array([1.0, 1.5, 1.0])
    cw = np.array([1.0, 1.0, 0.3])
    b = np.array([0.5, 0.25, 0.5])
    q = b / (cw + b)

    def cost_C(C):
        z = (C - d) / sd
        late = sd * (stats.norm.pdf(z) + z * stats.norm.cdf(z))            # E (C - D)^+
        early = late - (C - d)                                             # E (D - C)^+
        return float(np.sum(cw * late + b * early))

    def C_of(t):          # start times -> completion times, respecting the order (no overlap)
        C = np.zeros(3)
        prev = -np.inf
        for j in range(3):
            st = max(t[j], prev)
            C[j] = st + p[j]
            prev = C[j]
        return C

    res = optimize.minimize(lambda t: cost_C(C_of(t)), x0=d - p - 3.0, method="Nelder-Mead",
                            options=dict(xatol=1e-9, fatol=1e-12, maxiter=40000, maxfev=40000))
    C = C_of(res.x)
    own = d + sd * stats.norm.ppf(q)                  # each read's own quantile landing
    naive = C_of(own - p)                             # release each at its own quantile minus its own time; the line pushes the later ones
    print(f"   own-quantile landings (no line): {np.round(own, 3)}  with q = {np.round(q, 3)}")
    print(f"   optimal completions:             {np.round(C, 3)}   cost {cost_C(C):.4f}")
    print(f"   each released at own quantile:   {np.round(naive, 3)}   cost {cost_C(naive):.4f}")
    Fj = stats.norm.cdf((C - d) / sd)
    marg = (cw + b) * (Fj - q)
    # blocks = maximal runs with no idle time between them
    blocks, cur = [], [0]
    for j in range(1, 3):
        if abs(C[j] - p[j] - C[j - 1]) < 1e-5:
            cur.append(j)
        else:
            blocks.append(cur)
            cur = [j]
    blocks.append(cur)
    for blk in blocks:
        print(f"   block {blk}: sum of (c_j + b_j)(F_j(C_j) - q_j) = {marg[blk].sum():+.2e}; members' F_j(C_j) = {np.round(Fj[blk], 3)} against own q {np.round(q[blk], 3)}")


def I6():
    print("== I6 life cycle of a sharp hint (test run of about 20 s, sigma 0.1; L = 25 K; r = 3 s, W = 0, c = 1) ==")
    law = lognormal(20.0, 0.1)
    x, c = 3.0, 1.0
    for lam in [1e-6, 2e-5, 1e-4]:
        lamL = lam * 25_000
        s_g, j_g = argmin_J(law, x, c, lamL, 200.0)
        keep = lamL * float(law.M(0.0))               # keep in memory from the turn's end until arrival
        d = 20.0 * math.sqrt((math.exp(0.01) - 1.0) * math.exp(0.01))
        q = lamL / (c + lamL)
        print(f"   lambda = {lam:g}: keep all along {keep:6.3f} | drop and read at arrival {c * x:5.3f} | drop, read back at s* = {s_g:5.2f} s: {j_g:5.3f}"
              f"  (sharp-hint formula (c + lamL) d phi(z_q) = {(c + lamL) * d * stats.norm.pdf(stats.norm.ppf(q)):.3f}; F(x|s*) = {float(F_cond(law, x, s_g)):.3f}, q = {q:.3f})")


if __name__ == "__main__":
    I1()
    I2()
    I3()
    I4()
    I5()
    I6()
