"""Check 1: J_F with recourse, its closed form and its derivative.

(A) deterministic W:  J_F(s) = c x - c M(s) + (c + lamL) M(s + x),
                      dJ_F/ds = S(s) [ c F(x|s) - lamL (1 - F(x|s)) ]
(B) random W independent of the gap, a read started at arrival is not re-timed:
                      J(s) = kbar - c M(s) + (c + lamL) E_W[ M(s + (r - W)^+) ],
                      dJ/ds = S(s) [ (c + lamL) Phi(s) - lamL ],  Phi(s) = P(A + W <= s + r | A > s)
(C) random W, the read started at arrival is held to its best landing (newsvendor):
                      J(s) = J_B(s) - (kbar - kstar) F(s)
(D) defective law (mass at infinity), drop time theta, fixed cost K_io per read:
      G(s, theta) = J(arrival) - J(s, theta)
                  = c int_s^{s+x} [S(v) - S(theta)] dv - lamL [ int_{s+x}^{theta} S(v) dv - W S(theta) ] - K_io S(theta)

Each formula is compared with a Monte Carlo of the policy itself (not of the formula).
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script's folder

import numpy as np
from scipy import integrate, special

from fetchlib import TwoPieceLognormal, J_F, dJ_F, TOOL

rng = np.random.default_rng(20261126)
N = 2_000_000


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

    def sample(self, rng, n):
        return self.eta * (-np.log(rng.random(n))) ** (1.0 / self.k)


class Atoms:
    """A discrete law (a point hint, or a few support points)."""

    def __init__(self, pts, w):
        self.pts, self.w, self.cap = np.asarray(pts, float), np.asarray(w, float), None

    def sf(self, t):
        t = np.asarray(t, dtype=float)
        return (self.w[None, :] * (self.pts[None, :] > t.reshape(-1, 1))).sum(1).reshape(t.shape)

    def M(self, t):
        t = np.asarray(t, dtype=float)
        return (self.w[None, :] * np.maximum(self.pts[None, :] - t.reshape(-1, 1), 0.0)).sum(1).reshape(t.shape)

    def sample(self, rng, n):
        return rng.choice(self.pts, size=n, p=self.w)


def mc_cost_det(R, s, x, c, lamL):
    """Simulate the policy: read starts at min(s, R); stall and early hold against T = R + W."""
    start = np.minimum(s, R)
    lead = (R - start) - x          # T - landing = (R + W) - (start + r)
    stall = np.maximum(-lead, 0.0)
    hold = np.maximum(lead, 0.0)
    v = c * stall + lamL * hold
    return v.mean(), v.std() / math.sqrt(len(v))


def part_A():
    print("== (A) deterministic W: closed form and derivative vs Monte Carlo and finite differences ==")
    laws = {
        "lognormal sigma 0.3, median 20 s": TwoPieceLognormal(20 * math.exp(-0.3 * 1.2815515655446004), 20.0, 20 * math.exp(0.3 * 1.2815515655446004)),
        "tool entry 0.41/1.69/16.8": TwoPieceLognormal(**TOOL),
        "Weibull k=2, eta=10": Weibull(2.0, 10.0),
        "Weibull k=0.6, eta=5": Weibull(0.6, 5.0),
        "exponential mean 4": Weibull(1.0, 4.0),
        "atoms 3 s (0.3), 12 s (0.7)": Atoms([3.0, 12.0], [0.3, 0.7]),
    }
    worst = 0.0
    worst_d = 0.0
    for name, law in laws.items():
        R = law.sample(rng, N)
        for (x, c, lamL) in [(2.0, 1.0, 0.5), (4.0, 0.3, 0.025), (1.0, 1.0, 2.5)]:
            for s in [0.0, 0.7, 2.5, 6.0, 15.0, 40.0]:
                jf = float(J_F(law, s, x, c, lamL))
                mc, se = mc_cost_det(R, s, x, c, lamL)
                # se = 0 when every sample arrives before s (the cost is then the constant c x)
                z = (mc - jf) / se if se > 1e-9 else (0.0 if abs(mc - jf) < 1e-6 else float("inf"))
                worst = max(worst, abs(z))
                # derivative: central difference of the closed form vs the draft's formula
                eps = 1e-5
                fd = float(J_F(law, s + eps, x, c, lamL) - J_F(law, max(s - eps, 0.0), x, c, lamL)) / (eps + min(s, eps))
                dj = float(dJ_F(law, s, x, c, lamL))
                if not isinstance(law, Atoms):
                    worst_d = max(worst_d, abs(fd - dj))
        print(f"  {name:34s} max |z| so far {worst:5.2f}   max |finite diff - formula| so far {worst_d:.2e}")
    print(f"  worst Monte Carlo z-score over {len(laws) * 3 * 6} cells: {worst:.2f} (N = {N:,} per law)")
    print(f"  worst |dJ/ds finite difference - formula| (continuous laws): {worst_d:.2e}")
    # the closed form's quadrature: M(t) = int_t^inf S
    law = TwoPieceLognormal(**TOOL)
    for t in [0.0, 0.5, 1.69, 5.0, 60.0]:
        q, _ = integrate.quad(lambda v: float(law.sf(v)), t, np.inf, limit=400)
        print(f"  stop-loss M({t:5.2f}) closed form {float(law.M(t)):.6f}  quadrature {q:.6f}")
    # atoms: J is continuous, the derivative formula is the right derivative
    law = laws["atoms 3 s (0.3), 12 s (0.7)"]
    x, c, lamL = 2.0, 1.0, 0.5
    for s in [0.999, 1.0, 1.001, 2.999, 3.0, 3.001, 9.999, 10.0, 10.001]:
        print(f"  atoms: s={s:7.3f}  J={float(J_F(law, s, x, c, lamL)):.5f}  right-derivative formula {float(dJ_F(law, s, x, c, lamL)):+.3f}")


def part_B_C():
    print("== (B),(C) random W independent of the gap ==")
    law = TwoPieceLognormal(20 * math.exp(-0.15 * 1.2815515655446004), 20.0, 20 * math.exp(0.15 * 1.2815515655446004))
    cases = [
        ("(i)  P(W <= r) >= q: no hold after arrival", 4.0, 1.0, 0.5, 2.0, 0.75),
        ("(ii) P(W <= r) <  q: a read started at arrival would be held", 4.0, 1.0, 1.5, 4.0, 1.5),
    ]
    for (label, r, c, lamL, kW, thW) in cases:
        q = lamL / (c + lamL)
        R = law.sample(rng, N)
        W = rng.gamma(kW, thW, N)
        Wq = rng.gamma(kW, thW, 400_000)          # an independent sample for the W-integrals
        xw = np.maximum(r - Wq, 0.0)
        kbar = (c * np.maximum(r - Wq, 0.0) + lamL * np.maximum(Wq - r, 0.0)).mean()
        FWr = float(np.mean(Wq <= r))

        def J_B(s):
            return kbar - c * float(law.M(s)) + (c + lamL) * float(np.mean(law.M(s + xw)))

        def dJ_B(s):
            return c * float(law.sf(s)) - (c + lamL) * float(np.mean(law.sf(s + xw)))

        # best post-arrival delay delta*: land at the q-quantile of W, never before r
        deltas = np.linspace(0.0, 12.0, 2401)
        kd = [(c * np.maximum(d + r - Wq, 0.0) + lamL * np.maximum(Wq - d - r, 0.0)).mean() for d in deltas]
        kstar, dstar = float(np.min(kd)), float(deltas[int(np.argmin(kd))])
        qW = float(np.quantile(Wq, q))
        print(f"  {label}")
        print(f"    r={r}, W ~ gamma(mean {kW*thW}, shape {kW}), c={c}, lamL={lamL}, q={q:.3f}, P(W<=r)={FWr:.3f}:"
              f" kbar={kbar:.4f}, kstar={kstar:.4f} at delay {dstar:.2f} s (newsvendor: (Q_W(q) - r)^+ = {max(qW - r, 0.0):.2f} s)")
        worst = 0.0
        for s in [0.0, 8.0, 14.0, 17.0, 20.0, 24.0]:
            start = np.minimum(s, R)
            lead = R + W - start - r
            v = c * np.maximum(-lead, 0.0) + lamL * np.maximum(lead, 0.0)
            mcB, seB = v.mean(), v.std() / math.sqrt(N)
            # (C): arrivals before s are released dstar after arrival
            startC = np.where(R <= s, R + dstar, s)
            leadC = R + W - startC - r
            vC = c * np.maximum(-leadC, 0.0) + lamL * np.maximum(leadC, 0.0)
            mcC, seC = vC.mean(), vC.std() / math.sqrt(N)
            jB = J_B(s)
            jC = jB - (kbar - kstar) * float(law.cdf(s))
            eps = 1e-4
            fd = (J_B(s + eps) - J_B(max(s - eps, 0))) / (eps + min(s, eps))
            alive = R > s
            Phi_mc = np.mean((R[alive] + W[alive]) <= s + r)
            dj_phi = float(law.sf(s)) * ((c + lamL) * Phi_mc - lamL)
            zB, zC = (mcB - jB) / seB, (mcC - jC) / seC
            worst = max(worst, abs(zB), abs(zC))
            print(f"    s={s:5.1f}  (B) formula {jB:.4f} MC {mcB:.4f} (z {zB:+.2f})   (C) formula {jC:.4f} MC {mcC:.4f} (z {zC:+.2f})"
                  f"   dJ_B/ds: finite diff {fd:+.5f} | S(s)[(c+lamL)Phi-lamL], Phi by MC {dj_phi:+.5f} | formula {dJ_B(s):+.5f}   Phi={Phi_mc:.3f}")
        print(f"    worst z-score {worst:.2f}")
        g = np.linspace(0.0, 60.0, 1201)
        jb = np.array([J_B(s) for s in g])
        jc = jb - (kbar - kstar) * law.cdf(g)
        Phi_at = lambda s: 1.0 - float(np.mean(law.sf(s + xw))) / max(float(law.sf(s)), 1e-300)
        sB, sC = g[int(np.argmin(jb))], g[int(np.argmin(jc))]
        phis = np.array([Phi_at(s) for s in g[g <= 30.0]])
        print(f"    sup_s Phi(s) = {phis.max():.3f} (bounded by P(W<=r) = {FWr:.3f});"
              f" (B): min J {jb.min():.4f} at s={sB:.2f} vs arrival {kbar:.4f};"
              f" (C): min J {jc.min():.4f} at s={sC:.2f} vs arrival {kstar:.4f}")


def part_D():
    print("== (D) defective law, drop time theta, fixed read cost K_io: G(s, theta) vs Monte Carlo ==")
    base = TwoPieceLognormal(**TOOL)
    p_inf = 0.15                      # the session ended with this probability
    r, Wq, c, lamL, Kio = 3.0, 1.0, 1.0, 0.5, 0.4
    x = r - Wq
    S = lambda t: p_inf + (1 - p_inf) * float(base.sf(t))
    M_fin = lambda t: (1 - p_inf) * float(base.M(t))      # int_t^inf (S - p_inf)

    def G(s, th):
        # int_a^b S(v) dv = M_fin(a) - M_fin(b) + p_inf (b - a)
        I1 = M_fin(s) - M_fin(s + x) + p_inf * x - x * S(th)
        I2 = M_fin(s + x) - M_fin(th) + p_inf * (th - s - x)
        return c * I1 - lamL * (I2 - Wq * S(th)) - Kio * S(th)

    n = N
    worst = 0.0
    for k, (s, th) in enumerate([(0.0, 3.0), (0.0, 6.0), (0.0, 30.0), (1.0, 4.0), (1.0, 12.0), (5.0, 8.0), (5.0, 60.0)]):
        assert th >= s + r - 1e-12
        rk = np.random.default_rng(1000 + k)          # a fresh sample per cell, so the z-scores are independent
        R = base.sample(rk, n)
        R = np.where(rk.random(n) < p_inf, np.inf, R)
        # arrival policy: stall x and one read for every session that returns
        cost_arr = np.where(np.isfinite(R), c * x + Kio, 0.0)
        cost = np.zeros(n)
        early = R <= s
        cost[early] = c * x + Kio
        served = (R > s) & (R <= th)
        lead = (R[served] + Wq) - (s + r)
        cost[served] = c * np.maximum(-lead, 0.0) + lamL * np.maximum(lead, 0.0) + Kio
        gone = R > th
        back = gone & np.isfinite(R)
        cost[gone] = lamL * (th - s - r) + Kio
        cost[back] += c * x + Kio
        d = cost_arr - cost
        mc, se = d.mean(), d.std() / math.sqrt(n)
        z = (mc - G(s, th)) / se
        worst = max(worst, abs(z))
        print(f"  s={s:4.1f} theta={th:5.1f}  G formula {G(s, th):+.5f}  Monte Carlo {mc:+.5f} +- {se:.5f}  (z {z:+.2f})")
    print(f"  worst z-score {worst:.2f}")
    # decomposition: G(s, theta) = Flight(s) + int_{s+r}^{theta} S(u) [K_net h(u) - lamL] du
    Knet = c * x + Kio - lamL * Wq
    hS = lambda u: (1 - p_inf) * float(base.pdf(u))      # h(u) S(u) = f(u)
    for (s, th) in [(0.0, 6.0), (1.0, 12.0), (5.0, 60.0)]:
        flight = G(s, s + r)
        hold, _ = integrate.quad(lambda u: Knet * hS(u) - lamL * S(u), s + r, th, limit=400)
        print(f"  decomposition s={s} theta={th}: Flight {flight:+.5f} + Hold {hold:+.5f} = {flight + hold:+.5f}  vs G {G(s, th):+.5f}")
    # the draft's DFR condition with a defective law
    lhs_draft = c * (M_fin(0) - M_fin(x) + p_inf * x)     # c E[min(R, x)] with min(inf, x) = x
    lhs_exact = c * (M_fin(0) - M_fin(x))                # c E[min(R, x); R < inf]
    print(f"  defective law p_inf={p_inf}: the draft's c E[min(R,x)] = {lhs_draft:.4f}; the stall actually saved is"
          f" c E[min(R,x); R<inf] = {lhs_exact:.4f} (difference c x p_inf = {c * x * p_inf:.4f});"
          f" the draft's rent term lamL E[(R-x)^+] is infinite")


if __name__ == "__main__":
    part_A()
    part_B_C()
    part_D()
