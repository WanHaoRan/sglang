"""Check 2c: is "Fetch before arrival empty, the decision is Retain's" right on falling-hazard keys?

Two policies for one gap, both compared with "drop now, read at arrival" (gain > 0 = better):
  keep(s, theta):  the copy is in host memory at age s; keep it until theta, then drop.
      G_keep(s, theta)  = K_net [S(s) - S(theta)] - lamL [M(s) - M(theta)],   K_net = c x + K_io - lamL W
  fetch(s, theta): the copy is on the SSD only; start its read at s (if the turn has not arrived),
      keep the landed copy until theta >= s + r, then drop.
      G_fetch(s, theta) = c [M(s) - M(s+x) - x S(theta)] - lamL [M(s+x) - M(theta) - W S(theta)] - K_io S(theta)
Claim (identity, any law, any theta):
      G_keep(s, theta) - G_fetch(s, theta) = K_net S(s) - (c + lamL) [M(s) - M(s+x)]
                                           >= (K_io - lamL r) S(s)
so with K_io >= lamL r (a read holds its pages for its whole duration, or device time is priced)
a read before arrival pays only where keeping pays.

x = r - W > 0, S survival of the gap, M(t) = E[(G - t)^+], K_io = kappa L / beta + lamL r_m the fixed
cost of one read (device time plus the rent of its pages in flight).
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script's folder

import numpy as np

from fetchlib import TwoPieceLognormal, TOOL, HUMAN, CAP

L = 25_000
BETA = 21_400.0
rng = np.random.default_rng(7)


def G_keep(law, s, th, x, W, c, lamL, Kio):
    Knet = c * x + Kio - lamL * W
    return Knet * (law.sf(s) - law.sf(th)) - lamL * (law.M(s) - law.M(th))


def G_fetch(law, s, th, x, W, c, lamL, Kio):
    Sth = law.sf(th)
    return (c * (law.M(s) - law.M(s + x) - x * Sth)
            - lamL * (law.M(s + x) - law.M(th) - W * Sth) - Kio * Sth)


def mc_keep(R, s, th, x, W, c, lamL, Kio):
    """Simulate keep(s, theta) against 'drop at s, read at arrival', for sessions alive at s (others: 0)."""
    n = len(R)
    gain = np.zeros(n)
    served = (R > s) & (R <= th)
    gain[served] = c * x + Kio - lamL * (R[served] - s + W)     # no read, no stall; rent from s to admission
    gone = R > th
    gain[gone] = -lamL * (th - s)                               # rent until the drop, then as the baseline
    return gain.mean(), gain.std() / math.sqrt(n)


def part_identity():
    print("== identity G_keep - G_fetch = K_net S(s) - (c + lamL) [M(s) - M(s+x)], and Monte Carlo of keep ==")
    law = TwoPieceLognormal(**TOOL)
    R = law.sample(rng, 2_000_000)
    worst_id, worst_z, viol = 0.0, 0.0, 0
    for _ in range(4000):
        r = rng.uniform(0.5, 8.0)
        W = rng.uniform(0.0, r * 0.95)
        x = r - W
        c = rng.choice([1.0, 0.3, 0.05])
        lamL = rng.choice([0.0, 0.025, 0.5, 2.5])
        r_m = r * rng.choice([1.0, 1.0, 0.5])            # 0.5: half of r is spent parked without pages
        Kio = lamL * r_m + rng.choice([0.0, 0.3])
        s = rng.choice([0.0, 0.3, 1.0, 3.0, 10.0, 40.0])
        th = s + r + rng.choice([0.0, 1.0, 5.0, 30.0, 300.0])
        lhs = float(G_keep(law, s, th, x, W, c, lamL, Kio) - G_fetch(law, s, th, x, W, c, lamL, Kio))
        Knet = c * x + Kio - lamL * W
        rhs = float(Knet * law.sf(s) - (c + lamL) * (law.M(s) - law.M(s + x)))
        worst_id = max(worst_id, abs(lhs - rhs))
        if Kio >= lamL * r - 1e-12 and lhs < -1e-10:
            viol += 1
    print(f"  4,000 random cells: max |lhs - rhs| = {worst_id:.2e}; cells with K_io >= lamL r where fetch beat keep: {viol}")
    for (s, th, r, W, c, lamL, Kio) in [(0.0, 5.0, 3.0, 1.0, 1.0, 0.5, 1.5), (1.0, 20.0, 4.0, 0.0, 0.3, 0.025, 0.1), (3.0, 9.0, 2.0, 0.5, 1.0, 2.5, 5.0)]:
        x = r - W
        mc, se = mc_keep(R, s, th, x, W, c, lamL, Kio)
        g = float(G_keep(law, s, th, x, W, c, lamL, Kio))
        worst_z = max(worst_z, abs((mc - g) / se))
        print(f"  keep(s={s}, theta={th}; r={r}, W={W}, c={c}, lamL={lamL}, K_io={Kio}): formula {g:+.5f}  Monte Carlo {mc:+.5f} +- {se:.5f}")
    print(f"  worst z-score of G_keep {worst_z:.2f}")
    # when a parked read holds no pages (K_io < lamL r) a fetch can beat keeping
    r, W, c, lamL = 6.0, 0.0, 1.0, 0.5
    x = r - W
    Kio = lamL * 2.07                                    # pages held for 2.07 s of a 6 s read; 3.9 s parked in the device's line
    s = 30.0
    th = s + r
    print(f"  counterexample with K_io = {Kio:.3f} < lamL r = {lamL * r:.3f} (read parked 3.9 s holding no pages), s = {s}:"
          f" G_keep = {float(G_keep(law, s, th + 20, x, W, c, lamL, Kio)):+.4f}, G_fetch = {float(G_fetch(law, s, th + 20, x, W, c, lamL, Kio)):+.4f}")


def retention_value(law, ages, x, W, c, lamL, Kio, hi):
    """V_ret(a) = sup_theta G_keep(a, theta) / S(a) on a grid of drop times."""
    th = np.unique(np.concatenate([np.geomspace(1e-3, hi, 4000), np.linspace(0, hi, 4000)]))
    Knet = c * x + Kio - lamL * W
    A = Knet * law.sf(th) + lamL * law.M(th)       # G_keep(a, theta) = [Knet S(a) + lamL... ] - A(theta) + ...
    out = []
    for a in ages:
        m = th >= a
        g = Knet * (law.sf(a) - law.sf(th[m])) - lamL * (law.M(a) - law.M(th[m]))
        k = int(np.argmax(g))
        out.append((float(g[k]) / float(law.sf(a)), float(th[m][k])))
    return out


def fetch_value(law, a, x, W, r, c, lamL, Kio, hi):
    """sup over s >= a, theta >= s + r of G_fetch(s, theta) / S(a)."""
    s_grid = np.unique(np.concatenate([[a], a + np.geomspace(1e-3, max(hi - a, 1e-2), 160), [max(hi - x, a)]]))
    best = (0.0, None, None)
    for s in s_grid:
        th = s + r + np.concatenate([[0.0], np.geomspace(1e-3, max(hi - s - r, 1e-2) + 1.0, 400)])
        g = G_fetch(law, s, th, x, W, c, lamL, Kio)
        k = int(np.argmax(g))
        if g[k] > best[0] + 1e-12:
            best = (float(g[k]), float(s), float(th[k]))
    return best[0] / float(law.sf(a)), best[1], best[2]


def part_tool():
    print("== tool entry (uncapped): where Retain keeps, where a read before arrival would pay ==")
    law = TwoPieceLognormal(**TOOL)          # no cap: the cap's atom is a replay artifact
    hi = 4000.0
    ages = [0.0, 0.2, 0.5, 0.8, 1.2, 1.7, 2.5, 4.0, 6.0, 10.0, 15.0, 25.0, 40.0, 80.0, 200.0]
    for (r, W, c, lam, kappa) in [(3.0, 0.0, 1.0, 2e-5, 0.0), (3.0, 0.0, 1.0, 1e-6, 0.0), (3.0, 0.0, 1.0, 1e-4, 0.0),
                                  (5.0, 2.0, 1.0, 2e-5, 0.0), (3.0, 0.0, 0.3, 2e-5, 0.0), (3.0, 0.0, 1.0, 2e-5, 0.5)]:
        x = r - W
        lamL = lam * L
        Kio = lamL * r + kappa * L / BETA            # pages held from release (r_m = r)
        Knet = c * x + Kio - lamL * W
        rv = retention_value(law, ages, x, W, c, lamL, Kio, hi)
        print(f"  r={r}, W={W}, c={c}, lambda={lam:g} (lamL={lamL}), kappa={kappa}: K_io={Kio:.3f}, K_net={Knet:.3f};"
              f" Retain's local rule keeps while hazard >= lamL/K_net = {lamL / Knet:.4f} /s")
        print("     age (s)   hazard   V_ret (keep value, s)  drop at (s) | fetch value (s)  start   drop at | hysteresis (c+lamL)E[(x-R)^+] + kappa L/beta")
        for a, (v, thk) in zip(ages, rv):
            fv, fs, fth = fetch_value(law, a, x, W, r, c, lamL, Kio, hi)
            Sa = float(law.sf(a))
            hyst = (c + lamL) * (x - float(law.M(a) - law.M(a + x)) / Sa) + kappa * L / BETA
            fdesc = f"{fv:8.4f}      {fs:7.2f} {fth:8.2f}" if fs is not None else f"{0.0:8.4f}      (none)          "
            print(f"     {a:6.1f}   {float(law.hazard(a)):6.4f}   {v:8.4f}              {thk:8.2f}   | {fdesc} | {hyst:8.4f}")
    # a price fall after the drop: the copy was dropped at lambda = 1e-4; room frees and lambda falls to 1e-6
    print("  a price fall after the drop (r=3, W=0, c=1): dropped under lambda=1e-4, then lambda falls to 1e-6 at the given age")
    for a in [1.0, 3.0, 10.0, 30.0, 100.0]:
        lamL = 1e-6 * L
        fv, fs, fth = fetch_value(law, a, 3.0, 0.0, 3.0, 1.0, lamL, lamL * 3.0, hi)
        print(f"     age {a:6.1f} s: fetch value {fv:.3f} s of priced TTFT per session, start {fs - a if fs is not None else float('nan'):.2f} s from now, keep until age {fth if fth else float('nan'):.1f} s")


def part_human():
    print("== human entry (uncapped): the upward kink of the modelled hazard at the median (96 s) ==")
    law = TwoPieceLognormal(**HUMAN)
    hi = 20000.0
    r, W, c, lam = 3.0, 0.0, 1.0, 1e-6
    x, lamL = r - W, lam * L
    Kio = lamL * r
    Knet = c * x + Kio
    print(f"  r={r}, c={c}, lambda={lam:g}: Retain's local rule keeps while hazard >= {lamL / Knet:.5f} /s;"
          f" hazard just below / above the median: {float(law.hazard(95.999)):.5f} / {float(law.hazard(96.001)):.5f}")
    ages = [0.0, 1.0, 5.0, 20.0, 60.0, 90.0, 94.0, 96.5, 120.0, 300.0]
    rv = retention_value(law, ages, x, W, c, lamL, Kio, hi)
    for a, (v, thk) in zip(ages, rv):
        fv, fs, fth = fetch_value(law, a, x, W, r, c, lamL, Kio, hi)
        print(f"     age {a:6.1f}: hazard {float(law.hazard(a)):.5f}  V_ret {v:7.4f} (drop at {thk:8.1f})   fetch value {fv:7.4f}"
              + (f" (start at age {fs:.1f}, drop at {fth:.1f})" if fs is not None else ""))


if __name__ == "__main__":
    part_identity()
    part_tool()
    part_human()
