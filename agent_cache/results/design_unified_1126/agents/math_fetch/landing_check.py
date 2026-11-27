"""Check 1c: after arrival, where should the read land (DESIGN.md section 3.2)?

(i)   deterministic read time r: the best landing is max(now + r, Q_D(q)), q = lamL / (c + lamL).
(ii)  random read time r independent of D: the best RELEASE time is max(now, Q_{D - r}(q)).
(iii) a fixed charge eps for being late at all (the wait for the next opening after the landing):
      first-order condition F_D(l) + (1 - q) eps f_D(l) = q, so the landing is earlier than the q-quantile.
(iv)  which q the engine's fixed 2 s margin stands for, under a Laplace error of the admission
      estimate with median +0.45 s and |error| p90 of 3.5 s (campaign 34, readers; the shape is assumed).
(v)   an estimate of D that keeps moving: releasing at the first time P(D <= t + r | info) >= q is
      never too late, and can be too early.
"""

import math

import numpy as np
from scipy import optimize

rng = np.random.default_rng(5)
N = 200_000


def laplace(n, med, b):
    return med + rng.laplace(0.0, b, n)


def part_i_ii_iii():
    now = 0.0
    D = 10.0 + laplace(N, 0.45, 1.5)                 # admission time if resident
    print("== (i) deterministic r = 3 s ==")
    print("   c    lamL   q       grid-best landing   max(now + r, Q_D(q))")
    r = 3.0
    grid = np.linspace(now + r, 25.0, 1101)
    for (c, lamL) in [(1.0, 0.025), (1.0, 0.5), (1.0, 2.5), (0.3, 0.5), (0.05, 0.5), (1.0, 0.0)]:
        q = lamL / (c + lamL)
        cost = [c * np.maximum(l - D, 0).mean() + lamL * np.maximum(D - l, 0).mean() for l in grid]
        lbest = grid[int(np.argmin(cost))]
        rule = max(now + r, float(np.quantile(D, q)))
        print(f"   {c:4.2f} {lamL:5.3f}  {q:5.3f}   {lbest:7.2f}             {rule:7.2f}")
    print("   (q = 0: any landing up to the earliest possible admission is as good; the rule releases at once)")
    # constraint binding: D soon
    D2 = 2.0 + laplace(N, 0.45, 1.5)
    q = 0.5
    cost = [np.maximum(l - D2, 0).mean() + 1.0 * np.maximum(D2 - l, 0).mean() for l in grid]
    print(f"   when the q-quantile of D ({np.quantile(D2, q):.2f}) is before now + r ({now + r}), the best landing is now + r: grid {grid[int(np.argmin(cost))]:.2f}")
    print("== (ii) random read time (lognormal, mean 3 s, sd 1.5 s), independent of D ==")
    sig = math.sqrt(math.log(1 + 0.25))
    rr = np.exp(rng.normal(math.log(3.0) - 0.5 * sig * sig, sig, N))
    rel = np.linspace(0.0, 15.0, 751)
    print("   c    lamL   q       grid-best release   Q_{D-r}(q)    Q_D(q) - mean r (the draft's 'land at the quantile of D')   its extra cost")
    for (c, lamL) in [(1.0, 0.025), (1.0, 0.5), (1.0, 2.5)]:
        q = lamL / (c + lamL)
        f = lambda s: c * np.maximum(s + rr - D, 0).mean() + lamL * np.maximum(D - s - rr, 0).mean()
        cost = np.array([f(s) for s in rel])
        sbest = rel[int(np.argmin(cost))]
        rule = max(0.0, float(np.quantile(D - rr, q)))
        naive = max(0.0, float(np.quantile(D, q)) - 3.0)
        print(f"   {c:4.2f} {lamL:5.3f}  {q:5.3f}   {sbest:7.2f}             {rule:7.2f}       {naive:7.2f}                                                  {f(naive) - cost.min():+.4f} s")
    print("== (iii) a fixed charge eps = 0.5 s for landing after D (wait for the next opening) ==")
    r = 3.0
    eps = 0.5
    # Laplace(med, b) pdf and cdf for the first-order condition
    med, b = 10.45, 1.5
    Fd = lambda l: 0.5 * math.exp((l - med) / b) if l < med else 1 - 0.5 * math.exp(-(l - med) / b)
    fd = lambda l: 0.5 / b * math.exp(-abs(l - med) / b)
    print("   c    lamL   q       grid-best landing   root of F_D(l) + (1-q) eps f_D(l) = q    q-quantile of D")
    for (c, lamL) in [(1.0, 0.5), (1.0, 2.5), (0.3, 0.5)]:
        q = lamL / (c + lamL)
        cost = [c * (np.maximum(l - D, 0).mean() + eps * np.mean(D < l)) + lamL * np.maximum(D - l, 0).mean() for l in grid]
        lbest = grid[int(np.argmin(cost))]
        g = lambda l: Fd(l) + (1 - q) * eps * fd(l) - q
        ls = np.linspace(0.0, 30.0, 3001)
        gv = np.array([g(l) for l in ls])
        k = int(np.flatnonzero(gv >= 0)[0])               # first crossing from below
        root = optimize.brentq(g, ls[k - 1], ls[k]) if k > 0 else 0.0
        print(f"   {c:4.2f} {lamL:5.3f}  {q:5.3f}   {lbest:7.2f}             {root:7.2f}                                   {float(np.quantile(D, q)):7.2f}")


def part_iv():
    print("== (iv) the fixed 2 s margin as a quantile ==")
    # error = actual admission - estimate ~ Laplace(median m, b); |error| p90 = 3.5 s
    m = 0.45
    b = optimize.brentq(lambda b: 0.5 * math.exp(-(3.5 - m) / b) + 0.5 * math.exp(-(3.5 + m) / b) - 0.1, 0.1, 10.0)
    for margin in [1.0, 2.0]:
        p_late = 0.5 * math.exp(-(margin + m) / b)          # P(actual < estimate - margin)
        ratio = p_late / (1 - p_late)                          # lamL / c
        print(f"   Laplace scale b = {b:.2f} s: landing {margin:.0f} s before the estimate is late with probability {p_late:.3f} = q"
              f" -> right when lambda L / c = {ratio:.3f}, i.e. lambda = {ratio / 25_000:.1e} per token-second at L = 25 K and c = 1")
    for (c, lam) in [(1.0, 1e-6), (1.0, 2e-5), (0.1, 2e-5), (0.02, 2e-5)]:
        lamL = lam * 25_000
        q = lamL / (c + lamL)
        # quantile of the error at level q: landing = estimate + Q_err(q)
        Qe = m + b * math.log(2 * q) if q < 0.5 else m - b * math.log(2 * (1 - q))
        print(f"   c = {c}, lambda = {lam:g}: q = {q:.3f} -> land {(-Qe):+.2f} s before the estimate (negative = after it)")


def part_v():
    print("== (v) an estimate that keeps moving: first-crossing release against the best stopping rule (toy) ==")
    # D = now + position * service; the queue ahead may stall once (probability p_stall) for a known extra delay.
    # At time 0 the estimate is D0; at time t1 < D0 - r it is revealed whether a stall of `extra` seconds happens.
    r, D0, extra, p_stall, t1 = 3.0, 10.0, 6.0, 0.5, 5.0
    c, lamL = 1.0, 0.5
    q = lamL / (c + lamL)

    def cost(land, D):
        return c * max(land - D, 0.0) + lamL * max(D - land, 0.0)

    # rule A (first crossing on the law seen at time 0): P(D <= t + r) >= q first holds when t + r = D0 (mass 1 - p_stall >= q)
    tA = D0 - r
    cA = (1 - p_stall) * cost(tA + r, D0) + p_stall * cost(tA + r, D0 + extra)
    # rule B: wait for the news at t1, then release just in time
    cB = (1 - p_stall) * cost(max(t1, D0 - r) + r, D0) + p_stall * cost(max(t1, D0 + extra - r) + r, D0 + extra)
    print(f"   q = {q:.3f}; the law at time 0 is D = {D0} w.p. {1 - p_stall} or {D0 + extra} w.p. {p_stall}; the news comes at t = {t1} (before the first crossing at t = {tA})")
    print(f"   release at the first crossing of the time-0 law: cost {cA:.3f}; re-evaluate after the news: cost {cB:.3f}")
    # news after the first crossing: the first-crossing rule has already released
    t1 = 8.0
    cB2 = (1 - p_stall) * cost(max(t1, D0 - r) + r, D0) + p_stall * cost(max(t1, D0 + extra - r) + r, D0 + extra)
    cA2 = cA
    # best fixed release time when the news comes too late to use for the early branch
    best = min(((1 - p_stall) * cost(t + r, D0) + p_stall * cost(max(t, min(t1, t)) + r, D0 + extra), t) for t in np.linspace(0, 15, 1501))
    # stopping rule: either release at t <= t1 blind, or wait until t1 and then release at max(t1, D - r)
    wait_cost = (1 - p_stall) * cost(t1 + r, D0) + p_stall * cost(max(t1, D0 + extra - r) + r, D0 + extra)
    print(f"   news at t = {t1} (after the crossing at {tA}): first-crossing release costs {cA2:.3f}; waiting for the news costs {wait_cost:.3f}"
          f" -> the first crossing is {'too early' if wait_cost < cA2 else 'right'} here")


if __name__ == "__main__":
    part_i_ii_iii()
    part_iv()
    part_v()
