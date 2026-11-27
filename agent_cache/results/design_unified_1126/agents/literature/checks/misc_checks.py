#!/usr/bin/env python3
"""Numeric checks for three classical results as the draft uses them.

CPU only.  Run:
  CUDA_VISIBLE_DEVICES= nice -n 10 /home/wanhr/sglang/agent_cache/baselines/continuum/.venv/bin/python -I -B misc_checks.py > misc_checks.out

1. Newsvendor critical fractile (draft 3.2): the landing time l* that minimizes c E[(l-D)^+] + lamL E[(D-l)^+]
   satisfies P(D <= l*) = lamL / (c + lamL).            D = admission time, c = cost per second late (stall),
   lamL = lambda * L = cost per second early (host memory held).
2. Derivative of the start-time cost (draft 3.3) and the read-now-or-wait comparison.
3. Greedy by cost per token for freeing m tokens (draft 5.1): optimal for the divisible problem, within one holder
   (additively) otherwise; how far in ratio.
4. Open queue: one more second of work delays (1 + Q)/(1 - rho) turns on average, not 1 + Q (draft 1.5).
"""
import warnings
warnings.simplefilter("ignore")
import itertools
import numpy as np
from scipy import stats, optimize

rng = np.random.default_rng(20261126)

def newsvendor():
    print("== 1. Newsvendor quantile for the landing time (draft 3.2)")
    print("   D = 10 s + lognormal error (median 0.45 s, sigma 0.8); quadrature on a grid")
    t = np.linspace(0, 40, 800001); dt = t[1] - t[0]
    D = stats.lognorm(0.8, loc=10.0, scale=0.45)
    f = D.pdf(t)
    print("       c   lamL |  q = lamL/(c+lamL) |  argmin l*   P(D <= l*)  | quantile formula l_q")
    for c, lamL in ((1.0, 0.05), (1.0, 1.0), (0.2, 1.0), (0.05, 1.0), (1.0, 0.0025)):
        cost = lambda l: float(np.sum((c * np.maximum(l - t, 0) + lamL * np.maximum(t - l, 0)) * f) * dt)
        res = optimize.minimize_scalar(cost, bounds=(10.0, 30.0), method="bounded", options={"xatol": 1e-6})
        q = lamL / (c + lamL)
        print(f"   {c:5.2f} {lamL:6.4f} |      {q:6.4f}        |  {res.x:8.4f}    {D.cdf(res.x):6.4f}     |  {D.ppf(q):8.4f}")

def derivative():
    print("== 2. Start time before arrival (draft 3.3)")
    print("   J_F(s) = c E[(l-T)^+] + lamL E[(T-l)^+],  l = min(s, A) + r,  T = A + W,  x = r - W > 0")
    print("   claim: dJ_F/ds = c P(s < A <= s+x) - lamL P(A > s+x);  falling hazard -> optimum at an end; rising -> threshold")
    x, c, lamL = 2.0, 1.0, 0.05
    q = lamL / (c + lamL)
    for name, D, smax in (("Weibull k=0.5 scale 3 s (falling hazard)", stats.weibull_min(0.5, scale=3.0), 60.0),
                          ("Weibull k=3 scale 20 s (rising hazard)", stats.weibull_min(3.0, scale=20.0), 40.0),
                          ("lognormal median 1.69 s sigma 1.79 (rises then falls)", stats.lognorm(1.79, scale=1.69), 60.0)):
        t = np.linspace(0, 4000, 4000001); dt = t[1] - t[0]
        f = D.pdf(t); f[0] = 0.0 if not np.isfinite(f[0]) else f[0]
        def J(s):
            late = np.where(t <= s, x, np.maximum(0.0, x - (t - s)))
            early = np.where(t <= s, 0.0, np.maximum(0.0, t - s - x))
            return float(np.sum((c * late + lamL * early) * f) * dt)
        ss = np.array([0.05, 0.5, 2.0, 5.0, 10.0, 20.0, 30.0])
        ss = ss[ss < smax]
        worst = 0.0
        signs = []
        for s in ss:
            h = 1e-2
            fd = (J(s + h) - J(s - h)) / (2 * h)
            an = c * (D.cdf(s + x) - D.cdf(s)) - lamL * D.sf(s + x)
            worst = max(worst, abs(fd - an))
            Fx = (D.cdf(s + x) - D.cdf(s)) / D.sf(s)
            signs.append(f"s={s:g}: F(x|s)={Fx:.3f} dJ/ds={an:+.4f}")
        grid = np.linspace(0.0, smax, 241)
        vals = np.array([J(s) for s in grid])
        k = int(np.argmin(vals))
        J_inf = c * x                      # never read before arrival: stall x at arrival (session always returns here)
        print(f"   {name}: max |finite difference - formula| = {worst:.2e};  q = {q:.4f}")
        print("      " + ";  ".join(signs))
        print(f"      J_F(0) = {vals[0]:.4f}, min over s in [0,{smax:g}] = {vals[k]:.4f} at s = {grid[k]:.2f}, J_F(wait for arrival) = {J_inf:.4f}")
    print("   note: when the gap has mass at infinity (session ended), the left side of the draft's test must be")
    print("   c E[min(R, x); R < infinity] (min(R, x) is x, not 0, on R = infinity) and E[(R - x)^+] is infinite unless the hold is capped.")

def knapsack():
    print("== 3. Freeing m tokens at least cost: greedy by cost per token (draft 5.1)")
    n, trials = 12, 3000
    ratios, add_excess, viol = [], [], 0
    worst = (0, None)
    for _ in range(trials):
        L = rng.integers(2, 40, size=n).astype(float)            # tokens (thousands)
        C = L * rng.uniform(0.2, 2.0, size=n)                    # cost of displacing each holder
        m = float(rng.uniform(0.2, 0.7) * L.sum())
        order = np.argsort(C / L)
        tot_L = tot_C = 0.0; lp = 0.0; last = 0.0
        for j in order:
            if tot_L >= m: break
            need = m - tot_L
            lp += C[j] * min(1.0, need / L[j])                   # divisible optimum (fractional knapsack)
            tot_L += L[j]; tot_C += C[j]; last = C[j]
        best = np.inf
        for mask in range(1, 1 << n):
            idx = [j for j in range(n) if mask >> j & 1]
            if L[idx].sum() >= m:
                best = min(best, C[idx].sum())
        ratios.append(tot_C / best); add_excess.append((tot_C - lp) / last)
        if tot_C > lp + last + 1e-9: viol += 1
        if tot_C / best > worst[0]: worst = (tot_C / best, (L.copy(), C.copy(), m, tot_C, best))
    r = np.array(ratios)
    print(f"   {trials} random instances, {n} holders: greedy cost / integer optimum: mean {r.mean():.3f}, p90 {np.quantile(r, 0.9):.3f}, max {r.max():.3f};")
    print(f"   greedy <= divisible optimum + cost of the last holder taken in every instance: violations {viol};"
          f" (greedy - divisible optimum)/last holder: max {max(add_excess):.3f}")
    print("   two-holder example of the additive gap: need 10 K tokens; B holds 9 K at cost 9.0 (1.00 per K), A holds 10 K at cost 10.1")
    print("   (1.01 per K).  Greedy takes B then A: cost 19.1; the optimum takes A alone: 10.1.  'Within one holder' is 1.9 times here.")

def open_queue():
    print("== 4. Open single-server FIFO queue: turns delayed by one more unit of work (draft 1.5, 'mu about 1 + Q')")
    print("   M/M/1, Lindley recursion; an extra epsilon of work in turn k delays every later turn of the same busy period.")
    print("   Q_k = turns waiting when turn k starts service.  Prediction: E[delayed | Q] = (Q + rho)/(1 - rho), so mu = (1 + Q)/(1 - rho).")
    for rho in (0.5, 0.8, 0.95):
        n = 2_000_000
        A = rng.exponential(1.0, size=n)            # interarrival, rate 1
        S = rng.exponential(rho, size=n)            # service, mean rho
        W = np.empty(n); W[0] = 0.0
        # Lindley
        x = 0.0
        for i in range(1, n):
            x = max(0.0, x + S[i - 1] - A[i])
            W[i] = x
        arr = np.cumsum(A)
        start = arr + W
        # turns after k in k's busy period: next index j > k with W[j] == 0
        zero = np.flatnonzero(W == 0.0)
        nxt = zero[np.searchsorted(zero, np.arange(n), side="right").clip(max=len(zero) - 1)]
        after = np.maximum(nxt - np.arange(n) - 1, 0)
        # Q_k: arrivals before start_k among later turns
        Q = np.searchsorted(arr, start, side="right") - np.arange(n) - 1
        sel = slice(1000, n - 200000)
        mu_meas = 1 + after[sel].mean()
        print(f"   rho={rho:4.2f}: mean Q at service start {Q[sel].mean():7.3f}; mean turns delayed {after[sel].mean():8.3f};"
              f" measured mu = {mu_meas:8.3f};  1 + E[Q] = {1 + Q[sel].mean():7.3f};  (1 + E[Q])/(1 - rho) = {(1 + Q[sel].mean()) / (1 - rho):8.3f}")
        for qv in (0, 2, 5, 10):
            mk = (Q[sel] == qv)
            if mk.sum() > 2000:
                print(f"        given Q = {qv:2d}: turns delayed {after[sel][mk].mean():8.3f}  (prediction {(qv + rho) / (1 - rho):8.3f};  Q itself {qv})")

if __name__ == "__main__":
    newsvendor()
    derivative()
    knapsack()
    open_queue()
