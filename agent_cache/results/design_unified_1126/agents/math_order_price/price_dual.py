#!/usr/bin/env python3
"""Price (DESIGN.md section 6, results 6, 7 and 10) on the smallest model that has a room constraint.

Open model, Retain only.  N sessions; a session alternates a gap G (whole seconds, law given by its class:
the hint) and a return.  During the gap its prefix of L tokens is either held in host memory or dropped;
a dropped prefix cannot be taken back before the return (no read ahead in this toy).  A return that finds
the prefix dropped costs K (a read).  Hard constraint: tokens held <= C at every second.

Relaxed problem: tokens held <= C on average.  Lagrangian: each session minimises
    J_i(lam) = [ K_i P(G_i > tau) + lam L_i E min(G_i, tau) ] / E[G_i]       over its drop age tau,
and D(lam) = sum_i J_i(lam) - lam C is a lower bound on the miss cost per second of every policy that
meets the hard constraint with the same information (age and class), for every lam >= 0.

What the script checks
  P1  The index nu(a) = (K/L) max_h P(a < G <= a+h) / E[min(G-a, h); G > a] and the rule "keep while
      nu >= lam" attain the single-session optimum for every price (brute force over drop ages).
  P2  D is concave; at its maximiser lam* demand meets capacity (complementary slackness; lam* = 0 when the
      room does not bind); lam* is an index value: the index of the marginal holder.
  P3  Weak duality: the hard-constrained index policy and LRU cost at least max D.
  P4  The gap between the index policy and the bound shrinks like N^(-1/2) (result 10, in this toy).
  P5  Reading lam off the eviction frontier (water-filling) against lam*, with four estimators, and against
      dual ascent (the multiplicative update of the paper draft).
Output: price_dual.out
"""
import math
import os
import sys
import time

import numpy as np
from scipy.stats import norm

OUT = []
T_MAX = 1200


def P(s=""):
    OUT.append(s)
    print(s, flush=True)


def pmf_lognormal(median, sigma):
    n = np.arange(0, T_MAX + 1, dtype=float)
    cdf = np.zeros(T_MAX + 1)
    cdf[1:] = norm.cdf((np.log(n[1:]) - math.log(median)) / sigma)
    cdf[T_MAX] = 1.0
    pmf = np.diff(cdf, prepend=0.0)
    pmf[0] = 0.0
    return pmf / pmf.sum()


def pmf_normal(mean, sd):
    n = np.arange(0, T_MAX + 1, dtype=float)
    cdf = norm.cdf((n - mean) / sd)
    cdf[T_MAX] = 1.0
    pmf = np.diff(cdf, prepend=0.0)
    pmf[0] = 0.0
    return pmf / pmf.sum()


class Cls:
    """One class of sessions: gap law, size, restore cost; survival, index table and cost lines."""

    def __init__(self, name, share, pmf, tokens, cost):
        self.name, self.share, self.pmf, self.L, self.K = name, share, pmf, float(tokens), float(cost)
        cdf = np.cumsum(pmf)
        self.S = 1.0 - cdf                     # S[n] = P(G > n), n = 0..T_MAX
        self.S[self.S < 1e-15] = 0.0
        self.cumS = np.concatenate([[0.0], np.cumsum(self.S)])   # cumS[t] = sum_{j<t} S[j] = E min(G, t)
        self.EG = self.cumS[T_MAX]
        self.cdf = cdf
        self.nu = self._index()
        # lines for J(lam) = min_tau (A[tau] + lam * Bt[tau])
        self.A = self.K * self.S / self.EG                         # miss cost per second with drop age tau
        self.Bt = self.L * self.cumS[:T_MAX + 1] / self.EG         # tokens held on average with drop age tau

    def _index(self):
        nu = np.zeros(T_MAX + 1)
        S, cumS = self.S, self.cumS
        for a in range(T_MAX):
            if S[a] <= 0.0:
                break
            h = np.arange(1, T_MAX - a + 1)
            num = S[a] - S[a + h]
            den = cumS[a + h] - cumS[a]
            nu[a] = (self.K / self.L) * np.max(num / den)
        return nu

    def best(self, lam):
        v = self.A + lam * self.Bt
        tau = int(np.argmin(v))
        return v[tau], tau

    def tau_index_rule(self, lam):
        below = np.flatnonzero(self.nu[:T_MAX] < lam)
        return int(below[0]) if len(below) else T_MAX


def make_classes():
    return [
        Cls("tool (heavy tail)", 0.60, pmf_lognormal(1.7, 1.8), 20000, 0.9 + 20000 / 21400.0),
        Cls("person (heavy tail)", 0.10, pmf_lognormal(96.0, 1.9), 30000, 0.9 + 30000 / 21400.0),
        Cls("test run 20 +- 2 s", 0.30, pmf_normal(20.0, 2.0), 25000, 0.9 + 25000 / 21400.0),
    ]


def dual(classes, counts, cap, lam):
    val = -lam * cap
    dem = 0.0
    for c, n in zip(classes, counts):
        v, tau = c.best(lam)
        val += n * v
        dem += n * c.Bt[tau]
    return val, dem


def maximise_dual(classes, counts, cap):
    grid = np.concatenate([[0.0], np.logspace(-10, -1, 3000)])
    vals = [dual(classes, counts, cap, lam)[0] for lam in grid]
    i = int(np.argmax(vals))
    lo = grid[max(i - 1, 0)]
    hi = grid[min(i + 1, len(grid) - 1)]
    for _ in range(200):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if dual(classes, counts, cap, m1)[0] < dual(classes, counts, cap, m2)[0]:
            lo = m1
        else:
            hi = m2
    lam = 0.5 * (lo + hi)
    if vals[0] >= dual(classes, counts, cap, lam)[0]:
        lam = 0.0
    return lam, dual(classes, counts, cap, lam)[0]


def simulate(classes, counts, cap, steps, seed, policy="index", lam_fixed=None, ascent=None, warm=2400):
    """Hard-constrained simulation.  Returns cost per second, mean tokens held, frontier samples."""
    rng = np.random.default_rng(seed)
    cls = np.concatenate([np.full(n, k) for k, n in enumerate(counts)])
    n_tot = len(cls)
    L = np.array([classes[k].L for k in cls])
    K = np.array([classes[k].K for k in cls])
    nu_tab = np.stack([c.nu for c in classes])            # [class, age]
    cdfs = [c.cdf for c in classes]

    def draw(idx):
        out = np.empty(len(idx), dtype=np.int64)
        u = rng.random(len(idx))
        for k in range(len(classes)):
            sel = cls[idx] == k
            if sel.any():
                out[sel] = np.searchsorted(cdfs[k], u[sel], side="left")
        return np.maximum(out, 1)

    age = np.zeros(n_tot, dtype=np.int64)
    held = np.zeros(n_tot, dtype=bool)
    next_ret = draw(np.arange(n_tot))
    cost = 0.0
    held_sum = 0.0
    ev_idx, ev_tok = [], []
    f_ev_max = np.full(steps + 1, np.nan)
    f_held_min = np.full(steps + 1, np.nan)
    lam = ascent["lam0"] if ascent else lam_fixed
    lam_trace = []
    acc_held = acc_forced = 0.0
    for t in range(1, steps + 1):
        age += 1
        ret = np.flatnonzero(next_ret == t)
        if len(ret):
            if t > warm:
                cost += K[ret][~held[ret]].sum()
            age[ret] = 0
            held[ret] = True
            next_ret[ret] = t + draw(ret)
        if policy in ("threshold", "ascent"):
            h = np.flatnonzero(held)
            drop = h[nu_tab[cls[h], np.minimum(age[h], T_MAX)] < lam]
            held[drop] = False
        tok = L[held].sum()
        forced = 0.0
        if tok > cap:
            h = np.flatnonzero(held)
            if policy == "lru":
                key = -age[h].astype(float)
            elif policy == "exact":
                key = K[h] / (L[h] * np.maximum(next_ret[h] - t, 1))
            else:
                key = nu_tab[cls[h], np.minimum(age[h], T_MAX)]
            order = np.argsort(key, kind="stable")
            freed = np.cumsum(L[h][order])
            k_cut = int(np.searchsorted(freed, tok - cap, side="left")) + 1
            victims = h[order[:k_cut]]
            held[victims] = False
            vi = nu_tab[cls[victims], np.minimum(age[victims], T_MAX)]
            forced = L[victims].sum()
            if t > warm:
                ev_idx.append(vi)
                ev_tok.append(L[victims])
                f_ev_max[t] = vi.max()
            tok -= forced
        if t > warm:
            held_sum += tok
            h = np.flatnonzero(held)
            if len(h):
                f_held_min[t] = nu_tab[cls[h], np.minimum(age[h], T_MAX)].min()
        if ascent:
            acc_held += tok
            acc_forced += forced
            if t % ascent["every"] == 0:
                d = acc_held / ascent["every"] + acc_forced
                g = (d - cap) / cap
                lam = lam * math.exp(ascent["eta"] * g)
                lam = min(max(lam, 1e-12), 1.0)
                acc_held = acc_forced = 0.0
            lam_trace.append(lam)
    span = steps - warm
    res = dict(cost=cost / span, held=held_sum / span,
               ev_idx=np.concatenate(ev_idx) if ev_idx else np.array([]),
               ev_tok=np.concatenate(ev_tok) if ev_tok else np.array([]),
               f_ev_max=f_ev_max[warm + 1:], f_held_min=f_held_min[warm + 1:], lam_trace=np.array(lam_trace))
    return res


def rolling_max_mean(x, w):
    """Mean over time of the maximum of x over the last w seconds (NaN = no eviction that second)."""
    y = np.where(np.isnan(x), -np.inf, x)
    n = len(y) // w * w
    blocks = y[:n].reshape(-1, w).max(axis=1)
    blocks = blocks[np.isfinite(blocks)]
    return float(blocks.mean()) if len(blocks) else float("nan")


def main():
    t_start = time.time()
    classes = make_classes()
    P("Price of host room on the smallest model (open, Retain only)")
    P("")
    P("Classes (the hint is the class's gap law; whole seconds, capped at 1,200 s):")
    for c in classes:
        q = [int(np.searchsorted(c.cdf, x)) for x in (0.1, 0.5, 0.9)]
        P("  %-20s share %.2f, L = %5.0f tokens, K = %.2f s, gap p10/p50/p90 = %d/%d/%d s, mean %.1f s; "
          "index at age 0 / 5 / 30 / 300 s: %.2e %.2e %.2e %.2e" % (
              c.name, c.share, c.L, c.K, q[0], q[1], q[2], c.EG, c.nu[0], c.nu[5], c.nu[30], c.nu[300]))
    P("")
    P("P1. Single session: 'keep while nu(a) >= lam' against the best drop age (brute force), 60 prices:")
    worst = 0.0
    for c in classes:
        for lam in np.logspace(-8, -2.5, 60):
            v_best, _ = c.best(lam)
            tau = c.tau_index_rule(lam)
            v_rule = c.A[tau] + lam * c.Bt[tau]
            worst = max(worst, (v_rule - v_best) / max(v_best, 1e-300))
    P("    largest relative excess of the index rule over the optimum: %.2e" % worst)
    mono = []
    for c in classes:
        taus = [c.tau_index_rule(lam) for lam in np.logspace(-8, -2.5, 200)]
        mono.append(all(a >= b for a, b in zip(taus[:-1], taus[1:])))
    P("    the drop age of the rule never rises with the price (indexability of the stopping problem): %s" % mono)
    P("")
    n_ref = 1000
    counts = [int(round(c.share * n_ref)) for c in classes]
    tot_tokens = sum(n * c.L for n, c in zip(counts, classes))
    cap = 0.30 * tot_tokens
    lam_star, d_star = maximise_dual(classes, counts, cap)
    P("P2. N = %d sessions, all prefixes together %.1f M tokens, room C = 30 %% of that = %.2f M tokens." % (
        n_ref, tot_tokens / 1e6, cap / 1e6))
    P("    %10s %14s %16s %12s" % ("price", "D(price)", "demand (tokens)", "demand - C"))
    for f in (0.0, 0.25, 0.5, 0.9, 0.999, 1.0, 1.001, 1.1, 2.0, 4.0):
        lam = f * lam_star
        v, dem = dual(classes, counts, cap, lam)
        P("    %10.3e %14.5f %16.0f %+12.0f%s" % (lam, v, dem, dem - cap, "   <- lam*" if f == 1.0 else ""))
    _, dem_lo = dual(classes, counts, cap, lam_star * (1 - 1e-6))
    _, dem_hi = dual(classes, counts, cap, lam_star * (1 + 1e-6))
    P("    lam* = %.4e s per token-second; demand just below lam*: %.0f, just above: %.0f, room: %.0f" % (
        lam_star, dem_lo, dem_hi, cap))
    P("    (the room is met by holding the marginal class in part: complementary slackness with randomisation)")
    best = None
    for c in classes:
        for a in range(T_MAX):
            if c.nu[a] > 0:
                e = abs(c.nu[a] - lam_star) / lam_star
                if best is None or e < best[0]:
                    best = (e, c.name, a, c.nu[a])
    P("    nearest index value: class '%s' at idle age %d s, nu = %.4e (relative distance %.1e)  -> lam* is the"
      % (best[1], best[2], best[3], best[0]))
    P("    index of the marginal holder.")
    big = 1.01 * tot_tokens
    lam0, d0 = maximise_dual(classes, counts, big)
    P("    With room for everything (C = 1.01 x all prefixes): lam* = %.1e, bound = %.2e (room free -> price 0)." % (lam0, d0))
    P("")
    P("P3/P4. Hard-constrained policies against the bound max D (cost in seconds of read per session-second, x 1e3;")
    P("    three seeds a row, gap = (cost - bound) / bound with its standard error over seeds):")
    P("    %6s %10s %12s %16s %10s %10s %12s %22s" % (
        "N", "bound", "index policy", "gap % (se)", "LRU", "LRU gap %", "room used %", "exact gaps known, gap %"))
    rows = []
    for n_tot, steps in ((30, 400000), (100, 200000), (300, 100000), (1000, 60000), (3000, 30000)):
        cnt = [int(round(c.share * n_tot)) for c in classes]
        capn = 0.30 * sum(n * c.L for n, c in zip(cnt, classes))
        lam_n, d_n = maximise_dual(classes, cnt, capn)
        gaps, costs, used = [], [], []
        for seed in (7, 8, 9):
            r = simulate(classes, cnt, capn, steps, seed=seed, policy="index")
            gaps.append((r["cost"] - d_n) / d_n)
            costs.append(r["cost"])
            used.append(r["held"] / capn)
        r2 = simulate(classes, cnt, capn, steps, seed=7, policy="lru")
        r3 = simulate(classes, cnt, capn, min(steps, 60000), seed=7, policy="exact")
        nn = sum(cnt)
        gm = float(np.mean(gaps))
        gse = float(np.std(gaps, ddof=1) / math.sqrt(len(gaps)))
        rows.append((nn, gm, gse))
        P("    %6d %10.4f %12.4f %9.2f (%4.2f) %10.4f %10.2f %12.1f %22.1f" % (
            nn, 1e3 * d_n / nn, 1e3 * float(np.mean(costs)) / nn, 100 * gm, 100 * gse, 1e3 * r2["cost"] / nn,
            100 * (r2["cost"] - d_n) / d_n, 100 * float(np.mean(used)), 100 * (r3["cost"] - d_n) / d_n))
    xs = np.log([r[0] for r in rows[:3]])
    ys = np.log([max(r[1], 1e-12) for r in rows[:3]])
    slope = float(np.polyfit(xs, ys, 1)[0])
    P("    slope of log(gap) on log(N) over N = 30..300: %.2f (the general guarantee for index policies is -0.5;" % slope)
    P("    faster decay is known when the fluid fixed point is non-degenerate)")
    P("    A policy that knows each gap exactly beats the bound (negative gap): the bound is for policies with the")
    P("    same information, here age and class only.")
    P("")
    P("P5. Reading the price off the eviction frontier (N = 1,000; lam* = %.3e):" % lam_star)
    r = simulate(classes, counts, cap, 60000, seed=11, policy="index")
    ev, tk = r["ev_idx"], r["ev_tok"]
    P("    token-weighted mean index of evicted prefixes        %.3e  (%.2f lam*)" % (
        float(np.average(ev, weights=tk)), float(np.average(ev, weights=tk)) / lam_star))
    P("    median index of evicted prefixes                     %.3e  (%.2f lam*)" % (
        float(np.median(ev)), float(np.median(ev)) / lam_star))
    fm = r["f_ev_max"]
    P("    mean of the highest index evicted in a second        %.3e  (%.2f lam*)" % (
        float(np.nanmean(fm)), float(np.nanmean(fm)) / lam_star))
    for w in (5, 30, 120, 600):
        v = rolling_max_mean(fm, w)
        P("    'highest index given up lately', window %4d s      %.3e  (%.2f lam*)" % (w, v, v / lam_star))
    fh = r["f_held_min"]
    P("    time-average of the lowest index still held          %.3e  (%.2f lam*)" % (
        float(np.nanmean(fh)), float(np.nanmean(fh)) / lam_star))
    P("    median over time of the lowest index still held      %.3e  (%.2f lam*)" % (
        float(np.nanmedian(fh)), float(np.nanmedian(fh)) / lam_star))
    P("    seconds with at least one eviction: %.1f %%" % (100 * float(np.mean(~np.isnan(fm)))))
    P("    Cost of acting on a wrong price (threshold policy 'drop when nu < price', room still enforced):")
    for f in (0.1, 0.5, 1.0, 2.0, 10.0):
        rr = simulate(classes, counts, cap, 30000, seed=11, policy="threshold", lam_fixed=f * lam_star)
        P("      price = %5.1f lam*: cost %.4f (x 1e-3 per session), room used %.1f %%" % (
            f, 1e3 * rr["cost"] / n_ref, 100 * rr["held"] / cap))
    P("    Dual ascent, the draft paper's update lam <- lam exp(eta (d - C)/C) every 5 s, d = tokens held +")
    P("    tokens evicted by the room while their index was above the price:")
    for eta in (0.05, 0.3, 1.0):
        for lam0 in (0.01 * lam_star, 100 * lam_star):
            rr = simulate(classes, counts, cap, 30000, seed=11, policy="ascent",
                          ascent=dict(lam0=lam0, eta=eta, every=5), warm=0)
            tr = rr["lam_trace"] / lam_star
            inside = np.flatnonzero((tr > 1 / 1.5) & (tr < 1.5))
            t_in = int(inside[0]) if len(inside) else -1
            tail = tr[len(tr) // 2:]
            P("      eta = %.2f, start %6.2f lam*: first within a factor 1.5 of lam* after %6d s; second half: "
              "median %.2f lam*, p5-p95 %.2f-%.2f lam*" % (
                  eta, lam0 / lam_star, t_in, float(np.median(tail)), float(np.quantile(tail, 0.05)),
                  float(np.quantile(tail, 0.95))))
    P("")
    P("(%.0f s of CPU)" % (time.time() - t_start))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "price_dual.out"), "w") as f:
        f.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    sys.exit(main())
