#!/usr/bin/env python3
"""Open single-server FCFS queue: the marginal effect of one more second of work (the weight mu).

Claim checked (DESIGN.md 1.5): "In an open queue mu is about 1 + Q (queue length)".

Sample-path fact (Lindley): with fixed arrival times a_j and service times s_j,
    W_j = max over i <= j of [ sum_{l=i}^{j-1} s_l - (a_j - a_i) ]      (W_j >= 0, the i = j term is 0)
so each W_j is convex and piecewise linear in every s_k, with right-derivative 1 iff k and j share a busy
period (k before j) and 0 otherwise.  Hence
    d/d(delta) [ sum_j W_j ]  at delta = 0+  =  B_k := number of jobs served after k in k's busy period,
and mu_k = 1 + B_k when the extra work also delays k's own first token.

M/G/1 expectation, conditional on the state when the work is added (Q jobs waiting behind the job in
service, x seconds of its service left, arrival rate lam, load rho):
    E[B_k | Q, x] = (Q + lam * x) / (1 - rho),
because each waiting job, and each arrival during x, starts a sub-busy period of mean size 1/(1 - rho).
Averaged over jobs of an M/M/1 queue:  E[B_k] = rho / (1 - rho)^2,  against  E[Q at service start] =
rho^2 / (1 - rho).

This script checks all three by simulation (numpy), and the convexity bound for a finite injection.
"""
import os
import sys

import numpy as np


def lindley(inter, serv):
    """Waits W_j for interarrival times inter[j] = a_j - a_{j-1} (inter[0] unused) and services serv."""
    n = len(serv)
    w = np.empty(n)
    w[0] = 0.0
    x = 0.0
    for j in range(1, n):
        x = x + serv[j - 1] - inter[j]
        if x < 0.0:
            x = 0.0
        w[j] = x
    return w


def busy_stats(w):
    """For each job: index of its busy period and the number of later jobs in the same busy period."""
    n = len(w)
    start = (w == 0.0)
    start[0] = True
    bp = np.cumsum(start) - 1
    sizes = np.bincount(bp)
    first = np.flatnonzero(start)
    pos = np.arange(n) - first[bp]
    later = sizes[bp] - pos - 1
    return bp, later


def queue_at_service_start(arr, w):
    """Number of jobs waiting (arrived, not started) at the instant job j starts service."""
    start = arr + w
    # jobs with arrival <= start_j and index > j
    idx = np.searchsorted(arr, start, side="right")  # number of arrivals up to start_j
    return idx - (np.arange(len(arr)) + 1)


def main():
    out = []
    P = out.append
    rng = np.random.default_rng(20261126)
    P("Open FCFS single server: d(sum of waits)/d(work of job k) = B_k = later jobs in k's busy period")
    P("")
    P("1. Sample-path check (M/M/1, rho = 0.8, 4,000 jobs, 200 random k, delta = 1e-7):")
    n = 4000
    lam, mean_s = 0.8, 1.0
    inter = rng.exponential(1.0 / lam, n)
    serv = rng.exponential(mean_s, n)
    w0 = lindley(inter, serv)
    _, later = busy_stats(w0)
    worst = 0.0
    ks = rng.choice(n - 1, 200, replace=False)
    for k in ks:
        s2 = serv.copy()
        s2[k] += 1e-7
        d = (lindley(inter, s2).sum() - w0.sum()) / 1e-7
        worst = max(worst, abs(d - later[k]))
    P("   max |finite-difference slope - B_k| over 200 jobs = %.2e   (B_k ranges %d..%d)" % (
        worst, later[ks].min(), later[ks].max()))
    P("")
    P("2. Finite injection: convexity gives  d(sum W) >= delta * B_k  (extra busy periods merge):")
    for delta in (0.1, 1.0, 5.0):
        ratios = []
        viol = 0
        for k in ks:
            s2 = serv.copy()
            s2[k] += delta
            d = lindley(inter, s2).sum() - w0.sum()
            if d < delta * later[k] - 1e-9:
                viol += 1
            ratios.append(d / delta)
        P("   delta = %4.1f s: mean d(sum W)/delta = %7.2f  against mean B_k = %7.2f ; violations of the bound: %d" % (
            delta, float(np.mean(ratios)), float(np.mean(later[ks])), viol))
    P("")
    P("3. Averages over jobs, M/M/1 (2,000,000 jobs per load): simulated E[B_k], theory rho/(1-rho)^2,")
    P("   and the draft's proxy E[Q at service start] (theory rho^2/(1-rho)):")
    P("   %5s %12s %12s %12s %12s %10s" % ("rho", "E[B] sim", "rho/(1-rho)^2", "E[Q] sim", "rho^2/(1-rho)", "E[B]/E[Q]"))
    n = 2_000_000
    for rho in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
        inter = rng.exponential(1.0 / rho, n)
        serv = rng.exponential(1.0, n)
        w = lindley(inter, serv)
        _, later = busy_stats(w)
        arr = np.cumsum(inter)
        q = queue_at_service_start(arr, w)
        P("   %5.2f %12.2f %12.2f %12.2f %12.2f %10.1f" % (
            rho, later.mean(), rho / (1 - rho) ** 2, q.mean(), rho ** 2 / (1 - rho), later.mean() / max(q.mean(), 1e-9)))
    P("")
    P("4. Conditional law: E[B_k | Q at service start, own service s] = (Q + lam*s)/(1 - rho)")
    P("   (M/G/1 with lognormal service, sigma = 1; rho = 0.8; regression of B on the predictor):")
    rho = 0.8
    sig = 1.0
    n = 2_000_000
    serv = rng.lognormal(-sig * sig / 2.0, sig, n)  # mean 1
    inter = rng.exponential(1.0 / rho, n)
    w = lindley(inter, serv)
    _, later = busy_stats(w)
    arr = np.cumsum(inter)
    q = queue_at_service_start(arr, w)
    pred = (q + rho * serv) / (1 - rho)
    # binned comparison
    order = np.argsort(pred)
    P("   %14s %12s %12s %10s" % ("predictor bin", "mean pred", "mean B", "mean 1+Q"))
    edges = [0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = order[int(lo * n):int(hi * n)]
        P("   %5.0f%%-%3.0f%%    %12.2f %12.2f %10.2f" % (
            100 * lo, 100 * hi, pred[sel].mean(), later[sel].mean(), 1 + q[sel].mean()))
    slope = float(np.dot(pred, later) / np.dot(pred, pred))
    P("   least-squares slope of B on (Q + lam*s)/(1-rho) through the origin: %.3f (theory 1)" % slope)
    P("")
    P("Conclusion: in an open queue mu = 1 + (later jobs in the busy period) = 1 + (Q + lam*x)/(1 - rho) in")
    P("expectation, not 1 + Q.  At rho = 0.9 the queue length understates it about elevenfold (91 against 9).")
    text = "\n".join(out)
    print(text)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "open_queue_mu.out"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    sys.exit(main())
