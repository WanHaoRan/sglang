#!/usr/bin/env python3
"""Stability (DESIGN.md result 9): the smallest dynamical model of campaign 32's loop.

State: d = dirty tokens in host memory (pages whose SSD write is pending; they cannot be evicted).

    refusal share        p(d) = p_max * clip((d - d0) / (d1 - d0), 0, 1)     reads refused for lack of clean room
    GPU time a request   g(p) = g0 + gamma * L * p                           a refused read is recomputed
    request rate         x(p) = 1 / g(p)                                     saturated closed loop
    write inflow         x(p) * (Dw + (1 - b) * L * p)                       new tokens, plus rewrites unless born clean
    write outflow        w   (tokens per second while d > 0)

    d' = x(p(d)) * (Dw + (1 - b) * L * p(d)) - w,      0 <= d <= d_cap

Constants from the campaign records: g0 = 0.46 s (campaign 34, minutes 90-109), gamma = 0.043 ms/token,
L = 25 K tokens of history, Dw = 1.9 K tokens written per request.  The refusal ramp is an assumption fitted
to two records of campaign 32: first refusals with about 100-160 K tokens pending, about half of the
history recomputed with 500-650 K pending (d0 = 100 K, d1 = 600 K, p_max = 0.55).

Variants: b = 1 "born clean" (a recomputed page the SSD already holds is not written again);
"delay" (a read that finds no room waits; p = 0).

Parts
  S1  Fixed points in closed form and their stability, as a function of the write rate w.
  S2  A deterministic episode: the write rate dips below the new-token rate for six minutes (reads take the
      device), then recovers.  Without the two rules the loop stays; with either it drains.
  S3  Stochastic runs (Poisson requests, heavy-tailed writes, binomial refusals): chance of the loop in two
      hours against the write rate, per variant; and with room held by early pins (d0 lowered).
Output: stability.out
"""
import math
import os
import sys

import numpy as np

OUT = []
G0 = 0.46
GAMMA = 0.043e-3
L = 25000.0
DW = 1900.0
D0, D1, PMAX = 100e3, 600e3, 0.55
DCAP = 700e3


def P(s=""):
    OUT.append(s)
    print(s, flush=True)


def p_of(d, d0=D0, delay=False):
    if delay:
        return np.zeros_like(np.asarray(d, dtype=float))
    return PMAX * np.clip((np.asarray(d, dtype=float) - d0) / (D1 - d0), 0.0, 1.0)


def inflow(p, born_clean):
    return (DW + (0.0 if born_clean else L * p)) / (G0 + GAMMA * L * p)


def fixed_points(w, d0=D0):
    """Closed form for the plain system (no rule)."""
    a_max = DW / G0                                     # new tokens per second at full speed
    a_loop = (DW + L * PMAX) / (G0 + GAMMA * L * PMAX)  # write inflow with every possible read refused
    res = dict(a_max=a_max, a_loop=a_loop)
    if w <= a_max:
        res["kind"] = "writes below the new-token rate: d grows from 0, only the loop state exists"
        return res
    p_star = (w * G0 - DW) / (L * (1.0 - w * GAMMA)) if w * GAMMA < 1 else float("inf")
    res["p_star"] = p_star
    if p_star >= PMAX:
        res["kind"] = "only the clean state (d = 0) exists"
        return res
    d_u = d0 + (D1 - d0) * p_star / PMAX
    x = 1.0 / (G0 + GAMMA * L * p_star)
    rate = (PMAX / (D1 - d0)) * L * (G0 - GAMMA * DW) * x * x
    res.update(kind="bistable", d_u=d_u, efold=1.0 / rate)
    return res


def integrate(w_of_t, t_end, born_clean, delay, d_init=0.0, dt=0.05, d0=D0):
    n = int(t_end / dt)
    d = d_init
    trace = np.empty(n)
    for i in range(n):
        t = i * dt
        p = float(p_of(d, d0, delay))
        dd = inflow(p, born_clean) - (w_of_t(t) if d > 0 or inflow(p, born_clean) > w_of_t(t) else inflow(p, born_clean))
        d = min(max(d + dd * dt, 0.0), DCAP)
        trace[i] = d
    return trace


def stochastic(w, born_clean, delay, runs, t_end, seed, d0=D0, dt=0.5):
    """Time-stepped stochastic version: Poisson completions at rate x(p), lognormal tokens written per request
    (mean Dw; 2 % of requests write 18 K tokens: context rewrites), binomial refusals at p(d)."""
    rng = np.random.default_rng(seed)
    n = int(t_end / dt)
    d = np.zeros(runs)
    tipped_at = np.full(runs, np.nan)
    d_sum = np.zeros(runs)
    d_max = np.zeros(runs)
    served = np.zeros(runs)
    recomputed = np.zeros(runs)
    sig = 1.0
    mu_small = math.log((DW - 0.02 * 18000.0) / 0.98) - sig * sig / 2
    for i in range(n):
        p = p_of(d, d0, delay)
        x = 1.0 / (G0 + GAMMA * L * p)
        k = rng.poisson(x * dt)
        kmax = int(k.max())
        add = np.zeros(runs)
        if kmax > 0:
            sizes = rng.lognormal(mu_small, sig, size=(runs, kmax))
            big = rng.random((runs, kmax)) < 0.02
            sizes = np.where(big, 18000.0, sizes)
            mask = np.arange(kmax)[None, :] < k[:, None]
            add = (sizes * mask).sum(axis=1)
        refused = rng.binomial(k, p)
        if not born_clean:
            add = add + refused * L
        served += k
        recomputed += refused * L
        d = np.clip(d + add - w * dt, 0.0, DCAP)
        d_sum += d
        d_max = np.maximum(d_max, d)
        newly = np.isnan(tipped_at) & (d > 400e3)
        tipped_at[newly] = i * dt
    return dict(tip_share=float(np.mean(~np.isnan(tipped_at))),
                tip_median=float(np.nanmedian(tipped_at)) if np.any(~np.isnan(tipped_at)) else float("nan"),
                d_mean=float(np.mean(d_sum / n)), d_p95max=float(np.quantile(d_max, 0.95)),
                rate=float(np.mean(served / t_end)), recomputed=float(np.mean(recomputed / np.maximum(served, 1)) / L))


def main():
    P("Stability: the smallest model of campaign 32's loop")
    P("")
    a_max = DW / G0
    a_loop = (DW + L * PMAX) / (G0 + GAMMA * L * PMAX)
    P("Constants: g0 = %.2f s, recompute %.3f ms/token x %d K = %.3f s a refused read, %d tokens written a request."
      % (G0, GAMMA * 1e3, L / 1e3, GAMMA * L, DW))
    P("  new-token rate at full speed  a_max  = Dw / g0 = %.2f K tokens/s   (campaign 34 measured 4.0-4.8 K)" % (a_max / 1e3))
    P("  write inflow inside the loop  a_loop = (Dw + L p_max) / (g0 + gamma L p_max) = %.2f K tokens/s" % (a_loop / 1e3))
    P("  in the loop: request rate %.2f /s (campaign 32: 0.98), rewrites %.0f %% of writes (campaign 32: 74-93 %%)"
      % (1.0 / (G0 + GAMMA * L * PMAX), 100 * L * PMAX / (DW + L * PMAX)))
    P("  ceiling of the inflow as every request recomputes: 1/gamma = %.1f K tokens/s (above the device's 21.4 K)" % (1 / GAMMA / 1e3))
    P("")
    P("S1. Fixed points of the plain system against the write rate w (K tokens/s):")
    P("    %6s %10s %12s %14s  %s" % ("w", "p*", "tipping d_u", "e-folding s", "regime"))
    for w in (3.8e3, 4.2e3, 4.6e3, 5.5e3, 6.5e3, 8.0e3, 10.2e3, 13.0e3, 16.0e3):
        f = fixed_points(w)
        if f["kind"] == "bistable":
            P("    %6.1f %10.3f %10.0f K %14.0f  clean state d = 0 stable; above %.0f K dirty tokens the loop runs away"
              % (w / 1e3, f["p_star"], f["d_u"] / 1e3, f["efold"], f["d_u"] / 1e3))
        else:
            P("    %6.1f %10s %12s %14s  %s" % (w / 1e3, "-", "-", "-", f["kind"]))
    P("    With born-clean pages or with delay: d' = Dw x - w <= a_max - w, so for every w > a_max = %.2f K the" % (a_max / 1e3))
    P("    only fixed point is d = 0 and it attracts every start.")
    P("")
    P("S2. One episode: w = 8 K, then 3.4 K for six minutes (reads take the device), then 10.2 K (the loop's own")
    P("    read/write split).  Dirty tokens (K) at the end of each phase, and 20 minutes after recovery:")

    def w_of_t(t):
        if t < 600:
            return 8.0e3
        if t < 960:
            return 3.4e3
        return 10.2e3

    P("    %-34s %10s %10s %12s %12s" % ("variant", "t = 600 s", "t = 960 s", "t = 1,260 s", "t = 2,160 s"))
    for label, bc, dl in (("plain (no rule)", False, False), ("born clean", True, False),
                          ("delay instead of refusal", False, True), ("both", True, True)):
        tr = integrate(w_of_t, 2160.0, bc, dl)
        at = lambda t: tr[int(t / 0.05) - 1] / 1e3
        P("    %-34s %10.0f %10.0f %12.0f %12.0f" % (label, at(600), at(960), at(1260), at(2160)))
    P("    The dip puts (a_max - 3.4 K) x 360 s = %.0f K tokens behind; the plain system is then past its tipping" % (
        (a_max - 3.4e3) * 360 / 1e3))
    P("    point and stays in the loop although the write rate is back above a_max (hysteresis).")
    P("")
    P("S3. Stochastic runs, two hours each, 400 runs a row: share of runs that reach 400 K dirty tokens.")
    P("    %-26s %7s %8s %10s %10s %12s %9s %11s" % (
        "variant", "w (K)", "w/a_max", "loop share", "median s", "mean dirty K", "req / s", "recomputed"))
    for label, bc, dl in (("plain (no rule)", False, False), ("born clean", True, False),
                          ("delay instead of refusal", False, True)):
        for w in (4.3e3, 4.6e3, 5.0e3, 5.5e3, 6.5e3):
            r = stochastic(w, bc, dl, 400, 7200.0, 3)
            P("    %-26s %7.1f %8.2f %9.0f%% %10.0f %12.1f %9.2f %10.1f%%" % (
                label, w / 1e3, w / a_max, 100 * r["tip_share"], r["tip_median"], r["d_mean"] / 1e3, r["rate"],
                100 * r["recomputed"]))
    P("    Room held by early pins (refusals start 200 K tokens sooner: d0 = 0 instead of 100 K), plain system:")
    for w in (5.0e3, 5.5e3, 6.5e3):
        r = stochastic(w, False, False, 400, 7200.0, 3, d0=0.0)
        f = fixed_points(w, d0=0.0)
        P("      w = %.1f K: loop in %3.0f %% of runs (median onset %5.0f s); tipping point %3.0f K instead of %3.0f K"
          % (w / 1e3, 100 * r["tip_share"], r["tip_median"], f["d_u"] / 1e3, fixed_points(w)["d_u"] / 1e3))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stability.out"), "w") as f:
        f.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    sys.exit(main())
