"""Check 2a: the hazard of the trace's two gap entries, as the engine models them
(two-piece lognormal through p10/p50/p90; next_turn_policy.GapDistribution.from_quantiles),
and as the replayed gaps themselves show it (campaign 34 ec_full client log, read-only).

Questions: is the hazard decreasing everywhere, or only after an early mode? Where is
F(x | s) = P(G <= s + x | G > s) rising in s?
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script's folder

import numpy as np
from scipy import optimize

from fetchlib import TwoPieceLognormal, F_cond, Phi, Phibar, phi, TOOL, HUMAN, CAP, Z90

CLIENT = ("/home/wanhr/sglang/agent_cache/results/20261123_093604_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H160/"
          "programs_093918_ec_full/client.jsonl")


def mills_mode(sig):
    """z* with phi(z)/Phibar(z) - z = sig: the lognormal(sigma) hazard peaks at median * exp(sig z*)."""
    f = lambda z: float(phi(z) / Phibar(z)) - z - sig
    return optimize.brentq(f, -30.0, 30.0)


def lognormal_table():
    print("== lognormal(sigma): where the hazard peaks (it rises before, falls after) ==")
    print("   sigma  p90/p50   z*      quantile of the peak   peak / median")
    for sig in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.798, 0.9, 1.0, 1.105, 1.5, 1.792, 1.859, 2.0, 2.679, 3.0]:
        z = mills_mode(sig)
        print(f"   {sig:5.3f}  {math.exp(sig * Z90):7.2f}  {z:+7.3f}   {float(Phi(z)):9.5f}             {math.exp(sig * z):9.4g}")
    crit = float(phi(0.0) / 0.5)
    print(f"   the peak sits at the median when sigma = phi(0)/0.5 = {crit:.4f}, i.e. p90/p50 = {math.exp(crit * Z90):.3f}")


def entry_report(name, q3, cap):
    law = TwoPieceLognormal(**q3)
    lawc = TwoPieceLognormal(**q3, cap=cap)
    print(f"== {name}: p10/p50/p90 = {q3['p10']}/{q3['p50']}/{q3['p90']} s ==")
    print(f"   sigma_low {law.sl:.4f}, sigma_high {law.sh:.4f}; mean {law.mean():.2f} s (uncapped), {lawc.mean():.2f} s (capped at {cap:.0f} s)")
    print(f"   model p99 {float(law.ppf(0.99)):.1f} s; P(G > {cap:.0f} s) = {float(law.sf(cap)):.4g}")
    # hazard below and above the median
    zl = mills_mode(law.sl)
    t_mode = law.m * math.exp(law.sl * zl)
    zh = mills_mode(law.sh)
    t_mode_h = law.m * math.exp(law.sh * zh)
    h = lambda t: float(law.hazard(t))
    print(f"   lower piece (below the median): hazard peaks at {t_mode:.3f} s, the {100 * float(Phi(zl)):.1f}th percentile, h = {h(t_mode):.4f} /s")
    print(f"   upper piece alone would peak at {t_mode_h:.4f} s (< median), so above the median the hazard only falls")
    eps = 1e-9
    print(f"   at the median {law.m} s the density, and so the hazard, jumps by sigma_low/sigma_high = {law.sl / law.sh:.3f}:"
          f" h(median-) = {h(law.m * (1 - eps)):.4f}, h(median+) = {h(law.m * (1 + eps)):.4f} /s")
    ages = [0.05, 0.1, 0.2, 0.41, 0.79, 1.0, 1.69, 2.0, 3.0, 5.0, 10.0, 16.8, 30.0, 60.0, 96.0, 97.0, 120.0, 300.0, 600.0, 1040.0, 1199.0]
    print("   idle age (s):     " + " ".join(f"{a:8.4g}" for a in ages))
    print("   hazard (1/s):     " + " ".join(f"{h(a):8.4g}" for a in ages))
    print("   P(G <= age):      " + " ".join(f"{float(law.cdf(a)):8.4g}" for a in ages))
    # monotone pieces of the hazard on a fine grid
    g = np.geomspace(1e-3, cap * 0.9999, 400001)
    hz = law.hazard(g)
    up = np.diff(hz) > 0
    edges = np.flatnonzero(np.diff(up.astype(int)) != 0) + 1
    segs = np.split(np.arange(len(up)), edges)
    out = []
    for sgm in segs:
        if up[sgm[0]]:
            out.append(f"[{g[sgm[0]]:.4g}, {g[sgm[-1] + 1]:.4g}] s")
    print(f"   the hazard rises on: {', '.join(out) if out else 'nowhere'}; it falls elsewhere on (0.001, {cap:.0f}) s")
    share_rising = float(law.cdf(t_mode))
    print(f"   share of gaps that end while the hazard is still rising (before the early mode): {100 * share_rising:.1f} %")
    # F(x|s): where it rises
    print("   F(x|s) = P(G <= s+x | G > s), uncapped law:")
    print("      x (s)   F(x|0)   max_s F(x|s)  at s (s)   rises for s in")
    for x in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
        s = np.unique(np.concatenate([[0.0], np.geomspace(1e-4, 5000.0, 200001)]))
        fx = F_cond(law, x, s)
        k = int(np.argmax(fx))
        upx = np.diff(fx) > 1e-13
        e = np.flatnonzero(np.diff(upx.astype(int)) != 0) + 1
        sg = np.split(np.arange(len(upx)), e)
        ris = [f"[{s[a[0]]:.3g}, {s[a[-1] + 1]:.3g}]" for a in sg if upx[a[0]]]
        print(f"      {x:4.1f}   {float(fx[0]):6.4f}    {float(fx[k]):6.4f}      {s[k]:8.3f}   {', '.join(ris)}")
    return law, lawc


def empirical():
    print("== replayed gaps (campaign 34 ec_full client log): empirical hazard ==")
    waits = {"tool": [], "user": []}
    with open(CLIENT) as f:
        for line in f:
            d = json.loads(line)
            if d.get("kind") != "req":
                continue
            if d.get("gap") in waits:
                waits[d["gap"]].append(float(d["wait_s"]))
    for kind, w in waits.items():
        w = np.sort(np.asarray(w))
        n = len(w)
        qs = np.quantile(w, [0.1, 0.5, 0.9, 0.99])
        print(f"   {kind}: n={n}, p10/p50/p90/p99 = {qs[0]:.2f}/{qs[1]:.2f}/{qs[2]:.2f}/{qs[3]:.1f} s, mean {w.mean():.2f} s,"
              f" at the 1,200 s cap: {int((w >= 1199.99).sum())}")
        # life-table hazard on log-spaced bins: events in the bin / exposure in the bin
        edges = np.array([0.0, 0.1, 0.2, 0.3, 0.45, 0.65, 0.9, 1.3, 1.8, 2.6, 3.7, 5.3, 7.5, 11, 15, 22, 31, 44, 63, 90, 128, 181, 256, 362, 512, 724, 1024, 1199.9])
        rows = []
        for a, b in zip(edges[:-1], edges[1:]):
            ev = int(((w > a) & (w <= b)).sum())
            expo = float(np.clip(w, a, b).sum() - a * n)            # sum over gaps of time spent in (a, b]
            at_risk = int((w > a).sum())
            if at_risk < 30 or expo <= 0:
                continue
            hz = ev / expo
            se = math.sqrt(ev) / expo if ev > 0 else float("nan")
            rows.append((a, b, at_risk, ev, hz, se))
        print("      bin (s)            at risk  events  hazard (1/s)  +-")
        for (a, b, ar, ev, hz, se) in rows:
            print(f"      ({a:7.2f}, {b:7.2f}]  {ar:6d}  {ev:6d}   {hz:9.4f}   {se:7.4f}")
        hzs = np.array([r[4] for r in rows])
        kmax = int(np.argmax(hzs))
        print(f"      the empirical hazard peaks in bin ({rows[kmax][0]}, {rows[kmax][1]}] s;"
              f" share of gaps at or below that bin's upper edge: {100 * float((w <= rows[kmax][1]).mean()):.1f} %")
        # rises between consecutive bins that exceed two standard errors
        ups = []
        for i in range(1, len(rows)):
            d = rows[i][4] - rows[i - 1][4]
            se = math.hypot(rows[i][5], rows[i - 1][5])
            if d > 2 * se:
                ups.append(f"({rows[i - 1][0]}, {rows[i][1]}]")
        print(f"      bins where the hazard rises by more than 2 standard errors: {', '.join(ups) if ups else 'none'}")


if __name__ == "__main__":
    lognormal_table()
    entry_report("tool waits", TOOL, CAP)
    entry_report("human waits", HUMAN, CAP)
    empirical()
