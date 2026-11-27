"""Check 2b: J_F(s) on the trace's two gap entries for realistic values.

L = 25,000 tokens; r = 2..6 s; W = 0, 2, 5 s; c in {1, 0.3, 0}; lambda in {0, 1e-6, 2e-5, 1e-4}
per token-second (rent lamL = lambda L of 0, 0.025, 0.5, 2.5 s of TTFT per second held).

For each cell, from the turn's end (idle age 0), on the engine's law (two-piece lognormal
capped at 1,200 s):
  s_opt, J(s_opt)        global minimum of J_F over the start s (staged copy held until admission)
  gain                   J(arrival) - J(s_opt), with J(arrival) = c x
  J(0)                   read at once and hold until admission
  thr                    the threshold rule: first s with F(x|s) >= q, and its cost
  eng                    what the engine's 64-point read_start returns, and that start's true cost
  drop                   best (start, drop time) when the staged copy may be dropped (section "Fetch = flight + hold"),
                         with the read's in-flight memory charged: K_io = lamL * r_m, r_m = min(r, 0.9 + L / beta)
Writes jf_grid.csv next to this script.
"""

import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script's folder

import numpy as np

from fetchlib import (TwoPieceLognormal, Conditioned, J_F, F_cond, best_start, start_grid,
                      load_engine_policy, TOOL, HUMAN, CAP)

L = 25_000
BETA = 21_400.0          # tokens per second, the one SSD
LAGS = 0.9
HERE = os.path.dirname(os.path.abspath(__file__))

ntp = load_engine_policy()


def threshold_start(law, x, q, hi):
    g = start_grid(hi, n_log=20000, n_lin=20000)
    fx = F_cond(law, x, g)
    ok = np.flatnonzero(fx >= q)
    return None if len(ok) == 0 else float(g[ok[0]])


def engine_start(q3, r, W, lamL, c):
    """The engine's rule 1 on the same entry (64 equal-weight points capped at 1,200 s); its cost
    has weight 1 on stall, so the rent is passed as lamL / c."""
    if c <= 0.0:
        return "n/a"
    dist = ntp.GapDistribution.from_quantiles(p10=q3["p10"], p50=q3["p50"], p90=q3["p90"])
    rem = ntp.remaining_gaps(support=dist.support(), idle_s=0.0)
    return ntp.read_start(remaining=rem, read_s=r, wait_s=W, rent_per_s=lamL / c)


def best_with_drop(law, x, W, r, c, lamL, Kio, hi):
    """max over (s, theta >= s + r) of
       G = c [M(s) - M(s+x) - x S(theta)] - lamL [M(s+x) - M(theta) - W S(theta)] - K_io S(theta)."""
    s_grid = np.unique(np.concatenate([[0.0, max(hi - x, 0.0)], np.geomspace(1e-3, hi, 300), np.linspace(0, min(hi, 60.0), 241)]))
    best = (0.0, None, None)
    for s in s_grid:
        th = s + r + np.concatenate([[0.0], np.geomspace(1e-3, max(hi - s - r, 1e-2) + 1.0, 600)])
        Sth = law.sf(th)
        G = (c * (law.M(s) - law.M(s + x) - x * Sth)
             - lamL * (law.M(s + x) - law.M(th) - W * Sth) - Kio * Sth)
        k = int(np.argmax(G))
        if G[k] > best[0] + 1e-12:
            best = (float(G[k]), float(s), float(th[k]))
    return best


def run(name, q3):
    law = TwoPieceLognormal(**q3, cap=CAP)
    rows = []
    for W in [0.0, 2.0, 5.0]:
        for r in [2.0, 3.0, 4.0, 5.0, 6.0]:
            x = max(r - W, 0.0)
            for c in [1.0, 0.3, 0.0]:
                for lam in [0.0, 1e-6, 2e-5, 1e-4]:
                    lamL = lam * L
                    row = dict(entry=name, r=r, W=W, x=x, c=c, lam=lam, lamL=lamL)
                    if x <= 0.0:
                        row.update(q="", F_x0="", s_opt="arrival", J_opt=0.0, J_arr=0.0, gain=0.0, J_now="",
                                   thr_start="", J_thr="", eng_start="arrival (x = 0)", J_eng="",
                                   drop_gain=0.0, drop_s="", drop_theta="")
                        rows.append(row)
                        continue
                    q = lamL / (c + lamL) if (c + lamL) > 0 else float("nan")
                    f0 = float(F_cond(law, x, 0.0))
                    s_opt, J_opt, J_arr = best_start(law, x, c, lamL, CAP, extra=[CAP - x])
                    if J_opt >= J_arr - 1e-9:
                        s_show, J_opt = "arrival", J_arr
                    elif abs(s_opt + x - CAP) < 0.5:
                        s_show = "cap - x"          # the read lands on the 1,200 s cap's atom
                    else:
                        s_show = f"{s_opt:.3f}"
                    J_now = float(J_F(law, 0.0, x, c, lamL))
                    if c + lamL > 0:
                        ts = threshold_start(law, x, q, CAP)
                    else:
                        ts = None
                    J_thr = "" if ts is None else float(J_F(law, ts, x, c, lamL))
                    es = engine_start(q3, r, W, lamL, c)
                    if es == "n/a":
                        es_show, J_eng = "n/a (c = 0)", ""
                    elif es is None:
                        es_show, J_eng = "arrival", J_arr
                    else:
                        es_show, J_eng = f"{es:.3f}", float(J_F(law, es, x, c, lamL))
                    r_m = min(r, LAGS + L / BETA)
                    dg, ds, dth = best_with_drop(law, x, W, r, c, lamL, lamL * r_m, CAP)
                    row.update(q=q, F_x0=f0, s_opt=s_show, J_opt=J_opt, J_arr=J_arr, gain=J_arr - J_opt, J_now=J_now,
                               thr_start="" if ts is None else ts, J_thr=J_thr, eng_start=es_show, J_eng=J_eng,
                               drop_gain=dg, drop_s="" if ds is None else ds, drop_theta="" if dth is None else dth)
                    rows.append(row)
    return rows


def show(rows, name):
    print(f"== {name} entry, from the turn's end; L = {L:,} tokens ==")
    print("   W    r    x    c    lambda   lamL    q      F(x|0)  | best start   J(best)  J(arrival)  gain    J(read now) | threshold rule: start  cost"
          "   | engine 64-pt: start   true cost | with drop + in-flight rent: gain  start  drop at")
    for d in rows:
        if d["x"] <= 0.0:
            continue
        thr = f"{d['thr_start']:9.3f} {d['J_thr']:7.3f}" if d["thr_start"] != "" else "    never        "
        eng = f"{d['eng_start']:>12s} {d['J_eng']:7.3f}" if d["J_eng"] != "" else f"{d['eng_start']:>12s}        "
        drop = (f"{d['drop_gain']:6.3f} {d['drop_s']:8.2f} {d['drop_theta']:8.2f}" if d["drop_s"] != "" else "  0.000   (no early read)")
        qq = f"{d['q']:5.3f}" if d["q"] == d["q"] else "  n/a"
        print(f"  {d['W']:3.0f}  {d['r']:3.0f}  {d['x']:3.0f}  {d['c']:3.1f}  {d['lam']:7.0e}  {d['lamL']:5.3f}  {qq}  {d['F_x0']:6.3f}  |"
              f" {d['s_opt']:>10s}  {d['J_opt']:7.3f}  {d['J_arr']:7.3f}   {d['gain']:6.3f}   {d['J_now']:9.3f}   | {thr}          | {eng}     | {drop}")


def summary(rows):
    """Counts over the cells with x > 0 and c > 0."""
    print("== summary over the cells with x > 0 and c > 0 ==")
    for name in ["tool", "human"]:
        cells = [d for d in rows if d["entry"] == name and d["x"] > 0 and d["c"] > 0]
        now = [d for d in cells if d["s_opt"] == "0.000"]
        capx = [d for d in cells if d["s_opt"] == "cap - x"]
        arr = [d for d in cells if d["s_opt"] == "arrival"]
        other = [d for d in cells if d["s_opt"] not in ("0.000", "cap - x", "arrival")]
        thr_now = [d for d in cells if d["thr_start"] != "" and d["thr_start"] < 1.0]
        thr_wrong = [d for d in thr_now if d["J_thr"] > d["J_arr"] + 1e-9]
        worst = max(thr_wrong, key=lambda d: d["J_thr"] - d["J_arr"]) if thr_wrong else None
        print(f"   {name}: {len(cells)} cells; best start is the turn's end in {len(now)}, the cap's atom in {len(capx)}"
              f" (largest gain there {max([d['gain'] for d in capx], default=0):.4f} s), the arrival in {len(arr)}, elsewhere in {len(other)}")
        if now:
            print(f"      where the turn's end is best: lambda in {sorted(set(d['lam'] for d in now))}, gains {min(d['gain'] for d in now):.3f} to {max(d['gain'] for d in now):.3f} s")
        print(f"      the threshold rule starts within 1 s of the turn's end in {len(thr_now)} cells; in {len(thr_wrong)} of them that costs more than reading at arrival"
              + (f" (worst: x={worst['x']}, c={worst['c']}, lambda={worst['lam']:g}: {worst['J_thr']:.2f} against {worst['J_arr']:.2f} s)" if worst else ""))
        pos = [d for d in cells if d["lam"] > 0]
        dg = [d for d in pos if d["drop_gain"] > 1e-3 and d["drop_s"] != "" and d["drop_s"] < 1.0]
        print(f"      with a drop time and the in-flight rent charged, a read started at the turn's end and dropped later pays in {len(dg)} of the {len(pos)} cells with lambda > 0")
        for lam in [1e-6, 2e-5, 1e-4]:
            sel = [d for d in dg if d["lam"] == lam and d["W"] == 0.0]
            if sel:
                print(f"         lambda = {lam:g}, W = 0: " + "; ".join(f"x={d['x']:.0f}, c={d['c']}: gain {d['drop_gain']:.2f} s, drop at {d['drop_theta']:.1f} s" for d in sel))


def ages(name, q3):
    """The same decision taken later in the gap (the engine re-plans every 0.5 s on the conditional law)."""
    base = TwoPieceLognormal(**q3, cap=CAP)
    print(f"== {name} entry at later idle ages (r = 3 s, W = 0, c = 1): best start from now, gain over arrival ==")
    print("   age (s)   lambda   F(x|age)   q      best start (from now)   gain     J(now)-J(arrival)")
    for age in [0.0, 0.3, 0.8, 1.7, 3.0, 10.0, 30.0, 96.0, 300.0, 1100.0]:
        law = Conditioned(base, age)
        for lam in [1e-6, 2e-5, 1e-4]:
            lamL = lam * L
            x, c = 3.0, 1.0
            hi = CAP - age
            g = start_grid(hi)
            g = np.unique(np.concatenate([g, [max(hi - x, 0.0)]]))
            j = J_F(law, g, x, c, lamL)
            k = int(np.argmin(j))
            J_arr = c * x
            best = "arrival" if j[k] >= J_arr - 1e-9 else f"{g[k]:.3f}"
            print(f"   {age:7.1f}   {lam:7.0e}   {float(F_cond(law, x, 0.0)):6.3f}   {lamL / (c + lamL):5.3f}   {best:>12s}"
                  f"          {max(J_arr - j[k], 0.0):7.4f}   {float(J_F(law, 0.0, x, c, lamL)) - J_arr:+9.3f}")


if __name__ == "__main__":
    allrows = []
    for name, q3 in [("tool", TOOL), ("human", HUMAN)]:
        rows = run(name, q3)
        show(rows, name)
        allrows += rows
    with open(os.path.join(HERE, "jf_grid.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0].keys()))
        w.writeheader()
        w.writerows(allrows)
    print(f"wrote {os.path.join(HERE, 'jf_grid.csv')} ({len(allrows)} rows)")
    summary(allrows)
    ages("tool", TOOL)
    ages("human", HUMAN)
