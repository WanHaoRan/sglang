"""Check 2d: the Fetch rules on the hints the replay actually sent (campaign 34 ec_full client log,
16,118 turns with a p10/p50/p90 entry and the true next gap). Read-only on the log.

(a) shape of the per-key entries: how many have a hazard that still rises at the median
    (sigma_high < 0.798, i.e. p90/p50 < 2.78)?
(b) calibration of the engine's two-piece lognormal on these entries.
(c) realized priced cost per turn of each start rule, for a prefix that is on the SSD at the turn's end
    (L = 25 K tokens, W = 0, x = r, in-flight rent K_io = lamL r charged on every read):
      arrival     read when the turn arrives
      exact       oracle: start so the read lands at the arrival
      argmin      start at the minimiser of J_F on the entry's law (capped at 1,200 s), hold until admission
      threshold   start when F(x|s) >= q first holds, hold until admission
      p50         start at p50 - x (a point forecast), hold until admission
      p10         start at p10 - x - 1 s (what the jit_p50 arm does), hold until admission
      engine      the engine's rule re-run every 0.5 s on the 64-point support (read_start), hold until admission
      fetch+drop  start now, drop the landed copy at the entry's best drop time (the joint rule G(s, theta))
    and, for reference, a copy that never left memory:
      keep        keep until the entry's best drop time (Retain's rule), then read at arrival
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script's folder

import numpy as np

from fetchlib import TwoPieceLognormal, J_F, F_cond, load_engine_policy, Z90, CAP

CLIENT = ("/home/wanhr/sglang/agent_cache/results/20261123_093604_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H160/"
          "programs_093918_ec_full/client.jsonl")
L = 25_000
ntp = load_engine_policy()


def load():
    rows = []
    with open(CLIENT) as f:
        for line in f:
            d = json.loads(line)
            if d.get("kind") != "req":
                continue
            h = d.get("hint")
            if not h or "p10" not in h or h.get("true_gap_s") is None:
                continue
            m = h.get("matched")
            kind = "no_call" if h.get("reply_key") == "_no_call" else "tool"
            level = "tool/word" if (m and "/" in m) else ("*" if m == "*" else "tool")
            rows.append((h["p10"], h["p50"], h["p90"], h["n"], h["true_gap_s"], kind, level))
    return rows


def part_a(rows):
    print(f"== (a) shape of the {len(rows):,} entries sent ==")
    sl = np.array([max(0.05, math.log(max(r[1], 0.01) / max(min(r[0], r[1]), 0.01)) / Z90) for r in rows])
    sh = np.array([max(0.05, math.log(max(r[2], r[1]) / max(r[1], 0.01)) / Z90) for r in rows])
    kind = np.array([r[5] for r in rows])
    n = np.array([r[3] for r in rows])
    for name, m in [("all", np.ones(len(rows), bool)), ("tool calls", kind == "tool"), ("no call (human)", kind == "no_call")]:
        print(f"   {name:16s} n={int(m.sum()):6d}  sigma_high p10/p50/p90 = {np.quantile(sh[m], 0.1):.2f}/{np.quantile(sh[m], 0.5):.2f}/{np.quantile(sh[m], 0.9):.2f};"
              f"  sigma_low {np.quantile(sl[m], 0.1):.2f}/{np.quantile(sl[m], 0.5):.2f}/{np.quantile(sl[m], 0.9):.2f}")
        print(f"   {'':16s} share with sigma_high < 0.798 (hazard still rising at the median): {100 * np.mean(sh[m] < 0.798):.1f} %;"
              f" sigma_high < 0.5: {100 * np.mean(sh[m] < 0.5):.1f} %; both sigmas < 0.5: {100 * np.mean((sh[m] < 0.5) & (sl[m] < 0.5)):.1f} %;"
              f" sigma_high < 0.2: {100 * np.mean(sh[m] < 0.2):.1f} %")
    print(f"   samples behind an entry (n): p10/p50/p90 = {np.quantile(n, 0.1):.0f}/{np.quantile(n, 0.5):.0f}/{np.quantile(n, 0.9):.0f}")
    return sl, sh


def part_b(rows):
    print("== (b) calibration of the two-piece lognormal on these entries ==")
    u, mean_model, g_all = [], [], []
    for (p10, p50, p90, n, g, kind, level) in rows:
        law = TwoPieceLognormal(p10, p50, p90, cap=CAP)
        gg = min(max(g, 0.01), CAP)
        u.append(float(law.cdf(gg)) if gg < CAP else 1.0)
        mean_model.append(law.mean())
        g_all.append(gg)
    u, mean_model, g_all = np.array(u), np.array(mean_model), np.array(g_all)
    dec = np.histogram(u, bins=np.linspace(0, 1, 11))[0] / len(u)
    print("   share of true gaps per decile of the entry's own law (10 % each if calibrated): " + " ".join(f"{100 * v:4.1f}" for v in dec))
    print(f"   below the p10: {100 * np.mean(u < 0.1):.1f} %, inside [p10, p90]: {100 * np.mean((u >= 0.1) & (u <= 0.9)):.1f} %, above the p90: {100 * np.mean(u > 0.9):.1f} %,"
          f" above the model's p99: {100 * np.mean(u > 0.99):.1f} %")
    print(f"   mean true gap {g_all.mean():.2f} s; mean of the entries' model means (capped at 1,200 s) {mean_model.mean():.2f} s")
    for x in [2.0, 4.0]:
        over = np.maximum(g_all - x, 0.0).mean()
        mod = np.mean([float(TwoPieceLognormal(r[0], r[1], r[2], cap=CAP).M(x)) for r in rows[::8]])
        print(f"   E[(gap - x)^+] at x = {x}: realized {over:.2f} s, model {mod:.2f} s (every 8th entry)")


def engine_dynamic(dist_support, g, r, W, rent):
    """Tick every 0.5 s until the true arrival; returns the idle age at which the engine stages, or None."""
    a = 0.0
    while a < g:
        rem = ntp.remaining_gaps(support=dist_support, idle_s=a)
        if len(rem) == 0:
            return None
        st = ntp.read_start(remaining=rem, read_s=r, wait_s=W, rent_per_s=rent)
        if st is not None and st <= 0.0:
            return a
        if st is None:
            # nothing changes until the next support point is passed
            nxt = dist_support[len(dist_support) - len(rem)]
            a = max(a + 0.5, math.ceil(nxt / 0.5) * 0.5)
        else:
            a += 0.5
    return None


def realized_fetch(g, s, x, c, lamL, Kio):
    """Read planned at s (None = arrival), held until admission; W = 0."""
    if s is None or g <= s:
        return c * x + Kio
    return c * max(x - (g - s), 0.0) + lamL * max(g - s - x, 0.0) + Kio


def part_c(rows, sub):
    print("== (c) realized priced cost per turn (s of TTFT), prefix on the SSD at the turn's end; L = 25 K, W = 0, x = r ==")
    grid = np.unique(np.concatenate([[0.0], np.geomspace(1e-3, CAP, 1500), np.linspace(0.0, 60.0, 601)]))
    th_grid = np.unique(np.concatenate([np.geomspace(1e-2, CAP, 800), np.linspace(0.0, 60.0, 301), [CAP]]))
    for (x, lam) in [(2.0, 1e-6), (4.0, 1e-6), (2.0, 2e-5), (4.0, 2e-5), (4.0, 1e-4)]:
        c = 1.0
        lamL = lam * L
        q = lamL / (c + lamL)
        r, W = x, 0.0
        Kio = lamL * r
        tot = dict(arrival=0.0, exact=0.0, argmin=0.0, threshold=0.0, p50=0.0, p10=0.0, engine=0.0, fetch_drop=0.0, keep=0.0)
        n_stage = dict(argmin=0, threshold=0, engine=0, fetch_drop=0)
        ex_ante = 0.0
        for i in sub:
            (p10, p50, p90, n, g, kind, level) = rows[i]
            g = min(max(g, 0.01), CAP)
            law = TwoPieceLognormal(p10, p50, p90, cap=CAP)
            tot["arrival"] += c * x + Kio
            tot["exact"] += c * max(x - g, 0.0) + Kio
            # argmin on the capped law
            j = J_F(law, grid, x, c, lamL)
            k = int(np.argmin(j))
            s_arg = None if j[k] >= c * x - 1e-9 else float(grid[k])
            ex_ante += max(c * x - float(j[k]), 0.0)
            tot["argmin"] += realized_fetch(g, s_arg, x, c, lamL, Kio)
            n_stage["argmin"] += int(s_arg is not None and g > s_arg)
            # threshold rule
            fx = F_cond(law, x, grid)
            ok = np.flatnonzero(fx >= q)
            s_thr = None if len(ok) == 0 else float(grid[ok[0]])
            tot["threshold"] += realized_fetch(g, s_thr, x, c, lamL, Kio)
            n_stage["threshold"] += int(s_thr is not None and g > s_thr)
            tot["p50"] += realized_fetch(g, max(p50 - x, 0.0), x, c, lamL, Kio)
            tot["p10"] += realized_fetch(g, max(p10 - x - 1.0, 0.0), x, c, lamL, Kio)
            # the engine, re-run every 0.5 s
            dist = ntp.GapDistribution.from_quantiles(p10=p10, p50=p50, p90=p90)
            s_eng = engine_dynamic(dist.support(), g, r, W, lamL / c)
            tot["engine"] += realized_fetch(g, s_eng, x, c, lamL, Kio)
            n_stage["engine"] += int(s_eng is not None)
            # start now, drop at the best time theta >= r:  G(0, theta) = c [M(0) - M(x) - x S(theta)] - lamL [M(x) - M(theta)] - K_io S(theta)
            th = th_grid[th_grid >= r]
            G = c * (law.M(0.0) - law.M(x) - x * law.sf(th)) - lamL * (law.M(x) - law.M(th)) - Kio * law.sf(th)
            kk = int(np.argmax(G))
            if G[kk] > 1e-9:
                theta = float(th[kk])
                n_stage["fetch_drop"] += 1
                if g <= theta:
                    tot["fetch_drop"] += c * max(x - g, 0.0) + lamL * max(g - x, 0.0) + Kio
                else:
                    tot["fetch_drop"] += lamL * (theta - r) + Kio + c * x + Kio
            else:
                tot["fetch_drop"] += c * x + Kio
            # a copy that never left memory: keep until the best theta of G_keep(0, theta) = K_net (1 - S(theta)) - lamL (M(0) - M(theta))
            Knet = c * x + Kio
            Gk = Knet * (1.0 - law.sf(th_grid)) - lamL * (law.M(0.0) - law.M(th_grid))
            kk = int(np.argmax(Gk))
            if Gk[kk] > 1e-9:
                theta = float(th_grid[kk])
                tot["keep"] += lamL * g if g <= theta else lamL * theta + c * x + Kio
            else:
                tot["keep"] += c * x + Kio
        m = len(sub)
        print(f"   x = r = {x} s, lambda = {lam:g} (lamL = {lamL}, q = {q:.3f}, K_io = {Kio:.3f}); {m:,} turns")
        print("      " + "  ".join(f"{k}={v / m:7.3f}" for k, v in tot.items()))
        print(f"      reads started before arrival: argmin {n_stage['argmin']}, threshold {n_stage['threshold']}, engine {n_stage['engine']}, fetch+drop {n_stage['fetch_drop']}"
              f";  argmin's own expected gain per turn {ex_ante / m:.3f} s, realized {(tot['arrival'] - tot['argmin']) / m:+.3f} s")


if __name__ == "__main__":
    rows = load()
    part_a(rows)
    part_b(rows)
    rng = np.random.default_rng(11)
    sub = np.sort(rng.choice(len(rows), size=4000, replace=False))
    part_c(rows, sub)
