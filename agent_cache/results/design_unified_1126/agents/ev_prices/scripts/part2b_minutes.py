"""Part 2, minute by minute: does the mean TTFT move with GPU work per request at about mu, and with reader
stall at about c?

Per run and minute m (requests sent in m): T(m) mean TTFT; S(m) mean stall per request (readers' stall, 0 for
the others); G(m) GPU seconds of history computed again per request (ideal - cached, at the run's loaded cost).
For a pair of runs (A, B): y = T_A - T_B, x1 = S_A - S_B, x2 = G_A - G_B, minute by minute, and
   y = c_hat * x1 + mu_hat * x2      (least squares, no intercept; also each term alone)
within a phase of the run. Beside it: N - B (users in session minus running requests), 1 + Qr, 1 + Q, and the
measured c (share of admissions with no ready turn left waiting), averaged over the phase's minutes.
Bootstrap over minutes (2,000 resamples) for the intervals.
usage: part2b_minutes.py"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import lib  # noqa: E402
import qstate  # noqa: E402
import readers  # noqa: E402

FIT = json.load(open(f"{lib.OUT}/prefill_fit.json"))


def gamma(spec):
    f = FIT.get(spec, {}).get("loaded")
    return (f["ms_per_tok_20K"] / 1e3) if f else 0.043e-3


def series(spec):
    camp, arm = spec.split(":")
    st = qstate.build(camp, arm)
    rows = readers.annotate(st)
    g = gamma(spec)
    M = 120
    acc = [dict(n=0, T=0.0, S=0.0, G=0.0, c_n=0, c_s=0) for _ in range(M)]
    by_key = {}
    for r in rows:
        if r["ttft"] is None:
            continue
        m = int(r["t_send"] // 60)
        if not 0 <= m < M:
            continue
        a = acc[m]
        a["n"] += 1
        a["T"] += r["ttft"]
        a["S"] += r["stall"] if (r["k"] > 0 and r["busy_q"] > 0.02) else 0.0
        a["G"] += max(0, r["ideal"] - r["cached"]) * g
        ma = int(r["adm"] // 60)
        if 0 <= ma < M:
            acc[ma]["c_n"] += 1
            acc[ma]["c_s"] += 1 if r["left_rdy"] == 0 else 0
        by_key[(r["tree"], r["stream"], r["k"])] = r
    out = []
    for m in range(M):
        a = acc[m]
        lo, hi = m * 60.0, m * 60.0 + 60.0
        n = max(1, a["n"])
        out.append(dict(n=a["n"], T=a["T"] / n, S=a["S"] / n, G=a["G"] / n, N=st["n_live"].mean(lo, hi),
                        Q=st["tot"].mean(lo, hi), Qr=st["rdy"].mean(lo, hi), B=st["running"].mean(lo, hi),
                        c=a["c_s"] / a["c_n"] if a["c_n"] else float("nan")))
    return out, by_key


def fit(y, X):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


def boot(y, X, reps=2000, seed=1):
    rng = random.Random(seed)
    n = len(y)
    bs = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        try:
            bs.append(fit([y[i] for i in idx], [X[i] for i in idx]))
        except Exception:  # noqa: BLE001
            pass
    bs = np.asarray(bs)
    return np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0)


def main():
    specs = ["c32:ec_full", "c34:ec_full", "c32:jit_exact", "c34:jit_exact", "c32:jit_p50", "c34:jit_p50",
             "c32:ec_exact", "c34:ec_exact", "c34:hicache", "c32:hicache"]
    ser = {s: series(s) for s in specs}
    phases = [("28-60 GPU saturated, users not capped", 28, 60), ("60-85 same, read line filling", 60, 85),
              ("85-110 users capped at 128", 85, 110), ("28-110", 28, 110)]
    print("A. recompute pairs: y = T_A - T_B per minute against x2 = GPU seconds of recompute per request (A - B)")
    for a, b in (("c32:ec_full", "c34:ec_full"), ("c32:jit_exact", "c34:jit_exact"), ("c32:ec_exact", "c34:ec_exact"),
                 ("c32:jit_p50", "c34:jit_p50"), ("c34:hicache", "c34:ec_full"), ("c32:hicache", "c32:ec_full")):
        sa, sb = ser[a][0], ser[b][0]
        print(f"== {a} - {b}")
        for name, lo, hi in phases:
            ms = [m for m in range(lo, hi) if sa[m]["n"] >= 10 and sb[m]["n"] >= 10]
            y = [sa[m]["T"] - sb[m]["T"] for m in ms]
            x1 = [sa[m]["S"] - sb[m]["S"] for m in ms]
            x2 = [sa[m]["G"] - sb[m]["G"] for m in ms]
            if max(abs(v) for v in x2) < 0.05:
                print(f"   {name}: recompute differs by under 0.05 GPU s per request in every minute; not fitted "
                      f"(mean dT {np.mean(y):+.2f} s, mean d stall {np.mean(x1):+.2f} s)")
                continue
            b2 = fit(y, [[v] for v in x2])[0]
            lo2, hi2 = boot(y, [[v] for v in x2])
            bb = fit(y, list(zip(x1, x2)))
            lob, hib = boot(y, list(zip(x1, x2)))
            nb = np.mean([sa[m]["N"] - sa[m]["B"] for m in ms])
            print(f"   {name}: {len(ms)} minutes; mean dT {np.mean(y):+7.2f} s, mean dG {np.mean(x2):+.4f} GPU s/req, mean d stall {np.mean(x1):+.2f} s/req")
            print(f"        mu_hat (recompute alone) {b2:6.1f} [{lo2[0]:.1f}, {hi2[0]:.1f}]; with stall: c_hat {bb[0]:+.2f} [{lob[0]:+.2f}, {hib[0]:+.2f}], "
                  f"mu_hat {bb[1]:6.1f} [{lob[1]:.1f}, {hib[1]:.1f}] | N-B (A) {nb:5.1f}, N {np.mean([sa[m]['N'] for m in ms]):5.1f}, "
                  f"1+Q {1+np.mean([sa[m]['Q'] for m in ms]):5.1f}, 1+Qr {1+np.mean([sa[m]['Qr'] for m in ms]):5.1f}")
    print()
    print("B. stall pairs (campaign 34, same recompute): y = T_A - T_B per minute against x1 = stall per request (A - B)")
    for a, b in (("c34:jit_p50", "c34:ec_full"), ("c34:jit_p50", "c34:jit_exact"), ("c34:jit_p50", "c34:ec_exact"),
                 ("c34:jit_exact", "c34:ec_full"), ("c34:ec_exact", "c34:ec_full")):
        sa, sb = ser[a][0], ser[b][0]
        ka, kb = ser[a][1], ser[b][1]
        print(f"== {a} - {b}")
        for name, lo, hi in phases:
            ms = [m for m in range(lo, hi) if sa[m]["n"] >= 10 and sb[m]["n"] >= 10]
            y = [sa[m]["T"] - sb[m]["T"] for m in ms]
            x1 = [sa[m]["S"] - sb[m]["S"] for m in ms]
            x2 = [sa[m]["G"] - sb[m]["G"] for m in ms]
            b1 = fit(y, [[v] for v in x1])[0]
            lo1, hi1 = boot(y, [[v] for v in x1])
            bb = fit(y, list(zip(x1, x2)))
            lob, hib = boot(y, list(zip(x1, x2)))
            # paired on the requests both arms sent, by the minute the B arm sent them
            pair = [(ka[k]["ttft"] - kb[k]["ttft"]) for k in kb if k in ka and lo * 60 <= kb[k]["t_send"] < hi * 60]
            cm = np.nanmean([sb[m]["c"] for m in ms])
            print(f"   {name}: {len(ms)} minutes; mean dT {np.mean(y):+6.2f} s (paired on {len(pair)} requests {np.mean(pair):+.2f} s), "
                  f"mean d stall {np.mean(x1):+.2f} s/req, mean dG {np.mean(x2):+.4f} GPU s/req")
            print(f"        c_hat (stall alone) {b1:+.3f} [{lo1[0]:+.3f}, {hi1[0]:+.3f}]; with recompute: c_hat {bb[0]:+.3f} [{lob[0]:+.3f}, {hib[0]:+.3f}], "
                  f"mu_hat {bb[1]:7.1f} [{lob[1]:.0f}, {hib[1]:.0f}] | measured c (B arm) {cm:.3f}; ratio of means dT/dS {np.mean(y)/np.mean(x1) if abs(np.mean(x1))>0.05 else float('nan'):+.3f}")


if __name__ == "__main__":
    main()
