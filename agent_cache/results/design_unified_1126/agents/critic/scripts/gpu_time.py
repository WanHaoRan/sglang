"""GPU-time accounting per run: what each scheduler pass spends, and what the host-to-GPU load (h2d),
the GPU-to-host backup (d2h) and GPU eviction add to it.

Method: intervals between consecutive 'Decode batch' log lines hold decode_log_interval (40) decode steps
plus every prefill batch started in between. OLS of the interval length on
  40 x (1, running, context K)   [decode step]
  sum over prefill batches of (1, new tokens, new x cached K)   [prefill]
  sum of h2d tokens, sum of d2h tokens, sum of evict_device ms   [copies]
restricted to intervals with a queue at both ends (the GPU is never idle there).
Also prints the directly logged copy times (h2d_done ms, d2h_done ms, evict_device ms) per window.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xlib  # noqa: E402

WINDOWS = [(0, 28), (28, 70), (70, 90), (90, 110)]


def series(d, name, key):
    return [(t, f.get(key, 0)) for t, f in d["ev"].get(name, [])]


def window_sum(pairs, lo, hi):
    return sum(v for t, v in pairs if lo <= t < hi), sum(1 for t, v in pairs if lo <= t < hi)


def main(spec):
    camp, arm = spec.split(":")
    d = xlib.load(camp, arm)
    batch = d["batch"]
    h2d_tok = series(d, "h2d_submit", "tokens")
    h2d_ms = series(d, "h2d_done", "ms")
    h2d_done_tok = series(d, "h2d_done", "tokens")
    d2h_tok = series(d, "d2h_submit", "tokens")
    d2h_ms = series(d, "d2h_done", "ms")
    evd_ms = series(d, "evict_device", "ms")
    evd_tok = series(d, "evict_device", "tokens")
    lbi = series(d, "load_back_init", "tokens")
    rts = d["rts"]
    print(f"== {spec}")
    print("window  reqs  prefill_batches new_tok/batch  h2d: n tokens(M) ms_total ms/req us/tok | d2h: n tokens(M) ms_total ms/req us/tok | evict_dev: n ms_total ms/req | load_back_init n tok(M)")
    for lo, hi in WINDOWS:
        a, b = lo * 60, hi * 60
        nreq = sum(1 for r in rts.values() if a <= r[6] + r[4] < b)  # admitted in the window
        pb = [x for x in batch if x[1] == "P" and a <= x[0] < b]
        newtok = sum(x[3] for x in pb)
        ht, hn = window_sum(h2d_tok, a, b)
        hms, _ = window_sum(h2d_ms, a, b)
        dt_, dn = window_sum(d2h_tok, a, b)
        dms, _ = window_sum(d2h_ms, a, b)
        ems, en = window_sum(evd_ms, a, b)
        lt, ln = window_sum(lbi, a, b)
        nr = max(nreq, 1)
        print(f"{lo:3d}-{hi:<3d} {nreq:6d} {len(pb):7d} {newtok / max(len(pb), 1):9.0f}   "
              f"{hn:6d} {ht / 1e6:7.2f} {hms / 1e3:8.1f}s {hms / nr:6.1f} {1e3 * hms / max(ht, 1):6.2f} | "
              f"{dn:6d} {dt_ / 1e6:6.2f} {dms / 1e3:7.1f}s {dms / nr:6.2f} {1e3 * dms / max(dt_, 1):6.2f} | "
              f"{en:6d} {ems / 1e3:7.1f}s {ems / nr:6.2f} | {ln:6d} {lt / 1e6:7.2f}")
    # h2d rate distribution
    rates = [tok / ms for (t, tok), (_, ms) in zip(h2d_done_tok, h2d_ms) if ms > 0 and tok >= 4096]
    if rates:
        print(f"h2d copy rate (tokens/ms, copies of >= 4096 tokens, n={len(rates)}): p10 {xlib.pct(rates, .1):.0f} p50 {xlib.pct(rates, .5):.0f} "
              f"p90 {xlib.pct(rates, .9):.0f}  -> GB/s p50 {xlib.pct(rates, .5) * 98304 / 1e6:.1f}")
    # regression over decode-log intervals
    D = [(i, x) for i, x in enumerate(batch) if x[1] == "D"]
    rows, y, tmid = [], [], []

    def between(pairs, lo, hi, idx=[0]):
        s = 0
        return s

    # index event series for fast window sums
    def cum(pairs):
        ts = np.array([p[0] for p in pairs])
        cs = np.concatenate([[0.0], np.cumsum([p[1] for p in pairs])])
        return ts, cs

    def wsum(tc, lo, hi):
        ts, cs = tc
        i, j = np.searchsorted(ts, lo, "left"), np.searchsorted(ts, hi, "left")
        return cs[j] - cs[i]

    H, Dh, E = cum(h2d_tok), cum(d2h_tok), cum(evd_ms)
    Hn = cum([(t, 1) for t, _ in h2d_tok])
    for (i0, a), (i1, b) in zip(D, D[1:]):
        if a[5] < 5 or b[5] < 5:  # a queue at both ends
            continue
        if a[2] < 8 or b[2] < 8:
            continue
        pbs = [x for x in batch[i0 + 1:i1] if x[1] == "P"]
        dt = b[0] - a[0]
        if dt <= 0 or dt > 30:
            continue
        run = 0.5 * (a[2] + b[2])
        ctx = 0.5 * (a[3] + b[3]) / 1000.0
        rows.append([
            40.0, 40.0 * run, 40.0 * ctx,
            len(pbs), sum(x[3] for x in pbs), sum(x[3] * x[4] / 1000.0 for x in pbs),
            wsum(H, a[0], b[0]), wsum(Dh, a[0], b[0]), wsum(E, a[0], b[0]) / 1000.0,
        ])
        y.append(dt)
        tmid.append(0.5 * (a[0] + b[0]))
    X = np.array(rows)
    y = np.array(y)
    tmid = np.array(tmid)
    names = ["step_const_ms", "step_per_running_ms", "step_per_ctxK_ms", "prefill_batch_ms", "prefill_per_new_tok_ms",
             "prefill_per_new_x_cachedK_us", "h2d_per_tok_us", "d2h_per_tok_us", "evict_dev_ms_coef"]
    scale = [1e3, 1e3, 1e3, 1e3, 1e3, 1e6, 1e6, 1e6, 1.0]

    def fit(cols, label, mask=None):
        Xs = X[:, cols] if mask is None else X[mask][:, cols]
        ys = y if mask is None else y[mask]
        beta, res, rank, sv = np.linalg.lstsq(Xs, ys, rcond=None)
        pred = Xs @ beta
        ss = ((ys - pred) ** 2).sum()
        r2 = 1 - ss / ((ys - ys.mean()) ** 2).sum()
        # classic OLS standard errors
        dof = max(len(ys) - len(cols), 1)
        cov = ss / dof * np.linalg.inv(Xs.T @ Xs)
        se = np.sqrt(np.diag(cov))
        print(f"-- {label}: n={len(ys)} R2={r2:.3f} resid sd={np.sqrt(ss / dof) * 1e3:.0f} ms, mean interval {ys.mean():.3f} s")
        for c, b_, s_ in zip(cols, beta, se):
            print(f"   {names[c]:30s} {b_ * scale[c]:10.4f} +- {s_ * scale[c]:.4f}")
        # share of time each group explains
        tot = ys.sum()
        parts = (Xs * beta).sum(axis=0) / tot
        print("   share of time: " + ", ".join(f"{names[c]} {p * 100:.1f}%" for c, p in zip(cols, parts)))
        return beta

    if len(y) > 50:
        fit([0, 1, 2, 3, 4, 5], "without copies")
        fit([0, 1, 2, 3, 4, 5, 6], "with h2d tokens")
        fit([0, 1, 2, 3, 4, 5, 6, 7], "with h2d and d2h tokens")
        fit([0, 1, 2, 3, 4, 5, 6, 7, 8], "with h2d, d2h tokens and evict_device ms")
        late = tmid >= 70 * 60
        if late.sum() > 50:
            fit([0, 1, 2, 3, 4, 5, 6, 7], "minutes 70+: with h2d and d2h tokens", mask=late)
        mid = (tmid >= 28 * 60) & (tmid < 70 * 60)
        if mid.sum() > 50:
            fit([0, 1, 2, 3, 4, 5, 6, 7], "minutes 28-70: with h2d and d2h tokens", mask=mid)


if __name__ == "__main__":
    for spec in sys.argv[1:]:
        main(spec)
