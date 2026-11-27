"""How long does a prefill pass take with and without a host-to-GPU load, per batch?

Each prefill pass that loads history logs: load_back_init (per request), h2d_submit (the merged copy), then,
once the forward's result is processed, 'Prefill batch' (new and cached tokens of that batch) and h2d_done
(ms = the copy's own duration on its CUDA stream). pass_s = t('Prefill batch') - t(h2d_submit).
The forward waits layer by layer on the copy (cache_controller.py LayerDoneCounter.wait_until), so
pass_s ~ max(copy, forward) + const. Fits:
  A  pass = c + b x new + g x copy_ms            (linear)
  B  pass = c + max(g x copy_ms, b x new)        (grid over b, least squares for c and g)
and reports the blocked time per batch: pass - (c + b x new), i.e. what the same batch would take if its
history were already on the GPU, per regime window, per admitted request and as a share of wall time.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xlib  # noqa: E402

WINDOWS = [(0, 28), (28, 70), (70, 90), (90, 110)]


def main(spec):
    camp, arm = spec.split(":")
    d = xlib.load(camp, arm)
    P = [x for x in d["batch"] if x[1] == "P"]
    pt = np.array([x[0] for x in P])
    sub = d["ev"].get("h2d_submit", [])
    done = d["ev"].get("h2d_done", [])
    # acks are in submit order
    rows = []
    for (ts, fs), (td, fd) in zip(sub, done):
        if fs["tokens"] != fd["tokens"]:
            continue
        i = np.searchsorted(pt, ts, "left")
        if i >= len(P):
            continue
        p = P[i]
        if p[0] - ts > 5.0:
            continue
        # the batch's cached tokens must cover the load
        if p[4] < fs["tokens"]:
            continue
        rows.append((ts, p[0] - ts, p[3], p[4], fs["tokens"], fd["ms"] / 1000.0, p[2], p[6], p[7]))
    A = np.array(rows)
    print(f"== {spec}: {len(sub)} loads, {len(A)} paired with their prefill batch")
    t, dur, new, cached, ltok, copy, nseq, running, queue = A.T
    sat = queue >= 5
    print(f"paired loads with a queue >= 5: {int(sat.sum())}")

    def ols(X, y):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ beta
        dof = max(len(y) - X.shape[1], 1)
        s2 = (res ** 2).sum() / dof
        se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
        r2 = 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()
        return beta, se, r2, np.sqrt(s2)

    for label, m in (("all", np.ones(len(A), bool)), ("queue>=5", sat), ("no queue", ~sat)):
        if m.sum() < 30:
            continue
        y = dur[m]
        X = np.column_stack([np.ones(m.sum()), new[m], copy[m]])
        beta, se, r2, sd = ols(X, y)
        print(f"-- A linear, {label}: n={int(m.sum())} R2={r2:.3f} sd={sd * 1e3:.1f} ms | const {beta[0] * 1e3:.1f}+-{se[0] * 1e3:.1f} ms, "
              f"per new token {beta[1] * 1e6:.2f}+-{se[1] * 1e6:.2f} us, per copy-second {beta[2]:.3f}+-{se[2]:.3f}")
        X2 = np.column_stack([np.ones(m.sum()), new[m], ltok[m]])
        beta2, se2, r22, sd2 = ols(X2, y)
        print(f"   (on loaded tokens instead of copy time: const {beta2[0] * 1e3:.1f} ms, per new token {beta2[1] * 1e6:.2f} us, "
              f"per loaded token {beta2[2] * 1e6:.3f}+-{se2[2] * 1e6:.3f} us, R2={r22:.3f})")
        best = None
        for b in np.arange(10e-6, 80e-6, 1e-6):
            for g in np.arange(0.5, 1.6, 0.05):
                mx = np.maximum(g * copy[m], b * new[m])
                c = float(np.mean(y - mx))
                ss = float(((y - mx - c) ** 2).sum())
                if best is None or ss < best[0]:
                    best = (ss, b, g, c)
        ss, b, g, c = best
        r2b = 1 - ss / ((y - y.mean()) ** 2).sum()
        print(f"-- B max form, {label}: R2={r2b:.3f} sd={np.sqrt(ss / len(y)) * 1e3:.1f} ms | const {c * 1e3:.1f} ms, per new token {b * 1e6:.0f} us, "
              f"copy weight {g:.2f}")
        if label == "queue>=5":
            fit_b, fit_g, fit_c = b, g, c
            lin = beta
    # blocked time per batch: measured pass minus the same batch with nothing to load
    for name, base in (("max-form baseline", fit_c + fit_b * new), ("linear baseline", lin[0] + lin[1] * new)):
        blocked = np.maximum(dur - base, 0.0)
        print(f"-- blocked time by window ({name}: pass - (const + per-token x new), floored at 0)")
        print("   window   loads  copy_s  blocked_s  blocked/copy  wall%   admitted  blocked ms/req  copy ms/req  loaded tok/req")
        for lo, hi in WINDOWS:
            a, b_ = lo * 60, hi * 60
            m = (t >= a) & (t < b_)
            nreq = sum(1 for r in d["rts"].values() if a <= r[6] + r[4] < b_)
            if m.sum() == 0:
                continue
            print(f"   {lo:3d}-{hi:<3d} {int(m.sum()):7d} {copy[m].sum():7.1f} {blocked[m].sum():9.1f} {blocked[m].sum() / copy[m].sum():10.2f} "
                  f"{100 * blocked[m].sum() / (b_ - a):6.2f} {nreq:9d} {1e3 * blocked[m].sum() / max(nreq, 1):12.1f} "
                  f"{1e3 * copy[m].sum() / max(nreq, 1):11.1f} {ltok[m].sum() / max(nreq, 1):12.0f}")
    # pass time against new tokens, by copy-time bin, to show the max() shape
    print("-- median pass ms by new-token bin (rows) and copy-ms bin (columns), queue >= 5")
    nb = [0, 256, 512, 1024, 2048, 4096, 8192, 1 << 30]
    cb = [0, 0.02, 0.04, 0.06, 0.08, 0.12, 0.2, 10]
    print("   new\\copy  " + "  ".join(f"{int(cb[j] * 1e3):>3d}-{int(min(cb[j + 1], 9.999) * 1e3):<4d}" for j in range(len(cb) - 1)))
    for i in range(len(nb) - 1):
        cells = []
        for j in range(len(cb) - 1):
            m = sat & (new >= nb[i]) & (new < nb[i + 1]) & (copy >= cb[j]) & (copy < cb[j + 1])
            cells.append(f"{np.median(dur[m]) * 1e3:5.0f}({int(m.sum()):4d})" if m.sum() >= 5 else "     -     ")
        print(f"   {nb[i]:5d}+    " + " ".join(cells))


if __name__ == "__main__":
    for spec in sys.argv[1:]:
        main(spec)
