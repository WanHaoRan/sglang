"""Check of the blocking model 'a prefill pass with a load takes max(d, C) + F' (C the copy's own time, d what is
left of the batch already on the GPU, F the forward) on the decode-log intervals: regress the interval length on
40 decode steps, the prefill batches and the sum of max(0, C - d0); the coefficient of the last should be 1."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xlib

def main(spec):
    camp, arm = spec.split(":")
    d = xlib.load(camp, arm)
    batch = d["batch"]
    done = d["ev"].get("h2d_done", [])
    sub = d["ev"].get("h2d_submit", [])
    loads = [(ts, fd["ms"] / 1000.0, fs["tokens"]) for (ts, fs), (td, fd) in zip(sub, done) if fs["tokens"] == fd["tokens"]]
    lt = np.array([x[0] for x in loads]); lc = np.array([x[1] for x in loads]); ltok = np.array([x[2] for x in loads])
    D = [(i, x) for i, x in enumerate(batch) if x[1] == "D"]
    print("==", spec, "loads", len(loads))
    for d0 in (0.0, 0.015, 0.020, 0.025, 0.030):
        blk = np.maximum(lc - d0, 0.0)
        cs = np.concatenate([[0.0], np.cumsum(blk)])
        rows, y, tm = [], [], []
        for (i0, a), (i1, b) in zip(D, D[1:]):
            if a[5] < 5 or b[5] < 5 or a[2] < 8 or b[2] < 8:
                continue
            dt = b[0] - a[0]
            if dt <= 0 or dt > 30:
                continue
            pbs = [x for x in batch[i0 + 1:i1] if x[1] == "P"]
            i, j = np.searchsorted(lt, a[0]), np.searchsorted(lt, b[0])
            rows.append([40.0, 40.0 * 0.5 * (a[2] + b[2]), 40.0 * 0.5 * (a[3] + b[3]) / 1e3, len(pbs), sum(x[3] for x in pbs), cs[j] - cs[i]])
            y.append(dt); tm.append(0.5 * (a[0] + b[0]))
        X = np.array(rows); y = np.array(y); tm = np.array(tm)
        for label, m in (("all", np.ones(len(y), bool)), ("min 28-70", (tm >= 1680) & (tm < 4200)), ("min 70-110", tm >= 4200)):
            Xs, ys = X[m], y[m]
            beta, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
            res = ys - Xs @ beta
            dof = len(ys) - Xs.shape[1]
            se = np.sqrt(np.diag((res ** 2).sum() / dof * np.linalg.inv(Xs.T @ Xs)))
            r2 = 1 - (res ** 2).sum() / ((ys - ys.mean()) ** 2).sum()
            share = (Xs[:, 5] * beta[5]).sum() / ys.sum()
            raw = Xs[:, 5].sum() / ys.sum()
            print(f"d0={d0*1e3:4.0f} ms {label:10s} n={len(ys):5d} R2={r2:.4f} blocked coef {beta[5]:.3f}+-{se[5]:.3f}  prefill batch {beta[3]*1e3:5.1f} ms  per new tok {beta[4]*1e6:5.1f} us"
                  f"  share of time: fitted {share*100:.1f}%  (sum of max(0,C-d0) itself {raw*100:.1f}%)")
    # totals by window with d0 = 20 ms
    print("blocked = sum over loads of max(0, C - 20 ms), by window:")
    for lo, hi in [(0, 28), (28, 70), (70, 90), (90, 110)]:
        a, b = lo * 60, hi * 60
        m = (lt >= a) & (lt < b)
        nreq = sum(1 for r in d["rts"].values() if a <= r[6] + r[4] < b)
        blk = np.maximum(lc[m] - 0.020, 0).sum()
        print(f"  {lo:3d}-{hi:<3d} loads {int(m.sum()):5d} copy {lc[m].sum():7.1f} s  blocked {blk:7.1f} s = {100*blk/(b-a):5.2f}% of wall time, {1e3*blk/max(nreq,1):5.1f} ms per admitted request ({nreq}); loaded {ltok[m].sum()/max(nreq,1):6.0f} tokens per request")

for spec in sys.argv[1:]:
    main(spec)
