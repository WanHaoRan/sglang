"""Cleanest cut: decode-log intervals (40 decode steps) that hold exactly ONE prefill batch with few new tokens.
Interval length against that batch's host-to-GPU copy time: the slope is the wall time one second of copy costs."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xlib

def main(spec):
    camp, arm = spec.split(":")
    d = xlib.load(camp, arm)
    batch = d["batch"]
    sub = d["ev"].get("h2d_submit", []); done = d["ev"].get("h2d_done", [])
    loads = [(ts, fd["ms"] / 1000.0, fs["tokens"]) for (ts, fs), (td, fd) in zip(sub, done) if fs["tokens"] == fd["tokens"]]
    lt = np.array([x[0] for x in loads]); lc = np.array([x[1] for x in loads])
    D = [(i, x) for i, x in enumerate(batch) if x[1] == "D"]
    rows = []
    pure = []
    for (i0, a), (i1, b) in zip(D, D[1:]):
        if a[5] < 5 or b[5] < 5 or a[2] < 12 or b[2] < 12 or not (1680 <= a[0] < 6600):
            continue
        dt = b[0] - a[0]
        if dt > 10:
            continue
        pbs = [x for x in batch[i0 + 1:i1] if x[1] == "P"]
        if not pbs:
            pure.append(dt)
            continue
        if len(pbs) != 1:
            continue
        p = pbs[0]
        i, j = np.searchsorted(lt, a[0]), np.searchsorted(lt, b[0])
        if j - i > 1:
            continue
        copy = float(lc[i:j].sum())
        rows.append((dt, p[3], copy, p[4], 0.5 * (a[2] + b[2])))
    A = np.array(rows)
    base = float(np.median(pure))
    print(f"== {spec}: pure-decode interval median {base:.3f} s (n={len(pure)}); intervals with exactly one prefill batch: {len(A)}")
    print("   new tokens  copy ms bin   n   median interval s   minus pure-decode median (ms)   median copy ms")
    for nlo, nhi in [(0, 512), (512, 2048), (2048, 100000)]:
        for clo, chi in [(0, 0.001), (0.001, 0.025), (0.025, 0.05), (0.05, 0.075), (0.075, 0.1), (0.1, 0.15), (0.15, 9)]:
            m = (A[:, 1] >= nlo) & (A[:, 1] < nhi) & (A[:, 2] >= clo) & (A[:, 2] < chi)
            if m.sum() >= 8:
                print(f"   {nlo:5d}-{nhi:<6d} {clo*1e3:5.0f}-{min(chi,9)*1e3:<5.0f} {int(m.sum()):4d}   {np.median(A[m,0]):.3f}   {1e3*(np.median(A[m,0])-base):7.1f}   {1e3*np.median(A[m,2]):6.1f}")
    m = (A[:, 1] < 512)
    X = np.column_stack([np.ones(m.sum()), A[m, 1], A[m, 2]])
    beta, *_ = np.linalg.lstsq(X, A[m, 0], rcond=None)
    res = A[m, 0] - X @ beta
    se = np.sqrt(np.diag((res ** 2).sum() / (m.sum() - 3) * np.linalg.inv(X.T @ X)))
    print(f"   OLS on intervals with one batch of < 512 new tokens (n={int(m.sum())}): interval = {beta[0]:.3f} + {beta[1]*1e6:.1f} us x new + {beta[2]:.3f}+-{se[2]:.3f} x copy_s")
    mm = m & (A[:, 2] > 0.025)
    X = np.column_stack([np.ones(mm.sum()), A[mm, 1], A[mm, 2]])
    beta, *_ = np.linalg.lstsq(X, A[mm, 0], rcond=None)
    res = A[mm, 0] - X @ beta
    se = np.sqrt(np.diag((res ** 2).sum() / (mm.sum() - 3) * np.linalg.inv(X.T @ X)))
    print(f"   same, copies over 25 ms only (n={int(mm.sum())}): slope on copy_s {beta[2]:.3f}+-{se[2]:.3f}")

for spec in sys.argv[1:]:
    main(spec)
