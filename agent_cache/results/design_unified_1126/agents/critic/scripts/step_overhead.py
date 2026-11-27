"""Decode step time on pure-decode log intervals (40 steps, no prefill batch in between), fitted on running
requests and context, per arm: does the hinted engine's per-pass machinery (tick, read dispatch, gate) show up
against the LRU baseline arm at the same batch shape?"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xlib

rows_all = {}
for spec in sys.argv[1:]:
    camp, arm = spec.split(":")
    d = xlib.load(camp, arm)
    D = [(i, x) for i, x in enumerate(d["batch"]) if x[1] == "D"]
    rows = []
    for (i0, x), (i1, y) in zip(D, D[1:]):
        if any(p[1] == "P" for p in d["batch"][i0 + 1:i1]):
            continue
        dt = y[0] - x[0]
        if dt > 5 or x[2] < 12 or y[2] < 12 or x[5] < 5:
            continue
        if not (1680 <= x[0] < 6600):
            continue
        # skip cleaner passes
        if any(t_end - secs - 1 <= x[0] <= t_end + 1 for t_end, g, f, gib, secs in d["cleanup"]):
            continue
        rows.append((dt / 40 * 1e3, 0.5 * (x[2] + y[2]), 0.5 * (x[3] + y[3]) / 1e3, x[5], x[0]))
    A = np.array(rows)
    rows_all[spec] = A
    X = np.column_stack([np.ones(len(A)), A[:, 1], A[:, 2]])
    beta, *_ = np.linalg.lstsq(X, A[:, 0], rcond=None)
    res = A[:, 0] - X @ beta
    at = beta[0] + beta[1] * 20 + beta[2] * 640
    print(f"{spec:14s} n={len(A):4d} step ms = {beta[0]:.2f} + {beta[1]:.3f} x running + {beta[2]:.4f} x ctxK (sd {res.std():.2f}); at 20 running, 640K: {at:.2f} ms; "
          f"mean running {A[:,1].mean():.1f} ctx {A[:,2].mean():.0f}K queue {A[:,3].mean():.0f}; raw median {np.median(A[:,0]):.2f}")
# pooled fit with an arm dummy against the first spec
specs = list(rows_all)
base = specs[0]
for other in specs[1:]:
    A, B = rows_all[base], rows_all[other]
    y = np.concatenate([A[:, 0], B[:, 0]])
    X = np.column_stack([np.ones(len(y)), np.concatenate([A[:, 1], B[:, 1]]), np.concatenate([A[:, 2], B[:, 2]]),
                         np.concatenate([np.zeros(len(A)), np.ones(len(B))]), np.concatenate([A[:, 3], B[:, 3]])])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    se = np.sqrt(np.diag((res ** 2).sum() / (len(y) - 5) * np.linalg.inv(X.T @ X)))
    print(f"pooled {base} + {other}: arm effect of {other} = {beta[3]:+.3f} +- {se[3]:.3f} ms per step; per queued request {beta[4]*1e3:+.2f} +- {se[4]*1e3:.2f} us")
