"""Is the cleaner's recompute visible as lost throughput inside campaign 34?  Per minute (by admission time,
minutes 86-109, user cap binding): requests admitted, output tokens of the admitted, and history recomputed
(lost >= 256 tokens) by the requests admitted in that minute.  Pooled OLS with arm intercepts:
   admitted_m = a_arm + b * recomputed_K_m (+ c * new-text K) ;   the identity predicts b = -0.043/g per K tokens."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import vlib, runtab
rows = []
arms = ("ec_full", "jit_exact", "ec_exact", "jit_p50")
for ai, arm in enumerate(arms):
    tab = runtab.build(34, arm); reqs = tab["reqs"]
    for m in range(86, 110):
        R = [q for q in reqs.values() if m * 60 <= q["adm"] < (m + 1) * 60]
        lost = sum(max(q["ideal"] - q["cached"], 0) for q in R if q["ideal"] - q["cached"] >= 256)
        new = sum(q["prompt"] - q["ideal"] for q in R)
        out = sum(q["out"] for q in R)
        rows.append((ai, m, len(R), lost / 1e3, new / 1e3, out / 1e3))
rows = np.array(rows, float)
n = len(rows)
D = np.zeros((n, len(arms)))
for i, r in enumerate(rows): D[i, int(r[0])] = 1.0
y = rows[:, 2]
def fit(X, names):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    s2 = (res ** 2).sum() / (n - X.shape[1])
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return {nm: (b[len(arms) + i], se[len(arms) + i]) for i, nm in enumerate(names)}, float(np.sqrt(s2))
print(f"minutes 86-109, {n} arm-minutes: admitted per minute mean {y.mean():.1f} sd {y.std():.1f}; recomputed history per minute mean {rows[:,3].mean():.1f} K, max {rows[:,3].max():.0f} K")
f1, s = fit(np.hstack([D, rows[:, 3:4]]), ["recomputed K"])
print(f"  admitted ~ arm + recomputed:           b = {f1['recomputed K'][0]:+.4f} +- {f1['recomputed K'][1]:.4f} requests per K tokens (residual sd {s:.1f})")
f2, s = fit(np.hstack([D, rows[:, 3:5]]), ["recomputed K", "new text K"])
print(f"  admitted ~ arm + recomputed + new text: b = {f2['recomputed K'][0]:+.4f} +- {f2['recomputed K'][1]:.4f};  new text {f2['new text K'][0]:+.4f} +- {f2['new text K'][1]:.4f}")
g = 60.0 / y.mean()
print(f"  the identity's prediction: -0.043 s per K tokens / g ({g:.3f} s) = {-0.043/g:+.4f} requests per K tokens")
# bursts: minutes with >= 100 K recomputed against the other minutes of the same arm
hi = rows[rows[:, 3] >= 100]; lo = rows[rows[:, 3] < 20]
print(f"  minutes with >= 100 K recomputed: {len(hi)}, admitted {hi[:,2].mean():.1f} (recomputed {hi[:,3].mean():.0f} K); minutes with < 20 K: {len(lo)}, admitted {lo[:,2].mean():.1f}")
