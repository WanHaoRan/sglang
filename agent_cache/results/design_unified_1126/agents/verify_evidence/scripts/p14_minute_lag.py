"""p13 with a lag: admitted_m ~ arm + b0 * recomputed_m + b1 * recomputed_(m-1), and the same with completions
(requests finished in the minute) and with output tokens generated per minute (decode throughput)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import vlib, runtab
arms = ("ec_full", "jit_exact", "ec_exact", "jit_p50")
X = []; Y = {"admitted": [], "finished": []}
for ai, arm in enumerate(arms):
    tab = runtab.build(34, arm); reqs = tab["reqs"]
    L = {}; A = {}; F = {}
    for q in reqs.values():
        m = int(q["adm"] // 60); A[m] = A.get(m, 0) + 1
        if q["ideal"] - q["cached"] >= 256: L[m] = L.get(m, 0) + (q["ideal"] - q["cached"]) / 1e3
        f = int(q["fin"] // 60); F[f] = F.get(f, 0) + 1
    for m in range(87, 110):
        row = [0.0] * len(arms); row[ai] = 1.0
        X.append(row + [L.get(m, 0.0), L.get(m - 1, 0.0)])
        Y["admitted"].append(A.get(m, 0)); Y["finished"].append(F.get(m, 0))
X = np.array(X)
for nm, y in Y.items():
    y = np.array(y, float); n = len(y)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b; s2 = (res ** 2).sum() / (n - X.shape[1]); cov = s2 * np.linalg.inv(X.T @ X)
    k = len(arms)
    tot = b[k] + b[k + 1]; se_tot = float(np.sqrt(cov[k, k] + cov[k + 1, k + 1] + 2 * cov[k, k + 1]))
    print(f"{nm:9s}: b0 {b[k]:+.4f} +- {np.sqrt(cov[k,k]):.4f}, b1 {b[k+1]:+.4f} +- {np.sqrt(cov[k+1,k+1]):.4f}; sum {tot:+.4f} +- {se_tot:.4f} requests per K tokens (prediction -0.093)")
