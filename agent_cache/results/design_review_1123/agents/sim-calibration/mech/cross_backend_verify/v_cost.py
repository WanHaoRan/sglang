"""GPU cost per prefill token on each backend, reviewer's own estimates.
 (1) idle cold prompts: first requests (k=0, nothing cached) of 18-30 K tokens whose prefill ran with <= 1 other request
     running and an empty queue: TTFT / prompt tokens, and TTFT - queue wait.
 (2) busy intervals between consecutive decode log lines (40 decode steps each) in [lo, hi) s with running >= 8 at both
     ends: OLS of wall time on [1, running, K tokens, K prefill tokens, prefill batches] -> decode step cost and prefill
     cost per token; then the prefill share of wall time.
 (3) the same OLS with a prefix term (sum new x cached of the logged prefill lines, K x K).
usage: v_cost.py lo hi"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v_lib import *

lo, hi = float(sys.argv[1]), float(sys.argv[2])


def ols(X, y):
    k = len(X[0])
    A = [[0.0] * k for _ in range(k)]
    b = [0.0] * k
    for xi, yi in zip(X, y):
        for i in range(k):
            b[i] += xi[i] * yi
            for j in range(k):
                A[i][j] += xi[i] * xi[j]
    # gaussian elimination
    M = [A[i] + [b[i]] for i in range(k)]
    for c in range(k):
        p = max(range(c, k), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(k):
            if r != c:
                f = M[r][c] / M[c][c]
                for cc in range(c, k + 1):
                    M[r][cc] -= f * M[c][cc]
    beta = [M[i][k] / M[i][i] for i in range(k)]
    yhat = [sum(bb * xx for bb, xx in zip(beta, xi)) for xi in X]
    my = sum(y) / len(y)
    ss_res = sum((a - b_) ** 2 for a, b_ in zip(y, yhat))
    ss_tot = sum((a - my) ** 2 for a in y)
    return beta, 1 - ss_res / ss_tot


for be in ("tri", "fa3"):
    # (1) idle cold prompts
    cold = []
    for arm in ARMS:
        d = load(f"{arm}_{be}")
        dec = d["dec"]
        pre = d["pre"]
        for r in reqs(d):
            if r["k"] != 0 or not r["prompt"] or not (18000 <= r["prompt"] <= 30000) or (r["cached"] or 0) != 0 or r["ttft"] is None:
                continue
            ts, te = r["t_send"], r["t_send"] + r["ttft"]
            pl = [p for p in pre if ts - 0.05 <= p[0] <= te]
            if not pl or max(p[4] for p in pl) > 1 or max(p[5] for p in pl) > 0:
                continue
            q = d["rts"].get(r["rid"], [None] * 5)[4]
            cold.append((r["prompt"], r["ttft"], q / 1e3 if q is not None else 0.0, arm))
        drop(f"{arm}_{be}")
    if cold:
        tok = sum(c[0] for c in cold)
        print(f"{be}: idle cold prompts 18-30K: n={len(cold)}, median prompt {med(c[0] for c in cold):.0f}, median TTFT {med(c[1] for c in cold):.3f} s, "
              f"sum TTFT/sum tokens {1e3 * sum(c[1] for c in cold) / tok:.4f} ms/token, (TTFT-queue) {1e3 * sum(c[1] - c[2] for c in cold) / tok:.4f} ms/token, "
              f"20K-equivalent {20 * sum(c[1] - c[2] for c in cold) / tok * 1e3:.3f} s")
    # (2) busy intervals
    X, X6, y, meta = [], [], [], []
    for arm in HINTED:
        d = load(f"{arm}_{be}")
        dec = d["dec"]
        pre = d["pre"]
        j = 0
        for a, b in zip(dec, dec[1:]):
            if not (lo <= a[0] and b[0] < hi):
                continue
            dt = b[0] - a[0]
            if dt <= 0 or dt > 30 or a[1] < 8 or b[1] < 8:
                continue
            while j < len(pre) and pre[j][0] <= a[0]:
                j += 1
            k = j
            new = cachedx = 0.0
            nb = 0
            while k < len(pre) and pre[k][0] <= b[0]:
                new += pre[k][2]; cachedx += pre[k][2] * pre[k][3]; nb += 1
                k += 1
            run = (a[1] + b[1]) / 2
            tokk = (a[2] + b[2]) / 2 / 1e3
            X.append([1.0, run, tokk, new / 1e3, nb])
            X6.append([1.0, run, tokk, new / 1e3, nb, cachedx / 1e6])
            y.append(dt)
            meta.append((arm, a[0], new, nb, run, tokk))
        drop(f"{arm}_{be}")
    beta, r2 = ols(X, y)
    beta6, r26 = ols(X6, y)
    c, br, bt, bp, bn = beta
    print(f"{be}: {len(y)} busy decode-log intervals in [{lo:.0f},{hi:.0f}) s; dt = {c:.4f} + {br:.5f} x running + {bt:.6f} x Ktok + {bp:.5f} x Kprefill + {bn:.4f} x batches; R2 {r2:.3f}")
    print(f"    decode step at 22 running, 635 K tokens: {1e3 * (c + br * 22 + bt * 635) / 40:.2f} ms;  prefill per token {bp:.5f} ms (+{1e3 * bn:.1f} ms a batch)")
    tot_new = sum(m[2] for m in meta); tot_nb = sum(m[3] for m in meta); tot_t = sum(y)
    pre_t = bp * tot_new / 1e3 + bn * tot_nb
    print(f"    prefill tokens {tot_new / 1e6:.1f} M in {tot_nb} batches; prefill share of busy wall time {100 * pre_t / tot_t:.1f} %; "
          f"all-in prefill cost {1e3 * pre_t / tot_new:.4f} ms/token")
    print(f"    with prefix term: {', '.join(f'{v:.5f}' for v in beta6)}; R2 {r26:.3f}; per token at prefix 0 {beta6[3]:.5f} ms, at prefix 16K {beta6[3] + beta6[5] * 16:.5f} ms, at 25K {beta6[3] + beta6[5] * 25:.5f} ms")
    # pure decode intervals only (no prefill): decode model
    Xd = [x[:3] for x, m in zip(X, meta) if m[3] == 0]
    yd = [yy for yy, m in zip(y, meta) if m[3] == 0]
    if len(yd) > 50:
        bd, r2d = ols(Xd, yd)
        res = [yy - (bd[0] + bd[1] * x[1] + bd[2] * x[2]) for x, yy in zip(X, y)]
        rp = sum(r for r, m in zip(res, meta) if m[3] > 0)
        newp = sum(m[2] for m in meta if m[3] > 0)
        print(f"    pure-decode intervals {len(yd)}: step at 22 running/635K {1e3 * (bd[0] + bd[1] * 22 + bd[2] * 635) / 40:.2f} ms (R2 {r2d:.3f}); "
              f"residual of prefill intervals / their prefill tokens = {1e3 * rp / newp:.4f} ms/token")
