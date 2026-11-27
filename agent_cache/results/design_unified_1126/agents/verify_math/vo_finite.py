"""verify_math: finite injections in the saturated closed loop with one FCFS server, many independent paths.
Run: CUDA_VISIBLE_DEVICES= nice -n 10 <venv python> -I -B vo_finite.py > vo_finite.out
"""
import heapq
import math

import numpy as np

SEED = 31_1126


def run(N, M, pause, svc, inject_at=None, delta=0.0):
    nxt = [(pause[u][0], u) for u in range(N)]
    heapq.heapify(nxt)
    k = [0] * N
    tot = 0.0
    t = 0.0
    injected = False
    while nxt:
        arr, u = heapq.heappop(nxt)
        if arr > t:
            t = arr
        s = svc[u][k[u]]
        if inject_at is not None and not injected and t + s > inject_at:
            s += delta
            injected = True
        t += s
        tot += t - arr
        k[u] += 1
        if k[u] < M:
            heapq.heappush(nxt, (t + pause[u][k[u]], u))
    return tot


print("== Finite injections, closed loop with one FCFS server (independent paths): rise of the summed response time per injected second ==")
for label, sig in (("equal work (1 s)", 0.0), ("lognormal work (sigma 0.5)", 0.5)):
    for N in (8, 32):
        for d in (0.01, 0.5, 2.0, 5.0):
            vals = []
            for rep in range(400):
                rng = np.random.default_rng([SEED, N, rep, int(sig * 10)])
                M = 120
                pause = rng.exponential(0.5 * N, size=(N, M + 1))
                svc = np.ones((N, M + 1)) if sig == 0 else rng.lognormal(-sig * sig / 2, sig, size=(N, M + 1))
                base = run(N, M, pause, svc)
                T0 = rng.uniform(0.25, 0.5) * M * N
                pert = run(N, M, pause, svc, inject_at=T0, delta=d)
                vals.append((pert - base) / d)
            vals = np.array(vals)
            print(f"   {label}, N = {N}, injection {d} s: mean {vals.mean():.1f} (se {vals.std(ddof=1) / math.sqrt(len(vals)):.1f}), median {np.median(vals):.1f}  [first-order value N = {N}]")
