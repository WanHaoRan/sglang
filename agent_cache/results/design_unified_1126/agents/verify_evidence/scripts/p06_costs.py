"""Idle prefill cost and uncontended read cost (ev_prices 3a, ev_fetch 4b), my own selection.
 prefill: requests with queue wait < 20 ms, nothing from host or SSD (cached all on the GPU), at most 12
          running at the last batch line, sent in minutes 0-28; y = ttft - queue; x1 = new tokens (prompt -
          cached), x2 = sum of positions of the new tokens in K = (prompt^2 - cached^2)/2000.
 read:    reads of queued turns, never held, dispatched within 20 ms of the park, I/O started within 5 ms of the
          dispatch (read thread idle); y = pin - park; x = pages."""
import sys, os, bisect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import vlib, runtab

def ols(X, y):
    X = np.asarray(X, float); y = np.asarray(y, float)
    beta, res, rk, sv = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return beta, r2

Xp = []; yp = []; Xr = []; yr = []; nonio = []
per_arm = {}
for camp, arm in [(34, "ec_full"), (34, "jit_exact"), (34, "ec_exact"), (34, "jit_p50")]:
    tab = runtab.build(camp, arm); ev = vlib.load(camp, arm)
    reqs, reads, T = tab["reqs"], tab["reads"], tab["T"]
    bt = [(t, running) for t, kind, running, queue, usage, _, _ in vlib.batches(ev)]
    bts = [b[0] for b in bt]
    n0 = len(yp)
    for rid, q in reqs.items():
        if q["t_send"] >= 28 * 60 or q["ttft"] is None: continue
        if q["queue"] > 0.020 or q["host"] or q["sto"] or q["reads"]: continue
        j = bisect.bisect_right(bts, q["adm"]) - 1
        if j < 0 or bt[j][1] > 12: continue
        new = q["prompt"] - q["cached"]
        if new < 2000: continue
        Xp.append([1.0, new, (q["prompt"] ** 2 - q["cached"] ** 2) / 2000.0]); yp.append(q["ttft"] - q["queue"])
    n1 = len(yr)
    for r in reads:
        if r["staged"] or r["hold"] or r["pin"] is None or r.get("dead"): continue
        if r["disp"] - r["park"] > 0.020: continue
        if (r["io_end"] - r["io_ms"]) - r["disp"] > 0.005: continue
        Xr.append([1.0, r["pages"]]); yr.append(r["pin"] - r["park"]); nonio.append(r["pin"] - r["park"] - r["io_ms"])
    per_arm[arm] = (len(yp) - n0, len(yr) - n1)
print("samples per arm (prefill, reads):", per_arm)
b, r2 = ols(Xp, yp)
print(f"idle prefill (n {len(yp)}): {b[0]*1e3:.1f} ms + {b[1]*1e3:.4f} ms/token + {b[2]*1e6:.2f} us per token per K of position; R2 {r2:.4f}")
T = lambda L: b[0] + b[1] * L + b[2] * L * L / 2000.0
print(f"   cold prompt of 20 K: {T(20000):.2f} s; 30 K: {T(30000):.2f} s; 45 K: {T(45000):.2f} s ({T(45000)/45000*1e3:.4f} ms/token)")
b2, r22 = ols([[1.0, x[1]] for x in Xp], yp)
print(f"   linear only: {b2[0]*1e3:.1f} ms + {b2[1]*1e3:.4f} ms/token; R2 {r22:.4f}")
br, rr = ols(Xr, yr)
print(f"uncontended read (n {len(yr)}): {br[0]:.3f} s + {br[1]*1e3:.2f} ms/page = {br[1]*1e3/64:.4f} ms/token; R2 {rr:.3f}; non-I/O part median {np.median(nonio):.3f} s")
R = lambda L: br[0] + br[1] * L / 64.0
for L in (10000, 20000, 30000, 45000, 60000):
    print(f"   L = {L}: read {R(L):.2f} s  recompute on an idle GPU {T(L)-b[0]:.2f} s")
# crossover
import math
A = b[2] / 2000.0; Bc = b[1] - br[1] / 64.0; C = -br[0]
disc = Bc * Bc - 4 * A * C
print(f"   own crossover (recompute = read): L = {(-Bc + math.sqrt(disc)) / (2 * A):.0f} tokens")
