"""Calibration helper: GPU time per decode step and per prefill batch from a server log (read-only, streaming).

  TZ=UTC python -I -B calib_gpu_fit.py <run>/server.log <t_start of the client record, epoch seconds>

Intervals between consecutive 'Decode batch' lines hold 40 decode steps plus the prefill batches logged in between.
Under a full pool (token usage >= 0.90 at both ends) the interval length is regressed on 40 x [1, running, K tokens,
queue] and, per prefill batch, [1, K new tokens, K new x K prefix, K cached tokens]. Output used in CALIBRATION.md:
the pure-decode step (23.6 to 23.8 ms at 19 to 21 running), the non-decode time per prefilled token (0.061 to
0.063 ms), and the per-batch and per-cached-token terms (11 to 19 ms a batch, 0.6 to 1.3 ms per K cached tokens).
"""
import re, sys, numpy as np
from datetime import datetime
path = sys.argv[1]
t0 = float(sys.argv[2])   # client t_start (epoch)
def ts(s): return datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").timestamp()
pre = re.compile(r"^\[([\d\- :.]+)\] Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+), token usage: ([\d.]+), #running-req: (\d+), #queue-req: (\d+)")
dec = re.compile(r"^\[([\d\- :.]+)\] Decode batch, #running-req: (\d+), #token: (\d+), token usage: ([\d.]+).*#queue-req: (\d+)")
ev = []
import time, calendar
# server log is local time == UTC? derive offset from first Decode line vs t0 later; use naive timestamp() consistently
for line in open(path, errors="replace"):
    if "batch, #" not in line: continue
    m = pre.match(line)
    if m:
        g = m.groups(); ev.append(("p", ts(g[0]), int(g[1]), int(g[2]), int(g[3]), float(g[4]), int(g[5]), int(g[6]))); continue
    m = dec.match(line)
    if m:
        g = m.groups(); ev.append(("d", ts(g[0]), int(g[1]), int(g[2]), float(g[3]), int(g[4])))
print(len(ev), "batch lines")
rows = []
last_d = None; acc = None
for e in ev:
    if e[0] == "d":
        if last_d is not None and acc is not None:
            dt = e[1] - last_d[1]
            if 0 < dt < 30:
                run = (e[2] + last_d[2]) / 2; ktok = (e[3] + last_d[3]) / 2 / 1e3; q = (e[5] + last_d[5]) / 2
                rows.append((last_d[1] - t0, dt, run, ktok, q, acc["nb"], acc["new"] / 1e3, acc["newxc"] / 1e6, acc["cached"] / 1e3, acc["nseq"], min(e[4], last_d[4])))
        last_d = e; acc = dict(nb=0, new=0, newxc=0, cached=0, nseq=0)
    elif acc is not None:
        acc["nb"] += 1; acc["new"] += e[3]; acc["cached"] += e[4]; acc["nseq"] += e[2]
        acc["newxc"] += e[3] * (e[4] / max(1, e[2]))
R = np.array(rows)
print("intervals", len(R))
def fit(mask, label, cols):
    A = R[mask]
    names = ["40*1", "40*run", "40*ktok", "40*queue", "nbatch", "Knew", "Knew*Kprefix", "Kcached", "nseq"]
    X = np.column_stack([40 * np.ones(len(A)), 40 * A[:, 2], 40 * A[:, 3], 40 * A[:, 4], A[:, 5], A[:, 6], A[:, 7], A[:, 8], A[:, 9]])[:, cols]
    y = A[:, 1]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    r2 = 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"{label}: n={len(A)} R2={r2:.3f} total {y.sum():.0f} s; " + "; ".join(f"{names[c]}={b*1e3:.4f}ms" for c, b in zip(cols, beta)))
    # shares
    dec_t = (X[:, :sum(1 for c in cols if c < 4)] @ beta[:sum(1 for c in cols if c < 4)]).sum()
    print(f"      decode share {dec_t / y.sum():.3f}; prefill tokens {A[:,6].sum()/1e3:.2f} M in {A[:,5].sum():.0f} batches ({A[:,9].sum():.0f} seqs); non-decode per new token {(y.sum()-dec_t)/A[:,6].sum():.4f} ms; mean run {A[:,2].mean():.1f} ktok {A[:,3].mean():.0f} queue {A[:,4].mean():.1f}")
full = R[:, 10] >= 0.90
for lo, hi in ((1800, 3600), (3600, 5400), (5400, 6600), (1800, 6600)):
    m = full & (R[:, 0] >= lo) & (R[:, 0] < hi)
    fit(m, f"[{lo},{hi}) base", [0, 1, 2, 4, 5, 6])
    fit(m, f"[{lo},{hi}) +queue +cached", [0, 1, 2, 3, 4, 5, 6, 7])
pure = full & (R[:, 5] == 0)
for lo, hi in ((1800, 3600), (3600, 5400), (5400, 6600)):
    m = pure & (R[:, 0] >= lo) & (R[:, 0] < hi)
    A = R[m]
    print(f"pure decode intervals [{lo},{hi}): n={len(A)} ms/step median {np.median(A[:,1])/40*1e3:.2f} run {A[:,2].mean():.1f} ktok {A[:,3].mean():.0f} queue {A[:,4].mean():.1f}")
    fit(m, "   pure", [0, 1, 2, 3])
