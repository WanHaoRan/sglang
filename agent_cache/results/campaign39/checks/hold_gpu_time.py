"""What a prefill pass costs the running turns, with and without the load hold.

    python3 hold_gpu_time.py <label>=<run folder>:<programs folder> ... [--from S] [--to S]

Per run, over the requests' clock (seconds since the client's start) in [from, to):
1. Decode-log intervals (40 decode steps between two "Decode batch" lines) that hold exactly one prefill batch:
   their length against that batch's host-to-GPU copy time. The slope is the wall time a second of copy costs the
   running turns: near 1 when the pass waits on the copy, near 0 when decode steps run beside it.
2. Per window: decode steps a second, prefill batches a second, copy seconds a second, and the held batches.
Reads server.log once per run (no cache: a run may still be writing it).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/critic/scripts")
import xlib  # noqa: E402


def load(run, prog):
    name = f"_tmp_hold_{os.path.basename(run)}"
    d = xlib.load("tmp", name, rebuild=True, run=run, prog=prog)
    os.remove(f"{xlib.EXT}/tmp_{name}.pkl")
    return d


def one(label, run, prog, lo, hi):
    d = load(run, prog)
    batch = d["batch"]
    sub = d["ev"].get("h2d_submit", [])
    done = d["ev"].get("h2d_done", [])
    held = d["ev"].get("load_hold", [])
    # A copy's submit and done lines pair in order; the done line carries its duration.
    loads = [(ts, fd["ms"] / 1000.0, fs["tokens"]) for (ts, fs), (td, fd) in zip(sub, done) if fs["tokens"] == fd["tokens"]]
    lt = np.array([x[0] for x in loads])
    lc = np.array([x[1] for x in loads])
    ltok = np.array([x[2] for x in loads])
    last = max((x[0] for x in batch), default=0.0)
    hi = min(hi, last)
    D = [(i, x) for i, x in enumerate(batch) if x[1] == "D"]
    rows, pure = [], []
    for (i0, a), (i1, b) in zip(D, D[1:]):
        if a[5] < 5 or b[5] < 5 or a[2] < 12 or b[2] < 12 or not (lo <= a[0] < hi):
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
        i, j = np.searchsorted(lt, a[0]), np.searchsorted(lt, b[0])
        if j - i > 1:
            continue
        rows.append((dt, pbs[0][3], float(lc[i:j].sum()), pbs[0][4], 0.5 * (a[2] + b[2])))
    print(f"== {label}: {os.path.basename(run)}; log reaches {last:.0f} s; window {lo:.0f} to {hi:.0f} s")
    if len(pure) < 8 or len(rows) < 30:
        print(f"   too few intervals yet (pure {len(pure)}, with one batch {len(rows)})")
    else:
        A = np.array(rows)
        base = float(np.median(pure))
        print(f"   pure-decode interval of 40 steps: median {base:.3f} s (n={len(pure)}); running turns median {np.median(A[:, 4]):.0f}")
        print("   new tokens   copy ms     n   interval minus pure (ms)   median copy ms")
        for nlo, nhi in [(0, 512), (512, 2048), (2048, 100000)]:
            for clo, chi in [(0, 0.001), (0.001, 0.025), (0.025, 0.05), (0.05, 0.075), (0.075, 0.1), (0.1, 0.15), (0.15, 9)]:
                m = (A[:, 1] >= nlo) & (A[:, 1] < nhi) & (A[:, 2] >= clo) & (A[:, 2] < chi)
                if m.sum() >= 8:
                    print(f"   {nlo:5d}-{nhi:<6d} {clo * 1e3:4.0f}-{min(chi, 9) * 1e3:<5.0f} {int(m.sum()):4d}   {1e3 * (np.median(A[m, 0]) - base):7.1f}   {1e3 * np.median(A[m, 2]):6.1f}")
        for name, m in (("under 512 new tokens", A[:, 1] < 512), ("under 2,048 new tokens", A[:, 1] < 2048)):
            if m.sum() >= 30:
                X = np.column_stack([np.ones(m.sum()), A[m, 1], A[m, 2]])
                beta, *_ = np.linalg.lstsq(X, A[m, 0], rcond=None)
                res = A[m, 0] - X @ beta
                se = np.sqrt(np.diag((res ** 2).sum() / (m.sum() - 3) * np.linalg.inv(X.T @ X)))
                print(f"   fit, one batch of {name} (n={int(m.sum())}): interval = {beta[0]:.3f} s + {beta[1] * 1e6:.1f} us x new tokens + ({beta[2]:.2f} +- {se[2]:.2f}) x copy seconds")
    # Accounting over the window.
    span = hi - lo
    if span <= 0:
        return
    dec = [x for x in batch if x[1] == "D" and lo <= x[0] < hi]
    pre = [x for x in batch if x[1] == "P" and lo <= x[0] < hi]
    m = (lt >= lo) & (lt < hi)
    big = m & (ltok >= 8192)
    h = [f for t, f in held if lo <= t < hi]
    steps = np.array([f["steps"] for f in h]) if h else np.array([])
    print(f"   window totals: decode steps/s {40 * len(dec) / span:.2f}; prefill batches/s {len(pre) / span:.2f}; copies/s {m.sum() / span:.2f}, "
          f"copy seconds per second {lc[m].sum() / span:.4f} (loads of 8,192 tokens or more: {lc[big].sum() / span:.4f}); mean copy {1e3 * lc[m].mean() if m.sum() else 0:.1f} ms")
    if len(steps):
        landed = sum(1 for f in h if f["landed"] == 1)
        print(f"   held batches: {len(h)} ({len(h) / span:.2f}/s), landed {landed} ({100 * landed / len(h):.1f} %), decode steps beside the copy: "
              f"mean {steps.mean():.2f}, median {np.median(steps):.0f}, p90 {np.percentile(steps, 90):.0f}, at the bound of 16: {int((steps >= 16).sum())}")
    else:
        print("   held batches: 0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--from", dest="lo", type=float, default=1680.0)
    ap.add_argument("--to", dest="hi", type=float, default=6600.0)
    a = ap.parse_args()
    for spec in a.runs:
        label, _, rest = spec.partition("=")
        run, _, prog = rest.partition(":")
        one(label, run, f"{run}/{prog}", a.lo, a.hi)


if __name__ == "__main__":
    main()
