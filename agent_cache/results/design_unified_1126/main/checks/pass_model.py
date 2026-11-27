"""Which model of a blocked prefill pass fits the logs: the copy added to the compute, or the larger of the two?

    python3 pass_model.py <label>=<run folder>:<programs folder> [--from S] [--to S]

Decode-log intervals (40 decode steps) that hold exactly one prefill batch; y = interval minus the median pure-decode
interval = what the batch cost the running turns. r = the batch's host-to-GPU copy time (logged), n = its new tokens.
Models: additive y = a + k n + b r; meeting y = a + max(r, p0 + k n) with the pass's compute p = p0 + k n
(Proposition 23: the pass consumes layers in order and waits for a layer that has not landed).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/critic/scripts")
import xlib  # noqa: E402


def rows_of(d, lo, hi):
    batch = d["batch"]
    sub, done = d["ev"].get("h2d_submit", []), d["ev"].get("h2d_done", [])
    loads = [(ts, fd["ms"] / 1000.0) for (ts, fs), (td, fd) in zip(sub, done) if fs["tokens"] == fd["tokens"]]
    lt = np.array([x[0] for x in loads]); lc = np.array([x[1] for x in loads])
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
            pure.append(dt); continue
        if len(pbs) != 1:
            continue
        i, j = np.searchsorted(lt, a[0]), np.searchsorted(lt, b[0])
        if j - i > 1:
            continue
        rows.append((dt, pbs[0][3], float(lc[i:j].sum())))
    A = np.array(rows)
    return A[:, 0] - float(np.median(pure)), A[:, 1], A[:, 2], len(pure)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--from", dest="lo", type=float, default=1680.0)
    ap.add_argument("--to", dest="hi", type=float, default=6600.0)
    x = ap.parse_args()
    label, _, rest = x.run.partition("=")
    run, _, prog = rest.partition(":")
    name = f"_tmp_pass_{os.path.basename(run)}"
    d = xlib.load("tmp", name, rebuild=True, run=run, prog=f"{run}/{prog}")
    os.remove(f"{xlib.EXT}/tmp_{name}.pkl")
    y, n, r, npure = rows_of(d, x.lo, x.hi)
    keep = (r > 0.001) & (np.abs(y) < 1.0)
    y, n, r = y[keep], n[keep], r[keep]
    print(f"== {label}: {len(y)} intervals with one prefill batch and a copy ({npure} pure); copy median {1e3 * np.median(r):.0f} ms, new tokens median {np.median(n):.0f}")
    X = np.column_stack([np.ones(len(y)), n, r])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    print(f"   additive: y = {1e3 * beta[0]:.1f} ms + {1e6 * beta[1]:.1f} us x new + {beta[2]:.2f} x copy; rms error {1e3 * np.sqrt((res ** 2).mean()):.1f} ms, median abs {1e3 * np.median(np.abs(res)):.1f} ms")
    best = None
    for p0 in np.arange(0.0, 0.0605, 0.002):
        for k in np.arange(4e-6, 40.5e-6, 1e-6):
            m = np.maximum(r, p0 + k * n)
            a = float(np.median(y - m))
            e = float(np.median(np.abs(y - m - a)))
            if best is None or e < best[0]:
                best = (e, p0, k, a)
    e, p0, k, a = best
    m = np.maximum(r, p0 + k * n)
    res2 = y - m - a
    print(f"   meeting:  y = {1e3 * a:.1f} ms + max(copy, {1e3 * p0:.0f} ms + {1e6 * k:.0f} us x new); rms error {1e3 * np.sqrt((res2 ** 2).mean()):.1f} ms, median abs {1e3 * e:.1f} ms")
    wait = np.maximum(0.0, r - (p0 + k * n))
    print(f"   under the meeting model the pass waits (copy - compute)^+ = {1e3 * wait.mean():.1f} ms a batch on average ({1e3 * np.median(wait):.1f} median); "
          f"copy longer than compute in {100 * (wait > 0).mean():.0f} % of batches; compute mean {1e3 * (p0 + k * n).mean():.1f} ms, copy mean {1e3 * r.mean():.1f} ms")
    for nlo, nhi in ((0, 512), (512, 2048), (2048, 10 ** 6)):
        s = (n >= nlo) & (n < nhi)
        if s.sum() >= 20:
            print(f"   new tokens {nlo}-{nhi}: n={int(s.sum())}; median abs error additive {1e3 * np.median(np.abs(res[s])):.1f} ms, meeting {1e3 * np.median(np.abs(res2[s])):.1f} ms; mean error additive {1e3 * res[s].mean():+.1f}, meeting {1e3 * res2[s].mean():+.1f}")


if __name__ == "__main__":
    main()
