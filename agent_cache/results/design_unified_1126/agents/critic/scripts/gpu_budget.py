"""Wall-clock budget of the GPU at saturation: decode steps (step time taken from decode-log intervals that
hold no prefill batch), prefill forward, and the time prefill passes wait on the host-to-GPU copy."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xlib

def main(spec):
    camp, arm = spec.split(":")
    d = xlib.load(camp, arm)
    batch = d["batch"]
    done = d["ev"].get("h2d_done", []); sub = d["ev"].get("h2d_submit", [])
    loads = [(ts, fd["ms"] / 1000.0, fs["tokens"]) for (ts, fs), (td, fd) in zip(sub, done) if fs["tokens"] == fd["tokens"]]
    lt = np.array([x[0] for x in loads]); lc = np.array([x[1] for x in loads])
    D = [(i, x) for i, x in enumerate(batch) if x[1] == "D"]
    print("==", spec)
    print("window  intervals  pure-decode intervals: n, step ms p50 (p10-p90) | wall s = 40 x step x intervals + prefill passes;  prefill batches, new tok,"
          " blocked s (sum max(0,C-20ms)) | implied prefill forward us/new token if blocking is full / half / none | reqs, g ms/req, blocked % of g")
    for lo, hi in [(28, 70), (70, 90), (90, 110)]:
        a, b = lo * 60, hi * 60
        pure, n_int, wall, npb, newtok = [], 0, 0.0, 0, 0
        for (i0, x), (i1, y) in zip(D, D[1:]):
            if not (a <= x[0] < b) or x[5] < 5 or y[5] < 5:
                continue
            dt = y[0] - x[0]
            if dt > 30:
                continue
            pbs = [p for p in batch[i0 + 1:i1] if p[1] == "P"]
            n_int += 1; wall += dt; npb += len(pbs); newtok += sum(p[3] for p in pbs)
            if not pbs:
                pure.append(dt / 40.0 * 1e3)
        if not pure:
            continue
        step = xlib.pct(pure, .5) / 1e3
        m = (lt >= a) & (lt < b)
        blk = float(np.maximum(lc[m] - 0.020, 0).sum()) * wall / (b - a)  # scale to the intervals kept
        dec = 40 * step * n_int
        rest = wall - dec
        nreq = sum(1 for r in d["rts"].values() if a <= r[6] + r[4] < b) * wall / (b - a)
        print(f"{lo:3d}-{hi:<3d} {n_int:6d}   {len(pure):5d}  {step*1e3:5.2f} ({xlib.pct(pure,.1):.1f}-{xlib.pct(pure,.9):.1f}) | wall {wall:7.1f} = decode {dec:7.1f} + {rest:6.1f};  "
              f"{npb:5d} {newtok/1e6:6.2f}M blocked {blk:6.1f} | {1e6*(rest-blk)/newtok:5.1f} / {1e6*(rest-blk/2)/newtok:5.1f} / {1e6*rest/newtok:5.1f} | "
              f"{nreq:6.0f} {1e3*wall/nreq:6.1f} {100*blk/wall:5.1f}%")

for spec in sys.argv[1:]:
    main(spec)
