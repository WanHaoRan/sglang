"""Referee: GPU work per window in a simulated run (prefill seconds, batches, new tokens, decode share), to set
against the measured decode-line regression (the builder's calib_gpu_fit.py output on the server logs)."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import reflib

for folder in sys.argv[1:]:
    cols, rows = reflib.read_csv(os.path.join(folder, "requests.csv"))
    ix = {c: i for i, c in enumerate(cols)}
    ta = np.array([float(r[ix["t_admit"]]) for r in rows])
    new = np.array([float(r[ix["prompt"]]) - float(r[ix["tok_gpu"]]) - float(r[ix["tok_host"]]) - float(r[ix["tok_ssd"]]) for r in rows])
    host = np.array([float(r[ix["tok_host"]]) + float(r[ix["tok_ssd"]]) for r in rows])
    out = np.array([float(r[ix["out"]]) for r in rows])
    scols, srows = reflib.read_csv(os.path.join(folder, "series.csv"))
    six = {c: i for i, c in enumerate(scols)}
    st = np.array([float(r[six["t"]]) for r in srows])
    pf = np.array([float(r[six["gpu_prefill_s"]]) for r in srows])
    nb = np.array([float(r[six["gpu_batches"]]) for r in srows])
    step = np.array([float(r[six["step_ms"]]) for r in srows])
    run = np.array([float(r[six["running"]]) for r in srows])
    ctx = np.array([float(r[six["ctx_run"]]) for r in srows])
    print(folder)
    for lo, hi in ((1800, 3600), (3600, 5400), (5400, 6600)):
        i0, i1 = np.searchsorted(st, lo), np.searchsorted(st, hi)
        dpf, dnb = pf[i1] - pf[i0], nb[i1] - nb[i0]
        w = (ta >= lo) & (ta < hi)
        ww = (st > lo) & (st <= hi)
        print(f"  [{lo},{hi}): prefill {dpf:.0f} s of {hi - lo} (decode share {1 - dpf / (hi - lo):.3f}); new tokens {new[w].sum() / 1e6:.2f} M "
              f"in {dnb:.0f} batches ({w.sum()} requests; {new[w].sum() / max(1, dnb):.0f} new per batch); non-decode per new token "
              f"{1e3 * dpf / new[w].sum():.4f} ms; loaded from host/SSD {host[w].sum() / 1e6:.1f} M; step median {np.median(step[ww]):.2f} ms, "
              f"running mean {run[ww].mean():.1f}, context {np.median(ctx[ww]) / 1e3:.0f} K; output tokens {out[w].sum() / 1e3:.0f} K")
