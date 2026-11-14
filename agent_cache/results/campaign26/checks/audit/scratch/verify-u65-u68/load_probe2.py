#!/usr/bin/env python3
"""Read-only: per-run band rows for a few runs (minutes, mean TTFT), to check the probe against
results/campaign23/gain44/a4_gpu_knee.out (hint_free) and to show how many minutes the top bands hold."""
import runpy, sys, io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    g = runpy.run_path(sys.argv[1])
M, BANDS, WARM, st = g["M"], g["BANDS"], g["WARM"], g["st"]
for n in ("hintfree", "baseline", "nofc_noend", "nofc_noend_rep", "jit_exact", "jit_exact_d", "ec_r13", "ec_r13_rep"):
    cells = []
    for lo, hi in BANDS:
        x = [r for r in M[n] if r["m"] >= WARM and lo <= r["load"] < hi]
        cells.append(f"{lo}-{hi}: {len(x):3d} min {st.mean(r['ttft'] for r in x):6.2f} s" if x else f"{lo}-{hi}:   0 min")
    print(f"{n:15s} " + " | ".join(cells))
