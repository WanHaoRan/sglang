"""Reviewer check 7: new KV produced per run (what write-through sends to SSD) and SSD read volume."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
B = 98304
for a in ("nofc_noend", "ec_r13", "jit_exact", "baseline"):
    R = Run(a)
    w = [r for r in R.every if r[F["t_send"]] < 6900]
    new = sum((r[F["prompt"]] or 0) - (r[F["cached"]] or 0) + (r[F["completion"]] or 0) for r in w)
    delta = sum((r[F["completion"]] or 0) for r in w)
    rec = sum(r[F["rec"]] for r in w)
    pages = sum(p for (c, t, tend, p, ms) in R.raw["io"] if tend - R.t0 < 6900)
    print(f"{a:11s} first 115 min: tokens computed (uncached prompt + completion) {new / 1e6:6.2f} M = {new * B / 1e12:5.2f} TB of KV if all written through ({new * B / 1e12 * 24 * 60 / 115:5.1f} TB/day); of it recomputed history {rec / 1e6:5.2f} M; SSD read {pages * 64 * B / 1e12:5.2f} TB ({pages} pages)")
