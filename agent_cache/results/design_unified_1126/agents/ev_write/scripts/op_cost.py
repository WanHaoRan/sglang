"""Write-thread time per op by op size, with and without a read in progress (single, unmerged ops)."""
import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import PAGE, load_trees
from model import load
from q3_rw import Busy
camp, label = sys.argv[1], sys.argv[2]
m = load(camp, label, load_trees())
rb = Busy([(r["t0"], r["t1"]) for r in m.reads])
agg = collections.defaultdict(lambda: [0, 0.0, 0])
for o in m.ops.values():
    if o["t_io"] is None or o["group"] is not None: continue
    pages = o["io_pages"]; a, b = o["t_io"] - o["io_ms"] / 1000.0, o["t_io"]
    if b <= a: continue
    fr = rb.overlap(a, b) / (b - a)
    size = "1 page" if pages == 1 else ("2-4" if pages <= 4 else ("5-16" if pages <= 16 else ("17-64" if pages <= 64 else "65-128")))
    k = (size, "read in progress" if fr >= 0.5 else "no read")
    agg[k][0] += 1; agg[k][1] += o["io_ms"]; agg[k][2] += pages
print(f"{camp} {label}: unmerged write ops (groups in campaign 34 are left out): size, reads | ops, mean ms per op, ms per page")
for size in ("1 page", "2-4", "5-16", "17-64", "65-128"):
    for rd in ("no read", "read in progress"):
        n, ms, pg = agg[(size, rd)]
        if n: print(f"  {size:8s} {rd:17s} {n:6d} ops {ms / n:7.1f} ms/op {ms / pg:6.2f} ms/page")
