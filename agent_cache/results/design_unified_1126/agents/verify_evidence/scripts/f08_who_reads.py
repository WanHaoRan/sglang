"""ev_fetch 2b: who reads from the SSD -- the recorded pause before the request, by window."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib, runtab
from vlib import pct
for arm in ("ec_full", "jit_exact"):
    tab = runtab.build(34, arm); reqs, stall, reads = tab["reqs"], tab["stall"], tab["reads"]
    print(f"c34 {arm}")
    for lo, hi in [(0, 30), (30, 70), (70, 110)]:
        R = [(rid, q) for rid, q in reqs.items() if q["k"] >= 1 and lo * 60 <= q["t_send"] < hi * 60]
        isr = lambda rid, q: bool(q["reads"]) or q["sto"] > 0
        rg = [q["wait_s"] or 0.0 for rid, q in R if isr(rid, q)]
        long_ = [(rid, q) for rid, q in R if (q["wait_s"] or 0) >= 60]; short = [(rid, q) for rid, q in R if (q["wait_s"] or 0) < 10]
        print(f"   minutes {lo}-{hi}: readers {len(rg)} of {len(R)}; median pause before a reader {pct(rg,.5):.1f} s; of requests after a pause >= 60 s, {100*sum(1 for rid,q in long_ if isr(rid,q))/max(len(long_),1):.0f} % read ({len(long_)}); after < 10 s, {100*sum(1 for rid,q in short if isr(rid,q))/max(len(short),1):.1f} % ({len(short)})")
