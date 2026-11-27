"""Footprint of the false ends in ec_full: TTFT of the next requests of falsely ended turns against the same
requests in ec_exact (exact hints, no false ends), and the share of ec_full's lead deficit they carry."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib, runtab, chains
from vlib import kv, mean
A = runtab.build(34, "ec_full"); B = runtab.build(34, "ec_exact"); ch = chains.load_chains(34, "ec_full")
ev = vlib.load(34, "ec_full")
lastreg = {}
for t, r in ev["nue_register"]:
    d = kv(r); lastreg[d["rid"]] = d["next_use_ms"]
fn = set()
for rid, v in lastreg.items():
    if v == "ended" and rid in A["reqs"] and not ch[rid]["last"]:
        fn.add("%s-t%d" % (rid.rsplit("-t", 1)[0], A["reqs"][rid]["k"] + 1))
IB = {(q["play"], q["stream"], q["k"]): q for q in B["reqs"].values()}
for lo, hi in [(0, 110), (0, 28), (28, 70), (70, 110)]:
    d_f = []; d_o = []
    for rid, q in A["reqs"].items():
        if not (lo * 60 <= q["t_send"] < hi * 60): continue
        o = IB.get((q["play"], q["stream"], q["k"]))
        if o is None or o["t_send"] >= 6600: continue
        (d_f if rid in fn else d_o).append(q["ttft"] - o["ttft"])
    n = len(d_f) + len(d_o)
    print(f"minutes {lo}-{hi}: paired ec_full - ec_exact: next requests of false ends n {len(d_f)} mean {mean(d_f):+.2f} s; all others n {len(d_o)} mean {mean(d_o):+.3f} s; "
          f"overall {((sum(d_f)+sum(d_o))/n):+.3f} s, of which the false-end requests carry {sum(d_f)/n:+.3f} s")
