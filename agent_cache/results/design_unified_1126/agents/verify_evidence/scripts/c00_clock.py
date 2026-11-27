"""Clock check: server log (UTC wall clock) against client seconds-since-t_start, per run."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib
from vlib import pct
for camp, arm in [(34, "ec_full"), (34, "jit_exact"), (34, "jit_p50"), (34, "ec_exact"), (34, "hicache"), (32, "ec_full"), (32, "jit_exact")]:
    ev = vlib.load(camp, arm)
    rs = vlib.req_stats(ev, camp, arm)
    cl = vlib.client(camp, arm)
    # (a) the ReqTimeStats line is printed at entry+queue+forward
    d1 = [r["fin"] - (r["entry"] + r["queue"] + r["fwd"]) for r in rs.values()]
    # (b) client send time against the server's entry time
    d2 = []; miss = 0
    for q in cl["reqs"]:
        r = rs.get(q["rid"])
        if r is None:
            miss += 1; continue
        d2.append(r["entry"] - q["t_send"])
    # (c) client ttft against queue + (first token) : ttft >= queue
    d3 = [q["ttft"] - rs[q["rid"]]["queue"] for q in cl["reqs"] if q["rid"] in rs and q.get("ttft") is not None]
    print(f"c{camp} {arm}: reqs client {len(cl['reqs'])} server {len(rs)} unmatched {miss} | "
          f"log-(entry+q+fwd) p50 {pct(d1,.5)*1e3:.1f} ms max|.| {max(abs(x) for x in d1)*1e3:.1f} ms | "
          f"entry-t_send p1 {pct(d2,.01)*1e3:.1f} p50 {pct(d2,.5)*1e3:.1f} p99 {pct(d2,.99)*1e3:.1f} ms | "
          f"ttft-queue p1 {pct(d3,.01):.3f} p50 {pct(d3,.5):.3f} s")
