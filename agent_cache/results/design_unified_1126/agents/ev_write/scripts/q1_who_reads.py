"""Of the prompt pages read back from the SSD: read by the session that wrote them, by another session that shares the
prefix, or both (tokens, and token-reads)."""
import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import PAGE, load_trees
from model import load, prev_req
from pages import build_pages
trees = load_trees()
for label in sys.argv[2:]:
    m = load(sys.argv[1], label, trees); p = build_pages(m)
    c = collections.Counter(); reads = collections.Counter()
    for kid, ws in enumerate(p.writes):
        if not ws or not p.reads[kid]: continue
        w_inst = m.reqs[m.ops[ws[0][2]]["rid"]].inst
        same = other = 0
        for t0, t1, ri in p.reads[kid]:
            if t0 < ws[0][0]: continue
            rd = m.reads[ri]
            rid = rd["rid"]
            r = m.reqs.get(rid)
            inst = r.inst if r is not None else (prev_req(m, rid).inst if prev_req(m, rid) else None)
            if inst == w_inst: same += 1
            else: other += 1
        if same + other == 0: continue
        c["same session only" if other == 0 else ("other sessions only" if same == 0 else "both")] += PAGE
        reads["by the writing session"] += same * PAGE; reads["by another session"] += other * PAGE
    tot = sum(c.values()); rt = sum(reads.values())
    print(f"{sys.argv[1]} {label}: prompt pages read back {tot / 1e6:.2f} M tokens: " + ", ".join(f"{k} {v / 1e6:.2f} M ({v / tot:.1%})" for k, v in c.most_common())
          + f"; token-reads {rt / 1e6:.1f} M: " + ", ".join(f"{k} {v / 1e6:.1f} M ({v / rt:.1%})" for k, v in reads.most_common()))
