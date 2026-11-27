"""Pages seen missing from memory before their SSD write was acked: list them."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import PAGE, load_trees
from model import load
from pages import build_pages
from q2_due import snapshots
camp, label = sys.argv[1], sys.argv[2]
m = load(camp, label, load_trees()); p = build_pages(m)
snaps, _ = snapshots(m, p)
n = 0
for kid, ws in enumerate(p.writes):
    if not ws: continue
    t_sub, t_done, op = ws[0]
    if t_done is None: continue
    j = p.kj[kid]
    for u in p.uses[kid]:
        for t, res_pages in snaps[u]:
            if t_sub < t < t_done and j >= res_pages:
                n += 1
                if n <= 12:
                    o = m.ops[op]
                    print(f"page j {j} key {kid}: write op {op} node {o['node']} tokens {o['tokens']} p0 {o['p0']} by {o['rid']} t_sub {t_sub:.3f} t_done {t_done:.3f}; seen missing at {t:.3f} by snapshot on {p.rs[u].rid} (resident pages {res_pages})")
print("total page observations missing before the write was done:", n)
