"""Slack of prompt-page writes by what the session did next: the kind and length of the gap before its next request.
'sure < x' = the page was seen missing less than x seconds after its write was acked."""
import sys, os, math, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import PAGE, load_trees
from model import load
from pages import build_pages
from q2_due import snapshots
trees = load_trees()
for label in sys.argv[2:]:
    m = load(sys.argv[1], label, trees); p = build_pages(m)
    snaps, _ = snapshots(m, p)
    nxt = {}
    for inst, lst in m.sessions.items():
        for a, b in zip(lst, lst[1:]):
            nxt[a.rid] = b
    agg = collections.defaultdict(lambda: collections.Counter())
    for kid, ws in enumerate(p.writes):
        if not ws: continue
        t_sub, t_done, op = ws[0]
        if t_done is None: continue
        wr = m.ops[op]["rid"]; r = m.reqs[wr]
        b = nxt.get(wr)
        if b is None:
            kind = "final request" if r.last else "run ended first"
        else:
            g = b.wait_s or 0.0
            kind = f"{b.gap}, gap " + ("under 5 s" if g < 5 else ("5-30 s" if g < 30 else ("30-120 s" if g < 120 else "over 120 s")))
        j = p.kj[kid]
        U = math.inf
        obs = []
        for u in p.uses[kid]:
            for t, res_pages in snaps[u]:
                if t > t_sub: obs.append((t, j < res_pages))
        if len(ws) > 1: obs.append((ws[1][0], False))
        obs.sort()
        for t, res in obs:
            if not res: U = t; break
        c = agg[kind]
        c["pages"] += 1
        for x in (1, 5, 30, 120):
            if U - t_done < x: c[x] += 1
        if U < math.inf: c["ever"] += 1
    print(f"{sys.argv[1]} {label}: prompt pages first written, by the session's next step | pages (M tokens) | seen missing within 1 s, 5 s, 30 s, 120 s of the write's ack | ever seen missing")
    for kind in sorted(agg, key=lambda k: -agg[k]["pages"]):
        c = agg[kind]; n = c["pages"]
        print(f"   {kind:28s} {n:7d} ({n * PAGE / 1e6:5.2f} M) | {c[1] / n:6.2%} {c[5] / n:6.2%} {c[30] / n:6.2%} {c[120] / n:6.2%} | {c['ever'] / n:6.1%}")
