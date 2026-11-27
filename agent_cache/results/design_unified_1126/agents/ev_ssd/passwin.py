"""When inside a pass do the deletions happen?  Lookups made during a logged pass whose pages the pass deleted:
found (the files were still there) or not.  Usage: python3 -I -B passwin.py <arm>"""
import sys
sys.path.insert(0, "/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/ev_ssd")
from replay import *
arm = sys.argv[1]
d = load_arm(arm)
# victims of each pass under deletion at the very end (frac 1.0), then test each lookup in the window
sim = Sim(d, {}, passes="logged", del_frac=1.0)
sim.run()
# rerun, capturing victim sets per pass
import types
vict = {}
sim2 = Sim(d, {}, passes="logged", del_frac=1.0)
orig = sim2.ev_pass_del
def cap(t, k, info):
    before = [p for p in range(d["n_ids"]) if sim2.mt[p] >= 0]
    orig(t, k, info)
    vict[k] = set(p for p in before if sim2.mt[p] < 0)
sim2.ev_pass_del = cap
sim2.run()
for k, (t_end, secs, groups, p0, p1) in enumerate(d["passes"]):
    snap = t_end - secs
    rows = []
    for r in d["reqs"]:
        for lk in r.lookups:
            if snap - 1 <= lk.t <= t_end + 1:
                end = min(lk.m + lk.n, len(lk.chain))
                span = lk.chain[lk.m:end]
                first_v = next((j for j, p in enumerate(span) if p in vict[k]), None)
                if first_v is None: continue
                rows.append(((lk.t - snap) / secs, r.rid, lk.m, lk.n, lk.hit, first_v))
    rows.sort()
    print(f"pass {k}: snapshot {snap:.1f}, end {t_end:.1f} ({secs} s)")
    for f, rid, m, n, hit, fv in rows:
        state = "files still there" if hit > fv else "already deleted"
        print(f"    at {f:5.2f} of the pass: {rid:22s} start {m} asked {n} hit {hit}; first victim page at offset {fv} -> {state}")
