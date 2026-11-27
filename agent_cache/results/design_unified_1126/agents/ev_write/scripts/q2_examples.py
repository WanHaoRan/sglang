"""Examples: pages first seen missing within a few seconds of their write."""
import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import PAGE, load_trees
from model import load
from pages import build_pages
from q2_due import snapshots
camp, label = sys.argv[1], sys.argv[2]
amax = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
m = load(camp, label, load_trees()); p = build_pages(m)
snaps, _ = snapshots(m, p)
shown = 0; by_pair = collections.Counter(); same_sess = collections.Counter()
ex = {}
for kid, ws in enumerate(p.writes):
    if not ws: continue
    t_sub, t_done, op = ws[0]
    wr = m.ops[op]["rid"]; j = p.kj[kid]
    obs = []
    for u in p.uses[kid]:
        for t, res_pages in snaps[u]:
            if t > t_sub: obs.append((t, j < res_pages, u, res_pages))
    obs.sort()
    if obs and not obs[0][1] and obs[0][0] - t_done < amax:
        u = obs[0][2]
        rel = "same session next turn" if (p.rs[u].inst == m.reqs[wr].inst and p.rs[u].k == m.reqs[wr].k + 1) else ("same session" if p.rs[u].inst == m.reqs[wr].inst else ("same play" if p.rs[u].play == m.reqs[wr].play else "other play"))
        same_sess[rel] += 1
        key = (wr, p.rs[u].rid)
        by_pair[key] += 1
        ex.setdefault(key, (t_sub, t_done, obs[0][0], j, obs[0][3], m.ops[op]["p0"], m.ops[op]["tokens"]))
print(dict(same_sess))
for (wr, ob), n in by_pair.most_common(12):
    t_sub, t_done, t_obs, j, resp, p0, tk = ex[(wr, ob)]
    w = m.reqs[wr]; o = m.reqs[ob]
    print(f"{n:5d} pages  writer {wr} (P {w.P} out {w.out} cached {w.cached} t_first {w.t_first:.2f} done {w.t_done:.2f} last={w.last}) op p0 {p0} tokens {tk} t_sub {t_sub:.2f} t_done {t_done:.2f} | observer {ob} (P {o.P} cached {o.cached} dev/host/stor {o.dev}/{o.host}/{o.stor} t_send {o.t_send:.2f}) snapshot t {t_obs:.2f} resident pages {resp} page j {j}; reqstat entry {m.reqstat[ob]['entry']:.2f}")
