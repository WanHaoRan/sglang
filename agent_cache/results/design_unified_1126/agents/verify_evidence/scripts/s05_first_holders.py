"""First requests of sessions: the pages they share with earlier sessions, and whether the earlier holders were
over when the first request was sent -- under three readings of 'holder'."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib, ssd_replay
arm = sys.argv[1] if len(sys.argv) > 1 else "ec_full"
D = ssd_replay.Data(34, arm)
reqs = D.reqs
evs = []
for rid, q in reqs.items():
    evs.append((q["t_send"] + q["latency"], 1, rid))
    evs.append((q["t_send"], 0, rid))
evs.sort()
writer = {}        # page -> first session whose request completed with the page in its prompt
any_hold = {}      # page -> sessions whose completed request held it
latest = {}        # session -> page set of its latest completed prompt
res = {k: [0, 0] for k in ("all holders ever", "writer", "holders whose latest prompt still has the page", "at least one holder over")}
n = tok = 0
for t, kind, rid in evs:
    s = D.sess[rid]; q = reqs[rid]
    if kind == 1:
        for i in D.keys[rid]:
            writer.setdefault(i, s)
            any_hold.setdefault(i, set()).add(s)
        latest[s] = set(D.keys[rid])
    elif q["k"] == 0:
        npg = q["ideal"] // 64
        if npg == 0: continue
        ks = D.keys[rid][:npg]
        n += 1; tok += npg * 64
        over = lambda h: D.ended_at.get(h, 1e18) < t
        hs = set(); ws = set(); cur = set()
        for i in ks:
            hs |= any_hold.get(i, set()); ws.add(writer[i])
            cur |= {h for h in any_hold.get(i, set()) if not over(h) and i in latest.get(h, ())}
        hs.discard(s)
        if all(over(h) for h in hs): res["all holders ever"][0] += 1; res["all holders ever"][1] += npg * 64
        if all(over(h) for h in ws): res["writer"][0] += 1; res["writer"][1] += npg * 64
        if not cur: res["holders whose latest prompt still has the page"][0] += 1; res["holders whose latest prompt still has the page"][1] += npg * 64
        if any(over(h) for h in hs): res["at least one holder over"][0] += 1; res["at least one holder over"][1] += npg * 64
print(f"c34 {arm}: first requests sharing >= 1 page with earlier sessions: {n} ({tok/1e6:.2f} M tokens)")
for k, (a, b) in res.items():
    print(f"   {k:50s}: {a:4d} requests, {b/1e6:.2f} M tokens ({100*b/tok:.0f} %)")
