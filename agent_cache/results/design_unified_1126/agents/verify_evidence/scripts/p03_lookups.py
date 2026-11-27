"""ev_prices 4 (F5): lookups that find nothing, by half hour; turns whose every lookup found nothing; the
direct cost of an empty lookup at light load (queue wait of returning turns with and without one)."""
import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib, runtab
from vlib import kv, mean, pct
for camp, arm in [(34, "ec_full"), (34, "jit_exact"), (34, "ec_exact"), (34, "jit_p50"), (34, "hicache")]:
    tab = runtab.build(camp, arm); reqs = tab["reqs"]; ev = vlib.load(camp, arm)
    per = collections.defaultdict(list)
    wins = [(0, 30), (30, 60), (60, 90), (90, 110)]
    cnt = {w: [0, 0] for w in wins}
    for t, r in ev["s2h_query"]:
        d = kv(r); rid = d["rid"]
        hit = int(d["storage_hit"]); req = int(d["tokens_req"])
        per[rid].append((t, req, hit))
        for w in wins:
            if w[0] * 60 <= t < w[1] * 60:
                cnt[w][0] += 1; cnt[w][1] += (hit == 0)
    print(f"=== c{camp} {arm}: lookups {len(ev['s2h_query'])}; share finding nothing by half hour: " +
          ", ".join(f"{w[0]}-{w[1]}: {100*c[1]/max(c[0],1):.0f} % ({c[1]} of {c[0]})" for w, c in cnt.items()))
    allempty = sum(1 for rid, v in per.items() if rid in reqs and all(h == 0 for _, _, h in v))
    print(f"  turns whose every lookup found nothing: {allempty} of {sum(1 for rid in per if rid in reqs)} turns with a lookup")
    # direct cost at light load: returning turns (k>=1) sent in minutes 0-28, queue wait by 'had an empty lookup and no read'
    for lo, hi in [(0, 28)]:
        a = []; b = []
        for rid, q in reqs.items():
            if q["k"] < 1 or not (lo * 60 <= q["t_send"] < hi * 60) or q["reads"]:
                continue
            v = per.get(rid)
            if v and all(h == 0 for _, _, h in v):
                a.append(q["queue"])
            elif not v:
                b.append(q["queue"])
        n_ret = sum(1 for q in reqs.values() if q["k"] >= 1 and lo * 60 <= q["t_send"] < hi * 60)
        if a and b:
            print(f"  minutes {lo}-{hi}, returning turns with no read: with an empty lookup n={len(a)} ({100*len(a)/n_ret:.0f} % of returning) queue wait mean {1e3*mean(a):.1f} ms median {1e3*pct(a,.5):.1f} ms;"
                  f" with no lookup n={len(b)} mean {1e3*mean(b):.1f} ms median {1e3*pct(b,.5):.1f} ms; difference {1e3*(mean(a)-mean(b)):.1f} ms -> {1e3*(mean(a)-mean(b))*len(a)/n_ret:.1f} ms per returning turn")
