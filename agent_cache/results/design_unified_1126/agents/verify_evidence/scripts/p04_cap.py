"""ev_prices 4: history lost by turns that ran out of lookups (hicache: 'reissue cap reached'; gate arms: capped)."""
import sys, os, re, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib, runtab
from vlib import kv
for camp, arm in [(34, "hicache"), (34, "ec_full"), (34, "jit_exact"), (34, "ec_exact"), (34, "jit_p50")]:
    tab = runtab.build(camp, arm); reqs = tab["reqs"]; ev = vlib.load(camp, arm)
    kinds = collections.Counter()
    capped = set()
    for t, r in ev.get("other", []):
        key = re.sub(r"[0-9a-f]{16,}|\d+", "#", r)[:60]
        kinds[key] += 1
        if "reissue cap reached" in r:
            m = re.search(r"req=([^\s,]+)", r) or re.search(r"rid=([^\s,]+)", r)
            if m: capped.add(m.group(1))
    for t, r in ev.get("ntp_gate", []):
        d = kv(r)
        if d.get("decision") == "capped": capped.add(d["rid"])
    for t, r in ev.get("ntp_reevict", []):
        d = kv(r)
        if d.get("capped") == "1": capped.add(d["rid"])
    lost = lambda q: max(q["ideal"] - q["cached"], 0)
    R = {rid: q for rid, q in reqs.items() if q["t_send"] < 6600}
    tot = sum(lost(q) for q in R.values() if lost(q) >= 256)
    cap = sum(lost(q) for rid, q in R.items() if lost(q) >= 256 and rid in capped)
    print(f"c{camp} {arm}: capped turns {len(capped)} (in client records before 6,600 s: {sum(1 for r in capped if r in R)}); lost >= 256 tokens: all {tot/1e6:.2f} M, by capped turns {cap/1e6:.3f} M ({100*cap/max(tot,1):.0f} %) over {sum(1 for rid,q in R.items() if lost(q)>=256 and rid in capped)} turns")
    if arm == "hicache":
        for k, v in kinds.most_common(8): print("     ", v, k)
        for lo, hi in [(0, 30), (30, 60), (60, 90), (90, 110)]:
            c = sum(lost(q) for rid, q in R.items() if lost(q) >= 256 and rid in capped and lo*60 <= q["t_send"] < hi*60)
            n = sum(1 for rid, q in R.items() if lost(q) >= 256 and rid in capped and lo*60 <= q["t_send"] < hi*60)
            print(f"      minutes {lo}-{hi}: capped turns' lost history {c/1e6:.1f} M over {n} turns")
