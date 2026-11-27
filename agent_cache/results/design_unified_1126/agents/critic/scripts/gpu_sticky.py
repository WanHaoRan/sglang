"""How many returning turns reach the scheduler before the next admission after their previous turn finished
(i.e. while their KV is plausibly still on the GPU), by window? Proxy: the first admission after a finish evicts
the finished request's GPU copy when the pool is full (pool use >= 0.95 on the surrounding batch lines)."""
import os, sys, bisect, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xlib

RID = re.compile(r"^(.*-c\d+)-t(\d+)$")
for spec in sys.argv[1:]:
    camp, arm = spec.split(":")
    d = xlib.load(camp, arm)
    rts = d["rts"]
    adm = sorted(r[6] + r[4] for r in rts.values())
    print("==", spec)
    print("window  returning  back before the next admission  share | their loaded tokens (K, mean) | gap from finish to arrival p10 p50 | time finish->next admission p50 p90")
    lb = {}
    for t, f in d["ev"].get("load_back_init", []):
        lb[str(f.get("rid"))] = lb.get(str(f.get("rid")), 0) + f.get("tokens", 0)
    for lo, hi in [(0, 28), (28, 70), (70, 90), (90, 110)]:
        a, b = lo * 60, hi * 60
        n = k = 0
        gaps, nxt, ltok = [], [], []
        for rid, r in rts.items():
            m = RID.match(rid)
            if not m or int(m[2]) == 0:
                continue
            prev = rts.get(f"{m[1]}-t{int(m[2]) - 1}")
            if prev is None:
                continue
            entry = r[6]
            if not (a <= entry < b):
                continue
            fin = prev[0]
            i = bisect.bisect_right(adm, fin)
            if i >= len(adm):
                continue
            n += 1
            gaps.append(entry - fin)
            nxt.append(adm[i] - fin)
            if entry < adm[i]:
                k += 1
                ltok.append(lb.get(rid, 0) / 1e3)
        if n:
            print(f"{lo:3d}-{hi:<3d} {n:8d} {k:8d} {k/n:6.3f} | {xlib.mean(ltok):6.1f} | {xlib.pct(gaps,.1):.2f} {xlib.pct(gaps,.5):.2f} | {xlib.pct(nxt,.5):.2f} {xlib.pct(nxt,.9):.2f}")
