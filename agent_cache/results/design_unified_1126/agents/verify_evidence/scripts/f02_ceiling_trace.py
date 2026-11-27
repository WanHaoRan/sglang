"""ev_fetch 2a (trace level): exact-gap ceiling E[min(G, x)] per returning request, and the 'read at the turn's
end or wait for the arrival' comparison per key, population A.  No engine code: the plain cost of section 3.3,
   read now:  c * (x - G)+  +  lam * L * min((G - x)+, hold)      (hold: pin given back after `hold` s)
   arrival:   c * x
"""
import sys, os, gzip, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib
from vlib import pct, mean
CAP = 1200.0
P = collections.defaultdict(list)
hist = []
with gzip.open(vlib.EXT + "/trace_gaps.jsonl.gz", "rt") as f:
    for line in f:
        d = json.loads(line)
        if d["peak"] > 65536:
            continue
        for st in d["streams"]:
            rs = st["reqs"]
            for k in range(1, len(rs)):
                P[rs[k - 1][2] or "_no_call"].append(min(float(rs[k][0] or 0.0), CAP))
                hist.append(rs[k - 1][4] + rs[k - 1][3])
allg = [g for v in P.values() for g in v]
L = pct(hist, .5)
print(f"gaps {len(allg)}; median history (previous prompt + output) {L:.0f} tokens, mean {mean(hist):.0f}")
print("exact-gap ceiling E[min(G,x)] per returning request (s): " + ", ".join(f"x={x}: {mean(min(g, x) for g in allg):.2f}" for x in (1, 2, 4, 6)))
print("   same with capped gaps left out: " + ", ".join(f"x={x}: {mean(min(g, x) for g in allg if g < CAP):.2f}" for x in (1, 2, 4, 6)))
tool = [g for k, v in P.items() if k != "_no_call" for g in v]
def entry(key):
    v = P[key]
    if len(v) >= 5:
        return v
    return tool if key != "_no_call" else allg
for hold in (1e9, 30.0):
    print(f"-- pin held for at most {hold:g} s after x")
    for x in (2, 4):
        for lam in (0.0, 1e-6, 2e-5, 1e-4):
            now_n = 0; gain = 0.0; best = 0.0
            for key, v in P.items():
                e = entry(key)
                c_now = mean(max(x - g, 0.0) + lam * L * min(max(g - x, 0.0), hold) for g in e)
                c_arr = float(x)
                if c_now < c_arr:
                    now_n += len(v)
                    # scored on the key's own gaps (in sample)
                    gain += sum(x - (max(x - g, 0.0) + lam * L * min(max(g - x, 0.0), hold)) for g in v)
            print(f"   x={x} lam={lam:g}: 'now' for {100*now_n/len(allg):.0f} % of requests; net {gain/len(allg):+.2f} s per returning request (ceiling {mean(min(g, x) for g in allg):.2f})")
