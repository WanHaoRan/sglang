"""ev_fetch 1a/1b: gap hazards on the replayed population (trees with peak prompt+output <= 65,536) and on the
whole trace.  A gap sample = the wait of request k (k >= 1) of a stream, keyed by the reply key of request k-1
('_no_call' when the trace records no tool call), x1, capped at 1,200 s (a capped gap is censored)."""
import sys, os, gzip, json, math, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib
from vlib import pct
CAP = 1200.0
BINS = [0, 1, 2, 4, 6, 10, 20, 60, 300, 1200]

def hazards(gaps):
    """per-bin hazard: -ln(1 - events/at_risk)/width; gaps >= CAP are censored at CAP."""
    out = []
    for lo, hi in zip(BINS, BINS[1:]):
        risk = sum(1 for g in gaps if g >= lo)
        evn = sum(1 for g in gaps if lo <= g < hi and g < CAP)
        if risk == 0 or evn >= risk:
            out.append((float("nan"), evn, risk)); continue
        out.append((-math.log(1 - evn / risk) / (hi - lo), evn, risk))
    return out

def F(gaps_sorted, x, s):
    """empirical P(G <= s+x | G > s)"""
    import bisect
    n = len(gaps_sorted)
    a = bisect.bisect_right(gaps_sorted, s)
    risk = n - a
    if risk <= 0:
        return None, 0
    b = bisect.bisect_right(gaps_sorted, s + x)
    return (b - a) / risk, risk

pops = {"A": collections.defaultdict(list), "B": collections.defaultdict(list)}
ntree = {"A": 0, "B": 0}
ends = {"A": collections.Counter(), "B": collections.Counter()}; replies = {"A": collections.Counter(), "B": collections.Counter()}
with gzip.open(vlib.EXT + "/trace_gaps.jsonl.gz", "rt") as f:
    for line in f:
        d = json.loads(line)
        which = ["B"] + (["A"] if d["peak"] <= 65536 else [])
        for w in which:
            ntree[w] += 1
        for st in d["streams"]:
            rs = st["reqs"]
            for k in range(len(rs)):
                key = rs[k][2] or "_no_call"
                for w in which:
                    replies[w][key] += 1
                    if k == len(rs) - 1:
                        ends[w][key] += 1
            for k in range(1, len(rs)):
                key = rs[k - 1][2] or "_no_call"
                g = min(float(rs[k][0] or 0.0), CAP)
                for w in which:
                    pops[w][key].append(g)

for w in ("A", "B"):
    P = pops[w]
    tool = [g for k, v in P.items() if k != "_no_call" for g in v]
    hum = P.get("_no_call", [])
    print(f"=== population {w}: trees {ntree[w]}, gaps {len(tool)+len(hum)}, tool {len(tool)}, no-call {len(hum)}")
    for name, g in (("tool", tool), ("no_call", hum)):
        hz = hazards(g)
        print(f"  {name:8s} p10/p50/p90 {pct(g,.1):.2f}/{pct(g,.5):.2f}/{pct(g,.9):.1f} s; at the cap {100*sum(1 for x in g if x>=CAP)/len(g):.1f} %")
        print("     hazard per s by bin " + " ".join(f"{lo}-{hi}:{h[0]:.4f}" for (lo, hi), h in zip(zip(BINS, BINS[1:]), hz)))
    # stream ends by reply key
    te = sum(v for k, v in ends[w].items() if k != "_no_call"); tr = sum(v for k, v in replies[w].items() if k != "_no_call")
    print(f"  stream ends after a reply: all {100*sum(ends[w].values())/sum(replies[w].values()):.1f} %; tool keys {100*te/tr:.1f} %; _no_call {100*ends[w]['_no_call']/max(replies[w]['_no_call'],1):.1f} %; bash/echo {100*ends[w]['bash/echo']/max(replies[w]['bash/echo'],1):.1f} %")
    # per key
    keys = [(k, v) for k, v in P.items() if len(v) >= 100]
    cov = sum(len(v) for k, v in keys); allg = sum(len(v) for v in P.values())
    print(f"  keys with >= 100 gaps: {len(keys)} covering {100*cov/allg:.1f} % of gaps")
    peak = collections.Counter(); rise = {(2, .05): 0, (4, .05): 0, (4, .15): 0, (6, .05): 0, (2, .15): 0, (6, .15): 0}
    ratios = []
    grid = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 45, 60]
    for k, v in keys:
        hz = hazards(v)
        # the bin where the empirical hazard is highest (bins with >= 20 at risk)
        best = max(((h[0], i) for i, h in enumerate(hz) if h[2] >= 20 and h[0] == h[0]), default=(0, 0))
        peak[best[1]] += len(v)
        vs = sorted(v)
        ratios.append((pct(v, .9) / max(pct(v, .1), 1e-9), k, len(v), pct(v, .1), pct(v, .5), pct(v, .9)))
        for (x, thr) in rise:
            f0, _ = F(vs, x, 0.0)
            mx = max((F(vs, x, s)[0] for s in grid if F(vs, x, s)[1] >= 50), default=f0)
            if mx - f0 > thr:
                rise[(x, thr)] += len(v)
    print("  share of gaps by the bin where the key's hazard peaks: " + ", ".join(f"{BINS[i]}-{BINS[i+1]} s {100*c/cov:.1f} %" for i, c in sorted(peak.items())))
    print("  share of gaps in keys whose F(x|s) rises above F(x|0) by more than d at some idle age s (>= 50 at risk): " +
          ", ".join(f"x={x} d={thr}: {100*c/cov:.1f} %" for (x, thr), c in sorted(rise.items())))
    ratios.sort()
    rr = [r[0] for r in ratios]
    print(f"  p90/p10 over keys: min {rr[0]:.1f} p10 {pct(rr,.1):.1f} median {pct(rr,.5):.1f} p90 {pct(rr,.9):.1f}")
    print("  sharpest keys: " + "; ".join(f"{k} n={n} {a:.2f}/{b:.2f}/{c:.2f} ratio {r:.1f}" for r, k, n, a, b, c in ratios[:6]))
    for nm in ("web_search", "terminal/ps", "other"):
        if nm in P:
            v = P[nm]; print(f"  {nm}: n={len(v)} ({100*len(v)/allg:.1f} % of gaps) p10/p50/p90 {pct(v,.1):.2f}/{pct(v,.5):.2f}/{pct(v,.9):.2f}")
