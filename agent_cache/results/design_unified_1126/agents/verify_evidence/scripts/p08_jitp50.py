"""ev_prices 2b: jit_p50 against the other three hinted arms of campaign 34 -- does read lateness leave the
mean unchanged?  Window means by send time, and pairs on (play, stream, k) sent in the window in both arms."""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib, runtab
from vlib import mean
tabs = {a: runtab.build(34, a) for a in ("jit_p50", "ec_full", "jit_exact", "ec_exact")}
def idx(tab):
    return {(q["play"], q["stream"], q["k"]): (rid, q) for rid, q in tab["reqs"].items()}
I = {a: idx(t) for a, t in tabs.items()}
def wstats(a, lo, hi):
    tab = tabs[a]; st = tab["stall"]
    R = [(rid, q) for rid, q in tab["reqs"].items() if lo * 60 <= q["t_send"] < hi * 60]
    n = len(R)
    return dict(n=n, ttft=mean(q["ttft"] for _, q in R), stall=sum(st[rid]["stall"] for rid, q in R if rid in st) / n,
                lost=sum(max(q["ideal"] - q["cached"], 0) for _, q in R) / n, new=sum(q["prompt"] - q["cached"] for _, q in R) / n,
                out=sum(q["out"] for _, q in R) / n, readers=sum(1 for rid, q in R if rid in st) / n)
for lo, hi in [(28, 110), (28, 60), (60, 85), (85, 110), (0, 28)]:
    print(f"=== minutes {lo}-{hi}")
    S = {a: wstats(a, lo, hi) for a in tabs}
    for a, s in S.items():
        print(f"   {a:9s}: n {s['n']:5d}  TTFT {s['ttft']:6.2f} s  stall/request {s['stall']:6.2f} s  readers {100*s['readers']:4.1f} %  uncached tokens/request {s['new']:6.0f} (lost history {s['lost']:5.0f})  out {s['out']:.0f}")
    for b in ("ec_full", "jit_exact", "ec_exact"):
        keys = [k for k in I["jit_p50"] if k in I[b] and lo * 60 <= I["jit_p50"][k][1]["t_send"] < hi * 60 and lo * 60 <= I[b][k][1]["t_send"] < hi * 60]
        d = [I["jit_p50"][k][1]["ttft"] - I[b][k][1]["ttft"] for k in keys]
        # bootstrap over users (plays)
        byplay = {}
        for k, x in zip(keys, d):
            byplay.setdefault(k[0], []).append(x)
        plays = list(byplay)
        rng = random.Random(1); bs = []
        for _ in range(400):
            tot = cnt = 0.0
            for p in (rng.choice(plays) for _ in plays):
                tot += sum(byplay[p]); cnt += len(byplay[p])
            bs.append(tot / cnt)
        bs.sort()
        ds = S["jit_p50"]["stall"] - S[b]["stall"]
        dg = (S["jit_p50"]["new"] - S[b]["new"]) * 0.043e-3
        print(f"   jit_p50 - {b:9s}: window means {S['jit_p50']['ttft'] - S[b]['ttft']:+.2f} s; paired on {len(d)} requests {mean(d):+.2f} s [{bs[10]:+.2f}, {bs[389]:+.2f}]; stall/request {ds:+.2f} s; uncached tokens {S['jit_p50']['new'] - S[b]['new']:+.0f} -> {dg:+.4f} GPU-s -> x108 = {108*dg:+.2f} s; paired/stall = {mean(d)/ds if ds else float('nan'):.2f}")
