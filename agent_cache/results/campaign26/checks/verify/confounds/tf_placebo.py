"""Test (f7): a placebo. On the shared returning turns where no arm of the set queued (server queue wait under 10 ms),
read from SSD or recomputed more than 64 tokens, the cache policy had nothing to do, so TTFT should be the same in every
arm; a difference here would point at hardware, host or client, not policy.
  nice -n 19 python3 -I -B tf_placebo.py"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

rank = C.order_rank()
ARMS = [a for a in C.RUNS if a != "order"]
D = {a: C.load(a) for a in ARMS}
B = {a: C.by_key(C.kept(D[a])) for a in ARMS}
READ = {a: set((x[1], x[2]) for x in D[a]["s2h"] if x[3] > 0) for a in ARMS}


def clean(a, k):
    r = D[a]["req"].get(f"{k[0]}-{k[1]}")
    return r is not None and r[0] < 10 and k not in READ[a] and C.rec(B[a][k]) <= 64


for title, names in (("LEAD5", C.LEAD5), ("SEVEN", C.SEVEN), ("CUT7", C.CUT7), ("ALL11", C.ALL11)):
    keys = C.shared_keys(B, names, rank)
    ks = [k for k in keys if all(clean(a, k) for a in names)]
    print(f"{title}: {len(ks)} of {len(keys)} shared returning turns are clean in every arm ({100 * len(ks) / len(keys):.1f} %)")
    ref = names[0]
    print("  arm            mean TTFT  median   p90 | paired against " + ref + ": mean difference, median difference, slower / faster | by half of the run (order positions under / from 5000): mean")
    for a in names:
        v = [B[a][k]["ttft"] for k in ks]
        d = [B[a][k]["ttft"] - B[ref][k]["ttft"] for k in ks]
        h1 = [B[a][k]["ttft"] for k in ks if rank.get(k, 10 ** 9) < 5000]
        h2 = [B[a][k]["ttft"] for k in ks if rank.get(k, 10 ** 9) >= 5000]
        print(f"  {a:14s} {C.mean(v):7.4f} {C.pct(v, .5):7.4f} {C.pct(v, .9):6.3f} | {C.mean(d):+7.4f} {C.pct(d, .5):+7.4f} {sum(1 for x in d if x > 0):5d} / {sum(1 for x in d if x < 0):5d} | {C.mean(h1):6.4f} {C.mean(h2):6.4f}")
    m = [C.mean(B[a][k]["ttft"] for k in ks) for a in names]
    print(f"  range of the means {max(m) - min(m):.4f} s = {100 * (max(m) - min(m)) / min(m):.1f} % of the lowest")
