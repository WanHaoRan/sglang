"""Test (f3), intervals: paired differences in TTFT, decode time (latency - ttft) and latency between LEAD5 arms on their
shared returning turns, with a block bootstrap over 250-position bins (5 %, 95 %) and the bins in which the first arm is
lower / higher. Same-arm repeats (JIT p50, hint-free) give the scale of what a second run moves.
  nice -n 19 python3 -I -B tf_latency_ci.py"""
import itertools, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

rank = C.order_rank()
NPOS = len(rank)
names = C.LEAD5 + ["jit_p50", "jit_p50_ctl", "hintfree", "hintfree_ctl"]
D = {a: C.load(a) for a in names}
B = {a: C.by_key(C.kept(D[a])) for a in names}


def report(a, b, keys):
    present = [k for k in keys if k in rank]
    absent = sorted((k for k in keys if k not in rank), key=lambda k: B[a][k]["t_send"])
    bins = [[k for k in present if lo <= rank[k] < lo + 250] for lo in range(0, NPOS, 250)] + [absent[i:i + 250] for i in range(0, len(absent), 250)]
    bins = [x for x in bins if x]
    cells = []
    for name, f in (("ttft", lambda t: t["ttft"]), ("decode", lambda t: t["latency"] - t["ttft"]), ("latency", lambda t: t["latency"])):
        d = {k: f(B[b][k]) - f(B[a][k]) for k in keys}
        per = [(sum(d[k] for k in bn), len(bn)) for bn in bins]
        rng = random.Random(0)
        draws = []
        for _ in range(4000):
            pick = rng.choices(per, k=len(per))
            draws.append(sum(c[0] for c in pick) / sum(c[1] for c in pick))
        lo_ = sum(1 for s, n in per if s > 0)
        hi_ = sum(1 for s, n in per if s < 0)
        cells.append(f"{name} {sum(d.values()) / len(keys):+6.3f} [{C.pct(draws, .05):+6.3f}, {C.pct(draws, .95):+6.3f}] bins {lo_}/{hi_}")
    print(f"  {b:12s} - {a:11s} ({len(keys)} turns) " + " | ".join(cells))


keys = C.shared_keys(B, C.LEAD5, rank)
print("LEAD5 pairs (second-named minus first-named):")
for a, b in itertools.combinations(C.LEAD5, 2):
    report(a, b, keys)
print("same arm run twice:")
for a, b in (("jit_p50", "jit_p50_ctl"), ("hintfree", "hintfree_ctl")):
    report(a, b, C.shared_keys(B, [a, b], rank))
