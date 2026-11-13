"""Test (b), calibration: the same block and session sign tests between two runs of the SAME arm, to see what these tests
return when there is no policy difference.
  nice -n 19 python3 -I -B tb_blocks_repeats.py"""
import math, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

rank = C.order_rank()
NPOS = len(rank)
PAIRS = (("baseline", "base_rep"), ("hintfree", "hintfree_ctl"), ("jit_p50", "jit_p50_ctl"), ("jit_p50_ctl", "jit_p50_d_cut"), ("jit_p50", "jit_p50_d_cut"))
names = sorted(set(a for p in PAIRS for a in p))
D = {a: C.load(a) for a in names}
B = {a: C.by_key(C.kept(D[a])) for a in names}


def p2(k, n):
    lo = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(lo + 1)) / 2 ** n) if n else float("nan")


print("pair (second minus first): shared turns | mean, median | bins of 250 positions where the first is lower / higher (p) | sessions where the first is lower / higher (p) | outside the fixed episode windows: mean, bins lower / higher | block bootstrap 250 [5 %, 95 %]")
for a, b in PAIRS:
    keys = C.shared_keys(B, [a, b], rank)
    d = {k: B[b][k]["ttft"] - B[a][k]["ttft"] for k in keys}
    present = [k for k in keys if k in rank]
    absent = sorted((k for k in keys if k not in rank), key=lambda k: B[a][k]["t_send"])
    bins = [[k for k in present if lo <= rank[k] < lo + 250] for lo in range(0, NPOS, 250)] + [absent[i:i + 250] for i in range(0, len(absent), 250)]
    bins = [x for x in bins if x]
    lo_ = sum(1 for bn in bins if sum(d[k] for k in bn) > 0)
    hi_ = sum(1 for bn in bins if sum(d[k] for k in bn) < 0)
    sess = {}
    for k in keys:
        sess[k[0]] = sess.get(k[0], 0.0) + d[k]
    sl = sum(1 for v in sess.values() if v > 0)
    sh = sum(1 for v in sess.values() if v < 0)
    calm = [k for k in keys if not C.in_episode(rank, k)]
    cb = [[k for k in bn if not C.in_episode(rank, k)] for bn in bins]
    cb = [x for x in cb if x]
    cl = sum(1 for bn in cb if sum(d[k] for k in bn) > 0)
    ch = sum(1 for bn in cb if sum(d[k] for k in bn) < 0)
    cells = [(sum(d[k] for k in bn), len(bn)) for bn in bins]
    rng = random.Random(0)
    draws = []
    for _ in range(4000):
        pick = rng.choices(cells, k=len(cells))
        draws.append(sum(c[0] for c in pick) / sum(c[1] for c in pick))
    print(f"  {b:13s} - {a:11s} {len(keys):5d} | {C.mean(d.values()):+6.3f} {C.pct(list(d.values()), .5):+7.4f} | {lo_:2d} / {hi_:2d} (p {p2(lo_, lo_ + hi_):.3f}) | {sl:3d} / {sh:3d} (p {p2(sl, sl + sh):.3g}) | "
          f"{C.mean(d[k] for k in calm):+6.3f}, {cl:2d} / {ch:2d} (p {p2(cl, cl + ch):.3f}) | [{C.pct(draws, .05):+6.3f}, {C.pct(draws, .95):+6.3f}]")
