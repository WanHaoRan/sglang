"""Test (b), second unit: blocks of consecutive positions instead of sessions (a congestion event hits every session at once).
For each pair of LEAD5 arms: in how many 250-position bins the first arm's mean TTFT is lower / higher (sign test over bins),
the same over the bins with a five-arm mean over 0.75 s, and a block bootstrap of the paired mean difference over bins of
250 and 500 positions (5 %, 95 %). Bins follow the order run; shared turns absent from it are binned by consensus order.
  nice -n 19 python3 -I -B tb_blocks.py"""
import itertools, math, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

rank = C.order_rank()
NPOS = len(rank)
D = {a: C.load(a) for a in C.LEAD5}
B = {a: C.by_key(C.kept(D[a])) for a in C.LEAD5}
keys = C.shared_keys(B, C.LEAD5, rank)
present = [k for k in keys if k in rank]
absent = sorted((k for k in keys if k not in rank), key=lambda k: C.mean(B[a][k]["t_send"] for a in C.LEAD5))


def make_bins(width):
    bins = [[k for k in present if lo <= rank[k] < lo + width] for lo in range(0, NPOS, width)] + [absent[i:i + width] for i in range(0, len(absent), width)]
    return [b for b in bins if b]


def p2(k, n):
    lo = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(lo + 1)) / 2 ** n) if n else float("nan")


b250, b500 = make_bins(250), make_bins(500)
hot = [b for b in b250 if C.mean(C.mean(B[a][k]["ttft"] for k in b) for a in C.LEAD5) > 0.75]
print(f"{len(keys)} shared turns in {len(b250)} bins of 250 positions ({len(hot)} with a five-arm mean over 0.75 s) and {len(b500)} bins of 500")
print("pair (later-named minus earlier-named): mean | bins where the earlier-named arm is lower / higher (p) | among the slow bins | block bootstrap 250 [5 %, 95 %] | block bootstrap 500 | share of draws with the opposite sign (250)")
for a, b in itertools.combinations(C.LEAD5, 2):
    d = {k: B[b][k]["ttft"] - B[a][k]["ttft"] for k in keys}
    lo_ = sum(1 for bn in b250 if sum(d[k] for k in bn) > 0)
    hi_ = sum(1 for bn in b250 if sum(d[k] for k in bn) < 0)
    hlo = sum(1 for bn in hot if sum(d[k] for k in bn) > 0)
    hhi = sum(1 for bn in hot if sum(d[k] for k in bn) < 0)
    out = []
    opp = None
    for bins in (b250, b500):
        cells = [(sum(d[k] for k in bn), len(bn)) for bn in bins]
        rng = random.Random(0)
        draws = []
        for _ in range(4000):
            pick = rng.choices(cells, k=len(cells))
            draws.append(sum(c[0] for c in pick) / sum(c[1] for c in pick))
        out.append(f"[{C.pct(draws, .05):+6.3f}, {C.pct(draws, .95):+6.3f}]")
        if opp is None:
            m = sum(d.values())
            opp = sum(1 for x in draws if (x < 0) == (m > 0)) / len(draws)
    print(f"  {b:10s} - {a:10s} {sum(d.values()) / len(keys):+6.3f} | {lo_:2d} / {hi_:2d} (p {p2(lo_, lo_ + hi_):.3f}) | {hlo:2d} / {hhi:2d} | {out[0]} | {out[1]} | {100 * opp:5.1f} %")
