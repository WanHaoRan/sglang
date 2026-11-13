"""Test (f6): where each pairwise difference sits. For every pair of LEAD5 arms: the share of the summed paired TTFT
difference that falls inside the fixed episode windows and inside windows chosen from the data (250-position bins whose
five-arm mean exceeds 1 s or 0.75 s); plus a duplicate-key check of the client logs.
  nice -n 19 python3 -I -B tf_shares.py"""
import itertools, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

rank = C.order_rank()
NPOS = len(rank)
ARMS = [a for a in C.RUNS if a != "order"]
D = {a: C.load(a) for a in ARMS}
print("duplicate (conv, turn) records per run: " + "  ".join(f"{a} {len(D[a]['turns']) - len(set((t['conv'], t['turn']) for t in D[a]['turns']))}" for a in ARMS))
B = {a: C.by_key(C.kept(D[a])) for a in ARMS}
keys = C.shared_keys(B, C.LEAD5, rank)
present = [k for k in keys if k in rank]
absent = sorted((k for k in keys if k not in rank), key=lambda k: C.mean(B[a][k]["t_send"] for a in C.LEAD5))
bins = [[k for k in present if lo <= rank[k] < lo + 250] for lo in range(0, NPOS, 250)] + [absent[i:i + 250] for i in range(0, len(absent), 250)]
bins = [b for b in bins if b]
fixed = set(k for k in keys if C.in_episode(rank, k))
hot10 = set(k for b in bins if C.mean(C.mean(B[a][k]["ttft"] for k in b) for a in C.LEAD5) > 1.0 for k in b)
hot075 = set(k for b in bins if C.mean(C.mean(B[a][k]["ttft"] for k in b) for a in C.LEAD5) > 0.75 for k in b)
print(f"{len(keys)} shared turns; fixed windows {len(fixed)}; data windows over 1 s {len(hot10)}; over 0.75 s {len(hot075)}")
print("pair: mean difference | share of the summed difference inside the fixed windows, inside the data windows (1 s), (0.75 s) | mean difference outside each")
for a, b in itertools.combinations(C.LEAD5, 2):
    d = {k: B[b][k]["ttft"] - B[a][k]["ttft"] for k in keys}
    tot = sum(d.values())
    cells = []
    outs = []
    for w in (fixed, hot10, hot075):
        ins = sum(d[k] for k in w)
        cells.append(f"{100 * ins / tot:+7.1f} %")
        outs.append(f"{(tot - ins) / (len(keys) - len(w)):+6.3f}")
    print(f"  {b:10s} - {a:10s} {tot / len(keys):+6.3f} | " + " ".join(cells) + " | " + " ".join(outs))
print("per arm, mean TTFT outside the fixed windows, outside the data windows (1 s), (0.75 s):")
for a in C.LEAD5:
    print(f"  {a:11s} " + " ".join(f"{C.mean(B[a][k]['ttft'] for k in keys if k not in w):6.3f}" for w in (fixed, hot10, hot075)))
