"""Test (f14): the size of the differences between arms against what a second run of one arm moves, in relative terms.
Every pair of runs of the same arm that exists (eight pairs over four arms), each on the returning turns the two runs
share: mean TTFT of each, the difference and the ratio; inside the fixed episode windows and elsewhere. Then the same for
every pair of LEAD5 arms and the repeat of rules 1 and 3 on the six runs' shared turns.
  nice -n 19 python3 -I -B tg_relmoves.py"""
import itertools, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

rank = C.order_rank()
SIX = C.LEAD5 + ["ec_r13_rep"]
SAME = (("rules 1+3", "ec_r13", "ec_r13_rep", "unit 605dfc41 -> a7198925, same day"), ("JIT p50", "jit_p50", "jit_p50_ctl", "2026-11-01 -> 11-11, old build twice"),
        ("JIT p50", "jit_p50", "jit_p50_d", "2026-11-01 old build -> 11-11 new build"), ("JIT p50", "jit_p50_ctl", "jit_p50_d", "old -> new build, unit 605dfc41 -> a7198925"),
        ("JIT p50", "jit_p50_ctl", "jit_p50_d_cut", "old -> new build, same unit, 74.7 min"), ("JIT p50", "jit_p50_d_cut", "jit_p50_d", "new build twice, two units, 74.7 min"),
        ("hint-free", "hintfree", "hintfree_ctl", "2026-11-01 -> 11-11"), ("original HiCache", "baseline", "base_rep", "2026-11-01 -> 11-05"))
names = sorted(set(SIX) | set(a for p in SAME for a in p[1:3]))
B = {a: C.by_key(C.kept(C.load(a))) for a in names}


def row(a, b, keys):
    ep = [k for k in keys if C.in_episode(rank, k)]
    calm = [k for k in keys if not C.in_episode(rank, k)]
    out = []
    for ks in (keys, ep, calm):
        ma, mb = C.mean(B[a][k]["ttft"] for k in ks), C.mean(B[b][k]["ttft"] for k in ks)
        out.append((ma, mb, mb - ma, 100 * (mb / ma - 1)))
    return out


print("A. two runs of the same arm (second against first): turns | all: means, difference s, % | fixed windows: means, difference, % | elsewhere: means, difference, %")
rel = []
for arm, a, b, note in SAME:
    keys = C.shared_keys(B, [a, b], rank)
    r = row(a, b, keys)
    rel.append((abs(r[0][3]), abs(r[1][3]), abs(r[2][3]), abs(r[0][2])))
    print(f"  {arm:16s} {b:13s} / {a:13s} {len(keys):5d} | " + " | ".join(f"{x[0]:6.3f} {x[1]:6.3f} {x[2]:+6.3f} {x[3]:+6.1f} %" for x in r) + f" | {note}")
print(f"  |relative move| of the mean over the eight pairs: median {sorted(x[0] for x in rel)[len(rel) // 2]:.1f} %, range {min(x[0] for x in rel):.1f}-{max(x[0] for x in rel):.1f} %; "
      f"inside the fixed windows range {min(x[1] for x in rel):.1f}-{max(x[1] for x in rel):.1f} %; elsewhere {min(x[2] for x in rel):.1f}-{max(x[2] for x in rel):.1f} %")
full = [x for x, p in zip(rel, SAME) if "74.7" not in p[3]]
print(f"  the six pairs of complete runs: |relative move| {', '.join(f'{x[0]:.1f}' for x in full)} %; |move| {', '.join(f'{x[3]:.3f}' for x in full)} s")

keys = C.shared_keys(B, SIX, rank)
print(f"\nB. between arms, on the six runs' {len(keys)} shared returning turns (second against first): all: difference s, % | fixed windows | elsewhere | larger than every same-arm relative move of complete runs above (all turns)?")
big = max(x[0] for x in full)
big_nb = max(x[0] for x, p in zip(rel, SAME) if "74.7" not in p[3] and p[0] != "original HiCache")
for a, b in itertools.combinations(SIX, 2):
    r = row(a, b, keys)
    tag = "the same arm" if {a, b} == {"ec_r13", "ec_r13_rep"} else ("yes" if abs(r[0][3]) > big else ("only without original HiCache's pair" if abs(r[0][3]) > big_nb else "no"))
    print(f"  {b:11s} / {a:11s} " + " | ".join(f"{x[2]:+6.3f} {x[3]:+6.1f} %" for x in r) + f" | {tag}")
print(f"  largest same-arm relative move of complete runs: {big:.1f} % (original HiCache), {big_nb:.1f} % without it")
