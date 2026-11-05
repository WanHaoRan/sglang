"""Read-only: within-run block bootstrap (250 oracle positions, 2,000 draws) of the forecast's cost at 44 users.

Reuses the critic's parser and its existing per-run cache; refuses to run if a cached pickle is missing,
so nothing is parsed or written outside this folder.
"""
import collections
import importlib.util
import os
import random

LIB = "/home/wanhr/sglang/agent_cache/frozen/oracle_scripts_staging/c23_gain44/v6_critic_lib.py"
CACHE = "/tmp/claude-1001/-home-wanhr-sglang/7c93b1d5-dc1c-431d-8d3e-efb4ae23a30d/scratchpad/v6cache"

spec = importlib.util.spec_from_file_location("L", LIB)
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)
L.CACHE = CACHE
for name, _ in L.ARMS:
    if not os.path.exists(os.path.join(CACHE, f"{name}.pkl")):
        raise SystemExit(f"missing cache for {name}; not parsing")
D = L.load_all()
R = {n: L.returning(d) for n, d in D.items()}
pos = L.rank_oracle(D["oracle"])
k5 = L.shared(D, ("baseline", "oracle", "oldp50", "newexact", "newp50"))
kn = [k for k in L.shared(D, ("newexact", "newp50")) if k in pos]
ko = [k for k in L.shared(D, ("oracle", "oldp50")) if k in pos]


def boot(ks, a, b, seed=1):
    random.seed(seed)
    blocks = collections.defaultdict(list)
    for k in ks:
        blocks[pos[k] // 250].append(R[a][k]["ttft"] - R[b][k]["ttft"])
    bl = list(blocks.values())
    stats = []
    for _ in range(2000):
        draw = [random.choice(bl) for _ in bl]
        n = sum(len(x) for x in draw)
        stats.append(sum(sum(x) for x in draw) / n)
    stats.sort()
    m = sum(sum(x) for x in bl) / len(ks)
    return m, stats[50], stats[1949], len(bl)


for lab, ks, a, b in (("7,768", k5, "newp50", "newexact"),
                      (f"{len(kn):,} (both new arms)", kn, "newp50", "newexact"),
                      ("7,768", k5, "oldp50", "oracle"),
                      (f"{len(ko):,} (both old arms)", ko, "oldp50", "oracle"),
                      ("7,768", k5, "oldp50", "hintfree"),
                      ("7,768", k5, "baseline", "oldp50"),
                      ("7,768", k5, "baseline", "newp50")):
    m, lo, hi, nb = boot(ks, a, b)
    print(f"{a} - {b} on {lab}: mean {m:+.3f} s, 95 % within-run interval {lo:+.3f} to {hi:+.3f} s, {nb} blocks")
