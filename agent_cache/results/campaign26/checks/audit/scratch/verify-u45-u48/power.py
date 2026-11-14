#!/usr/bin/env python3
"""Monte Carlo of the record's rule for repeats (DECISIONS.md L380-381), standard library only.
Shown = every pair of runs (one from each arm) has the same sign AND the smallest |difference|
exceeds the largest move between two runs of either arm."""
import random, itertools, sys
random.seed(12345)
def shown(a, b):
    d = [x - y for x in a for y in b]
    if not (all(v > 0 for v in d) or all(v < 0 for v in d)):
        return False
    mv = 0.0
    for arm in (a, b):
        for x, y in itertools.combinations(arm, 2):
            mv = max(mv, abs(x - y))
    return min(abs(v) for v in d) > mv
N = 200000
for na, nb in ((2, 2), (3, 3), (4, 4), (2, 1), (3, 1)):
    out = []
    for gap in (0, 1, 2, 3, 4, 6):
        k = 0
        for _ in range(N):
            a = [random.gauss(0, 1) for _ in range(na)]
            b = [random.gauss(gap, 1) for _ in range(nb)]
            k += shown(a, b)
        out.append(f"gap {gap} sd: {100*k/N:5.1f} %")
    print(f"runs {na} vs {nb}: " + " | ".join(out))
