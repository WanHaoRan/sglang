#!/usr/bin/env python3
"""Power of a two-sided pooled t test on run means (alpha 0.05), normal run-to-run noise; standard library only."""
import random, math
random.seed(7)
TCRIT = {2: 4.303, 4: 2.776, 6: 2.447, 8: 2.306, 10: 2.228, 14: 2.145, 18: 2.101}
N = 100000
for n in (2, 3, 4, 5, 6, 8, 10):
    df = 2 * n - 2
    out = []
    for gap in (0, 1, 2, 3):
        k = 0
        for _ in range(N):
            a = [random.gauss(0, 1) for _ in range(n)]
            b = [random.gauss(gap, 1) for _ in range(n)]
            ma, mb = sum(a) / n, sum(b) / n
            va = sum((x - ma) ** 2 for x in a) / (n - 1)
            vb = sum((x - mb) ** 2 for x in b) / (n - 1)
            sp = math.sqrt((va + vb) / 2)
            t = (mb - ma) / (sp * math.sqrt(2 / n)) if sp > 0 else 0
            k += abs(t) > TCRIT[df]
        out.append(f"gap {gap} sd: {100*k/N:5.1f} %")
    print(f"{n} runs per arm: " + " | ".join(out))
