"""How often the record's repeat rule (DECISIONS.md L380-381) says 'shown', for runs with normal run-to-run noise of sd 1.
Rule: every pair of runs of the two arms has the same sign, and the smallest pair difference exceeds the largest move
between two runs of either arm. n runs per arm; true gap d in units of the run-to-run sd."""
import itertools, random
rng = random.Random(0)
def fires(n, d):
    a = [rng.gauss(0, 1) for _ in range(n)]
    b = [rng.gauss(d, 1) for _ in range(n)]
    diffs = [y - x for x in a for y in b]
    if not (all(v > 0 for v in diffs) or all(v < 0 for v in diffs)):
        return False
    move = max(max(abs(p - q) for p, q in itertools.combinations(a, 2)), max(abs(p - q) for p, q in itertools.combinations(b, 2)))
    return min(abs(v) for v in diffs) > move
N = 200000
for n in (2, 3):
    print(f"{n} runs per arm:", "  ".join(f"gap {d} sd: {100 * sum(fires(n, d) for _ in range(N)) / N:.1f} %" for d in (0, 1, 2, 3)))
