"""Proposition 23, checked by direct simulation of the layer recursion.

A copy lands layer i (0-based, n layers) at (i + 1) r / n. A pass started at s computes layer i for p / n seconds, in
order, and cannot start a layer before it has landed. Checked on random (n, r, p, s):
  end(s) = max(s + p, max(r, p) + min(r, p) / n);
  the pass waits end(s) - s - p in total;
  s* = (r - p)^+ + min(r, p) / n is the latest start with end(s) = end(0), and the wait is 0 from s* on.
Then the step-boundary rules: starts only at multiples of delta. For the rule built (first boundary at or after the
copy's end) and the meeting rule (first boundary where the copy's remaining time is at most p), the batch's delay
against a start at 0 and the GPU time left to waiting.
"""
import math
import random


def end_time(n, r, p, s):
    t = s
    for i in range(n):
        t = max(t, (i + 1) * r / n) + p / n
    return t


def main():
    rng = random.Random(0)
    worst = 0.0
    for _ in range(200000):
        n = rng.choice([1, 2, 12, 24, 48, 64])
        r = rng.uniform(0.0, 0.3)
        p = rng.uniform(0.001, 0.3)
        s = rng.uniform(0.0, 0.5)
        e = end_time(n, r, p, s)
        closed = max(s + p, max(r, p) + min(r, p) / n)
        worst = max(worst, abs(e - closed))
        star = max(0.0, r - p) + min(r, p) / n
        assert abs(end_time(n, r, p, star) - end_time(n, r, p, 0.0)) < 1e-12
        assert end_time(n, r, p, star + 1e-6) > end_time(n, r, p, 0.0) + 5e-7
        assert abs(end_time(n, r, p, star) - star - p) < 1e-12
        assert abs(end_time(n, r, p, star + 0.01) - (star + 0.01) - p) < 1e-12
    print(f"closed form of the end time: largest error over 200,000 draws {worst:.2e} s")
    print("latest free start s* = (r - p)^+ + min(r, p) / n and zero wait from s* on: holds in every draw")
    n, delta = 48, 0.024
    print("step-boundary rules, 48 layers, decode step 24 ms, copy start uniform within a step (means over 100,000 draws)")
    print("  copy ms  pass ms | start at 0: wait | built rule: delay, wait | meeting rule: delay, wait")
    for r, p in ((0.060, 0.026), (0.060, 0.045), (0.095, 0.053), (0.120, 0.030), (0.040, 0.060)):
        acc = [0.0] * 5
        m = 100000
        for _ in range(m):
            off = rng.uniform(0.0, delta)          # the copy starts this long after a step boundary
            base = end_time(n, r, p, 0.0)
            acc[0] += base - p
            sb = math.ceil((r + off) / delta) * delta - off      # first boundary at or after the copy's end
            eb = end_time(n, r, p, sb)
            acc[1] += eb - base
            acc[2] += eb - sb - p
            k = 0
            while True:                                         # first boundary where the remaining copy time <= p
                sm = max(0.0, k * delta - off)
                if r - sm <= p:
                    break
                k += 1
            if k == 0:                                          # the batch is built at a boundary: no hold at all
                sm = 0.0
            em = end_time(n, r, p, sm)
            acc[3] += em - base
            acc[4] += em - sm - p
        a = [1e3 * v / m for v in acc]
        print(f"  {1e3 * r:6.0f} {1e3 * p:8.0f} | {a[0]:6.1f} | {a[1]:6.1f}, {a[2]:4.1f} | {a[3]:6.1f}, {a[4]:4.1f}")


if __name__ == "__main__":
    main()
