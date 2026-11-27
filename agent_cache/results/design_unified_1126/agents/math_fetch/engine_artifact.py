"""Check 2e: the engine's 64-point support against the continuous law (read-only on the engine).

(1) For x = read_s - wait_s > 0 and a non-empty support, read_start never returns None: starting
    at the last support point minus x always costs stall_at_arrival * (count - 1) / count at most.
(2) The planned start at the last support point is never reached on a continuous tail: remaining_gaps
    returns () once fewer than 4 points are left. It IS reached when 4 or more points share the
    1,200 s cap (the human entry: 6 points).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script's folder

import numpy as np

from fetchlib import load_engine_policy, TOOL, HUMAN

ntp = load_engine_policy()
rng = np.random.default_rng(0)

none_count, total, far = 0, 0, 0
for _ in range(20000):
    p50 = float(np.exp(rng.uniform(np.log(0.05), np.log(600.0))))
    p10 = p50 * float(np.exp(-rng.uniform(0.0, 4.0)))
    p90 = p50 * float(np.exp(rng.uniform(0.0, 4.0)))
    dist = ntp.GapDistribution.from_quantiles(p10=p10, p50=p50, p90=p90)
    sup = dist.support()
    idle = float(rng.choice([0.0, 0.0, 1.0, 10.0, 100.0]))
    rem = ntp.remaining_gaps(support=sup, idle_s=idle)
    if not rem:
        continue
    r = float(rng.uniform(0.5, 8.0))
    W = float(rng.uniform(0.0, r * 0.99))
    rent = float(np.exp(rng.uniform(np.log(1e-3), np.log(50.0))))
    st = ntp.read_start(remaining=rem, read_s=r, wait_s=W, rent_per_s=rent)
    total += 1
    none_count += st is None
    if st is not None and abs(st - max(0.0, rem[-1] - (r - W))) < 1e-9 and st > 0:
        far += 1
print(f"(1) {total:,} random entries with x > 0: read_start returned None {none_count} times;"
      f" it returned 'the last support point minus x' {far:,} times ({100 * far / total:.1f} %)")

for name, q3 in [("tool", TOOL), ("human", HUMAN)]:
    dist = ntp.GapDistribution.from_quantiles(**q3)
    sup = dist.support()
    at_cap = sum(1 for g in sup if g >= ntp.MAX_PLANNED_GAP_S)
    print(f"(2) {name} entry: last four support points {tuple(round(g, 1) for g in sup[-4:])}; points at the 1,200 s cap: {at_cap};"
          f" the plan dies (fewer than {ntp.MIN_REMAINING_POINTS} points left) once idle > {sup[-ntp.MIN_REMAINING_POINTS]:.1f} s")
    r, W, rent = 3.0, 0.0, 0.5
    staged_at = None
    a = 0.0
    while a < 1200.0:
        rem = ntp.remaining_gaps(support=sup, idle_s=a)
        if not rem:
            break
        st = ntp.read_start(remaining=rem, read_s=r, wait_s=W, rent_per_s=rent)
        if st is not None and st <= 0.0:
            staged_at = a
            break
        a += 0.5
    print(f"    a session that stays idle (r = 3 s, W = 0, rent 0.5/s): the engine stages at idle = {staged_at} s"
          f" ({'the cap atom' if staged_at and staged_at > 1000 else 'never before the plan dies' if staged_at is None else 'early'})")

# (3) the stall_weight argument (present in the working tree from 18:23 UTC on 2026-11-26; not in HEAD):
#     weighting the stall by c must give the same start as dividing the rent by c.
import inspect

if "stall_weight" in inspect.signature(ntp.read_start).parameters:
    diff = 0
    n3 = 0
    for _ in range(20000):
        p50 = float(np.exp(rng.uniform(np.log(0.05), np.log(600.0))))
        dist = ntp.GapDistribution.from_quantiles(p10=p50 * float(np.exp(-rng.uniform(0.0, 4.0))), p50=p50,
                                                  p90=p50 * float(np.exp(rng.uniform(0.0, 4.0))))
        rem = ntp.remaining_gaps(support=dist.support(), idle_s=float(rng.choice([0.0, 1.0, 10.0])))
        if not rem:
            continue
        r = float(rng.uniform(0.5, 8.0))
        W = float(rng.uniform(0.0, r * 1.2))
        rent = float(np.exp(rng.uniform(np.log(1e-3), np.log(50.0))))
        c = float(rng.choice([1.0, 0.3, 0.05]))
        a = ntp.read_start(remaining=rem, read_s=r, wait_s=W, rent_per_s=rent, stall_weight=c)
        b = ntp.read_start(remaining=rem, read_s=r, wait_s=W, rent_per_s=rent / c)
        n3 += 1
        if (a is None) != (b is None) or (a is not None and abs(a - b) > 1e-9):
            diff += 1
    print(f"(3) read_start(stall_weight=c, rent) against read_start(rent / c): {diff} differences in {n3:,} random cases")
else:
    print("(3) read_start has no stall_weight argument in this tree")
