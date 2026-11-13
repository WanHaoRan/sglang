"""Test (f9): R1's order with other statistics: mean with TTFT capped at 2, 5 and 10 s, median, p75, p90, p95, p99, share of
turns over 1, 2, 5 and 10 s, on LEAD5's shared returning turns.
  nice -n 19 python3 -I -B tf_caps.py"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

rank = C.order_rank()
D = {a: C.load(a) for a in C.LEAD5}
B = {a: C.by_key(C.kept(D[a])) for a in C.LEAD5}
keys = C.shared_keys(B, C.LEAD5, rank)
print(f"{len(keys)} shared returning turns")
print("arm          mean | capped at 10 s, 5 s, 2 s | median p75 p90 p95 p99 | turns over 1 s, 2 s, 5 s, 10 s | sum of TTFT over 10 s turns as a share of the arm's total")
for a in C.LEAD5:
    v = [B[a][k]["ttft"] for k in keys]
    print(f"  {a:11s} {C.mean(v):.3f} | {C.mean(min(x, 10) for x in v):.3f} {C.mean(min(x, 5) for x in v):.3f} {C.mean(min(x, 2) for x in v):.3f} | "
          f"{C.pct(v, .5):.3f} {C.pct(v, .75):.3f} {C.pct(v, .9):.3f} {C.pct(v, .95):.3f} {C.pct(v, .99):.2f} | "
          f"{sum(1 for x in v if x > 1):5d} {sum(1 for x in v if x > 2):5d} {sum(1 for x in v if x > 5):5d} {sum(1 for x in v if x > 10):5d} | {100 * sum(x for x in v if x > 10) / sum(v):.1f} %")
