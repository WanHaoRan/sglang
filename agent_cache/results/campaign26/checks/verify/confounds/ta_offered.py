"""Test (a), detail: the work offered to the server during the two congestion events, per arm, by wall clock.
An event's window in an arm runs from the median send time of the shared turns at the 100 positions before it to the
median send time of those at the 100 positions after it. Inside: every turn the arm sent (shared or not), first turns,
new prompt tokens the workload brings (prompt tokens of first turns, delta_tokens of returning turns: independent of the
cache policy), completion tokens asked, and the sessions that ended and started.
  nice -n 19 python3 -I -B ta_offered.py"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

rank = C.order_rank()
D = {a: C.load(a) for a in C.LEAD5}
B = {a: C.by_key(C.kept(D[a])) for a in C.LEAD5}
keys = C.shared_keys(B, C.LEAD5, rank)


def med(v):
    v = sorted(v)
    return v[len(v) // 2]


for lo, hi in ((5750, 6750), (8750, 9250)):
    print(f"event at positions {lo}-{hi - 1}")
    print("  arm          window (min)   length | turns sent, per min | first turns | new prompt tokens K, per min | completion tokens K, per min | sessions ended | mean TTFT of all turns sent in the window | shared-turn mean TTFT at these positions")
    for a in C.LEAD5:
        t1 = med([B[a][k]["t_send"] for k in keys if lo - 100 <= rank.get(k, -1) < lo])
        t2 = med([B[a][k]["t_send"] for k in keys if hi <= rank.get(k, -1) < hi + 100])
        w = [t for t in D[a]["turns"] if t1 <= t["t_send"] < t2 and t["ttft"] is not None]
        new = sum((t["pt"] if t["turn"] == 0 else (t["delta"] or 0)) for t in w)
        ends = sum(1 for c, e in D[a]["conv_end"].items() if t1 <= e["end_s"] < t2)
        ks = [k for k in keys if lo <= rank.get(k, -1) < hi]
        L = (t2 - t1) / 60
        print(f"  {a:11s} {t1 / 60:6.1f}-{t2 / 60:6.1f}  {L:5.1f} | {len(w):5d} {len(w) / L:6.1f} | {sum(1 for t in w if t['turn'] == 0):3d} | {new / 1000:8.1f} {new / 1000 / L:7.1f} | "
              f"{sum(t['ct'] or 0 for t in w) / 1000:7.1f} {sum(t['ct'] or 0 for t in w) / 1000 / L:6.1f} | {ends:3d} | {C.mean(t['ttft'] for t in w):6.3f} | {C.mean(B[a][k]['ttft'] for k in ks):6.3f}")
