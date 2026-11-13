"""The read stall of ec_r13 and of its repeat split as in r2_stall_queue.py (own read running, read thread on another
turn's read, read thread idle), on the 11,145 turns of the regenerated stall_ec.out, from the cache; plus what the
repeat's extra TTFT is made of, and the read thread's waits in each run.
  nice -n 19 python3 -I -B r2_stall_queue_rep.py"""
import bisect, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import arm, shared, rank, mean, rec

LANDED, SLACK, BOUND = 0.05, 0.1, 0.1
NAMES = ["jit_exact", "ec_r13", "ec_r13_rep", "ec_r1", "nofc", "nofc_noend"]
keys = shared(NAMES)
N = len(keys)
rk = rank()
print(f"{N} returning turns shared by {', '.join(NAMES)}; seconds per shared turn")
print("arm: TTFT = stall + rest of queue + after admission | stall = own read + behind other reads + thread idle | class 'read was the last wait' = own + other + idle (turns) | "
      "class 'GPU queue held it as well': stall = own + other + idle (turns), wait after landing | recomputed tokens per turn")
out = {}
for a in NAMES:
    d = arm(a)
    allio = sorted((t - ms, t, k) for k, v in d["ios"].items() for t, ms, _ in v)
    starts = [x[0] for x in allio]
    tot, last, both = [0.0] * 4, [0.0] * 4, [0.0] * 5
    nl = nb = 0
    per = {}
    for k in keys:
        arrive, wait = d["entry"][k], d["queue"][k]
        per[k] = [0.0, 0.0, 0.0, 0.0, wait, d["by"][k]["ttft"]]
        if not d["pages"][k] > 0:
            continue
        admit = arrive + wait
        mine = [(t - ms, t) for t, ms, _ in d["ios"][k] if t <= admit + SLACK]
        if not mine:
            continue
        stall = min(max(max(e for _, e in mine) - arrive, 0.0), wait)
        if stall < LANDED:
            per[k][0] = stall
            continue
        lo, hi = arrive, arrive + stall
        own = sum(max(0.0, min(e, hi) - max(s, lo)) for s, e in mine)
        i = max(0, bisect.bisect_right(starts, lo) - 1)
        other = 0.0
        while i < len(allio) and allio[i][0] < hi:
            s, e, kk = allio[i]
            if kk != k:
                other += max(0.0, min(e, hi) - max(s, lo))
            i += 1
        row = (stall, own, other, stall - own - other)
        per[k][:4] = row
        for j in range(4):
            tot[j] += row[j]
        if wait - stall < BOUND:
            nl += 1
            for j in range(4):
                last[j] += row[j]
        else:
            nb += 1
            for j in range(4):
                both[j] += row[j]
            both[4] += wait - stall
    out[a] = per
    t = mean(d["by"][k]["ttft"] for k in keys)
    w = mean(d["queue"][k] for k in keys)
    st_all = mean(per[k][0] for k in keys)
    print(f"  {a:11s} {t:.3f} = {st_all:.3f} + {w - st_all:.3f} + {t - w:.3f} | {tot[0] / N:.3f} = {tot[1] / N:.3f} + {tot[2] / N:.3f} + {tot[3] / N:.3f} | "
          f"{last[0] / N:.3f} = {last[1] / N:.3f} + {last[2] / N:.3f} + {last[3] / N:.3f} ({nl:4d}) | {both[0] / N:.3f} = {both[1] / N:.3f} + {both[2] / N:.3f} + {both[3] / N:.3f} ({nb:4d}), {both[4] / N:.3f} | {mean(rec(d['by'][k]) for k in keys):.0f}")

print("\nec_r13_rep minus ec_r13, paired, seconds per shared turn, by stretch of the order run: turns | TTFT | stall: own read, behind other reads, thread idle | rest of the queue wait | after admission")
A, B = out["ec_r13"], out["ec_r13_rep"]
for lo, hi in ((0, 5500), (5500, 7000), (7000, 8750), (8750, 11000), (-1, 0)):
    ks = [k for k in keys if (lo <= rk.get(k, -1) < hi)]
    if not ks:
        continue
    f = lambda j: sum(B[k][j] - A[k][j] for k in ks) / N
    stall = f(0)
    rest = sum((B[k][4] - B[k][0]) - (A[k][4] - A[k][0]) for k in ks) / N
    after = sum((B[k][5] - B[k][4]) - (A[k][5] - A[k][4]) for k in ks) / N
    name = "absent" if lo < 0 else f"{lo}-{hi}"
    print(f"   {name:11s} {len(ks):5d} | {f(5):+.3f} | {stall:+.3f}: {f(1):+.3f} {f(2):+.3f} {f(3):+.3f} | {rest:+.3f} | {after:+.3f}")
f = lambda j: sum(B[k][j] - A[k][j] for k in keys) / N
print(f"   all         {N:5d} | {f(5):+.3f} | {f(0):+.3f}: {f(1):+.3f} {f(2):+.3f} {f(3):+.3f} | {sum((B[k][4] - B[k][0]) - (A[k][4] - A[k][0]) for k in keys) / N:+.3f} | {sum((B[k][5] - B[k][4]) - (A[k][5] - A[k][4]) for k in keys) / N:+.3f}")
