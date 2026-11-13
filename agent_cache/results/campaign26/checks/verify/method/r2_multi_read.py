"""Inside the class stall_ec.py calls 'the read was the last thing it waited for' (admitted within 0.1 s of its last
read's end): how much of that stall belongs to turns that read more than once after arriving, and how much of it is the
time between the end of one of the turn's reads and the start of its next one (the turn was waiting for something
else, lost what it had, and read again). From the cache; definitions as stall_ec.py.
  nice -n 19 python3 -I -B r2_multi_read.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import arm, shared, LEAD5, mean

LANDED, SLACK, BOUND = 0.05, 0.1, 0.1
keys = shared(LEAD5)
N = len(keys)
print(f"{N} shared returning turns (LEAD5); seconds per shared turn")
print("arm: stall of the class 'read was the last wait' (turns) | of it in turns with one read after arrival (turns) | in turns with two or more (turns) | "
      "of the latter: before the first of those reads ended, between reads, in the last read | the class without the time before the last read began in multi-read turns")
for a in LEAD5:
    d = arm(a)
    one = [0, 0.0]
    multi = [0, 0.0, 0.0, 0.0, 0.0]
    for k in keys:
        if not d["pages"][k] > 0:
            continue
        arrive, wait = d["entry"][k], d["queue"][k]
        admit = arrive + wait
        mine = sorted((t - ms, t) for t, ms, _ in d["ios"][k] if t <= admit + SLACK)
        if not mine:
            continue
        stall = min(max(max(e for _, e in mine) - arrive, 0.0), wait)
        if stall < LANDED or wait - stall >= BOUND:
            continue
        after = [(s, e) for s, e in mine if e > arrive]
        if len(after) <= 1:
            one[0] += 1
            one[1] += stall
        else:
            first_end = after[0][1]
            last_start = after[-1][0]
            multi[0] += 1
            multi[1] += stall
            multi[2] += min(stall, max(0.0, first_end - arrive))
            multi[3] += max(0.0, min(arrive + stall, last_start) - max(arrive, first_end)) - sum(max(0.0, min(e, last_start) - max(s, first_end)) for s, e in after[1:-1])
            multi[4] += max(0.0, min(arrive + stall, after[-1][1]) - max(arrive, last_start))
    tot = one[1] + multi[1]
    print(f"  {a:11s} {tot / N:.3f} ({one[0] + multi[0]:4d}) | {one[1] / N:.3f} ({one[0]:4d}) | {multi[1] / N:.3f} ({multi[0]:3d}) | {multi[2] / N:.3f} {multi[3] / N:.3f} {multi[4] / N:.3f} | {(tot - multi[1] + multi[4]) / N:.3f}")
