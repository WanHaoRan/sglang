"""What a reading turn's read stall is made of. For every shared returning turn whose read had not landed at arrival,
its stall window [arrival, end of its last read before admission] is split into: its own reads running; the read thread
busy with other turns' reads; the read thread idle (the read is parked, held, refused room, or not yet issued).
Also the read thread's busy share at finer grain than 5 minutes, and inside the stall windows.
  nice -n 19 python3 -I -B stall_parts.py"""
import bisect, collections, os, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *

LANDED, SLACK, BOUND = 0.05, 0.1, 0.1
rk = rank()


def merged(iv):
    out = []
    for lo, hi in sorted(iv):
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return out


def overlap(busy, starts, lo, hi):
    if hi <= lo:
        return 0.0
    i = max(0, bisect.bisect_right(starts, lo) - 1)
    tot = 0.0
    while i < len(busy) and busy[i][0] < hi:
        tot += max(0.0, min(hi, busy[i][1]) - max(lo, busy[i][0]))
        i += 1
    return tot


def parts(names, label):
    keys = shared(names)
    N = len(keys)
    print(f"\n== {label}: {N} shared returning turns; seconds per shared turn")
    print("   arm: read stall | own reads running | read thread busy with other reads | read thread idle || the same three for the class admitted within 0.1 s of landing || "
          "read thread busy: share of the run, max over 5 min / 1 min / 30 s windows, minutes over 90 % busy, share of all stall-window time")
    for a in names:
        d = arm(a)
        busy = merged([(t - ms, t) for v in d["ios"].values() for t, ms, _ in v])
        starts = [b[0] for b in busy]
        tot = collections.Counter()
        last = collections.Counter()
        win_time = win_busy = 0.0
        for k in keys:
            if not d["pages"][k] > 0:
                continue
            arrive, wait = d["entry"][k], d["queue"][k]
            admit = arrive + wait
            mine = [(end - ms, end) for end, ms, _ in d["ios"][k] if end <= admit + SLACK]
            if not mine:
                continue
            end = min(max(e for _, e in mine), admit)
            stall = min(max(end - arrive, 0.0), wait)
            if stall < LANDED:
                continue
            lo, hi = arrive, arrive + stall
            own = merged([(max(s, lo), min(e, hi)) for s, e in mine if min(e, hi) > max(s, lo)])
            own_t = sum(e - s for s, e in own)
            b = overlap(busy, starts, lo, hi)
            row = (stall, own_t, max(0.0, b - own_t), stall - max(b, own_t))
            for i, x in enumerate(row):
                tot[i] += x
                if wait - stall < BOUND:
                    last[i] += x
            win_time += stall
            win_busy += b
        t0, t1 = d["t0"], d["t0"] + d["end"]
        def peak(w):
            return max(overlap(busy, starts, t0 + i * w / 2, t0 + i * w / 2 + w) / w for i in range(int(2 * d["end"] // w) - 1))
        mins90 = sum(overlap(busy, starts, t0 + 60 * m, t0 + 60 * (m + 1)) / 60 > 0.9 for m in range(int(d["end"] // 60)))
        print(f"   {a:12s} {tot[0] / N:.3f} | {tot[1] / N:.3f} | {tot[2] / N:.3f} | {tot[3] / N:.3f} || {last[0] / N:.3f}: {last[1] / N:.3f} {last[2] / N:.3f} {last[3] / N:.3f} || "
              f"{100 * overlap(busy, starts, t0, t1) / d['end']:.0f} %, {100 * peak(300):.0f} % / {100 * peak(60):.0f} % / {100 * peak(30):.0f} %, {mins90} min, {100 * win_busy / max(win_time, 1e-9):.0f} %")


parts(LEAD5, "LEAD5 (stall_ec.out's set)")
parts(SEVEN, "SEVEN (stall_ec_ctl.out's set)")
