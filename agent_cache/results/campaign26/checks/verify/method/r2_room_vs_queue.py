"""Read stall by cause, on the leading arms' shared returning turns (stall_ec.py's definitions): the stall carried by turns
that had a read launch refused host room, against the stall time spent behind other turns' reads, from the cache.
For a refused turn, 'after the first refusal' is the part of its stall window after its first launched=0 dispatch.
  nice -n 19 python3 -I -B r2_room_vs_queue.py"""
import bisect, collections, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import arm, shared, LEAD5, SEVEN, mean

LANDED, SLACK, BOUND = 0.05, 0.1, 0.1


def table(names, label):
    keys = shared(names)
    N = len(keys)
    print(f"\n== {label}: {N} shared returning turns; seconds per shared turn")
    print("   arm: read stall | late reading turns | of them with a refused launch: turns, their stall, of it after the first refusal, of that behind other reads | "
          "turns never refused: turns, their stall, of it behind other reads | all: behind other reads")
    for a in names:
        d = arm(a)
        allio = sorted((t - ms, t, k) for k, v in d["ios"].items() for t, ms, _ in v)
        starts = [x[0] for x in allio]
        first_ref = {}
        for e in d["server"]["evts"].get("srs_dispatch", []):
            if e["launched"] == 0:
                k = (e["conv"], e["turn"])
                first_ref[k] = min(first_ref.get(k, e["t"]), e["t"])

        def other(k, lo, hi):
            i = max(0, bisect.bisect_right(starts, lo) - 1)
            tot = 0.0
            while i < len(allio) and allio[i][0] < hi:
                s, e, kk = allio[i]
                if kk != k:
                    tot += max(0.0, min(e, hi) - max(s, lo))
                i += 1
            return tot

        ref = [0, 0.0, 0.0, 0.0]
        non = [0, 0.0, 0.0]
        late = 0
        tot_stall = tot_other = 0.0
        for k in keys:
            if not d["pages"][k] > 0:
                continue
            arrive, wait = d["entry"][k], d["queue"][k]
            admit = arrive + wait
            mine = [(t - ms, t) for t, ms, _ in d["ios"][k] if t <= admit + SLACK]
            if not mine:
                continue
            stall = min(max(max(e for _, e in mine) - arrive, 0.0), wait)
            if stall < LANDED:
                continue
            late += 1
            lo, hi = arrive, arrive + stall
            o = other(k, lo, hi)
            tot_stall += stall
            tot_other += o
            if k in first_ref and first_ref[k] < hi:
                r = max(first_ref[k], lo)
                ref[0] += 1
                ref[1] += stall
                ref[2] += hi - r
                ref[3] += other(k, r, hi)
            else:
                non[0] += 1
                non[1] += stall
                non[2] += o
        print(f"   {a:12s} {tot_stall / N:.3f} | {late:5d} | {ref[0]:4d} {ref[1] / N:.3f} {ref[2] / N:.3f} {ref[3] / N:.3f} | {non[0]:5d} {non[1] / N:.3f} {non[2] / N:.3f} | {tot_other / N:.3f}")


table(LEAD5, "LEAD5")
table(SEVEN, "SEVEN")
