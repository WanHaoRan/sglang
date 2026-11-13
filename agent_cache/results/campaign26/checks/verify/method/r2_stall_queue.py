"""What the read stall of stall_ec.py is made of, recomputed from the raw server.log (reads and ReqTimeStats) with the
kept turns from the cache: inside each late reading turn's stall window [arrival, last own read end before admission],
time its own read ran, time the (single) read thread ran another turn's read, time the thread was idle.
  nice -n 19 python3 -I -B r2_stall_queue.py"""
import bisect, calendar, collections, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import arm, shared, rank, LEAD5, SEVEN, mean, pct
from parse_cache_names import RUNS, RES, CURRENT

LANDED, SLACK, BOUND = 0.05, 0.1, 0.1
IO = re.compile(r"^\[(\d{4})-(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)\.(\d{3})\] HICACHE_EVT s2h_io rid=\S*-c(\d+)-t(\d+) pages=(\d+) tokens=(\d+) ms=(\d+)\s*$")
RT = re.compile(r"^\[[^\]]+\] ReqTimeStats\(rid=[^,]*-c(\d+)-t(\d+), .*queue_duration=([\d.]+)ms, forward_duration=([\d.]+)ms, entry_time=([\d.]+)\s*$")


def raw(a):
    assert RUNS[a] != CURRENT
    entry, ios, allio = {}, collections.defaultdict(list), []
    with open(os.path.join(RES, RUNS[a], "server.log"), errors="replace") as f:
        for line in f:
            if "HICACHE_EVT s2h_io" in line:
                m = IO.match(line)
                t = calendar.timegm((int(m[1]), int(m[2]), int(m[3]), int(m[4]), int(m[5]), int(m[6]))) + int(m[7]) / 1000
                k = (int(m[8]), int(m[9]))
                ios[k].append((t - int(m[12]) / 1000, t))
                allio.append((t - int(m[12]) / 1000, t, k))
            elif "ReqTimeStats(" in line:
                m = RT.match(line)
                if m:
                    entry[(int(m[1]), int(m[2]))] = (float(m[5]), float(m[3]) / 1000)
    allio.sort()
    return entry, ios, allio


def table(names, label):
    keys = shared(names)
    N = len(keys)
    print(f"\n== {label}: {N} shared returning turns; seconds per shared turn unless said")
    print("   arm: mean TTFT | read stall = own read running + thread on another turn's read + thread idle | share of stall behind other reads | "
          "class admitted within 0.1 s of landing: stall = own + other + idle | late reading turns, of them with >=0.1 s behind other reads, their mean s behind | "
          "stall if no turn waited behind another read")
    for a in names:
        d = arm(a)
        entry, ios, allio = raw(a)
        starts = [x[0] for x in allio]
        tot = [0.0] * 4
        last = [0.0] * 4
        late = behind = 0
        behind_s = 0.0
        for k in keys:
            if k not in ios:
                continue
            arrive, wait = entry[k]
            admit = arrive + wait
            mine = [(s, e) for s, e in ios[k] if e <= admit + SLACK]
            if not mine:
                continue
            stall = min(max(max(e for _, e in mine) - arrive, 0.0), wait)
            if stall < LANDED:
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
            # own reads after admission+slack are not in `mine` but are this turn's: none exist (stall_ec.out prints 0)
            row = (stall, own, other, stall - own - other)
            late += 1
            if other >= 0.1:
                behind += 1
                behind_s += other
            for j in range(4):
                tot[j] += row[j]
                if wait - stall < BOUND:
                    last[j] += row[j]
        t = mean(d["by"][k]["ttft"] for k in keys)
        print(f"   {a:12s} {t:.3f} | {tot[0] / N:.3f} = {tot[1] / N:.3f} + {tot[2] / N:.3f} + {tot[3] / N:.3f} | {100 * tot[2] / tot[0]:4.1f} % | "
              f"{last[0] / N:.3f} = {last[1] / N:.3f} + {last[2] / N:.3f} + {last[3] / N:.3f} | {late:5d} {behind:5d} {behind_s / max(behind, 1):5.2f} | {(tot[0] - tot[2]) / N:.3f}")


table(LEAD5, "LEAD5 (stall_ec.out's set)")
table(SEVEN, "SEVEN (stall_ec_ctl.out's set)")
