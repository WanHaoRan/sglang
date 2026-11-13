"""Are SSD-to-host reads served one at a time? From the raw server.log of each arm (not the cache): every s2h_io line gives
an interval [line time - ms, line time]. Counts intervals that overlap the one before by more than the 1 ms rounding of
the log, the busy share of the union, and the longest chain of back-to-back reads (gap under 20 ms).
  nice -n 19 python3 -I -B r2_serial_reads.py"""
import calendar, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_cache_names import RUNS, RES, CURRENT

TS = re.compile(r"^\[(\d{4})-(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)\.(\d{3})\] HICACHE_EVT s2h_io rid=(\S+) pages=(\d+) tokens=(\d+) ms=(\d+)\s*$")
print("arm: s2h_io lines | parsed | overlapping the previous read by >2 ms, by >50 ms | median, max overlap of those (ms) | reads starting within 20 ms of the previous end | longest back-to-back chain (reads, seconds)")
for a in ("nofc", "nofc_noend", "ec_r13", "ec_r1", "jit_exact", "jit_p50_ctl", "hintfree_ctl", "baseline"):
    assert RUNS[a] != CURRENT
    n = 0
    iv = []
    with open(os.path.join(RES, RUNS[a], "server.log"), errors="replace") as f:
        for line in f:
            if "HICACHE_EVT s2h_io" not in line:
                continue
            n += 1
            m = TS.match(line)
            if not m:
                continue
            t = calendar.timegm((int(m[1]), int(m[2]), int(m[3]), int(m[4]), int(m[5]), int(m[6]))) + int(m[7]) / 1000
            iv.append((t - int(m[11]) / 1000, t, m[8]))
    iv.sort()
    over = []
    b2b = 0
    chain = best = 1
    chain_start = iv[0][0]
    best_s = 0.0
    prev_end = iv[0][1]
    for s, e, rid in iv[1:]:
        if s < prev_end - 0.002:
            over.append(prev_end - s)
        if s - prev_end < 0.020:
            b2b += 1
            chain += 1
        else:
            chain = 1
            chain_start = s
        if chain > best:
            best, best_s = chain, e - chain_start
        prev_end = max(prev_end, e)
    over.sort()
    print(f"  {a:13s} {n:5d} | {len(iv):5d} | {len(over):4d} {sum(x > 0.05 for x in over):4d} | {1000 * over[len(over) // 2] if over else 0:6.0f} {1000 * over[-1] if over else 0:6.0f} | {b2b:5d} | {best:3d} {best_s:6.1f}")
