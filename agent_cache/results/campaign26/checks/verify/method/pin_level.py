"""Mean pinned tokens over each arm's kept window, integrating the 'pinned=' level of srs_pin / srs_unpin lines in the
log's own line order (the cache keeps the two event types apart, which loses the order of lines that share a millisecond).
  nice -n 19 python3 -I -B pin_level.py"""
import calendar, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
from parse_cache_names import RUNS, RES, CURRENT

TS = re.compile(r"^\[(\d{4})-(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)\.(\d{3})\] ")
for a in ("jit_p50", "jit_p50_ctl", "jit_p50_d_cut", "jit_exact", "hintfree", "hintfree_ctl", "ec_r13", "ec_r1", "nofc", "nofc_noend"):
    assert RUNS[a] != CURRENT
    d = arm(a)
    total, t0, v0, n, ties = 0.0, 0.0, 0, 0, 0
    last_t = None
    with open(os.path.join(RES, RUNS[a], "server.log"), errors="replace") as f:
        for line in f:
            if "HICACHE_EVT srs_pin " not in line and "HICACHE_EVT srs_unpin " not in line:
                continue
            m = TS.match(line)
            p = re.search(r" pinned=(\d+)", line)
            if not m or not p:
                continue
            t = calendar.timegm((int(m[1]), int(m[2]), int(m[3]), int(m[4]), int(m[5]), int(m[6]))) + int(m[7]) / 1000 - d["t0"]
            if not 0 <= t < d["end"]:
                continue
            n += 1
            ties += t == last_t
            last_t = t
            total += v0 * (t - t0)
            t0, v0 = t, int(p[1])
    print(f"  {a:13s} pin/unpin lines in the window {n:5d} (sharing a millisecond with the line before: {ties:4d}) | mean pinned tokens {(total + v0 * (d['end'] - t0)) / d['end']:7.0f}")
