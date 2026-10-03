#!/usr/bin/env python3
"""requery_share.py RUN_DIR...: per run, SSD reads (s2h_io) whose last lookup before the read was the arrival-time lookup
vs a re-issued one (#39283 admission re-query or a paced miss-retry)."""
import re, sys, datetime
EV = re.compile(r"^\[(\S+ \S+)\] HICACHE_EVT (prefetch_start|s2h_io) rid=(cmp\S+)")
def ts(s):
    return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").timestamp()
for run in sys.argv[1:]:
    ps, io = {}, {}
    for line in open(f"{run}/server.log", errors="replace"):
        m = EV.match(line)
        if not m:
            continue
        if m[2] == "prefetch_start":
            ps.setdefault(m[3], []).append(ts(m[1]))
        else:
            io.setdefault(m[3], ts(m[1]))
    first = later = 0
    delay = []
    for rid, t in io.items():
        before = [x for x in ps.get(rid, ()) if x <= t]
        if not before:
            continue
        if len(before) == 1:
            first += 1
        else:
            later += 1
            delay.append(before[-1] - before[0])
    delay.sort()
    med = delay[len(delay) // 2] if delay else float("nan")
    multi = sum(len(v) > 1 for v in ps.values())
    print(f"{run}: requests with >1 lookup {multi}, SSD reads {first + later}: arrival lookup {first}, re-issued lookup {later}"
          f" (median {med:.1f} s after arrival)")
