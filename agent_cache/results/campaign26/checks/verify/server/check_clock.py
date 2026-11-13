#!/usr/bin/env python3
"""Is (s2h_io line time - ms) a believable read start?  Pair every s2h_io line
with the launched srs_dispatch line of the same key and compare.

Run as:  nice -n 19 python3 -I -B check_clock.py
"""
import os
import sys

OUT = "/home/wanhr/sglang/agent_cache/results/campaign26/checks/verify/server"
sys.path.insert(0, OUT)
import analyze as A  # noqa: E402

lines_out = []


def P(s=""):
    lines_out.append(s)
    print(s)


P("read start (s2h_io line time - ms) minus the time of the launched srs_dispatch line of the same key, paired in order")
P("%-13s %7s %9s %9s | %9s %9s %9s %9s %9s | %10s %10s" % (
    "arm", "s2h", "launched", "paired", "min", "p01", "p50", "p99", "max", "n < -5 ms", "n > 1 s"))
for a in A.LEAD5 + ["jit_p50_ctl", "hintfree_ctl"]:
    r = A.Run(a)
    d = []
    n_s2h = len(r.s2h_all)
    n_l = sum(1 for v in r.srs.values() for x in v if x[7] == 1)
    for k, reads in r.s2h.items():
        ends = sorted(reads, key=lambda x: x[0])
        launches = sorted(x[0] for x in r.srs.get(k, ()) if x[7] == 1)
        if len(ends) != len(launches):
            continue
        for e, l in zip(ends, launches):
            d.append((e[0] - e[5] / 1000.0) - l)
    P("%-13s %7d %9d %9d | %9.4f %9.4f %9.4f %9.4f %9.4f | %10d %10d" % (
        a, n_s2h, n_l, len(d), min(d), A.pct(d, 0.01), A.pct(d, 0.5), A.pct(d, 0.99), max(d),
        sum(1 for z in d if z < -0.005), sum(1 for z in d if z > 1.0)))
with open(os.path.join(OUT, "check_clock.out"), "w") as f:
    f.write("\n".join(lines_out) + "\n")
