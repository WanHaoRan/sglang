#!/usr/bin/env python3
"""Explain every difference between recheck.json (exact integer-microsecond arithmetic,
raw logs) and frozen_results_items1to5.json (float arithmetic on epoch seconds, cache).

Run as:  nice -n 19 python3 -I -B recheck_diff.py
"""
import json
import os

OUT = "/home/wanhr/sglang/agent_cache/results/campaign26/checks/verify/server"
out = []


def P(s=""):
    out.append(s)
    print(s)


with open(os.path.join(OUT, "recheck.json")) as f:
    rc = json.load(f)
with open(os.path.join(OUT, "frozen_results_items1to5.json")) as f:
    fz = {(r["item"], r["arm"], r["arm_set"], r["metric"]): r["value"] for r in json.load(f)}
with open(os.path.join(OUT, "recheck_turns.json")) as f:
    turns = json.load(f)

n = same_int = int_diff = f_tiny = f_small = f_big = 0
int_rows = []
big_rows = []
worst_tiny = 0.0
for r in rc:
    key = (r["item"], r["arm"], r["arm_set"], r["metric"])
    a, b = r["value"], fz[key]
    n += 1
    if isinstance(a, int) and isinstance(b, int):
        if a == b:
            same_int += 1
        else:
            int_diff += 1
            int_rows.append((key, a, b))
        continue
    d = abs(a - b)
    if d <= 1e-6:
        f_tiny += 1
        worst_tiny = max(worst_tiny, d)
    elif d <= 1e-4:
        f_small += 1
        big_rows.append((key, a, b))
    else:
        f_big += 1
        big_rows.append((key, a, b))
P("values compared: %d" % n)
P("integers equal: %d   integers different: %d" % (same_int, int_diff))
P("floats within 1e-6 absolute: %d (worst %.3g)   1e-6..1e-4: %d   above 1e-4: %d" % (
    f_tiny, worst_tiny, f_small, f_big))
P()
P("INTEGER DIFFERENCES (recheck exact - frozen float)")
for key, a, b in int_rows:
    P("  %-8s %-13s %-6s %-55s exact=%d float=%d  (%+d)" % (key[0], key[1], key[2], key[3], a, b, a - b))
P()
P("FLOAT DIFFERENCES ABOVE 1e-6 (recheck exact - frozen float)")
for key, a, b in big_rows:
    P("  %-17s %-13s %-6s %-50s exact=%.6f float=%.6f  (%+.6f)" % (key[0], key[1], key[2], key[3], a, b, a - b))
P()
P("KNIFE-EDGE TURNS in the exact per-turn table (reading turns only)")
P("%-13s %7s %14s %16s %16s %16s %16s" % ("arm", "turns", "stall==0.050", "q-st==0.050", "q-st==0.100",
                                          "q-st==0.200", "q-st==0.500"))
for arm, rows in sorted(turns.items()):
    e_thr = [k for k, v in rows.items() if v[2] > 0 and v[1] == 50000]
    cnt = {}
    for b in (50000, 100000, 200000, 500000):
        cnt[b] = [k for k, v in rows.items() if v[2] > 0 and v[1] >= 50000 and (v[0] - v[1]) == b]
    P("%-13s %7d %14s %16d %16d %16d %16d" % (
        arm, len(rows), "%d %s" % (len(e_thr), ",".join(sorted(e_thr))[:60]),
        len(cnt[50000]), len(cnt[100000]), len(cnt[200000]), len(cnt[500000])))
P()
P("NEAR-EDGE COUNTS: reading turns with stall within 1 ms / 5 ms of 0.05 s (exact table)")
for arm, rows in sorted(turns.items()):
    rd = [v for v in rows.values() if v[2] > 0]
    P("%-13s reading=%d  stall in [0.049,0.051]: %d   in [0.045,0.055]: %d   stall in (0,0.05): %d   stall==0: %d" % (
        arm, len(rd), sum(1 for v in rd if 49000 <= v[1] <= 51000),
        sum(1 for v in rd if 45000 <= v[1] <= 55000),
        sum(1 for v in rd if 0 < v[1] < 50000), sum(1 for v in rd if v[1] == 0)))
with open(os.path.join(OUT, "recheck_diff.out"), "w") as f:
    f.write("\n".join(out) + "\n")
