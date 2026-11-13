#!/usr/bin/env python3
"""Item 5, part j: the "hardware control" decode figure of compare_ec*.out, split by when the lines were logged.

Decode ms per step = 1000 x running / gen throughput on "Decode batch" lines with no queue (the definition the
file states).  If the gap between runs (33 against 37 ms at 8-15 running) is the machine, it shows in every
part of the run; if it comes from which lines enter the median, it goes away inside a fixed window.
Read-only pass over server.log of the finished runs; caches the lines in decode_lines.pkl.
"""
import calendar
import json
import math
import os
import pickle
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = "/home/wanhr/sglang/agent_cache/results"
with open(os.path.join(HERE, "parsed.pkl"), "rb") as f:
    P = pickle.load(f)
with open(os.path.join(RESULTS, ".current_run")) as f:
    CURRENT = f.read().strip()
RX = re.compile(r"^\[(\d{4})-(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)\.(\d{3})\] Decode batch, #running-req: (\d+), #token: (\d+), token usage: ([\d.]+),"
                r".*?gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)")
CACHE = os.path.join(HERE, "decode_lines.pkl")


def pct(v, q):
    s = sorted(v)
    return s[min(len(s) - 1, int(q * len(s)))]


def med(v):
    s = sorted(v)
    n = len(s)
    if not n:
        return float("nan")
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


if os.path.exists(CACHE):
    with open(CACHE, "rb") as f:
        D = pickle.load(f)
else:
    D = {}
    for a, d in P.items():
        if CURRENT and d["folder"] == CURRENT:
            sys.exit(2)
        t0 = d["run"][0]["t_start"]
        rows = []
        with open(os.path.join(RESULTS, d["folder"], "server.log"), "r", errors="replace") as f:
            for line in f:
                if "Decode batch" not in line:
                    continue
                m = RX.match(line)
                if not m:
                    continue
                g = m.groups()
                ts = calendar.timegm((int(g[0]), int(g[1]), int(g[2]), int(g[3]), int(g[4]), int(g[5]))) + int(g[6]) / 1000.0
                rows.append((ts - t0, int(g[7]), int(g[8]), float(g[9]), float(g[10]), int(g[11])))
        D[a] = rows
    with open(CACHE, "wb") as f:
        pickle.dump(D, f, protocol=pickle.HIGHEST_PROTOCOL)

PRINTED = {"baseline": 33.07, "base_rep": 33.39, "hintfree": 37.54, "jit_p50": 37.10, "jit_exact": 37.98, "jit_p50_ctl": 36.82,
           "hintfree_ctl": 37.25, "ec_r13": 37.59, "nofc": 37.12, "ec_r1": 37.50, "nofc_noend": 37.35, "jit_p50_d_cut": 33.57}
RES = {}
print("=== decode ms per step, lines with no queue and 8-15 running: whole run (to set against the printed figure), then by window")
print("   %-14s %8s %7s %8s | %14s | %14s | %14s | %14s | %s" % (
    "run", "printed", "n", "median", "min 0-15", "min 15-80", "min 80-120", "after min 120", "median #token K in the same four windows"))
for a, rows in D.items():
    sel = [(t, 1000.0 * r / thr, tok) for t, r, tok, use, thr, q in rows if q == 0 and 8 <= r <= 15 and thr > 0]
    w = {"0-15": [x for x in sel if x[0] < 900], "15-80": [x for x in sel if 900 <= x[0] < 4800],
         "80-120": [x for x in sel if 4800 <= x[0] < 7200], "120+": [x for x in sel if x[0] >= 7200]}
    r = {"printed": PRINTED.get(a), "n": len(sel), "median_all": med([x[1] for x in sel])}
    for k, v in w.items():
        r["n_" + k] = len(v)
        r["median_" + k] = med([x[1] for x in v])
        r["median_tokens_k_" + k] = med([x[2] for x in v]) / 1000.0 if v else float("nan")
    RES[a] = r
    print("   %-14s %8s %7d %8.2f | %6.2f (%5d) | %6.2f (%5d) | %6.2f (%5d) | %6.2f (%5d) | %s" % (
        a, ("%.2f" % PRINTED[a]) if a in PRINTED else "-", len(sel), r["median_all"],
        r["median_0-15"], r["n_0-15"], r["median_15-80"], r["n_15-80"], r["median_80-120"], r["n_80-120"], r["median_120+"], r["n_120+"],
        " ".join("%.0f" % r["median_tokens_k_" + k] if r["n_" + k] else "-" for k in w)))

print()
print("=== the same for 16-23 running, minute 10-25 only (all arms in step there)")
for a, rows in D.items():
    sel = [1000.0 * r / thr for t, r, tok, use, thr, q in rows if q == 0 and 16 <= r <= 23 and thr > 0 and 600 <= t < 1500]
    allr = [1000.0 * r / thr for t, r, tok, use, thr, q in rows if q == 0 and 16 <= r <= 23 and thr > 0]
    RES[a]["r16_23_min10_25_median"] = med(sel)
    RES[a]["r16_23_min10_25_n"] = len(sel)
    RES[a]["r16_23_all_median"] = med(allr)
    print("   %-14s minute 10-25: %6.2f (n=%4d)   whole run: %6.2f (n=%5d)" % (a, med(sel), len(sel), med(allr), len(allr)))

with open(os.path.join(HERE, "item5j_decode_control.json"), "w") as f:
    json.dump(RES, f, indent=1, sort_keys=True)
print()
print("wrote item5j_decode_control.json")
