#!/usr/bin/env python3
"""Item 5, part k: the "SSD ms per page" figure of the hardware-control line, inside a fixed window.

SSD ms per page in reads of 500 pages or more (the definition the file states), from HICACHE_EVT s2h_io lines.
Whole run against the first 75 minutes only, which every arm including the cut one has.
"""
import calendar
import json
import os
import pickle
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = "/home/wanhr/sglang/agent_cache/results"
with open(os.path.join(HERE, "parsed.pkl"), "rb") as f:
    P = pickle.load(f)
with open(os.path.join(RESULTS, ".current_run")) as f:
    CURRENT = f.read().strip()
RX = re.compile(r"^\[(\d{4})-(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)\.(\d{3})\].*HICACHE_EVT s2h_io rid=\S+ pages=(\d+) tokens=(\d+) ms=([\d.]+)")
CACHE = os.path.join(HERE, "s2h_lines.pkl")


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
                if "s2h_io" not in line:
                    continue
                m = RX.match(line)
                if not m:
                    continue
                g = m.groups()
                ts = calendar.timegm((int(g[0]), int(g[1]), int(g[2]), int(g[3]), int(g[4]), int(g[5]))) + int(g[6]) / 1000.0
                rows.append((ts - t0, int(g[7]), int(g[8]), float(g[9])))
        D[a] = rows
    with open(CACHE, "wb") as f:
        pickle.dump(D, f, protocol=pickle.HIGHEST_PROTOCOL)

PRINTED = {"baseline": 2.87, "base_rep": 3.05, "hintfree": 2.25, "jit_p50": 2.47, "jit_exact": 2.20, "jit_p50_ctl": 2.49,
           "hintfree_ctl": 2.22, "ec_r13": 2.18, "nofc": 2.18, "ec_r1": 2.22, "nofc_noend": 2.24, "jit_p50_d_cut": 2.79}
RES = {}
print("=== SSD ms per page, reads of 500 pages or more: whole run (against the printed figure), then by window (end time of the read)")
print("   %-14s %8s %6s %8s | %13s | %13s | %13s | %13s | reads overlapping another read (share, whole run)" % (
    "run", "printed", "n", "median", "min 0-45", "min 45-75", "min 75-120", "episode1 own"))
for a, rows in D.items():
    big = [(t, ms / pg, pg, ms) for t, pg, tok, ms in rows if pg >= 500]
    w = {"0-45": [x[1] for x in big if x[0] < 2700], "45-75": [x[1] for x in big if 2700 <= x[0] < 4500],
         "75-120": [x[1] for x in big if 4500 <= x[0] < 7500]}
    # overlap: a read [end-ms, end] that intersects any other read of the run (all sizes)
    iv = sorted((t - ms / 1000.0, t) for t, pg, tok, ms in rows)
    ov = 0
    for i, (s, e) in enumerate(iv):
        hit = (i > 0 and iv[i - 1][1] > s) or (i + 1 < len(iv) and iv[i + 1][0] < e)
        ov += 1 if hit else 0
    r = {"printed": PRINTED.get(a), "reads_all": len(rows), "reads_500": len(big), "median_all": med([x[1] for x in big]),
         "share_overlapping": ov / len(iv) if iv else None}
    for k, v in w.items():
        r["n_" + k] = len(v)
        r["median_" + k] = med(v)
    RES[a] = r
    print("   %-14s %8s %6d %8.2f | %5.2f (%5d) | %5.2f (%5d) | %5.2f (%5d) | %s | %.3f of %d reads" % (
        a, ("%.2f" % PRINTED[a]) if a in PRINTED else "-", len(big), r["median_all"], r["median_0-45"], r["n_0-45"],
        r["median_45-75"], r["n_45-75"], r["median_75-120"], r["n_75-120"], "", r["share_overlapping"], len(iv)))

with open(os.path.join(HERE, "item5k_ssd_control.json"), "w") as f:
    json.dump(RES, f, indent=1, sort_keys=True)
print()
print("wrote item5k_ssd_control.json")
