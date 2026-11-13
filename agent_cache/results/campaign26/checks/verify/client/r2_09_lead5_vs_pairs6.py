#!/usr/bin/env python3
"""r2_09_lead5_vs_pairs6.py -- my LEAD5 numbers (11198 turns) set against the numbers the
regenerated pairs_ec.out prints under the same arm and pair labels (it is now on six arms,
11145 turns).  Shows which of my LEAD5 numbers no longer equal that file at printed precision
only because the turn set changed.

Run as:  nice -n 19 python3 -I -B r2_09_lead5_vs_pairs6.py
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
M = json.load(open(os.path.join(HERE, "r2_items_1_4.json")))
L = open(os.path.join(HERE, "r2_out_snapshot", "pairs_ec.out")).read().split("\n")
L5 = ["jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend"]
rows = []


def dec(s):
    s = s.strip().lstrip("+-")
    return len(s.split(".")[1]) if "." in s else 0


def chk(what, mine, theirs, signed=False):
    fmt = ("%+." if signed else "%.") + "%df" % dec(theirs)
    ms = fmt % mine
    rows.append((what, ms, theirs, ms == theirs or (float(ms) == 0 and float(theirs) == 0), mine))


def find(start):
    return next(k for k, s in enumerate(L) if s.startswith(start))


n = re.match(r"^(\d+) returning", L[0]).group(1)
chk("shared turns", M["sets"]["LEAD5"]["n_shared"], n)
i = find("per arm on them:")
k = i + 1
while L[k].strip():
    m = re.match(r"^\s+(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+) \|\s+([\d.]+)\s+"
                 r"([\d.]+) \|\s+(\d+)\s+(\d+)\s+(\d+)", L[k])
    a = m.group(1)
    if a in L5:
        st = M["sets"]["LEAD5"]["per_arm"][a]
        for name, key, g in (("mean", "mean", 2), ("median", "median", 3), ("p90", "p90", 4),
                             ("p99", "p99", 5), ("over 10 s", "n_over_10s", 6),
                             ("mean episodes", "mean_episodes", 7),
                             ("mean elsewhere", "mean_elsewhere", 8),
                             ("recomputed", "mean_recomputed", 11)):
            chk("%s %s" % (a, name), st[key], m.group(g))
    k += 1
for start, field, cname, ckey in (("all shared returning turns:", "mean_diff", None, None),
                                  ("the two load episodes", "mean_diff_episodes",
                                   "turns in load episodes", "n_shared_in_episodes"),
                                  ("the other positions:", "mean_diff_elsewhere",
                                   "turns elsewhere", "n_shared_elsewhere")):
    i = find(start)
    if cname:
        chk(cname, M["sets"]["LEAD5"][ckey], re.search(r": (\d+) turns", L[i]).group(1))
    cols = L[i + 1].split()
    k = i + 2
    while L[k].strip():
        row = L[k].split()[0]
        ents = re.findall(r"([+-]\d+\.\d+) \[\s*([+-][\d.]+),\s*([+-][\d.]+)\]", L[k])
        for col, (val, _a, _b) in zip(cols, ents):
            name = "%s-%s" % (row, col)
            if name in M["pairs"]["LEAD5"]:
                chk("%s %s" % (name, field), M["pairs"]["LEAD5"][name][field], val, signed=True)
        k += 1
ok = sum(1 for r in rows if r[3])
print("my LEAD5 numbers that pairs_ec.out prints under the same label: %d; equal at printed "
      "precision %d; different %d" % (len(rows), ok, len(rows) - ok))
for what, ms, th, good, raw in rows:
    if not good:
        print("   %-45s mine (LEAD5, 11198 turns) %9s (raw %.5f)   file (six arms, 11145 turns) %9s"
              % (what, ms, raw, th))
json.dump([{"what": r[0], "mine": r[1], "theirs": r[2], "equal": r[3], "raw": r[4]} for r in rows],
          open(os.path.join(HERE, "r2_lead5_vs_pairs6.json"), "w"), indent=1)
