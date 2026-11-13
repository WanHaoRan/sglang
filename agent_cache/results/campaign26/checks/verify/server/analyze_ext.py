#!/usr/bin/env python3
"""Run the unchanged code of analyze.py on the arm sets the regenerated files use.

stall_ec.out (01:35Z 2026-11-12) is on six arms: LEAD5 + ec_r13_rep ("LEAD6");
compare_ec.out (01:33Z) is on thirteen: ALL11 + jit_p50_d + ec_r13_rep ("ALL13").
analyze.main() indexes its sets by the names LEAD5 and ALL11, so those two names are
given the extended contents here and renamed on the way out.  Output goes to ext/,
nothing of the frozen results is touched.  Everything here is "after opening".

Also a reproducibility check: every value of the unchanged sets (SEVEN, CUT7) and
every per-run value of the thirteen original runs must equal the frozen value exactly.

Run as:  nice -n 19 python3 -I -B analyze_ext.py
"""
import contextlib
import io
import json
import os
import sys

OUT = "/home/wanhr/sglang/agent_cache/results/campaign26/checks/verify/server"
EXT = OUT + "/ext"
sys.path.insert(0, OUT)
import analyze as A  # noqa: E402

os.makedirs(EXT, exist_ok=True)
LEAD6 = ["jit_exact", "ec_r13", "ec_r13_rep", "ec_r1", "nofc", "nofc_noend"]
ALL13 = list(A.ALL11) + ["jit_p50_d", "ec_r13_rep"]
OLD_ALL11 = list(A.ALL11)
OLD_LEAD5 = list(A.LEAD5)
A.ALL11 = ALL13
A.LEAD5 = LEAD6
A.SETS = [("LEAD5", LEAD6), ("SEVEN", A.SEVEN), ("ALL11", ALL13), ("CUT7", A.CUT7)]
A.OUT = EXT
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    A.main()

with open(os.path.join(EXT, "results_items1to5.json")) as f:
    raw = json.load(f)
ren = {"LEAD5": "LEAD6", "ALL11": "ALL13"}
vals = []
for r in raw:
    r = dict(r)
    r["arm_set"] = ren.get(r["arm_set"], r["arm_set"])
    vals.append(r)
with open(os.path.join(EXT, "ext_values.json"), "w") as f:
    json.dump(vals, f, indent=0)
with open(os.path.join(EXT, "results_items1to5.txt")) as f:
    txt = f.read()
txt = ("NOTE: in this file 'LEAD5' means LEAD6 = %s and 'ALL11' means ALL13 = %s\n\n" % (
    " ".join(LEAD6), " ".join(ALL13))) + txt
with open(os.path.join(EXT, "ext_results.txt"), "w") as f:
    f.write(txt)
os.remove(os.path.join(EXT, "results_items1to5.json"))
os.remove(os.path.join(EXT, "results_items1to5.txt"))

with open(os.path.join(OUT, "frozen_results_items1to5.json")) as f:
    fz = {(r["item"], r["arm"], r["arm_set"], r["metric"]): r["value"] for r in json.load(f)}
same = diff = absent = 0
bad = []
for r in vals:
    key = (r["item"], r["arm"], r["arm_set"], r["metric"])
    if r["arm_set"] in ("LEAD6", "ALL13"):
        continue
    if key not in fz:
        absent += 1
        continue
    a, b = r["value"], fz[key]
    if a == b or (a != a and b != b):
        same += 1
    else:
        diff += 1
        bad.append((key, a, b))
lines = ["analyze_ext.py: %d values written to ext/ext_values.json" % len(vals),
         "reproducibility of the frozen results (sets SEVEN, CUT7 and per-run values): identical %d, different %d, "
         "no frozen counterpart (new arms) %d" % (same, diff, absent)]
for key, a, b in bad[:40]:
    lines.append("   DIFFERENT %s now=%r frozen=%r" % (key, a, b))
with open(os.path.join(EXT, "analyze_ext.out"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
