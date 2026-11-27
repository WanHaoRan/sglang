"""Part 3 summary from ../out/f4_readers_*.csv: readers by the GPU's state at arrival, and the c the readers saw
(share of their stall time with no other ready turn waiting, stall-weighted) by regime.
usage: part3_state_summary.py"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402

RUNS = (("c34", "ec_full", 40), ("c34", "jit_exact", 40), ("c34", "hicache", 40), ("c34", "jit_p50", 40), ("c34", "ec_exact", 40),
        ("c33", "ec_full", 110), ("c33", "jit_exact", 110), ("c33", "hicache", 110), ("c33", "jit_p50", 110), ("c33", "ec_exact", 110),
        ("c33", "oracle_v2", 110))
print("Returning requests with an SSD read (issued at or after arrival), by the GPU's state at arrival; first 40 minutes of campaign 34, "
      "whole run (to minute 110) of campaign 33")
print("run | readers | ready queue empty | of them pool use < 0.90 | ready queue 1-3 | 4+ | stall s by group (empty&room / empty no room / queue) | "
      "tokens read K | idle recompute s | loaded recompute s")
for camp, arm, hi in RUNS:
    rows = [r for r in csv.DictReader(open(f"{lib.OUT}/f4_readers_{camp}_{arm}.csv")) if float(r["arr"]) < hi * 60 and r["before"] == "False"]
    if not rows:
        print(f"{camp}:{arm} | 0")
        continue
    q0 = [r for r in rows if int(r["q_rdy"]) == 0]
    q0room = [r for r in q0 if float(r["use"]) < 0.90]
    q0full = [r for r in q0 if float(r["use"]) >= 0.90]
    q13 = [r for r in rows if 1 <= int(r["q_rdy"]) <= 3]
    q4 = [r for r in rows if int(r["q_rdy"]) >= 4]
    m = lambda xs, k: (sum(float(x[k]) for x in xs) / len(xs)) if xs else float("nan")  # noqa: E731
    print(f"{camp}:{arm} | {len(rows)} | {len(q0)} | {len(q0room)} | {len(q13)} | {len(q4)} | {m(q0room,'stall'):.2f} / {m(q0full,'stall'):.2f} / "
          f"{m(q13+q4,'stall'):.2f} | {m(rows,'X')/1e3:.1f} | {m(rows,'gi'):.2f} | {m(rows,'gl'):.2f}")
print()
print("c seen by the readers themselves: share of their stall time [due, admission] with no other ready turn waiting (stall-weighted), by regime")
print("run | minutes: readers, total stall s, stall-weighted c_i, mean c_i")
for camp, arm in (("c34", "ec_full"), ("c34", "jit_exact"), ("c34", "jit_p50"), ("c34", "ec_exact"), ("c34", "hicache"), ("c33", "hicache"),
                  ("c33", "jit_p50"), ("c33", "ec_full")):
    rows = list(csv.DictReader(open(f"{lib.OUT}/f4_readers_{camp}_{arm}.csv")))
    out = []
    for lo, hi in ((0, 8), (8, 28), (28, 70), (70, 90), (90, 110)):
        w = [r for r in rows if lo * 60 <= float(r["arr"]) < hi * 60]
        s = sum(float(r["stall"]) for r in w)
        cw = sum(float(r["ci"]) * float(r["stall"]) for r in w) / s if s > 0 else float("nan")
        cm = sum(float(r["ci"]) for r in w) / len(w) if w else float("nan")
        out.append(f"{lo}-{hi}: {len(w)}, {s:.0f} s, {cw:.3f}, {cm:.2f}")
    print(f"{camp}:{arm} | " + " | ".join(out))
