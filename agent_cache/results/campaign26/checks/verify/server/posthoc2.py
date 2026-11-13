#!/usr/bin/env python3
"""More figures computed after opening, for the concerns list:
pre-entry delay inside and outside the load episodes, and the second
episode's share of each arm's queue wait.

Run as:  nice -n 19 python3 -I -B posthoc2.py
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


labels = sorted(set(A.ALL11 + A.CUT7 + ["order_run"]))
R = {lab: A.Run(lab) for lab in labels}
pos = {(t["conv"], t["turn"]): i for i, t in enumerate(R["order_run"].ok_sorted)}
shared = {}
for sname, arms in A.SETS:
    s = None
    for a in arms:
        s = set(R[a].kept_ret) if s is None else (s & R[a].kept_ret)
    shared[sname] = sorted(s)


def scope(k):
    p = pos.get(k)
    if p is None:
        return "else"
    if 5500 <= p <= 6999:
        return "ep1"
    if 8750 <= p <= 10999:
        return "ep2"
    return "else"


P("PRE-ENTRY DELAY (entry_time - client send) and TTFT - queue wait, mean by scope, LEAD5 shared turns")
P("%-11s | ep1: pre-entry, TTFT-QW | ep2: pre-entry, TTFT-QW | elsewhere: pre-entry, TTFT-QW | all: pre-entry, TTFT-QW, share" % "arm")
keys = shared["LEAD5"]
for a in A.LEAD5:
    r = R[a]
    cells = []
    for sc in ("ep1", "ep2", "else", None):
        ks = [k for k in keys if sc is None or scope(k) == sc]
        pe = sum(r.arrival(k) - (r.t_start + r.cli[k]["t_send"]) for k in ks) / len(ks)
        aa = sum(r.cli[k]["ttft"] - r.qw(k) for k in ks) / len(ks)
        cells.append("%.4f %.4f" % (pe, aa) + ((" %.3f" % (pe / aa)) if sc is None else ""))
    P("%-11s | %s" % (a, " | ".join(cells)))
P()
P("SECOND EPISODE'S SHARE of each arm's summed queue wait, read stall and SSD pages, LEAD5 shared turns (1519 of 11198 turns)")
for a in A.LEAD5:
    r = R[a]
    ks = [k for k in keys if scope(k) == "ep2"]
    q = sum(r.qw(k) for k in ks) / sum(r.qw(k) for k in keys)
    s_ = sum(r.stall(k, 0.1)[0] for k in ks) / sum(r.stall(k, 0.1)[0] for k in keys)
    pg = sum(x[3] for k in ks for x in r.s2h.get(k, ())) / sum(x[3] for k in keys for x in r.s2h.get(k, ()))
    P("  %-11s queue wait %.3f  stall %.3f  pages %.3f" % (a, q, s_, pg))
P()
P("SAME FIGURES ON THREE SETS: mean queue wait, mean stall, SSD pages of the leading arms on ALL11 (6715), SEVEN (9702), LEAD5 (11198)")
for sname in ("ALL11", "SEVEN", "LEAD5"):
    for a in A.LEAD5:
        o = A.item23(R[a], shared[sname])
        i = A.item1(R[a], shared[sname])
        P("  %-6s %-11s qw %.4f  stall %.4f  pages %7d  reading %5d  lastWait %.4f  gpuQueue %.4f" % (
            sname, a, o["mean_qw"], o["mean_stall"], i["ssd_pages"], i["reading_turns"],
            o["lastwait_sum_per_turn"], o["gpuq_sum_per_turn"]))
with open(os.path.join(OUT, "posthoc2.out"), "w") as f:
    f.write("\n".join(lines_out) + "\n")
