#!/usr/bin/env python3
"""Figures computed after the existing outputs were opened, for the concerns list.

Run as:  nice -n 19 python3 -I -B posthoc.py
"""
import json
import os
import statistics as st
import sys
from collections import Counter

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
SETS = dict(A.SETS)
shared = {}
for sname, arms in A.SETS:
    s = None
    for a in arms:
        s = set(R[a].kept_ret) if s is None else (s & R[a].kept_ret)
    shared[sname] = sorted(s)

with open(os.path.join(OUT, "compare.json")) as f:
    rows = json.load(f)
P("COMPARISON ROWS: %d; numeric %d; by provenance %s; by file %s" % (
    len(rows), sum(1 for r in rows if r["mine_raw"] is not None),
    dict(Counter(r["prov"] for r in rows if r["mine_raw"] is not None)),
    dict(Counter(r["file"] for r in rows if r["mine_raw"] is not None))))
P("  matched numeric: %d; mismatched: %d; non-numeric rows (arm lists, run folders): %d, all equal: %s" % (
    sum(1 for r in rows if r["mine_raw"] is not None and r["match"]),
    sum(1 for r in rows if not r["match"]),
    sum(1 for r in rows if r["mine_raw"] is None), all(r["match"] for r in rows if r["mine_raw"] is None)))
P("  per-15-min rows: %d" % sum(1 for r in rows if r["section"].startswith("per 15 min")))
# medians: would the task's percentile rule print the same?
med = [r for r in rows if "pct(v,0.5) gives" in r["note"] or "statistics.median gives" in r["note"]]
diff = 0
for r in med:
    alt = float(r["note"].split("gives")[1])
    d = len(r["theirs"].split(".")[1]) if "." in r["theirs"] else 0
    if abs(alt - float(r["theirs"])) > 0.5 * 10 ** (-d) + 1e-12:
        diff += 1
        P("  median definition changes the printed value: %s %s %s theirs %s other rule %.5f" % (
            r["file"], r["section"], r["arm"], r["theirs"], alt))
P("  printed medians checked under both rules (statistics.median and sorted(v)[int(0.5*len)]): %d, printed value differs under the other rule: %d" % (len(med), diff))
P()

P("LEAD5 shared turns absent from the order run (in no stretch row of stall_ec.out / split_ec_lead.out)")
absent = [k for k in shared["LEAD5"] if k not in pos]
P("  n = %d of %d (%.1f %%)" % (len(absent), len(shared["LEAD5"]), 100.0 * len(absent) / len(shared["LEAD5"])))
for a in A.LEAD5:
    r = R[a]
    o = A.item23(r, absent)
    ts = [r.cli[k]["t_send"] for k in absent]
    P("  %-11s mean TTFT %.4f  queue wait %.4f  stall %.4f  reading turns %d  t_send min/median/max %.0f/%.0f/%.0f s" % (
        a, o["mean_ttft"], o["mean_qw"], o["mean_stall"], o["reading_n"], min(ts), A.pct(ts, 0.5), max(ts)))
P()

P("WHERE THE WAIT SITS: share of each arm's summed queue wait / read stall / TTFT on LEAD5 shared turns")
P("%-11s | in the two episodes (2972 turns, 26.5 %%): qw, stall, ttft | in the 112 turns (1 %%) with the largest queue wait: qw, stall | sessions holding those 112 turns | in the 11 largest (0.1 %%): qw" % "arm")
keys = shared["LEAD5"]
for a in A.LEAD5:
    r = R[a]
    per = [(r.qw(k), r.stall(k, 0.1)[0], r.cli[k]["ttft"], k) for k in keys]
    tq = sum(x[0] for x in per)
    tsl = sum(x[1] for x in per)
    tt = sum(x[2] for x in per)
    ep = [x for x in per if (5500 <= pos.get(x[3], -1) <= 6999) or (8750 <= pos.get(x[3], -1) <= 10999)]
    top = sorted(per, reverse=True)[:len(per) // 100]
    top01 = sorted(per, reverse=True)[:len(per) // 1000]
    P("%-11s | %.3f %.3f %.3f | %.3f %.3f | %d | %.3f" % (
        a, sum(x[0] for x in ep) / tq, sum(x[1] for x in ep) / tsl, sum(x[2] for x in ep) / tt,
        sum(x[0] for x in top) / tq, sum(x[1] for x in top) / tsl, len(set(x[3][0] for x in top)),
        sum(x[0] for x in top01) / tq))
P()

P("REFUSED LAUNCHES WITHOUT ANY READ: shared turns with a refused srs_dispatch and no s2h_io line (their wait is counted as 'rest of the queue wait')")
for sname in ("LEAD5", "SEVEN"):
    for a in SETS[sname]:
        r = R[a]
        ks = [k for k in shared[sname] if k not in r.s2h and any(x[7] == 0 for x in r.srs.get(k, ()))]
        qw = sum(r.qw(k) for k in ks)
        # part of the queue wait that had elapsed when the last refusal was logged
        till = 0.0
        for k in ks:
            last = max(x[0] for x in r.srs[k] if x[7] == 0)
            till += min(max(last - r.arrival(k), 0.0), r.qw(k))
        rec = sum(max(r.cli[k]["reply_reprefilled"] or 0, 0) for k in ks)
        P("  %-6s %-13s turns %4d  queue wait sum %8.1f s = %.4f s per shared turn; of it before the last refusal %8.1f s = %.4f s per shared turn; recomputed tokens per such turn %.0f" % (
            sname, a, len(ks), qw, qw / len(shared[sname]), till, till / len(shared[sname]), rec / len(ks) if ks else 0))
P()

P("LABELS IN compare_ec.out: whole-log line counts against the same lines on the arm's kept returning turns; 's2h reads' against s2h_io lines")
P("%-13s | ntp_stage lines: whole log, on kept returning turns | refused launches: whole log, on kept returning turns | kept turns that read (file's 's2h reads'), s2h_io lines on those turns, s2h_io lines whole log | pages: kept returning (item 1), kept first turns, sum (file's 'SSD pages read')" % "arm")
for a in A.ALL11:
    r = R[a]
    st_all = sum(len(v) for v in r.ntp.values())
    st_kr = sum(len(r.ntp[k]) for k in r.kept_ret if k in r.ntp)
    rf_all = sum(1 for v in r.srs.values() for x in v if x[7] == 0)
    rf_kr = sum(1 for k in r.kept_ret for x in r.srs.get(k, ()) if x[7] == 0)
    rd = [k for k in r.kept_keys if k in r.s2h]
    ln = sum(len(r.s2h[k]) for k in rd)
    p_ret = sum(x[3] for k in r.kept_ret for x in r.s2h.get(k, ()))
    p_first = sum(x[3] for k in r.kept_keys if k[1] == 0 for x in r.s2h.get(k, ()))
    P("%-13s | %5d %5d | %5d %5d | %5d %5d %5d | %7d %5d %7d" % (
        a, st_all, st_kr, rf_all, rf_kr, len(rd), ln, len(r.s2h_all), p_ret, p_first, p_ret + p_first))
P()

P("BOUND SENSITIVITY of statements built on the 'read was the last wait' sum (LEAD5, s per shared turn)")
P("%-8s | nofc (the '0.17 s') | nofc - ec_r13 | nofc - ec_r1 | nofc - jit_exact | nofc_noend - jit_exact | ec_r13 - ec_r1" % "bound")
for b in (0.05, 0.1, 0.2, 0.5, 1.0):
    v = {a: A.item23(R[a], keys, bound=b)["lastwait_sum_per_turn"] for a in A.LEAD5}
    P("%-8s | %.4f | %+.4f | %+.4f | %+.4f | %+.4f | %+.4f" % (
        b, v["nofc"], v["nofc"] - v["ec_r13"], v["nofc"] - v["ec_r1"], v["nofc"] - v["jit_exact"],
        v["nofc_noend"] - v["jit_exact"], v["ec_r13"] - v["ec_r1"]))
P()

P("LANDED THRESHOLD SENSITIVITY (LEAD5): reading turns counted as landed by arrival with the 0.05 s threshold at 0, 0.02, 0.05, 0.1, 0.2 s; and turns whose last read truly ended at or before arrival")
for a in A.LEAD5:
    r = R[a]
    stl = [r.stall(k, 0.1)[0] for k in keys if k in r.s2h]
    before = A.item23(r, keys)["last_read_ended_before_arrival_n"]
    P("  %-11s %s | ended at or before arrival %d" % (
        a, " ".join("%d" % sum(1 for z in stl if z < thr) for thr in (1e-9, 0.02, 0.05, 0.1, 0.2)), before))
P()

P("TOTALS for the sanity list")
tot_turns = sum(len(R[a].turns) for a in labels)
P("  runs %d, turn records %d, ReqTimeStats keyed lines %d, s2h_io lines %d, ntp_stage lines %d, srs_dispatch lines %d, evict_device lines %d, evict_host lines %d" % (
    len(labels), tot_turns, sum(sum(R[a].rts_n.values()) for a in labels), sum(len(R[a].s2h_all) for a in labels),
    sum(sum(len(v) for v in R[a].ntp.values()) for a in labels), sum(sum(len(v) for v in R[a].srs.values()) for a in labels),
    sum(len(R[a].evd) for a in labels), sum(len(R[a].evh) for a in labels)))
P("  order-run t_send ties: %d groups; keys in tied groups whose rank is within 1 of a stretch edge (3000, 5500, 7000, 8750): %d" % (
    sum(1 for c in Counter(t["t_send"] for t in R["order_run"].ok_sorted).values() if c > 1),
    sum(1 for i, t in enumerate(R["order_run"].ok_sorted)
        if any(abs(i - e) <= 1 for e in (3000, 5500, 7000, 8750))
        and Counter(x["t_send"] for x in R["order_run"].ok_sorted[max(0, i - 2):i + 3])[t["t_send"]] > 1)))
with open(os.path.join(OUT, "posthoc.out"), "w") as f:
    f.write("\n".join(lines_out) + "\n")
