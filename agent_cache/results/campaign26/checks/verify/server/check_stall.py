#!/usr/bin/env python3
"""Checks behind item 3/4: where read ends fall relative to admission, how often
the clamp is active, and a few raw per-turn traces.

Run as:  nice -n 19 python3 -I -B check_stall.py
"""
import json
import os
import sys

OUT = "/home/wanhr/sglang/agent_cache/results/campaign26/checks/verify/server"
RES = "/home/wanhr/sglang/agent_cache/results"
sys.path.insert(0, OUT)
import analyze as A  # noqa: E402

lines_out = []


def P(s=""):
    lines_out.append(s)
    print(s)


labels = sorted(set(A.ALL11 + A.CUT7))
R = {lab: A.Run(lab) for lab in labels}
shared = {}
for sname, arms in A.SETS:
    s = None
    for a in arms:
        s = set(R[a].kept_ret) if s is None else (s & R[a].kept_ret)
    shared[sname] = s

P("READ END RELATIVE TO ADMISSION (admission = entry_time + queue_duration), s2h_io lines of the set's shared turns")
P("%-6s %-13s %7s %9s %9s %9s %9s %10s %10s %10s | %8s %8s" % (
    "set", "arm", "lines", "end>adm", ">adm+.1", ">adm+.5", "end<arr", "max(e-adm)", "p99(e-adm)",
    "min(e-adm)", "clampLo", "clampHi"))
for sname in ("LEAD5", "SEVEN", "ALL11", "CUT7"):
    arms = dict(A.SETS)[sname]
    for a in arms:
        r = R[a]
        dv = []
        before_arr = 0
        clamp_lo = clamp_hi = 0
        for k in sorted(shared[sname]):
            if k not in r.s2h:
                continue
            x = r.rts_last[k]
            arr = x[5]
            qw = x[3] / 1000.0
            adm = arr + qw
            ends = [y[0] for y in r.s2h[k]]
            for e in ends:
                dv.append(e - adm)
                if e < arr:
                    before_arr += 1
            q = [e for e in ends if e <= adm + 0.1]
            if q:
                st = max(q) - arr
                if st < 0:
                    clamp_lo += 1
                if st > qw:
                    clamp_hi += 1
        P("%-6s %-13s %7d %9d %9d %9d %9d %10.4f %10.4f %10.4f | %8d %8d" % (
            sname, a, len(dv), sum(1 for z in dv if z > 0), sum(1 for z in dv if z > 0.1),
            sum(1 for z in dv if z > 0.5), before_arr, max(dv), A.pct(dv, 0.99), min(dv),
            clamp_lo, clamp_hi))
P()

P("GAP BETWEEN LAST READ END AND ADMISSION (queue wait - stall) for reading turns with stall >= 0.05 s, LEAD5 shared turns")
P("%-13s %6s %8s %8s %8s %8s %8s %8s %8s  share<0.05 <0.1 <0.2 <0.5 <1 <5" % (
    "arm", "n", "min", "p10", "p25", "p50", "p75", "p90", "max"))
for a in A.LEAD5 + ["jit_p50_ctl", "hintfree_ctl"]:
    r = R[a]
    sh = shared["LEAD5"] if a in A.LEAD5 else shared["SEVEN"]
    g = []
    for k in sorted(sh):
        if k not in r.s2h:
            continue
        st, has_q, qw = r.stall(k, 0.1)
        if st >= 0.05:
            g.append(qw - st)
    P("%-13s %6d %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.3f  %s" % (
        a, len(g), min(g), A.pct(g, 0.10), A.pct(g, 0.25), A.pct(g, 0.5), A.pct(g, 0.75),
        A.pct(g, 0.9), max(g),
        " ".join("%.3f" % (sum(1 for z in g if z < b) / len(g)) for b in (0.05, 0.1, 0.2, 0.5, 1, 5))))
P()

P("FIRST READ START RELATIVE TO ARRIVAL, per reading turn (LEAD5 shared turns), and srs_dispatch waited_ms of the same key")
P("%-13s %6s %9s %9s %9s %9s %9s | %9s %9s %9s" % (
    "arm", "turns", "min", "p10", "p50", "p90", "max", "wait p50", "wait p90", "wait max"))
for a in A.LEAD5:
    r = R[a]
    fs = []
    wm = []
    for k in sorted(shared["LEAD5"]):
        if k not in r.s2h:
            continue
        arr = r.rts_last[k][5]
        fs.append(min(y[0] - y[5] / 1000.0 for y in r.s2h[k]) - arr)
        for y in r.srs.get(k, ()):
            if y[7] == 1 and y[4] is not None:
                wm.append(float(y[4]) / 1000.0)
    P("%-13s %6d %9.4f %9.4f %9.4f %9.4f %9.4f | %9.4f %9.4f %9.4f" % (
        a, len(fs), min(fs), A.pct(fs, 0.1), A.pct(fs, 0.5), A.pct(fs, 0.9), max(fs),
        A.pct(wm, 0.5) if wm else float("nan"), A.pct(wm, 0.9) if wm else float("nan"),
        max(wm) if wm else float("nan")))
P()

P("STALL BY NUMBER OF READS OF THE TURN, LEAD5 shared turns: reading turns / mean stall of those turns / mean read ms summed per turn")
for a in A.LEAD5:
    r = R[a]
    by = {}
    for k in sorted(shared["LEAD5"]):
        if k not in r.s2h:
            continue
        n = len(r.s2h[k])
        st, has_q, qw = r.stall(k, 0.1)
        o = by.setdefault(n, [0, 0.0, 0.0, 0.0])
        o[0] += 1
        o[1] += st
        o[2] += sum(y[5] for y in r.s2h[k]) / 1000.0
        o[3] += qw
    P("%-13s %s" % (a, "  ".join("reads=%d: n=%d stall=%.3f io=%.3f qw=%.3f" % (
        n, o[0], o[1] / o[0], o[2] / o[0], o[3] / o[0]) for n, o in sorted(by.items()))))
P()

# raw traces of a few turns of nofc
with open(RES + "/.current_run") as f:
    current = f.read().strip()
r = R["nofc"]
assert r.folder != current
cands = []
for k in sorted(shared["LEAD5"]):
    if k in r.s2h:
        st, has_q, qw = r.stall(k, 0.1)
        cands.append((st, k, qw))
cands.sort()
picks = [cands[len(cands) // 2][1], cands[int(len(cands) * 0.9)][1], cands[-1][1], cands[0][1]]
P("RAW TRACES, nofc (median, p90, max, min stall among reading shared turns)")
want = {"-c%d-t%d" % k: k for k in picks}
found = {k: [] for k in picks}
with open(os.path.join(RES, r.folder, "server.log"), "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        if "rid=" not in line:
            continue
        for suffix, k in want.items():
            i = line.find(suffix)
            if i >= 0:
                nxt = line[i + len(suffix):i + len(suffix) + 1]
                if not nxt.isdigit():
                    found[k].append(line.rstrip()[:330])
for k in picks:
    st, has_q, qw = r.stall(k, 0.1)
    t = r.cli[k]
    P("-- key %s  arrival=%.3f  queue_wait=%.3f  admission=%.3f  stall=%.3f  client ttft=%.3f latency=%.3f t_send=%.3f (abs %.3f) gap_slept=%s cached=%s reprefilled=%s"
      % (k, r.arrival(k), qw, r.arrival(k) + qw, st, t["ttft"], t["latency"], t["t_send"],
         r.t_start + t["t_send"], t["gap_slept"], t["cached_tokens"], t["reply_reprefilled"]))
    for ln in found[k]:
        P("   " + ln)
with open(os.path.join(OUT, "check_stall.out"), "w") as f:
    f.write("\n".join(lines_out) + "\n")
