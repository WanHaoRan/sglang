#!/usr/bin/env python3
"""Item 6 again, against the four files as they stood at 01:38Z on 2026-11-12
(copies in existing_snapshot/).  Two of them were regenerated at 01:33-01:35Z, after
compare.py had been run on their earlier versions (compare.out, 21:07Z 2026-11-11):

  stall_ec.out       now six arms (LEAD5 + ec_r13_rep = "LEAD6"), 11145 shared turns
  compare_ec.out     now thirteen arms (ALL11 + jit_p50_d + ec_r13_rep = "ALL13")
  stall_ec_ctl.out   unchanged (SEVEN)
  split_ec_lead.out  unchanged (LEAD5)

Same comparison logic as compare.py.  My side of each comparison:
  frozen   the value, or plain arithmetic on values, of frozen_results_items1to5.json
           (for ALL13 rows of the eleven original arms: the ALL11 value, used only after
           checking that the 13-arm shared set is the same 6715 turns and that the value
           recomputed on it is identical)
  text     printed in a text output saved before opening, recomputed here
  after    computed after the existing files and scripts were opened; this covers every
           number on the six-arm set and every number of the two new arms

Run as:  nice -n 19 python3 -I -B compare2.py
"""
import json
import os
import re
import statistics as st
import sys

OUT = "/home/wanhr/sglang/agent_cache/results/campaign26/checks/verify/server"
SNAP = OUT + "/existing_snapshot"
sys.path.insert(0, OUT)
import analyze as A  # noqa: E402

LEAD5 = list(A.LEAD5)
SEVEN = list(A.SEVEN)
ALL11 = list(A.ALL11)
LEAD6 = ["jit_exact", "ec_r13", "ec_r13_rep", "ec_r1", "nofc", "nofc_noend"]
ALL13 = ALL11 + ["jit_p50_d", "ec_r13_rep"]
NEW_ARMS = ("jit_p50_d", "ec_r13_rep")
SETS = {"LEAD5": LEAD5, "SEVEN": SEVEN, "ALL11": ALL11, "LEAD6": LEAD6, "ALL13": ALL13}

FZ = {}
with open(os.path.join(OUT, "frozen_results_items1to5.json")) as f:
    for v in json.load(f):
        FZ[(v["item"], v["arm"], v["arm_set"], v["metric"])] = v["value"]
FX = {}
with open(os.path.join(OUT, "ext", "ext_values.json")) as f:
    for v in json.load(f):
        FX[(v["item"], v["arm"], v["arm_set"], v["metric"])] = v["value"]

labels = sorted(set(ALL13 + ["order_run"]))
R = {lab: A.Run(lab) for lab in labels}
pos = {(t["conv"], t["turn"]): i for i, t in enumerate(R["order_run"].ok_sorted)}
shared = {}
for sname, arms in SETS.items():
    s = None
    for a in arms:
        s = set(R[a].kept_ret) if s is None else (s & R[a].kept_ret)
    shared[sname] = sorted(s)
ALL13_IS_ALL11 = shared["ALL13"] == shared["ALL11"]

# ALL13 values of the eleven original arms against the frozen ALL11 values
all13_same = all13_diff = 0
for key, val in FX.items():
    if key[2] != "ALL13" or key[1] in NEW_ARMS:
        continue
    k11 = (key[0], key[1], "ALL11", key[3])
    if k11 in FZ:
        fv = FZ[k11]
        if isinstance(val, float) and isinstance(fv, float):
            # the two means against jit_exact etc. are plain sums in the same key order: identical
            ok = (val == fv) or abs(val - fv) <= 1e-12 * max(1.0, abs(fv))
        else:
            ok = val == fv
        all13_same += ok
        all13_diff += (not ok)

CUR = {"set": None}
ROWS = []


def G(item, arm, sname, metric):
    """Value for a set as the file defines it: frozen where it exists, else the extended run."""
    key = (item, arm, sname, metric)
    if sname == "ALL13" and arm not in NEW_ARMS and ALL13_IS_ALL11 and all13_diff == 0:
        k11 = (item, arm, "ALL11", metric)
        if k11 in FZ:
            return FZ[k11]
    if key in FZ:
        return FZ[key]
    return FX[key]


def frozen_ok(arm):
    sname = CUR["set"]
    if sname in ("LEAD5", "SEVEN"):
        return True
    if sname == "ALL13":
        if arm == "all":
            return ALL13_IS_ALL11
        return arm not in NEW_ARMS and ALL13_IS_ALL11 and all13_diff == 0
    return False


def decimals(s):
    s = s.strip().lstrip("+-")
    return len(s.split(".")[1]) if "." in s else 0


def cmp(fname, section, arm, what, theirs, mine, prov, note=""):
    d = decimals(theirs)
    tv = float(theirs)
    ok = abs(mine - tv) <= 0.5 * 10 ** (-d) * (1 + 1e-6) + 1e-12
    if prov == "frozen" and not frozen_ok(arm):
        prov = "after"
    if prov == "text" and (CUR["set"] == "LEAD6" or arm in NEW_ARMS):
        prov = "after"
    ROWS.append({"file": fname, "set": CUR["set"], "section": section, "arm": arm, "what": what,
                 "theirs": theirs.strip(),
                 "mine": ("%+.*f" if theirs.strip()[:1] in "+-" else "%.*f") % (d, mine),
                 "mine_raw": mine, "prov": prov, "match": bool(ok), "note": note})
    return ok


def blocks(path):
    with open(path) as f:
        txt = f.read()
    return [b.split("\n") for b in txt.strip("\n").split("\n\n")]


def my_turn(run, k, slack=0.1):
    """(stall, read time inside [arrival, admission] capped at the stall, queue wait, reads ending after admission + slack)."""
    arr = run.arrival(k)
    qw = run.qw(k)
    adm = arr + qw
    reads = run.s2h.get(k, ())
    q = [x for x in reads if x[0] <= adm + slack]
    stall = 0.0
    if q:
        stall = min(max(max(x[0] for x in q) - arr, 0.0), qw)
    inside = 0.0
    for x in q:
        inside += max(0.0, min(x[0], adm) - max(x[0] - x[5] / 1000.0, arr))
    return stall, min(inside, stall), qw, len(reads) - len(q)


def pages_of(run, k):
    return sum(x[3] for x in run.s2h.get(k, ()))


# ------------------------------------------------------------------ stall files
def do_stall(fname, sname):
    CUR["set"] = sname
    B = blocks(os.path.join(SNAP, fname))
    arms = SETS[sname]
    keys = shared[sname]
    N = len(keys)
    m = re.match(r"(\d+) returning turns shared by (.*)", B[0][0])
    cmp(fname, "header", "all", "shared returning turns", m.group(1), G("0", "all", sname, "shared_returning_turns"), "frozen")
    listed = [x.strip() for x in m.group(2).split(",")]
    ROWS.append({"file": fname, "set": sname, "section": "header", "arm": "all", "what": "arm list equals " + sname,
                 "theirs": " ".join(listed), "mine": " ".join(arms), "mine_raw": None,
                 "prov": "frozen" if sname in ("LEAD5", "SEVEN") else "after",
                 "match": listed == arms, "note": ""})
    T = {a: {k: my_turn(R[a], k) for k in keys} for a in arms}
    for line in B[1][1:]:
        g = re.match(r"^\s+(\S+)\s+([\d.]+) = ([\d.]+) \+ ([\d.]+) \+ ([\d.]+) \|\s+([\d.]+) \|\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+) \|\s+(\d+)\s*$", line)
        a = g.group(1)
        t = G("2", a, sname, "mean_ttft")
        qw = G("2", a, sname, "mean_queue_wait")
        stl = G("3", a, sname, "mean_read_stall_per_shared_turn")
        lw = G("3", a, sname, "stall_read_was_last_wait_per_shared_turn")
        gq = G("3", a, sname, "stall_gpu_queue_held_as_well_per_shared_turn")
        lwn = G("3", a, sname, "turns_read_was_last_wait")
        gqn = G("3", a, sname, "turns_gpu_queue_held_as_well")
        S = "per arm"
        cmp(fname, S, a, "mean TTFT", g.group(2), t, "frozen")
        cmp(fname, S, a, "read stall per shared turn (item 3)", g.group(3), stl, "frozen")
        cmp(fname, S, a, "rest of the queue wait = mean queue wait - stall (items 2,3)", g.group(4), qw - stl, "frozen")
        cmp(fname, S, a, "after admission = mean(TTFT - queue wait) (item 2)", g.group(5), G("2", a, sname, "mean_ttft_minus_queue_wait"), "frozen")
        cmp(fname, S, a, "mean TTFT with no stall = TTFT - stall", g.group(6), t - stl, "frozen")
        cmp(fname, S, a, "reading turns (item 1)", g.group(7), G("1", a, sname, "turns_that_read_from_ssd"), "frozen")
        cmp(fname, S, a, "landed by arrival (item 3)", g.group(8), G("3", a, sname, "reading_turns_landed_by_arrival"), "frozen")
        cmp(fname, S, a, "stall per reading turn not landed = (two sums x N)/(two counts)", g.group(9), (lw + gq) * N / (lwn + gqn), "frozen")
        late = [T[a][k] for k in keys if k in R[a].s2h and T[a][k][0] >= 0.05]
        cmp(fname, S, a, "of it inside the read", g.group(10), sum(x[1] for x in late) / len(late), "after")
        cmp(fname, S, a, "reads ending after admission + 0.1 s", g.group(11), sum(T[a][k][3] for k in keys), "text")
    for line in B[2][1:]:
        g = re.match(r"^\s+(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+) \|\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", line)
        a = g.group(1)
        late = sorted(T[a][k][0] for k in keys if k in R[a].s2h and T[a][k][0] >= 0.05)
        after = sorted(T[a][k][2] - T[a][k][0] for k in keys if k in R[a].s2h)
        S = "not-landed stall / wait after landing"
        cmp(fname, S, a, "stall of not-landed turns, median (statistics.median)", g.group(2), st.median(late), "after",
            "pct(v,0.5) gives %.4f" % A.pct(late, 0.5))
        cmp(fname, S, a, "stall of not-landed turns, p90", g.group(3), A.pct(late, 0.9), "after")
        cmp(fname, S, a, "stall of not-landed turns, p99", g.group(4), A.pct(late, 0.99), "after")
        cmp(fname, S, a, "queue wait after landing, reading turns, mean = rd QW - rd stall", g.group(5),
            G("x-23", a, sname, "reading_mean_qw") - G("x-23", a, sname, "reading_mean_stall"), "frozen")
        cmp(fname, S, a, "queue wait after landing, reading turns, median (statistics.median)", g.group(6), st.median(after), "after",
            "pct(v,0.5) gives %.4f" % A.pct(after, 0.5))
        cmp(fname, S, a, "queue wait after landing, reading turns, p90", g.group(7), A.pct(after, 0.9), "after")
    for line in B[3][1:]:
        parts = line.split(" | ")
        g = re.match(r"\s*(\d+)-\s*(\d+)\s+(\d+)", parts[0])
        lo, hi = int(g.group(1)), int(g.group(2))
        ks = [k for k in keys if lo <= pos.get(k, -1) < hi]
        ep = {(5500, 7000): "ep1", (8750, 11000): "ep2"}.get((lo, hi))
        S = "stretch %d-%d" % (lo, hi)
        if ep:
            cmp(fname, S, "all", "turns in stretch", g.group(3), G("x-split:" + ep, arms[0], sname, "n_turns"), "frozen")
        else:
            cmp(fname, S, "all", "turns in stretch", g.group(3), len(ks), "after")
        for cell in parts[1:]:
            a, t, s_, r_ = cell.split()
            if ep:
                mt = G("x-split:" + ep, a, sname, "mean_ttft")
                ms = G("x-split:" + ep, a, sname, "mean_read_stall")
                mq = G("x-split:" + ep, a, sname, "mean_queue_wait")
                prov = "frozen"
            else:
                mt = sum(R[a].cli[k]["ttft"] for k in ks) / len(ks)
                ms = sum(T[a][k][0] for k in ks) / len(ks)
                mq = sum(T[a][k][2] for k in ks) / len(ks)
                prov = "after"
            cmp(fname, S, a, "mean TTFT", t, mt, prov)
            cmp(fname, S, a, "read stall per turn", s_, ms, prov)
            cmp(fname, S, a, "rest of the queue wait per turn", r_, mq - ms, prov)
    for line in B[4][1:]:
        g = re.match(r"^\s+(\S+)\s+([\d.]+) \(\s*(\d+)\) \| ([\d.]+) \(\s*(\d+)\) ([\d.]+)\s*$", line)
        a = g.group(1)
        S = "stall split"
        cmp(fname, S, a, "stall, read was the last wait, s per shared turn (item 3)", g.group(2), G("3", a, sname, "stall_read_was_last_wait_per_shared_turn"), "frozen")
        cmp(fname, S, a, "turns, read was the last wait (item 3)", g.group(3), G("3", a, sname, "turns_read_was_last_wait"), "frozen")
        cmp(fname, S, a, "stall, GPU queue held the turn as well, s per shared turn (item 3)", g.group(4), G("3", a, sname, "stall_gpu_queue_held_as_well_per_shared_turn"), "frozen")
        cmp(fname, S, a, "turns, GPU queue held the turn as well (item 3)", g.group(5), G("3", a, sname, "turns_gpu_queue_held_as_well"), "frozen")
        both = [T[a][k] for k in keys if k in R[a].s2h and T[a][k][0] >= 0.05 and T[a][k][2] - T[a][k][0] >= 0.1]
        cmp(fname, S, a, "those turns' wait after landing, s per shared turn", g.group(6), sum(x[2] - x[0] for x in both) / N, "after")
    ref = arms[0]
    for line in B[5][1:]:
        g = re.match(r"^\s+(\S+)\s+([+-][\d.]+) \| ([+-][\d.]+) \| ([+-][\d.]+) \| ([+-][\d.]+)\s*$", line)
        a = g.group(1)
        S = "paired against " + ref

        def f(arm, mname, item="2"):
            return G(item, arm, sname, mname)
        dt = f(a, "mean_ttft") - f(ref, "mean_ttft")
        ds = f(a, "mean_read_stall_per_shared_turn", "3") - f(ref, "mean_read_stall_per_shared_turn", "3")
        dq = f(a, "mean_queue_wait") - f(ref, "mean_queue_wait")
        cmp(fname, S, a, "TTFT difference", g.group(2), dt, "frozen")
        cmp(fname, S, a, "read stall difference", g.group(3), ds, "frozen")
        cmp(fname, S, a, "rest of the queue wait difference", g.group(4), dq - ds, "frozen")
        cmp(fname, S, a, "after admission difference", g.group(5), dt - dq, "frozen")


# ------------------------------------------------------------------ split_ec_lead.out
def do_split():
    fname = "split_ec_lead.out"
    sname = "LEAD5"
    CUR["set"] = sname
    B = blocks(os.path.join(SNAP, fname))
    arms = SETS[sname]
    keys = shared[sname]
    N = len(keys)
    m = re.match(r"(\d+) returning turns shared by (.*)", B[0][0])
    cmp(fname, "header", "all", "shared returning turns", m.group(1), G("0", "all", sname, "shared_returning_turns"), "frozen")
    listed = [x.strip() for x in m.group(2).split(",")]
    ROWS.append({"file": fname, "set": sname, "section": "header", "arm": "all", "what": "arm list equals LEAD5",
                 "theirs": " ".join(listed), "mine": " ".join(arms), "mine_raw": None, "prov": "frozen",
                 "match": listed == arms, "note": ""})
    for line in B[1][1:]:
        g = re.match(r"^\s+(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+) \|\s+(\d+)\s+(\d+) \|\s+([\d.]+) %", line)
        a = g.group(1)
        v = sorted(R[a].cli[k]["ttft"] for k in keys)
        S = "TTFT distribution (client side)"
        cmp(fname, S, a, "mean TTFT", g.group(2), G("x-shared", a, sname, "mean_ttft"), "frozen")
        cmp(fname, S, a, "median TTFT (file: statistics.median; mine: pct 0.5)", g.group(3), G("x-shared", a, sname, "p50_ttft"), "frozen",
            "statistics.median gives %.5f" % st.median(v))
        cmp(fname, S, a, "p90 TTFT", g.group(4), G("x-shared", a, sname, "p90_ttft"), "frozen")
        cmp(fname, S, a, "p99 TTFT", g.group(5), G("x-shared", a, sname, "p99_ttft"), "frozen")
        cmp(fname, S, a, "p99.9 TTFT", g.group(6), A.pct(v, 0.999), "after")
        cmp(fname, S, a, "max TTFT", g.group(7), v[-1], "after")
        cmp(fname, S, a, "turns over 2 s", g.group(8), sum(1 for x in v if x > 2), "after")
        cmp(fname, S, a, "turns over 10 s", g.group(9), sum(1 for x in v if x > 10), "after")
        cmp(fname, S, a, "share of summed TTFT in slowest 1 %", g.group(10), 100 * sum(v[-max(1, N // 100):]) / sum(v), "after")
    for line in B[2][1:]:
        g = re.match(r"^\s+(\S+)\s+(read|no read)\s+(\d+)\s+([\d.]+) =\s+([\d.]+) \+\s+([\d.]+)\s+(\d+) \|\s+([\d.]+)", line)
        a, cls = g.group(1), g.group(2)
        S = "by whether the turn read (" + cls + ")"
        rn = G("1", a, sname, "turns_that_read_from_ssd")
        if cls == "read":
            n, t, q = rn, G("x-23", a, sname, "reading_mean_ttft"), G("x-23", a, sname, "reading_mean_qw")
            ks = [k for k in keys if k in R[a].s2h]
        else:
            n, t, q = N - rn, G("x-23", a, sname, "nonreading_mean_ttft"), G("x-23", a, sname, "nonreading_mean_qw")
            ks = [k for k in keys if k not in R[a].s2h]
        cmp(fname, S, a, "turns", g.group(3), n, "frozen")
        cmp(fname, S, a, "mean TTFT", g.group(4), t, "frozen")
        cmp(fname, S, a, "mean queue wait", g.group(5), q, "frozen")
        cmp(fname, S, a, "after admission", g.group(6), t - q, "frozen")
        cmp(fname, S, a, "recomputed tokens per turn", g.group(7),
            sum(max(R[a].cli[k]["reply_reprefilled"] or 0, 0) for k in ks) / len(ks), "after")
        cmp(fname, S, a, "class's part of the arm's mean TTFT", g.group(8), n * t / N, "frozen")
    ref = arms[0]
    for line in B[3][2:]:
        g = re.match(r"^\s+(\S+)\s+([+-][\d.]+) \|\s+(\d+) ([+-][\d.]+) \|\s+(\d+) ([+-][\d.]+) \|\s+(\d+) ([+-][\d.]+) \|\s+(\d+) ([+-][\d.]+)", line)
        a = g.group(1)
        S = "paired against %s by read class" % ref
        cmp(fname, S, a, "total TTFT difference", g.group(2), G("2", a, sname, "mean_ttft") - G("2", ref, sname, "mean_ttft"), "frozen")
        for i, (here, there, name) in enumerate(((True, True, "read in both"), (True, False, "read only here"),
                                                 (False, True, "read only in " + ref), (False, False, "read in neither"))):
            ks = [k for k in keys if (k in R[a].s2h) == here and (k in R[ref].s2h) == there]
            cmp(fname, S, a, name + ": turns", g.group(3 + 2 * i), len(ks), "after")
            cmp(fname, S, a, name + ": s of the difference per shared turn", g.group(4 + 2 * i),
                sum(R[a].cli[k]["ttft"] - R[ref].cli[k]["ttft"] for k in ks) / N, "after")
    for line in B[4][1:]:
        a = line.split()[0]
        S = "difference against %s by stretch" % ref
        for lo, hi, n, d, dq in re.findall(r"(\d+)-(\d+):\s+(\d+) ([+-][\d.]+) \| ([+-][\d.]+)", line):
            lo, hi = int(lo), int(hi)
            ep = {(5500, 7000): "ep1", (8750, 11000): "ep2"}.get((lo, hi))
            ks = [k for k in keys if lo <= pos.get(k, -1) < hi]
            if ep:
                nn = G("x-split:" + ep, a, sname, "n_turns")
                md = (G("x-split:" + ep, a, sname, "sum_ttft") - G("x-split:" + ep, ref, sname, "sum_ttft")) / N
                mq = (G("x-split:" + ep, a, sname, "sum_queue_wait") - G("x-split:" + ep, ref, sname, "sum_queue_wait")) / N
                prov = "frozen"
            else:
                nn = len(ks)
                md = sum(R[a].cli[k]["ttft"] - R[ref].cli[k]["ttft"] for k in ks) / N
                mq = sum(R[a].qw(k) - R[ref].qw(k) for k in ks) / N
                prov = "after"
            cmp(fname, S, a, "%d-%d turns" % (lo, hi), n, nn, prov)
            cmp(fname, S, a, "%d-%d TTFT difference, s per shared turn" % (lo, hi), d, md, prov)
            cmp(fname, S, a, "%d-%d of it in queue wait" % (lo, hi), dq, mq, prov)
    for line in B[5][1:]:
        g = re.match(r"^\s+(\S+)\s+(\d+) staged: (.*) \| read from SSD in each: (.*)$", line)
        a = g.group(1)
        S = "staged reads of " + a
        ks = [k for k in keys if k in R[a].ntp]
        cmp(fname, S, a, "turns with a staged read (item 1)", g.group(2), G("1", a, sname, "turns_with_staged_read"), "frozen")
        for b, val in re.findall(r"(\S+)\s+([\d.]+)", g.group(3)):
            cmp(fname, S, b, "mean TTFT of those turns in this arm", val, sum(R[b].cli[k]["ttft"] for k in ks) / len(ks), "after")
        for b, val in re.findall(r"(\S+)\s+(\d+)", g.group(4)):
            cmp(fname, S, b, "of those turns, read from SSD in this arm", val, sum(1 for k in ks if k in R[b].s2h), "after")
    for line in B[6][1:]:
        f = line.split()
        a = f[0]
        S = "reading turns"
        rn = G("1", a, sname, "turns_that_read_from_ssd")
        cmp(fname, S, a, "reading turns (item 1)", f[1], rn, "frozen")
        cmp(fname, S, a, "SSD pages per reading turn = pages / reading turns (item 1)", f[2], G("1", a, sname, "ssd_pages_shared_turns") / rn, "frozen")
        cmp(fname, S, a, "reads per reading turn (item 1)", f[3], G("1", a, sname, "reads_per_reading_turn"), "frozen")
        cmp(fname, S, a, "turns that read more than once", f[4], sum(1 for k in keys if len(R[a].s2h.get(k, ())) > 1), "text")
        cmp(fname, S, a, "turns refused room at a launch (item 1)", f[5], G("1", a, sname, "turns_with_refused_launch"), "frozen")


# ------------------------------------------------------------------ compare_ec.out
def do_compare():
    fname = "compare_ec.out"
    sname = "ALL13"
    CUR["set"] = sname
    B = blocks(os.path.join(SNAP, fname))
    arms = SETS[sname]
    keys = shared[sname]
    listed = [ln.split()[0] for ln in B[0][1:]]
    ROWS.append({"file": fname, "set": sname, "section": "runs", "arm": "all", "what": "arm list and order equal ALL11 + jit_p50_d + ec_r13_rep",
                 "theirs": " ".join(listed), "mine": " ".join(arms), "mine_raw": None, "prov": "after",
                 "match": listed == arms, "note": ""})
    for ln in B[0][1:]:
        f = ln.split()
        ROWS.append({"file": fname, "set": sname, "section": "runs", "arm": f[0], "what": "run folder", "theirs": f[1],
                     "mine": R[f[0]].folder, "mine_raw": None, "prov": "after" if f[0] in NEW_ARMS else "frozen",
                     "match": f[1] == R[f[0]].folder, "note": ""})
    for line in B[1][1:]:
        f = line.split()
        a = f[0]
        r = R[a]
        S = "per arm, kept turns"
        tt = [t["ttft"] for t in r.kept]
        allk = [(t["conv"], t["turn"]) for t in r.kept]
        cmp(fname, S, a, "kept turns", f[1], G("0", a, "", "kept_turns"), "frozen")
        cmp(fname, S, a, "up to minute = kept window / 60", f[2], G("0", a, "", "kept_window_length") / 60.0, "frozen")
        cmp(fname, S, a, "users started (sessions with a kept turn)", f[3], len(set(k[0] for k in allk)), "after")
        cmp(fname, S, a, "errors", f[4], sum(1 for t in r.turns if t["error"]), "text")
        cmp(fname, S, a, "mean TTFT, kept turns", f[5], sum(tt) / len(tt), "after")
        cmp(fname, S, a, "median TTFT, kept turns (statistics.median)", f[6], st.median(tt), "after", "pct(v,0.5) gives %.5f" % A.pct(tt, 0.5))
        cmp(fname, S, a, "p90 TTFT, kept turns", f[7], A.pct(tt, 0.9), "after")
        cmp(fname, S, a, "p99 TTFT, kept turns", f[8], A.pct(tt, 0.99), "after")
        cmp(fname, S, a, "mean queue wait, kept turns", f[9], sum(r.qw(k) for k in allk) / len(allk), "after")
        rk = [k for k in allk if k[1] > 0]
        cmp(fname, S, a, "recomputed tokens per kept returning turn", f[10],
            sum(max(r.cli[k]["reply_reprefilled"] or 0, 0) for k in rk) / len(rk), "after")
        first_pages = sum(pages_of(r, k) for k in allk if k[1] == 0)
        cmp(fname, S, a, "SSD pages read, kept turns incl. first turns = item 1 (kept returning) + kept first turns", f[11],
            G("1", a, "", "ssd_pages_kept_returning_turns") + first_pages, "after",
            "item 1 figure (returning only) %d + first-turn pages %d" % (G("1", a, "", "ssd_pages_kept_returning_turns"), first_pages))
        first_reading = sum(1 for k in allk if k[1] == 0 and pages_of(r, k) > 0)
        cmp(fname, S, a, "'s2h reads' = kept turns that read, incl. first turns", f[12],
            G("1", a, "", "turns_that_read_from_ssd_kept_returning") + first_reading, "after",
            "item 1 kept-returning reading turns %d + first turns that read %d" % (G("1", a, "", "turns_that_read_from_ssd_kept_returning"), first_reading))
    for line in B[2][1:]:
        g = re.match(r"^\s+(\S+)\s+decode 8-15\s+([\d.]+) \(\s*(\d+)\) 16-23\s+([\d.]+) \(\s*(\d+)\) 24-31\s+([\d.]+) \(\s*(\d+)\) \| first-turn prefill\s+([\d.]+) \(\s*(\d+)\) \| SSD\s+([\d.]+) \(\s*(\d+)\)", line)
        a = g.group(1)
        r = R[a]
        S = "hardware control"
        dec = [x for x in r.dec if 0.0 <= x[0] - r.t_start < r.cut]
        for i, (lo, hi) in enumerate(((8, 15), (16, 23), (24, 31))):
            step = [1000.0 * x[1] / x[4] for x in dec if lo <= x[1] <= hi and x[5] == 0 and x[4] > 0]
            cmp(fname, S, a, "decode ms per step, %d-%d running, median" % (lo, hi), g.group(2 + 2 * i), st.median(step), "after")
            cmp(fname, S, a, "decode lines, %d-%d running, no queue" % (lo, hi), g.group(3 + 2 * i), len(step), "after")
        fresh = []
        for t in r.kept:
            if t["turn"] != 0:
                continue
            k = (t["conv"], 0)
            unc = t["prompt_tokens"] - (t["cached_tokens"] or 0)
            if r.qw(k) < 0.05 and unc > 2000:
                fresh.append((t["ttft"] - r.qw(k)) / unc * 1000.0)
        cmp(fname, S, a, "first-turn prefill ms per token, median", g.group(8), st.median(fresh), "after")
        cmp(fname, S, a, "first turns used", g.group(9), len(fresh), "after")
        big = [x[5] / x[3] for x in r.s2h_all if x[3] >= 500]
        cmp(fname, S, a, "SSD ms per page, reads of 500 pages or more, median", g.group(10), st.median(big), "after")
        cmp(fname, S, a, "reads of 500 pages or more", g.group(11), len(big), "after")
    blk = B[3]
    m = re.match(r"all arms: (\d+) returning turns", blk[0])
    cmp(fname, "all arms", "all", "shared returning turns", m.group(1), G("0", "all", sname, "shared_returning_turns"), "frozen")
    i = 1
    while i < len(blk) and "by stretch" not in blk[i]:
        g = re.match(r"^\s+(\S+)\s+([\d.]+) =\s+([\d.]+) \+\s+([\d.]+) \|\s+(\d+) \|\s+([\d.]+) % \|(.*)$", blk[i])
        a = g.group(1)
        S = "all arms"
        cmp(fname, S, a, "mean TTFT", g.group(2), G("2", a, sname, "mean_ttft"), "frozen")
        cmp(fname, S, a, "mean queue wait (item 2 definition, all-arms turns)", g.group(3), G("2", a, sname, "mean_queue_wait"), "frozen")
        cmp(fname, S, a, "after admission = mean(TTFT - queue wait)", g.group(4), G("2", a, sname, "mean_ttft_minus_queue_wait"), "frozen")
        cmp(fname, S, a, "recomputed tokens per turn", g.group(5), G("x-shared", a, sname, "recomputed_tokens_mean"), "frozen")
        gains = re.findall(r"(-?\d+) %", g.group(7))
        mj = G("2", "jit_exact", sname, "mean_ttft")
        for base, val in zip(("baseline", "base_rep"), gains):
            mb = G("2", base, sname, "mean_ttft")
            cmp(fname, S, a, "share of the gain from %s to jit_exact" % base, val,
                100 * (mb - G("2", a, sname, "mean_ttft")) / (mb - mj), "frozen")
        i += 1
    order = [x.strip() for x in blk[i].split("|")[1:]]
    for line in blk[i + 1:]:
        parts = line.split("|")
        g = re.match(r"\s*(\d+)-\s*(\d+):\s+(\d+)", parts[0])
        lo, hi = int(g.group(1)), int(g.group(2))
        ks = [k for k in keys if lo <= pos.get(k, -1) < hi]
        S = "all arms, stretch %d-%d" % (lo, hi)
        ep = "ep1" if (lo, hi) == (5500, 7000) else None
        cmp(fname, S, "all", "turns in stretch", g.group(3), G("x-split:ep1", arms[0], sname, "n_turns") if ep else len(ks), "frozen" if ep else "after")
        for a, val in zip(order, parts[1:]):
            mt = G("x-split:ep1", a, sname, "mean_ttft") if ep else sum(R[a].cli[k]["ttft"] for k in ks) / len(ks)
            cmp(fname, S, a, "mean TTFT", val, mt, "frozen" if ep else "after")
    for line in B[5][1:]:
        parts = line.split("|")
        a, n = parts[0].split()
        S = "mechanisms, own returning turns"
        cmp(fname, S, a, "kept returning turns", n, G("0", a, "", "kept_returning_turns"), "frozen")
        cmp(fname, S, a, "reads per reading turn = s2h lines / reading turns (kept returning)", parts[5].strip(),
            G("1", a, "", "s2h_io_lines_kept_returning") / G("1", a, "", "turns_that_read_from_ssd_kept_returning"), "frozen")
        gpu, host = parts[7].split()
        cmp(fname, S, a, "GPU tokens evicted in the kept window, M (item 1)", gpu, G("1", a, "", "evict_device_tokens_kept_window") / 1e6, "frozen")
        cmp(fname, S, a, "host tokens evicted in the kept window, M (item 1)", host, G("1", a, "", "evict_host_tokens_kept_window") / 1e6, "frozen")
    blk = B[7]
    order = [x.strip() for x in blk[1].split("|")[1:]]
    for line in blk[2:]:
        cells = line.split("|")
        mnt = int(cells[0])
        S = "per 15 min, minute %d" % mnt
        rest = cells[1:]
        j = 0
        for a in order:
            r = R[a]
            c1 = rest[j].split() if j < len(rest) else []
            if len(c1) != 3:
                j += 1
                continue
            c3 = rest[j + 2].split()
            j += 3
            x = [t for t in r.kept if mnt * 60 <= t["t_send"] < (mnt + 15) * 60]
            dd = [d for d in r.dec if 0.0 <= d[0] - r.t_start < r.cut and mnt * 60 <= d[0] - r.t_start < (mnt + 15) * 60]
            pp = sum(p[2] for p in r.pre if 0.0 <= p[0] - r.t_start < r.cut and mnt * 60 <= p[0] - r.t_start < (mnt + 15) * 60)
            cmp(fname, S, a, "turns sent", c1[0], len(x), "after")
            cmp(fname, S, a, "mean TTFT", c1[1], sum(t["ttft"] for t in x) / len(x), "after")
            cmp(fname, S, a, "mean queue wait", c1[2], sum(r.qw((t["conv"], t["turn"])) for t in x) / len(x), "after")
            cmp(fname, S, a, "GPU mean running requests (Decode lines)", c3[0], sum(d[1] for d in dd) / len(dd), "after")
            cmp(fname, S, a, "GPU mean queued requests (Decode lines)", c3[1], sum(d[5] for d in dd) / len(dd), "after")
            cmp(fname, S, a, "prefill K tokens computed (Prefill lines)", c3[2], pp / 1000.0, "after")
    for line in B[9][1:]:
        g = re.match(r"^\s+(\S+)\s+([\d.]+) % \|\s+(\d+) \(\s*(\d+)\) \|\s+(\d+) \(\s*(\d+)\) \|", line)
        a = g.group(1)
        r = R[a]
        S = "reads and pins, own returning turns"
        cmp(fname, S, a, "share of kept returning turns that read from SSD, %", g.group(2),
            100.0 * G("1", a, "", "turns_that_read_from_ssd_kept_returning") / G("0", a, "", "kept_returning_turns"), "frozen")
        cmp(fname, S, a, "staged reads, ntp_stage lines in the whole log", g.group(3), G("x-run", a, "", "ntp_stage_lines_whole_log"), "frozen")
        cmp(fname, S, a, "kept returning turns with a staged read", g.group(4), sum(1 for k in r.kept_ret if k in r.ntp), "after")
        cmp(fname, S, a, "refused launches, lines in the whole log", g.group(5), G("x-run", a, "", "refused_launches_whole_log"), "frozen")
        cmp(fname, S, a, "kept returning turns with a refused launch", g.group(6),
            sum(1 for k in r.kept_ret if any(x[7] == 0 for x in r.srs.get(k, ()))), "after")


do_stall("stall_ec.out", "LEAD6")
do_stall("stall_ec_ctl.out", "SEVEN")
do_split()
do_compare()

FILES = ("stall_ec.out", "stall_ec_ctl.out", "split_ec_lead.out", "compare_ec.out")
num = [r for r in ROWS if r["mine_raw"] is not None]
out = []
out.append("compare2.py against existing_snapshot/ (files as of 01:38Z 2026-11-12): %d comparisons, %d equal at the printed precision, %d different" % (
    len(ROWS), sum(1 for r in ROWS if r["match"]), sum(1 for r in ROWS if not r["match"])))
out.append("  numeric rows %d (equal %d); non-numeric rows (arm lists, run folders) %d (equal %d)" % (
    len(num), sum(1 for r in num if r["match"]), len(ROWS) - len(num),
    sum(1 for r in ROWS if r["mine_raw"] is None and r["match"])))
for fname in FILES:
    rr = [r for r in ROWS if r["file"] == fname]
    out.append("  %-18s %4d compared, %4d equal, %3d different  | by provenance: %s" % (
        fname, len(rr), sum(1 for r in rr if r["match"]), sum(1 for r in rr if not r["match"]),
        ", ".join("%s %d/%d" % (p, sum(1 for r in rr if r["prov"] == p and r["match"]), sum(1 for r in rr if r["prov"] == p))
                  for p in ("frozen", "text", "after"))))
old = [r for r in ROWS if r["file"] == "compare_ec.out" and r["arm"] not in NEW_ARMS]
new = [r for r in ROWS if r["file"] == "compare_ec.out" and r["arm"] in NEW_ARMS]
out.append("  compare_ec.out rows of the eleven original arms (and 'all' rows): %d compared, %d equal; rows of jit_p50_d and ec_r13_rep: %d compared, %d equal" % (
    len(old), sum(1 for r in old if r["match"]), len(new), sum(1 for r in new if r["match"])))
out.append("13-arm shared set identical to ALL11's: %s (%d turns); ALL13 values of the eleven original arms identical to the frozen ALL11 values: %d of %d" % (
    ALL13_IS_ALL11, len(shared["ALL13"]), all13_same, all13_same + all13_diff))
out.append("six-arm set (LEAD5 + ec_r13_rep): %d shared returning turns; LEAD5: %d; LEAD5 turns not in the six-arm set: %d" % (
    len(shared["LEAD6"]), len(shared["LEAD5"]), len(set(shared["LEAD5"]) - set(shared["LEAD6"]))))
out.append("")
out.append("DIFFERENT:")
for r in ROWS:
    if not r["match"]:
        out.append("  %s | %s | %s | %s | theirs %s | mine %s (raw %r) | %s | %s" % (
            r["file"], r["section"], r["arm"], r["what"], r["theirs"], r["mine"], r["mine_raw"], r["prov"], r["note"]))
out.append("")
out.append("ALL COMPARISONS:")
for r in ROWS:
    out.append("  %-5s %-17s | %-38s | %-13s | %-70s | theirs %-10s | mine %-10s | %-6s %s" % (
        "ok" if r["match"] else "DIFF", r["file"], r["section"][:38], r["arm"], r["what"][:70], r["theirs"][:60],
        r["mine"][:60], r["prov"], ("| " + r["note"]) if r["note"] else ""))
with open(os.path.join(OUT, "compare2.out"), "w") as f:
    f.write("\n".join(out) + "\n")
with open(os.path.join(OUT, "compare2.json"), "w") as f:
    json.dump(ROWS, f, indent=0)
print("\n".join(out[:12 + sum(1 for r in ROWS if not r["match"])]))
