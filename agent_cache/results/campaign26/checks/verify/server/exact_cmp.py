#!/usr/bin/env python3
"""The threshold-dependent numbers of the stall files under exact integer-microsecond
arithmetic (loader and per-turn figures of recheck.py, raw logs), against what the
files print.  The files, like my frozen analyze.py, subtract epoch seconds held as
floats (resolution 2.4e-7 s at 1.79e9), so a turn whose stall is exactly 0.050 s falls
on either side of "stall < 0.05" by rounding.

Sets: LEAD6 (stall_ec.out as regenerated 01:35Z 2026-11-12), SEVEN (stall_ec_ctl.out),
LEAD5 (stall_ec.out as it stood at 21:07Z 2026-11-11; its printed values are taken from
compare.json, which quoted them then).

Run as:  nice -n 19 python3 -I -B exact_cmp.py
"""
import json
import os
import statistics as st
import sys

OUT = "/home/wanhr/sglang/agent_cache/results/campaign26/checks/verify/server"
sys.path.insert(0, OUT)
import recheck as RC  # noqa: E402

RC.RUNS["ec_r13_rep"] = "20261111_224624_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70"
LEAD5 = list(RC.LEAD5)
SEVEN = list(RC.SEVEN)
LEAD6 = ["jit_exact", "ec_r13", "ec_r13_rep", "ec_r1", "nofc", "nofc_noend"]
JOBS = [("stall_ec.out (01:35Z 2026-11-12, six arms)", "compare2.json", "stall_ec.out", "LEAD6", LEAD6),
        ("stall_ec_ctl.out", "compare2.json", "stall_ec_ctl.out", "SEVEN", SEVEN),
        ("stall_ec.out (earlier version, five arms, as quoted 21:07Z 2026-11-11)", "compare.json", "stall_ec.out", "LEAD5", LEAD5)]

out = []


def P(s=""):
    out.append(s)
    print(s)


with open(RC.RES + "/.current_run") as f:
    current = f.read().strip()
R = {a: RC.load(a, current) for a in sorted(set(LEAD6 + SEVEN))}


def decimals(s):
    s = s.strip().lstrip("+-")
    return len(s.split(".")[1]) if "." in s else 0


def same(theirs, mine):
    d = decimals(theirs)
    return abs(mine - float(theirs)) <= 0.5 * 10 ** (-d) * (1 + 1e-6) + 1e-12


def figures(r, keys):
    n = len(keys)
    late_st = []
    late_inside = []
    landed = 0
    lw = gq = 0.0
    lw_n = gq_n = 0
    both_after = 0.0
    edge = []
    for k in keys:
        q_us, st_us, n_reads, n_qual, e_us = RC.turn_row(r, k)
        if n_reads == 0:
            continue
        if st_us == 50000:
            edge.append("c%d-t%d" % k)
        if st_us < 50000:
            landed += 1
            continue
        adm = e_us + q_us
        inside = 0
        for end_us, pages, ms in r.s2h[k]:
            if end_us <= adm + 100000:
                inside += max(0, min(end_us, adm) - max(end_us - ms * 1000, e_us))
        late_st.append(st_us / 1e6)
        late_inside.append(min(inside, st_us) / 1e6)
        if q_us - st_us < 100000:
            lw += st_us / 1e6
            lw_n += 1
        else:
            gq += st_us / 1e6
            gq_n += 1
            both_after += (q_us - st_us) / 1e6
    s = sorted(late_st)
    return {
        "landed by arrival (item 3)": landed,
        "stall per reading turn not landed = (two sums x N)/(two counts)": sum(late_st) / len(late_st),
        "of it inside the read": sum(late_inside) / len(late_inside),
        "stall of not-landed turns, median (statistics.median)": st.median(s),
        "stall of not-landed turns, p90": RC.pct(s, 0.9),
        "stall of not-landed turns, p99": RC.pct(s, 0.99),
        "stall, read was the last wait, s per shared turn (item 3)": lw / n,
        "turns, read was the last wait (item 3)": lw_n,
        "stall, GPU queue held the turn as well, s per shared turn (item 3)": gq / n,
        "turns, GPU queue held the turn as well (item 3)": gq_n,
        "those turns' wait after landing, s per shared turn": both_after / n,
    }, edge


summary = []
for title, src, fname, sname, arms in JOBS:
    keys = None
    for a in arms:
        keys = set(R[a].kept_ret) if keys is None else (keys & R[a].kept_ret)
    keys = sorted(keys)
    with open(os.path.join(OUT, src)) as f:
        rows = [r for r in json.load(f) if r["file"] == fname]
    P("== %s   set %s, %d shared turns" % (title, sname, len(keys)))
    n_cmp = n_diff = 0
    for a in arms:
        fig, edge = figures(R[a], keys)
        diffs = []
        for r in rows:
            if r["arm"] != a or r["what"] not in fig:
                continue
            n_cmp += 1
            mine = fig[r["what"]]
            if not same(r["theirs"], mine):
                n_diff += 1
                d = decimals(r["theirs"])
                diffs.append("%s: file %s, exact %.*f (float arithmetic of mine %s)" % (r["what"], r["theirs"], d, mine, r["mine"]))
                summary.append({"file": title, "set": sname, "arm": a, "what": r["what"], "theirs": r["theirs"],
                                "exact": ("%.*f" % (d, mine)), "float_mine": r["mine"],
                                "turns_with_stall_exactly_0.050": edge})
        P("  %-13s reading turns with stall exactly 0.050 s: %d %s; printed numbers that change under exact arithmetic: %d" % (
            a, len(edge), ",".join(edge), len(diffs)))
        for d in diffs:
            P("       " + d)
    P("  -> %d printed threshold-dependent numbers compared, %d differ under exact arithmetic" % (n_cmp, n_diff))
    P()
with open(os.path.join(OUT, "exact_cmp.json"), "w") as f:
    json.dump(summary, f, indent=0)
with open(os.path.join(OUT, "exact_cmp.out"), "w") as f:
    f.write("\n".join(out) + "\n")
