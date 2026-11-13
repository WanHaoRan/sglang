#!/usr/bin/env python3
"""Item 6: compare my frozen numbers (items_1_4.json) with what the existing .out files print.

A number "matches" when my value, formatted with the same number of decimals, gives the printed string.
Part A reads the five files the task names.  Part B reads three more files of the same folder that print
further numbers of mine (same shared keys); it is reported apart.
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
C26 = "/home/wanhr/sglang/agent_cache/results/campaign26"
with open(os.path.join(HERE, "items_1_4.json")) as f:
    MINE = json.load(f)

V = {}
for e in MINE["values"]:
    if e["item"] in ("1", "2", "3"):
        V[(e["item"], e["arm_set"], e["arm"], e["metric"])] = e["value"]

ROWS = []  # dict rows
PART = "A"


def decimals(s):
    s = s.strip().lstrip("+-")
    return len(s.split(".")[1]) if "." in s else 0


def check(fname, key, theirs, signed=False, scale=1.0, note=""):
    theirs = theirs.strip()
    mine = V[key]
    fmt = ("%+." if signed else "%.") + "%df" % decimals(theirs)
    ms = fmt % (mine * scale)
    ROWS.append({"part": PART, "file": fname, "key": key, "mine": ms, "theirs": theirs, "ok": ms == theirs,
                 "raw": mine, "note": note})


def read(name):
    with open(os.path.join(C26, name)) as f:
        return f.read().split("\n")


def find(L, start):
    return next(k for k, s in enumerate(L) if s.startswith(start))


# ------------------------------------------------------------------ compare_ec.out (ALL11)
def all_arms_all11(fname):
    L = read(fname)
    i = find(L, "all arms:")
    n = re.match(r"all arms: (\d+) returning turns", L[i]).group(1)
    check(fname, ("1", "ALL11", "(set)", "shared_turns"), n)
    k = i + 1
    while not L[k].lstrip().startswith("by stretch"):
        m = re.match(r"^\s+(\S+)\s+([\d.]+) =\s+([\d.]+) \+\s+([\d.]+) \|\s+(\d+) \|\s+([\d.]+) % \|\s+(-?\d+) %\s+(-?\d+) %", L[k])
        assert m, L[k]
        arm = m.group(1)
        check(fname, ("1", "ALL11", arm, "mean_ttft"), m.group(2))
        check(fname, ("1", "ALL11", arm, "mean_recomputed_tokens"), m.group(5))
        # group 6 is the "landed reads lost" share (a server-log number, not mine); the gain shares are groups 7 and 8
        check(fname, ("3", "ALL11", arm, "share_of_gain_vs_baseline"), m.group(7), scale=100.0)
        check(fname, ("3", "ALL11", arm, "share_of_gain_vs_base_rep"), m.group(8), scale=100.0)
        k += 1
    hdr = [x.strip() for x in L[k].split("|")[1:]]
    k += 1
    stretch_counts = {}
    while L[k].strip():
        m = re.match(r"^\s+(\d+)-\s*(\d+):\s+(\d+) \|(.*)", L[k])
        assert m, L[k]
        lo, hi, cnt = int(m.group(1)), int(m.group(2)), m.group(3)
        stretch_counts[(lo, hi)] = int(cnt)
        vals = [x.strip() for x in m.group(4).split("|")]
        if (lo, hi) == (5500, 7000):
            # ALL11 holds no turn of the second episode, so this stretch is the whole "load episodes" class
            check(fname, ("1", "ALL11", "(set)", "shared_turns_in_load_episodes"), cnt, note="stretch 5500-7000")
            for arm, x in zip(hdr, vals):
                check(fname, ("1", "ALL11", arm, "mean_ttft_load_episodes"), x, note="stretch 5500-7000")
        k += 1
    other = sum(c for (lo, hi), c in stretch_counts.items() if (lo, hi) != (5500, 7000))
    check(fname, ("1", "ALL11", "(set)", "shared_turns_elsewhere"), str(other), note="sum of the other printed stretches")


def run_to_run(fname):
    # baseline against base_rep on their shared turns, which are the ALL11 shared turns (6715)
    L = read(fname)
    i = find(L, "run to run:")
    k = i + 1
    while L[k].strip():
        f = L[k].split()
        if f[0] == "all":
            check(fname, ("1", "ALL11", "(set)", "shared_turns"), f[1], note="run-to-run block")
            check(fname, ("1", "ALL11", "baseline", "mean_ttft"), f[2], note="run-to-run block")
            check(fname, ("1", "ALL11", "base_rep", "mean_ttft"), f[3], note="run-to-run block")
        else:
            m = re.match(r"^\s+(\d+)-\s*(\d+)\s+(.*)", L[k])
            if (int(m.group(1)), int(m.group(2))) == (5500, 7000):
                nums = m.group(3).split()
                check(fname, ("1", "ALL11", "(set)", "shared_turns_in_load_episodes"), nums[0], note="run-to-run block")
                check(fname, ("1", "ALL11", "baseline", "mean_ttft_load_episodes"), nums[1], note="run-to-run block")
                check(fname, ("1", "ALL11", "base_rep", "mean_ttft_load_episodes"), nums[2], note="run-to-run block")
        k += 1


all_arms_all11("compare_ec.out")
run_to_run("compare_ec.out")
run_to_run("compare_ec_cut.out")   # its "all arms" block is on 12 arms (6571 turns), not one of my four sets


# ------------------------------------------------------------------ split files
def split_file(fname, sname):
    L = read(fname)
    m = re.match(r"^(\d+) returning turns shared by (.*)", L[0])
    check(fname, ("1", sname, "(set)", "shared_turns"), m.group(1))
    i = find(L, "per arm: mean TTFT, median, p90, p99, p99.9, max")
    k = i + 1
    while L[k].strip():
        m = re.match(r"^\s+(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+) \|\s+(\d+)\s+(\d+) \|\s+([\d.]+) %", L[k])
        assert m, L[k]
        arm = m.group(1)
        check(fname, ("1", sname, arm, "mean_ttft"), m.group(2))
        check(fname, ("1", sname, arm, "median_ttft"), m.group(3))
        check(fname, ("1", sname, arm, "p90_ttft"), m.group(4))
        check(fname, ("1", sname, arm, "p99_ttft"), m.group(5))
        check(fname, ("1", sname, arm, "turns_over_10s"), m.group(9))
        k += 1
    if sname == "LEAD5":
        i = find(L, "against jit_exact, paired by turn")
        k = i + 2
        while L[k].strip():
            m = re.match(r"^\s+(\S+)\s+([+-][\d.]+) \|", L[k])
            assert m, L[k]
            check(fname, ("2", "LEAD5", "%s-jit_exact" % m.group(1), "paired_mean_ttft_diff"), m.group(2), signed=True)
            k += 1
        # the stretch line of one arm gives the two episode counts
        i = find(L, "the same difference against jit_exact by stretch")
        c = dict(((int(lo), int(hi)), int(n)) for lo, hi, n in re.findall(r"(\d+)-(\d+): (\d+) ", L[i + 1]))
        check(fname, ("1", "LEAD5", "(set)", "shared_turns_in_load_episodes"), str(c[(5500, 7000)] + c[(8750, 11000)]),
              note="sum of the two printed episode stretches")


split_file("split_ec_lead.out", "LEAD5")


# ------------------------------------------------------------------ pairs files
def pairs_file(fname, sname, wanted_pairs, per_arm_set=None):
    L = read(fname)
    m = re.match(r"^(\d+) returning turns shared by (.*)", L[0])
    check(fname, ("1", sname, "(set)", "shared_turns"), m.group(1))
    i = find(L, "per arm on them:")
    k = i + 1
    while L[k].strip():
        m = re.match(r"^\s+(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+) \|\s+([\d.]+)\s+([\d.]+) \|\s+(\d+)\s+(\d+)\s+(\d+)", L[k])
        assert m, L[k]
        arm = m.group(1)
        check(fname, ("1", sname, arm, "mean_ttft"), m.group(2))
        check(fname, ("1", sname, arm, "median_ttft"), m.group(3))
        check(fname, ("1", sname, arm, "p90_ttft"), m.group(4))
        check(fname, ("1", sname, arm, "p99_ttft"), m.group(5))
        check(fname, ("1", sname, arm, "turns_over_10s"), m.group(6))
        check(fname, ("1", sname, arm, "mean_ttft_load_episodes"), m.group(7))
        check(fname, ("1", sname, arm, "mean_ttft_elsewhere"), m.group(8))
        check(fname, ("1", sname, arm, "mean_recomputed_tokens"), m.group(11))
        k += 1
    sections = [("all shared returning turns:", "paired_mean_ttft_diff", "shared_turns", "all"),
                ("the two load episodes", "paired_mean_ttft_diff_load_episodes", "shared_turns_in_load_episodes", "load episodes"),
                ("the other positions:", "paired_mean_ttft_diff_elsewhere", "shared_turns_elsewhere", "elsewhere")]
    for start, metric, cmetric, label in sections:
        i = find(L, start)
        cnt = re.search(r": (\d+) turns", L[i]).group(1)
        if label != "all":
            check(fname, ("1", sname, "(set)", cmetric), cnt)
        cols = L[i + 1].split()
        k = i + 2
        while L[k].strip():
            row = L[k].split()[0]
            ents = re.findall(r"([+-]\d+\.\d+) \[\s*([+-][\d.]+),\s*([+-][\d.]+)\]", L[k])
            for col, (val, _lo, _hi) in zip(cols, ents):
                name = "%s-%s" % (row, col)
                if wanted_pairs is not None and name in wanted_pairs:
                    check(fname, ("2", sname, name, metric), val, signed=True, note="pair matrix")
            k += 1
    i = find(L, "the same arm run again")
    k = i + 1
    while L[k].strip():
        m = re.match(r"^\s+.*?(\S+)\s+- (\S+)\s+([+-][\d.]+) \[.*?\] ([+-][\d.]+) ([+-][\d.]+)\s*$", L[k])
        if m:
            name = "%s-%s" % (m.group(1), m.group(2))
            if wanted_pairs is not None and name in wanted_pairs:
                check(fname, ("2", sname, name, "paired_mean_ttft_diff"), m.group(3), signed=True, note="repeat line")
                check(fname, ("2", sname, name, "paired_mean_ttft_diff_load_episodes"), m.group(4), signed=True, note="repeat line")
                check(fname, ("2", sname, name, "paired_mean_ttft_diff_elsewhere"), m.group(5), signed=True, note="repeat line")
        k += 1


L5 = ["jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend"]
lead_pairs = {"%s-%s" % (L5[j], L5[i]) for i in range(5) for j in range(i + 1, 5)}
pairs_file("pairs_ec.out", "LEAD5", lead_pairs)
pairs_file("pairs_ec_cut.out", "CUT7", {"jit_p50_d_cut-jit_p50_ctl", "jit_p50_d_cut-jit_p50", "jit_p50_ctl-jit_p50"})

# ------------------------------------------------------------------ part B: other files of the folder
PART = "B"
# pairs_ec_ctl.out: 9 arms, 6715 shared turns; 06_extras.py checks that these are the ALL11 shared keys
pairs_file("pairs_ec_ctl.out", "ALL11", None)
# split_ec.out: 5 arms, 9702 shared turns; 06_extras.py checks that these are the SEVEN shared keys
split_file("split_ec.out", "SEVEN")
L = read("stall_ec_ctl.out")
m = re.match(r"^(\d+) returning turns shared by (.*)", L[0])
check("stall_ec_ctl.out", ("1", "SEVEN", "(set)", "shared_turns"), m.group(1))
i = find(L, "per arm: mean TTFT = read stall")
k = i + 1
while L[k].strip():
    f = L[k].split()
    check("stall_ec_ctl.out", ("1", "SEVEN", f[0], "mean_ttft"), f[1])
    k += 1
i = find(L, "by stretch of the oracle's send order")
c = {}
k = i + 1
while L[k].strip():
    m = re.match(r"^\s+(\d+)-\s*(\d+)\s+(\d+)", L[k])
    c[(int(m.group(1)), int(m.group(2)))] = int(m.group(3))
    k += 1
check("stall_ec_ctl.out", ("1", "SEVEN", "(set)", "shared_turns_in_load_episodes"), str(c[(5500, 7000)] + c[(8750, 11000)]),
      note="sum of the two printed episode stretches")
check("stall_ec_ctl.out", ("1", "SEVEN", "(set)", "shared_turns_elsewhere"), str(c[(0, 3000)] + c[(3000, 5500)] + c[(7000, 8750)]),
      note="sum of the three other printed stretches")

# ------------------------------------------------------------------ report
A = [r for r in ROWS if r["part"] == "A"]
B = [r for r in ROWS if r["part"] == "B"]
for name, rows in (("A: the five named files", A), ("B: pairs_ec_ctl.out, split_ec.out, stall_ec_ctl.out", B)):
    nok = sum(1 for r in rows if r["ok"])
    print("part %s: %d printed numbers compared, %d match, %d do not" % (name, len(rows), nok, len(rows) - nok))
    byfile = {}
    for r in rows:
        b = byfile.setdefault((r["file"], r["key"][1]), [0, 0])
        b[0] += 1
        b[1] += 1 if r["ok"] else 0
    for (fn, sn), (tot, ok) in sorted(byfile.items()):
        print("    %-22s %-6s compared %3d  matched %3d" % (fn, sn, tot, ok))
keysA = {r["key"] for r in A}
keysA_ok = {r["key"] for r in A if r["ok"]} - {r["key"] for r in A if not r["ok"]}
keysB = {r["key"] for r in B} - keysA
keysB_ok = ({r["key"] for r in B if r["ok"]} - {r["key"] for r in B if not r["ok"]}) - keysA
print()
print("my numbers in items 1-3: %d" % len(V))
print("  distinct numbers printed by the five named files: %d, all occurrences equal: %d" % (len(keysA), len(keysA_ok)))
print("  further distinct numbers printed only by the three other files: %d, equal: %d" % (len(keysB), len(keysB_ok)))
rest = sorted(set(V) - keysA - keysB)
print("  printed by none of these eight files: %d" % len(rest))
cnt = {}
for k in rest:
    cnt[(k[1], k[3])] = cnt.get((k[1], k[3]), [])
    cnt[(k[1], k[3])].append(k[2])
for (s, metric), arms in sorted(cnt.items()):
    print("      %-6s %-36s %s" % (s, metric, ", ".join(arms)))
print()
print("mismatches:")
for r in ROWS:
    if not r["ok"]:
        print("  part %s %-20s %s mine %s (raw %.6f) theirs %s %s" % (r["part"], r["file"], r["key"], r["mine"], r["raw"], r["theirs"], r["note"]))
print()
print("all comparisons:")
for r in ROWS:
    print("  %s %s %-18s %-6s %-28s %-38s mine %9s theirs %9s raw %.6f %s" % (
        "ok " if r["ok"] else "BAD", r["part"], r["file"], r["key"][1], r["key"][2], r["key"][3], r["mine"], r["theirs"], r["raw"], r["note"]))
with open(os.path.join(HERE, "item6_compare.json"), "w") as f:
    json.dump({"part_A_compared": len(A), "part_A_matched": sum(1 for r in A if r["ok"]),
               "part_B_compared": len(B), "part_B_matched": sum(1 for r in B if r["ok"]),
               "my_numbers_items_1_3": len(V), "distinct_in_five_files": len(keysA), "distinct_in_five_files_equal": len(keysA_ok),
               "distinct_only_in_other_three": len(keysB), "distinct_only_in_other_three_equal": len(keysB_ok),
               "not_printed": [list(k) for k in rest],
               "rows": [dict(r, key=list(r["key"])) for r in ROWS]}, f, indent=1)
