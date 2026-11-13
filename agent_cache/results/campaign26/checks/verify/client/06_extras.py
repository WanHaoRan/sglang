#!/usr/bin/env python3
"""Beyond items 1-3: recompute the other client-side numbers the existing .out files print, and compare.

Where the task gave no definition, the definition tried is written next to the table.  A row that does not
match under the definition tried is listed; it is a failure to reproduce, not proof of an error.
"""
import json
import math
import os
import pickle
import re
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
C26 = "/home/wanhr/sglang/agent_cache/results/campaign26"
with open(os.path.join(HERE, "parsed.pkl"), "rb") as f:
    P = pickle.load(f)
with open(os.path.join(HERE, "items_1_4.json")) as f:
    MINE = json.load(f)
V = {(e["item"], e["arm_set"], e["arm"], e["metric"]): e["value"] for e in MINE["values"]}

EPISODES = [(5500, 6999), (8750, 10999)]
STRETCHES = [(0, 3000), (3000, 5500), (5500, 7000), (7000, 8750), (8750, 11000)]
ROWS = []


def pct(v, q):
    s = sorted(v)
    return s[min(len(s) - 1, int(q * len(s)))]


def mean(v):
    return math.fsum(v) / len(v) if v else float("nan")


def recomputed(t):
    return max(t.get("reply_reprefilled") or 0, 0)


def with_ttft_sorted(turns):
    v = [t for t in turns if t.get("ttft") is not None]
    v.sort(key=lambda t: t["t_send"])
    return v


KEPT = {}
KM = {}
MAXS = {}
for a in P:
    v = with_ttft_sorted(P[a]["turns"])
    MAXS[a] = v[-1]["t_send"]
    KEPT[a] = [t for t in v if t["t_send"] < v[-1]["t_send"] - 300.0]
    KM[a] = {(t["conv"], t["turn"]): t for t in KEPT[a]}
POS = {(t["conv"], t["turn"]): i for i, t in enumerate(with_ttft_sorted(P["ORDER"]["turns"]))}


def in_episode(k):
    p = POS.get(k)
    return p is not None and any(lo <= p <= hi for lo, hi in EPISODES)


def shared_of(arms):
    return sorted(set.intersection(*[{k for k in KM[a] if k[1] > 0} for a in arms]))


def decimals(s):
    s = s.strip().lstrip("+-")
    return len(s.split(".")[1]) if "." in s else 0


def check(table, what, mine, theirs, signed=False, scale=1.0):
    theirs = theirs.strip()
    fmt = ("%+." if signed else "%.") + "%df" % decimals(theirs)
    ms = fmt % (mine * scale)
    ROWS.append((table, what, ms, theirs, ms == theirs, mine))


def read(name):
    with open(os.path.join(C26, name)) as f:
        return f.read().split("\n")


def find(L, start, after=0):
    return next(k for k in range(after, len(L)) if L[k].startswith(start))


# ---------------------------------------------------------------- 1. first table: each arm's own kept turns
def first_table(fname):
    L = read(fname)
    i = find(L, "per arm, turns sent at least 300 s before its last send")
    k = i + 1
    while L[k].strip():
        f = L[k].split()
        arm = f[0]
        kept = KEPT[arm]
        ret = [t for t in kept if t["turn"] > 0]
        tt = [t["ttft"] for t in kept]
        T = fname + " own-kept table"
        check(T, "%s kept turns" % arm, len(kept), f[1])
        check(T + " [cut minute = (largest t_send - 300)/60]", "%s up to minute" % arm, (MAXS[arm] - 300.0) / 60.0, f[2])
        check(T + " [my item 4: minute of the last kept send]", "%s minute of last kept send vs printed 'up to minute'" % arm,
              kept[-1]["t_send"] / 60.0, f[2])
        check(T, "%s users started (kept first turns)" % arm, len(kept) - len(ret), f[3])
        check(T, "%s errors" % arm, sum(1 for t in P[arm]["turns"] if t.get("error")), f[4])
        check(T + " [all kept turns, first turns included]", "%s mean TTFT" % arm, mean(tt), f[5])
        check(T + " [all kept turns, first turns included]", "%s median TTFT" % arm, pct(tt, 0.5), f[6])
        check(T + " [all kept turns, first turns included]", "%s p90 TTFT" % arm, pct(tt, 0.9), f[7])
        check(T + " [all kept turns, first turns included]", "%s p99 TTFT" % arm, pct(tt, 0.99), f[8])
        check(T + " [kept returning turns]", "%s recomputed per returning turn" % arm, mean([recomputed(t) for t in ret]), f[10])
        k += 1
    i = find(L, "mechanisms on each arm's own returning turns")
    k = i + 1
    while L[k].strip():
        f = L[k].split()
        check(fname + " mechanisms table", "%s own kept returning turns" % f[0], sum(1 for t in KEPT[f[0]] if t["turn"] > 0), f[1])
        k += 1


first_table("compare_ec.out")
first_table("compare_ec_cut.out")


# ---------------------------------------------------------------- 2. the "all arms" block and its stretches
def all_arms_block(fname, arms, frozen_set=None):
    L = read(fname)
    i = find(L, "all arms:")
    keys = shared_of(arms)
    T = fname + " all-arms block (%d arms)" % len(arms)
    check(T, "shared turns", len(keys), re.match(r"all arms: (\d+) returning turns", L[i]).group(1))
    mu = {a: mean([KM[a][k]["ttft"] for k in keys]) for a in arms}
    k = i + 1
    while not L[k].lstrip().startswith("by stretch"):
        m = re.match(r"^\s+(\S+)\s+([\d.]+) =\s+([\d.]+) \+\s+([\d.]+) \|\s+(\d+) \|\s+([\d.]+) % \|\s+(-?\d+) %\s+(-?\d+) %", L[k])
        arm = m.group(1)
        if frozen_set is None:
            check(T, "%s mean TTFT" % arm, mu[arm], m.group(2))
            check(T, "%s recomputed tokens" % arm, mean([recomputed(KM[arm][x]) for x in keys]), m.group(5))
            check(T, "%s share of gain vs baseline (%%)" % arm, (mu["baseline"] - mu[arm]) / (mu["baseline"] - mu["jit_exact"]), m.group(7), scale=100)
            check(T, "%s share of gain vs base_rep (%%)" % arm, (mu["base_rep"] - mu[arm]) / (mu["base_rep"] - mu["jit_exact"]), m.group(8), scale=100)
        k += 1
    hdr = [x.strip() for x in L[k].split("|")[1:]]
    k += 1
    while L[k].strip():
        m = re.match(r"^\s+(\d+)-\s*(\d+):\s+(\d+) \|(.*)", L[k])
        lo, hi = int(m.group(1)), int(m.group(2))
        ks = [x for x in keys if x in POS and lo <= POS[x] < hi]
        check(T + " stretches [lo <= position < hi]", "stretch %d-%d turns" % (lo, hi), len(ks), m.group(3))
        for arm, x in zip(hdr, [y.strip() for y in m.group(4).split("|")]):
            check(T + " stretches [lo <= position < hi]", "stretch %d-%d %s mean TTFT" % (lo, hi, arm), mean([KM[arm][q]["ttft"] for q in ks]), x)
        k += 1
    # run to run
    i = find(L, "run to run:")
    keys = shared_of(["baseline", "base_rep"])
    k = i + 1
    T = fname + " run-to-run block"
    while L[k].strip():
        f = L[k].split()
        if f[0] == "all":
            ks = keys
            nums = f[1:]
            label = "all"
        else:
            m = re.match(r"^\s+(\d+)-\s*(\d+)\s+(.*)", L[k])
            lo, hi = int(m.group(1)), int(m.group(2))
            ks = [x for x in keys if x in POS and lo <= POS[x] < hi]
            nums = m.group(3).split()
            label = "%d-%d" % (lo, hi)
        d = [KM["base_rep"][x]["ttft"] - KM["baseline"][x]["ttft"] for x in ks]
        check(T, "%s n" % label, len(ks), nums[0])
        check(T, "%s mean baseline" % label, mean([KM["baseline"][x]["ttft"] for x in ks]), nums[1])
        check(T, "%s mean base_rep" % label, mean([KM["base_rep"][x]["ttft"] for x in ks]), nums[2])
        check(T, "%s paired mean change" % label, mean(d), nums[3], signed=True)
        check(T + " [statistics.median of |change|]", "%s median |change|" % label, statistics.median([abs(x) for x in d]), nums[4])
        check(T, "%s faster in the repeat" % label, sum(1 for x in d if x < 0), nums[5])
        check(T, "%s slower in the repeat" % label, sum(1 for x in d if x > 0), nums[6])
        k += 1


ALL11 = ["baseline", "base_rep", "hintfree", "jit_p50", "jit_exact", "jit_p50_ctl", "hintfree_ctl", "ec_r13", "nofc", "ec_r1", "nofc_noend"]
all_arms_block("compare_ec.out", ALL11)
all_arms_block("compare_ec_cut.out", ALL11 + ["jit_p50_d_cut"])


# ---------------------------------------------------------------- 3. own send order: bins of 500, minutes reached, 15-minute rows
def own_order_tables(fname):
    L = read(fname)
    i = find(L, "by position in each arm's own send order")
    arms = [x.strip() for x in L[i + 1].split("|")[1:]]
    k = i + 2
    T = fname + " bins of 500 own-order positions [all kept turns; minute = send time of the bin's first turn]"
    while L[k].strip():
        cells = L[k].split("|")
        p = int(cells[0])
        for arm, cell in zip(arms, cells[1:]):
            f = cell.split()
            if not f:
                continue
            part = KEPT[arm][p:p + 500]
            check(T, "%s position %d mean TTFT" % (arm, p), mean([t["ttft"] for t in part]), f[0])
            check(T, "%s position %d minute reached" % (arm, p), part[0]["t_send"] / 60.0, f[1])
        k += 1
    i = find(L, "per 15 min of each arm's own clock")
    arms = [x.strip() for x in L[i + 1].split("|")[1:]]
    # every arm takes three '|' separated cells: "turns mean queue" | "load" | "gpu..."
    k = i + 2
    T = fname + " 15-minute rows [all kept turns by send time]"
    while L[k].strip():
        cells = L[k].split("|")
        m0 = int(cells[0])
        body = cells[1:]
        for j, arm in enumerate(arms):
            if 3 * j >= len(body):
                break
            f = body[3 * j].split()
            if not f:
                continue
            part = [t for t in KEPT[arm] if m0 * 60 <= t["t_send"] < (m0 + 15) * 60]
            check(T, "%s minute %d turns" % (arm, m0), len(part), f[0])
            check(T, "%s minute %d mean TTFT" % (arm, m0), mean([t["ttft"] for t in part]), f[1])
        k += 1
    i = find(L, "minute each arm reached positions of its own send order")
    k = i + 1
    T = fname + " minute reached at own-order positions [all kept turns]"
    while L[k].strip():
        f = L[k].split()
        arm = f[0]
        for j in range(1, len(f), 2):
            p = int(f[j].rstrip(":"))
            check(T, "%s position %d" % (arm, p), KEPT[arm][p]["t_send"] / 60.0, f[j + 1])
        k += 1


own_order_tables("compare_ec.out")
own_order_tables("compare_ec_cut.out")


# ---------------------------------------------------------------- 4. pairs files: per-arm table and every pair
def pairs_all(fname, frozen_set=None):
    L = read(fname)
    m = re.match(r"^(\d+) returning turns shared by (.*)", L[0])
    arms = [x.strip() for x in m.group(2).split(",")]
    keys = shared_of(arms)
    T = fname + " (%d arms)" % len(arms)
    check(T, "shared turns", len(keys), m.group(1))
    epi = [k for k in keys if in_episode(k)]
    els = [k for k in keys if not in_episode(k)]
    i = find(L, "per arm on them:")
    k = i + 1
    while L[k].strip():
        m = re.match(r"^\s+(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+) \|\s+([\d.]+)\s+([\d.]+) \|\s+(\d+)\s+(\d+)\s+(\d+)", L[k])
        arm = m.group(1)
        v = [KM[arm][x]["ttft"] for x in keys]
        if frozen_set:
            # these turns are the same keys as my frozen set, so the frozen numbers are the ones compared
            g = lambda metric: V[("1", frozen_set, arm, metric)]
            src = " [my frozen %s numbers]" % frozen_set
            check(T + src, "%s mean TTFT" % arm, g("mean_ttft"), m.group(2))
            check(T + src, "%s median TTFT" % arm, g("median_ttft"), m.group(3))
            check(T + src, "%s p90 TTFT" % arm, g("p90_ttft"), m.group(4))
            check(T + src, "%s p99 TTFT" % arm, g("p99_ttft"), m.group(5))
            check(T + src, "%s turns over 10 s" % arm, g("turns_over_10s"), m.group(6))
            check(T + src, "%s mean TTFT in load episodes" % arm, g("mean_ttft_load_episodes"), m.group(7))
            check(T + src, "%s mean TTFT elsewhere" % arm, g("mean_ttft_elsewhere"), m.group(8))
            check(T + src, "%s recomputed tokens" % arm, g("mean_recomputed_tokens"), m.group(11))
        k += 1
    for start, ks, label in (("all shared returning turns:", keys, "all"), ("the two load episodes", epi, "load episodes"),
                             ("the other positions:", els, "elsewhere")):
        i = find(L, start)
        check(T + " pair matrices", "turns in class %s" % label, len(ks), re.search(r": (\d+) turns", L[i]).group(1))
        cols = L[i + 1].split()
        k = i + 2
        while L[k].strip():
            row = L[k].split()[0]
            ents = re.findall(r"([+-]\d+\.\d+) \[\s*([+-][\d.]+),\s*([+-][\d.]+)\]", L[k])
            for col, (val, _lo, _hi) in zip(cols, ents):
                d = mean([KM[row][x]["ttft"] - KM[col][x]["ttft"] for x in ks])
                check(T + " pair matrices", "%s minus %s, %s" % (row, col, label), d, val, signed=True)
            k += 1
    i = find(L, "the same arm run again")
    k = i + 1
    while L[k].strip():
        m = re.match(r"^\s+.*?(\S+)\s+- (\S+)\s+([+-][\d.]+) \[.*?\] ([+-][\d.]+) ([+-][\d.]+)\s*$", L[k])
        if m:
            a, b = m.group(1), m.group(2)
            for ks, label, val in ((keys, "all", m.group(3)), (epi, "load episodes", m.group(4)), (els, "elsewhere", m.group(5))):
                check(T + " repeat lines", "%s minus %s, %s" % (a, b, label), mean([KM[a][x]["ttft"] - KM[b][x]["ttft"] for x in ks]), val, signed=True)
        k += 1
    return keys


k9 = pairs_all("pairs_ec_ctl.out", frozen_set="ALL11")
print("pairs_ec_ctl.out: its 9-arm shared keys are the ALL11 shared keys: %s" % (k9 == shared_of(ALL11)))
pairs_all("pairs_ec_cut.out")
pairs_all("pairs_ec.out")

# ---------------------------------------------------------------- 5. split files and the stall file (SEVEN keys)
LEAD5 = ["jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend"]
SEVEN = LEAD5 + ["jit_p50_ctl", "hintfree_ctl"]


def split_file(fname, frozen_set):
    L = read(fname)
    m = re.match(r"^(\d+) returning turns shared by (.*)", L[0])
    arms = [x.strip() for x in m.group(2).split(",")]
    keys = shared_of(arms)
    same = (keys == shared_of({"SEVEN": SEVEN, "LEAD5": LEAD5}[frozen_set]))
    T = fname + " (%d arms)" % len(arms)
    check(T, "shared turns", len(keys), m.group(1))
    print("%s: its shared keys are the %s shared keys: %s" % (fname, frozen_set, same))
    i = find(L, "per arm: mean TTFT, median, p90, p99, p99.9, max")
    k = i + 1
    while L[k].strip():
        m = re.match(r"^\s+(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+) \|\s+(\d+)\s+(\d+) \|\s+([\d.]+) %", L[k])
        arm = m.group(1)
        v = sorted((KM[arm][x]["ttft"] for x in keys), reverse=True)
        g = lambda metric: V[("1", frozen_set, arm, metric)]
        src = " [my frozen %s numbers]" % frozen_set
        if same and fname != "split_ec_lead.out":   # split_ec_lead.out was already compared in 05_compare.py
            check(T + src, "%s mean TTFT" % arm, g("mean_ttft"), m.group(2))
            check(T + src, "%s median TTFT" % arm, g("median_ttft"), m.group(3))
            check(T + src, "%s p90 TTFT" % arm, g("p90_ttft"), m.group(4))
            check(T + src, "%s p99 TTFT" % arm, g("p99_ttft"), m.group(5))
            check(T + src, "%s turns over 10 s" % arm, g("turns_over_10s"), m.group(9))
        check(T + " [pct 0.999]", "%s p99.9 TTFT" % arm, pct(v, 0.999), m.group(6))
        check(T, "%s max TTFT" % arm, v[0], m.group(7))
        check(T, "%s turns over 2 s" % arm, sum(1 for x in v if x > 2.0), m.group(8))
        check(T + " [slowest len//100 turns]", "%s share of summed TTFT in the slowest 1 %% (%%)" % arm,
              math.fsum(v[:max(1, len(v) // 100)]) / math.fsum(v), m.group(10), scale=100)
        k += 1
    i = find(L, "against jit_exact, paired by turn")
    k = i + 2
    while L[k].strip():
        m = re.match(r"^\s+(\S+)\s+([+-][\d.]+) \|", L[k])
        arm = m.group(1)
        if fname != "split_ec_lead.out":
            check(T, "%s minus jit_exact, paired mean" % arm, mean([KM[arm][x]["ttft"] - KM["jit_exact"][x]["ttft"] for x in keys]), m.group(2), signed=True)
        k += 1
    i = find(L, "the same difference against jit_exact by stretch")
    k = i + 1
    while L[k].strip():
        arm = L[k].split()[0]
        for lo, hi, cnt, val in re.findall(r"(\d+)-(\d+): (\d+) ([+-][\d.]+) \|", L[k]):
            lo, hi = int(lo), int(hi)
            ks = [x for x in keys if x in POS and lo <= POS[x] < hi]
            check(T + " stretch parts [sum over the stretch / all shared turns]", "%s stretch %d-%d turns" % (arm, lo, hi), len(ks), cnt)
            check(T + " stretch parts [sum over the stretch / all shared turns]", "%s stretch %d-%d part of the difference" % (arm, lo, hi),
                  math.fsum(KM[arm][x]["ttft"] - KM["jit_exact"][x]["ttft"] for x in ks) / len(keys), val, signed=True)
        k += 1
    return keys


split_file("split_ec.out", "SEVEN")
split_file("split_ec_lead.out", "LEAD5")

L = read("stall_ec_ctl.out")
m = re.match(r"^(\d+) returning turns shared by (.*)", L[0])
arms = [x.strip() for x in m.group(2).split(",")]
keys = shared_of(arms)
T = "stall_ec_ctl.out (7 arms)"
check(T, "shared turns", len(keys), m.group(1))
i = find(L, "per arm: mean TTFT = read stall")
k = i + 1
while L[k].strip():
    f = L[k].split()
    check(T + " [my frozen SEVEN numbers]", "%s mean TTFT" % f[0], V[("1", "SEVEN", f[0], "mean_ttft")], f[1])
    k += 1
i = find(L, "by stretch of the oracle's send order")
k = i + 1
while L[k].strip():
    cells = L[k].split("|")
    m = re.match(r"^\s+(\d+)-\s*(\d+)\s+(\d+)", cells[0])
    lo, hi = int(m.group(1)), int(m.group(2))
    ks = [x for x in keys if x in POS and lo <= POS[x] < hi]
    check(T + " stretches", "stretch %d-%d turns" % (lo, hi), len(ks), m.group(3))
    for cell in cells[1:]:
        f = cell.split()
        check(T + " stretches", "stretch %d-%d %s mean TTFT" % (lo, hi, f[0]), mean([KM[f[0]][x]["ttft"] for x in ks]), f[1])
    k += 1
i = find(L, "against jit_exact, paired by turn, seconds per shared turn")
k = i + 1
while k < len(L) and L[k].strip():
    f = L[k].split()
    check(T, "%s minus jit_exact, paired mean TTFT" % f[0], mean([KM[f[0]][x]["ttft"] - KM["jit_exact"][x]["ttft"] for x in keys]), f[1], signed=True)
    k += 1

# ---------------------------------------------------------------- report
nok = sum(1 for r in ROWS if r[4])
print()
print("extra comparisons: %d printed numbers; %d match; %d do not" % (len(ROWS), nok, len(ROWS) - nok))
bt = {}
for r in ROWS:
    b = bt.setdefault(r[0], [0, 0])
    b[0] += 1
    b[1] += 1 if r[4] else 0
for t in bt:
    print("  %4d of %4d  %s" % (bt[t][1], bt[t][0], t))
print()
print("rows that do not match:")
for r in ROWS:
    if not r[4]:
        print("  %s | %s | mine %s (raw %.6f) | theirs %s" % (r[0], r[1], r[2], r[5], r[3]))
with open(os.path.join(HERE, "item6_extras.json"), "w") as f:
    json.dump({"compared": len(ROWS), "matched": nok,
               "by_table": {t: {"compared": c, "matched": o} for t, (c, o) in bt.items()},
               "mismatches": [{"table": r[0], "what": r[1], "mine": r[2], "theirs": r[3], "mine_raw": r[5]} for r in ROWS if not r[4]]},
              f, indent=1)
