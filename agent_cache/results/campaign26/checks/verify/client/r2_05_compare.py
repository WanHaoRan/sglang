#!/usr/bin/env python3
"""r2_05_compare.py -- item 6: my numbers against what the existing .out files print.

Reads the copies in r2_out_snapshot/ (taken 2026-11-12 01:37Z; compare_ec.out and pairs_ec.out
were regenerated at 01:33Z with the arms jit_p50_d and ec_r13_rep added).
A number matches when my value, formatted with the printed number of decimals, gives the
printed string.

Categories:
  A   a number of items 1-3 (my four arm sets) that a file prints
  A4  a number of item 4 (kept turns, minute of the last kept send, errors) that a file prints
  X   any other number I can recompute from the client logs (other arm sets, other columns)

Run as:  nice -n 19 python3 -I -B r2_05_compare.py
"""
import json
import math
import os
import pickle
import re
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP = os.path.join(HERE, "r2_out_snapshot")

SETS = {
    "ALL11": ["baseline", "base_rep", "hintfree", "jit_p50", "jit_exact", "jit_p50_ctl",
              "hintfree_ctl", "ec_r13", "nofc", "ec_r1", "nofc_noend"],
    "LEAD5": ["jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend"],
    "SEVEN": ["jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend", "jit_p50_ctl", "hintfree_ctl"],
    "CUT7": ["jit_p50", "jit_p50_ctl", "jit_p50_d_cut", "hintfree_ctl", "nofc_noend", "ec_r13",
             "jit_exact"],
}
EPISODES = [(5500, 6999), (8750, 10999)]


def pct(v, q):
    s = sorted(v)
    return s[min(len(s) - 1, int(q * len(s)))]


def mean(v):
    return math.fsum(v) / len(v) if v else float("nan")


def recomputed(r):
    return max(r.get("reply_reprefilled") or 0, 0)


with open(os.path.join(HERE, "r2_parsed.pkl"), "rb") as f:
    P = pickle.load(f)
with open(os.path.join(HERE, "r2_parsed_extra.pkl"), "rb") as f:
    P.update(pickle.load(f))

OK, KEPT, KM, CUTMIN = {}, {}, {}, {}
for a, d in P.items():
    t = sorted((r for r in d["records"] if r.get("kind") == "turn" and r.get("ttft") is not None),
               key=lambda r: r["t_send"])
    OK[a] = t
    thr = t[-1]["t_send"] - 300.0
    KEPT[a] = [r for r in t if r["t_send"] < thr]
    KM[a] = {(r["conv"], r["turn"]): r for r in KEPT[a]}
    CUTMIN[a] = thr / 60.0
POS = {(r["conv"], r["turn"]): i for i, r in enumerate(OK["ORDER"])}


def in_ep(k):
    p = POS.get(k)
    return p is not None and any(lo <= p <= hi for lo, hi in EPISODES)


def shared(arms):
    s = None
    for a in arms:
        ks = set(k for k in KM[a] if k[1] > 0)
        s = ks if s is None else s & ks
    return sorted(s)


def arm_stats(a, keys):
    t = [KM[a][k]["ttft"] for k in keys]
    ep = [KM[a][k]["ttft"] for k in keys if in_ep(k)]
    el = [KM[a][k]["ttft"] for k in keys if not in_ep(k)]
    srt = sorted(t, reverse=True)
    return {"mean": mean(t), "median": pct(t, 0.5), "p90": pct(t, 0.9), "p99": pct(t, 0.99),
            "p999": pct(t, 0.999), "max": max(t), "over2": sum(1 for x in t if x > 2.0),
            "over10": sum(1 for x in t if x > 10.0), "mean_ep": mean(ep), "mean_el": mean(el),
            "n_ep": len(ep), "n_el": len(el),
            "recomp": mean([recomputed(KM[a][k]) for k in keys]),
            "top1share": math.fsum(srt[:len(srt) // 100]) / math.fsum(srt)}


def paired(second, first, keys):
    d = [KM[second][k]["ttft"] - KM[first][k]["ttft"] for k in keys]
    dep = [x for x, k in zip(d, keys) if in_ep(k)]
    del_ = [x for x, k in zip(d, keys) if not in_ep(k)]
    return {"all": mean(d), "ep": mean(dep), "el": mean(del_), "d": d}


SH = {s: shared(a) for s, a in SETS.items()}
ROWS = []


def decimals(s):
    s = s.strip().lstrip("+-")
    return len(s.split(".")[1]) if "." in s else 0


def check(cat, fname, block, what, mine, theirs, signed=False, scale=1.0):
    theirs = theirs.strip()
    fmt = ("%+." if signed else "%.") + "%df" % decimals(theirs)
    ms = fmt % (mine * scale)
    # "-0.000" and "+0.000" print the same quantity
    same = ms == theirs or (float(ms) == 0.0 and float(theirs) == 0.0)
    ROWS.append({"cat": cat, "file": fname, "block": block, "what": what, "mine": ms,
                 "theirs": theirs, "ok": same, "raw": mine * scale})


def read(name):
    with open(os.path.join(SNAP, name)) as f:
        return f.read().split("\n")


def find(L, start, frm=0):
    for k in range(frm, len(L)):
        if L[k].startswith(start):
            return k
    raise KeyError(start)


def arms_of_file(L):
    i = find(L, "runs:")
    arms = []
    k = i + 1
    while L[k].strip():
        arms.append(L[k].split()[0])
        k += 1
    return arms


def cat_for(sname, arm=None, sets_arms=None):
    return "A" if sname in SETS and (arm is None or arm in SETS[sname]) else "X"


# ---------------------------------------------------------------- compare_ec*.out
def compare_file(fname, my_set_name):
    """my_set_name: the task's arm set whose shared keys equal this file's 'all arms' keys,
    or None when the file's set is not one of the four."""
    L = read(fname)
    arms = arms_of_file(L)
    missing = [a for a in arms if a not in KM]
    assert not missing, missing
    keys = shared(arms)
    same_as = None
    for s in SETS:
        if SH[s] == keys:
            same_as = s
    print("%s: %d arms; their shared keys by my parser: %d; identical to my set: %s"
          % (fname, len(arms), len(keys), same_as))
    setcat = "A" if same_as else "X"
    # --- own-kept table
    i = find(L, "per arm, turns sent at least 300 s before its last send")
    k = i + 1
    while L[k].strip():
        f = L[k].split()
        a = f[0]
        kept = KEPT[a]
        ret = [r for r in kept if r["turn"] > 0]
        allt = [r["ttft"] for r in kept]
        check("A4", fname, "own-kept", "%s kept turns" % a, len(kept), f[1])
        check("A4", fname, "own-kept", "%s minute of last kept send vs 'up to minute'" % a,
              kept[-1]["t_send"] / 60.0, f[2])
        check("X", fname, "own-kept", "%s cut-off minute (max t_send - 300)/60 vs 'up to minute'"
              % a, CUTMIN[a], f[2])
        check("X", fname, "own-kept", "%s kept first turns vs 'users started'" % a,
              len(kept) - len(ret), f[3])
        nerr = sum(1 for r in P[a]["records"] if r.get("kind") == "turn"
                   and (r.get("error") or r.get("ttft") is None))
        check("A4", fname, "own-kept", "%s turn records with an error or null ttft" % a, nerr, f[4])
        check("X", fname, "own-kept", "%s mean TTFT all kept turns" % a, mean(allt), f[5])
        check("X", fname, "own-kept", "%s median all kept turns" % a, pct(allt, 0.5), f[6])
        check("X", fname, "own-kept", "%s p90 all kept turns" % a, pct(allt, 0.9), f[7])
        check("X", fname, "own-kept", "%s p99 all kept turns" % a, pct(allt, 0.99), f[8])
        check("X", fname, "own-kept", "%s recomputed per kept returning turn" % a,
              mean([recomputed(r) for r in ret]), f[10])
        k += 1
    # --- all-arms block
    i = find(L, "all arms:")
    n = re.match(r"all arms: (\d+) returning turns", L[i]).group(1)
    check(setcat, fname, "all-arms", "shared turns (%s)" % (same_as or "%d-arm set" % len(arms)),
          len(keys), n)
    st = {a: arm_stats(a, keys) for a in arms}
    k = i + 1
    while not L[k].lstrip().startswith("by stretch"):
        m = re.match(r"^\s+(\S+)\s+([\d.]+) =\s+([\d.]+) \+\s+([\d.]+) \|\s+(\d+) \|\s+([\d.]+) % "
                     r"\|\s+(-?\d+) %\s+(-?\d+) %", L[k])
        assert m, L[k]
        a = m.group(1)
        c = "A" if (same_as and a in SETS[same_as]) else "X"
        check(c, fname, "all-arms", "%s mean TTFT" % a, st[a]["mean"], m.group(2))
        check(c, fname, "all-arms", "%s mean recomputed tokens" % a, st[a]["recomp"], m.group(5))
        g1 = (st["baseline"]["mean"] - st[a]["mean"]) / (st["baseline"]["mean"] - st["jit_exact"]["mean"])
        g2 = (st["base_rep"]["mean"] - st[a]["mean"]) / (st["base_rep"]["mean"] - st["jit_exact"]["mean"])
        check(c, fname, "all-arms", "%s share of gain vs baseline" % a, g1, m.group(7), scale=100.0)
        check(c, fname, "all-arms", "%s share of gain vs base_rep" % a, g2, m.group(8), scale=100.0)
        k += 1
    hdr = [x.strip() for x in L[k].split("|")[1:]]
    assert hdr == arms, (hdr, arms)
    k += 1
    cnt_other = 0
    while L[k].strip():
        m = re.match(r"^\s+(\d+)-\s*(\d+):\s+(\d+) \|(.*)", L[k])
        assert m, L[k]
        lo, hi, cnt = int(m.group(1)), int(m.group(2)), m.group(3)
        sk = [x for x in keys if x in POS and lo <= POS[x] < hi]
        vals = [x.strip() for x in m.group(4).split("|")]
        is_ep1 = (lo, hi) == (5500, 7000)
        if is_ep1:
            # none of these sets holds a turn at positions 8750+, so the stretch is the whole
            # "load episodes" class
            n_ep2 = sum(1 for x in keys if x in POS and POS[x] >= 8750)
            assert n_ep2 == 0, n_ep2
            check(setcat, fname, "stretches", "turns in load episodes (stretch 5500-7000)",
                  len(sk), cnt)
        else:
            cnt_other += int(cnt)
            check("X", fname, "stretches", "turns in stretch %d-%d" % (lo, hi), len(sk), cnt)
        for a, x in zip(arms, vals):
            c = "A" if (is_ep1 and same_as and a in SETS[same_as]) else "X"
            check(c, fname, "stretches", "%s mean TTFT stretch %d-%d" % (a, lo, hi),
                  mean([KM[a][q]["ttft"] for q in sk]), x)
        k += 1
    check(setcat, fname, "stretches", "turns elsewhere (sum of the other printed stretches)",
          sum(1 for x in keys if not in_ep(x)), str(cnt_other))
    # --- run to run block: baseline against base_rep on their own shared turns
    i = find(L, "run to run:")
    rk = shared(["baseline", "base_rep"])
    rr_is_all11 = rk == SH["ALL11"]
    print("   run-to-run keys (baseline, base_rep): %d; identical to ALL11: %s" % (len(rk), rr_is_all11))
    k = i + 1
    while L[k].strip():
        f = L[k].split()
        if f[0] == "all":
            sk, nums, lab = rk, f[1:], "all"
            is_item = rr_is_all11
        else:
            m = re.match(r"^\s+(\d+)-\s*(\d+)\s+(.*)", L[k])
            lo, hi = int(m.group(1)), int(m.group(2))
            sk = [x for x in rk if x in POS and lo <= POS[x] < hi]
            nums, lab = m.group(3).split(), "%d-%d" % (lo, hi)
            is_item = rr_is_all11 and (lo, hi) == (5500, 7000)
        d = [KM["base_rep"][q]["ttft"] - KM["baseline"][q]["ttft"] for q in sk]
        c = "A" if is_item else "X"
        check(c, fname, "run-to-run", "%s n" % lab, len(sk), nums[0])
        check(c, fname, "run-to-run", "%s baseline mean" % lab,
              mean([KM["baseline"][q]["ttft"] for q in sk]), nums[1])
        check(c, fname, "run-to-run", "%s base_rep mean" % lab,
              mean([KM["base_rep"][q]["ttft"] for q in sk]), nums[2])
        check("X", fname, "run-to-run", "%s paired mean change" % lab, mean(d), nums[3], signed=True)
        check("X", fname, "run-to-run", "%s median |change| (statistics.median)" % lab,
              statistics.median([abs(x) for x in d]), nums[4])
        check("X", fname, "run-to-run", "%s faster in the repeat" % lab,
              sum(1 for x in d if x < 0), nums[5])
        check("X", fname, "run-to-run", "%s slower in the repeat" % lab,
              sum(1 for x in d if x > 0), nums[6])
        k += 1
    # --- mechanisms: first column is the arm's kept returning turns
    i = find(L, "mechanisms on each arm's own returning turns")
    k = i + 1
    while L[k].strip():
        f = L[k].split()
        check("X", fname, "mechanisms", "%s kept returning turns" % f[0],
              sum(1 for r in KEPT[f[0]] if r["turn"] > 0), f[1])
        k += 1
    # --- bins of 500 positions of each arm's own send order
    i = find(L, "by position in each arm's own send order")
    hdr = [x.strip() for x in L[i + 1].split("|")[1:]]
    assert hdr == arms, hdr
    k = i + 2
    while L[k].strip():
        cells = L[k].split("|")
        p0 = int(cells[0])
        for a, cell in zip(arms, cells[1:]):
            f = cell.split()
            if not f:
                continue
            b = KEPT[a][p0:p0 + 500]
            check("X", fname, "bins500", "%s bin %d mean TTFT" % (a, p0),
                  mean([r["ttft"] for r in b]), f[0])
            check("X", fname, "bins500", "%s bin %d minute reached" % (a, p0),
                  b[0]["t_send"] / 60.0, f[1])
        k += 1
    # --- 15 minute rows: turns and mean TTFT (all kept turns by send time)
    i = find(L, "per 15 min of each arm's own clock")
    k = i + 2
    while L[k].strip():
        cells = L[k].split("|")
        m0 = int(cells[0])
        # each arm takes three cells: "turns mean queue", "load K", "GPU ..."
        for j, a in enumerate(arms):
            f = cells[1 + 3 * j].split() if 1 + 3 * j < len(cells) else []
            if not f:
                continue
            b = [r["ttft"] for r in KEPT[a] if m0 * 60 <= r["t_send"] < (m0 + 15) * 60]
            check("X", fname, "rows15", "%s min %d turns" % (a, m0), len(b), f[0])
            check("X", fname, "rows15", "%s min %d mean TTFT" % (a, m0), mean(b), f[1])
        k += 1
    # --- minute reached at positions of the own send order
    i = find(L, "minute each arm reached positions of its own send order")
    k = i + 1
    while k < len(L) and L[k].strip():
        f = L[k].split()
        a = f[0]
        for pos, val in re.findall(r"(\d+):\s+([\d.]+)", L[k]):
            check("X", fname, "reached", "%s minute at own position %s" % (a, pos),
                  KEPT[a][int(pos)]["t_send"] / 60.0, val)
        k += 1
    return arms, keys


# ---------------------------------------------------------------- split_ec_lead.out
def split_file(fname, named=True):
    L = read(fname)
    m = re.match(r"^(\d+) returning turns shared by (.*)", L[0])
    arms = [x.strip() for x in m.group(2).split(",")]
    keys = shared(arms)
    sname = None
    for s in SETS:
        if SH[s] == keys and (set(SETS[s]) == set(arms) or not named):
            sname = s
    print("%s: arms %s; my shared keys %d; same keys as my set: %s" % (fname, arms, len(keys), sname))
    base = ("A" if named else "B") if sname else "X"
    c = base
    check(c, fname, "head", "shared turns (%s)" % sname, len(keys), m.group(1))
    st = {a: arm_stats(a, keys) for a in arms}
    i = find(L, "per arm: mean TTFT, median, p90, p99, p99.9, max")
    k = i + 1
    while L[k].strip():
        m = re.match(r"^\s+(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+) "
                     r"\|\s+(\d+)\s+(\d+) \|\s+([\d.]+) %", L[k])
        assert m, L[k]
        a = m.group(1)
        c = base if (sname and a in SETS[sname]) else "X"
        check(c, fname, "per-arm", "%s mean TTFT" % a, st[a]["mean"], m.group(2))
        check(c, fname, "per-arm", "%s median" % a, st[a]["median"], m.group(3))
        check(c, fname, "per-arm", "%s p90" % a, st[a]["p90"], m.group(4))
        check(c, fname, "per-arm", "%s p99" % a, st[a]["p99"], m.group(5))
        check("X", fname, "per-arm", "%s p99.9" % a, st[a]["p999"], m.group(6))
        check("X", fname, "per-arm", "%s max" % a, st[a]["max"], m.group(7))
        check("X", fname, "per-arm", "%s turns over 2 s" % a, st[a]["over2"], m.group(8))
        check(c, fname, "per-arm", "%s turns over 10 s" % a, st[a]["over10"], m.group(9))
        check("X", fname, "per-arm", "%s share of summed TTFT in slowest 1 %%" % a,
              st[a]["top1share"], m.group(10), scale=100.0)
        k += 1
    # by whether the turn read from SSD: client-side proxy cached_details.storage > 0
    i = find(L, "per arm, by whether the turn read from SSD in that arm")
    k = i + 1
    while L[k].strip():
        m = re.match(r"^\s+(\S+)\s+(read|no read)\s+(\d+)\s+([\d.]+) =\s+([\d.]+) \+\s+([\d.]+)\s+(\d+) "
                     r"\|\s+([\d.]+)", L[k])
        assert m, L[k]
        a, cls = m.group(1), m.group(2)

        def is_read(r):
            cd = r.get("cached_details") or {}
            return (cd.get("storage") or 0) > 0
        sel = [q for q in keys if is_read(KM[a][q]) == (cls == "read")]
        check("X", fname, "read-class", "%s %s turns (client proxy: cached_details.storage > 0)"
              % (a, cls), len(sel), m.group(3))
        k += 1
    i = find(L, "against jit_exact, paired by turn")
    k = i + 2
    while L[k].strip():
        m = re.match(r"^\s+(\S+)\s+([+-][\d.]+) \|", L[k])
        assert m, L[k]
        p = paired(m.group(1), "jit_exact", keys)
        check(base if sname == "LEAD5" else "X", fname, "paired-vs-jit_exact",
              "%s minus jit_exact" % m.group(1), p["all"], m.group(2), signed=True)
        k += 1
    i = find(L, "the same difference against jit_exact by stretch")
    k = i + 1
    first = True
    while L[k].strip():
        a = L[k].split()[0]
        d = paired(a, "jit_exact", keys)["d"]
        parts = re.findall(r"(\d+)-(\d+): (\d+) ([+-][\d.]+) \|", L[k])
        tot_ep = 0
        for lo, hi, cnt, val in parts:
            lo, hi = int(lo), int(hi)
            idx = [j for j, q in enumerate(keys) if q in POS and lo <= POS[q] < hi]
            if first:
                check("X", fname, "stretch-parts", "turns in stretch %d-%d" % (lo, hi), len(idx), cnt)
            if (lo, hi) in ((5500, 7000), (8750, 11000)):
                tot_ep += int(cnt)
            check("X", fname, "stretch-parts", "%s minus jit_exact, part in stretch %d-%d "
                  "(sum over the stretch / all shared turns)" % (a, lo, hi),
                  math.fsum(d[j] for j in idx) / len(keys), val, signed=True)
        if first:
            check(base, fname, "stretch-parts", "turns in load episodes (sum of the two printed "
                  "episode stretches)", sum(1 for q in keys if in_ep(q)), str(tot_ep))
            n_listed = sum(int(p[2]) for p in parts)
            print("   stretches list %d of the %d shared turns; %d have no order position"
                  % (n_listed, len(keys), len(keys) - n_listed))
        first = False
        k += 1


# ---------------------------------------------------------------- pairs_ec*.out
def pairs_file(fname, wanted_pairs, named=True):
    L = read(fname)
    m = re.match(r"^(\d+) returning turns shared by (.*)", L[0])
    arms = [x.strip() for x in m.group(2).split(",")]
    keys = shared(arms)
    sname = None
    for s in SETS:
        if SH[s] == keys and (set(SETS[s]) == set(arms) or not named):
            sname = s
    print("%s: arms %s; my shared keys %d; same keys as my set: %s" % (fname, arms, len(keys), sname))
    base = ("A" if named else "B") if sname else "X"
    c = base
    check(c, fname, "head", "shared turns (%s)" % sname, len(keys), m.group(1))
    st = {a: arm_stats(a, keys) for a in arms}
    i = find(L, "per arm on them:")
    k = i + 1
    while L[k].strip():
        m = re.match(r"^\s+(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+) \|\s+([\d.]+)\s+"
                     r"([\d.]+) \|\s+(\d+)\s+(\d+)\s+(\d+)", L[k])
        assert m, L[k]
        a = m.group(1)
        c = base if (sname and a in SETS[sname]) else "X"
        check(c, fname, "per-arm", "%s mean TTFT" % a, st[a]["mean"], m.group(2))
        check(c, fname, "per-arm", "%s median" % a, st[a]["median"], m.group(3))
        check(c, fname, "per-arm", "%s p90" % a, st[a]["p90"], m.group(4))
        check(c, fname, "per-arm", "%s p99" % a, st[a]["p99"], m.group(5))
        check(c, fname, "per-arm", "%s turns over 10 s" % a, st[a]["over10"], m.group(6))
        check(c, fname, "per-arm", "%s mean TTFT load episodes" % a, st[a]["mean_ep"], m.group(7))
        check(c, fname, "per-arm", "%s mean TTFT elsewhere" % a, st[a]["mean_el"], m.group(8))
        check(c, fname, "per-arm", "%s mean recomputed tokens" % a, st[a]["recomp"], m.group(11))
        k += 1
    sections = [("all shared returning turns:", "all", None),
                ("the two load episodes", "ep", "turns in load episodes"),
                ("the other positions:", "el", "turns elsewhere")]
    for start, part, cname in sections:
        i = find(L, start)
        cnt = re.search(r": (\d+) turns", L[i]).group(1)
        if cname:
            mine = sum(1 for q in keys if in_ep(q)) if part == "ep" else sum(1 for q in keys if not in_ep(q))
            check(base, fname, "matrix-%s" % part, cname, mine, cnt)
        cols = L[i + 1].split()
        k = i + 2
        while L[k].strip():
            row = L[k].split()[0]
            ents = re.findall(r"([+-]\d+\.\d+) \[\s*([+-][\d.]+),\s*([+-][\d.]+)\]", L[k])
            for col, (val, _lo, _hi) in zip(cols, ents):
                name = "%s-%s" % (row, col)
                cc = base if (sname and wanted_pairs is not None and name in wanted_pairs) else "X"
                check(cc, fname, "matrix-%s" % part, "%s minus %s" % (row, col),
                      paired(row, col, keys)[part], val, signed=True)
            k += 1
    i = find(L, "the same arm run again")
    k = i + 1
    while L[k].strip():
        m = re.match(r"^\s+.*?(\S+)\s+- (\S+)\s+([+-][\d.]+) \[.*?\] ([+-][\d.]+) ([+-][\d.]+)\s*$", L[k])
        assert m, L[k]
        name = "%s-%s" % (m.group(1), m.group(2))
        p = paired(m.group(1), m.group(2), keys)
        cc = base if (sname and wanted_pairs is not None and name in wanted_pairs) else "X"
        check(cc, fname, "repeat-lines", "%s minus %s" % (m.group(1), m.group(2)), p["all"],
              m.group(3), signed=True)
        check(cc, fname, "repeat-lines", "%s minus %s, load episodes" % (m.group(1), m.group(2)),
              p["ep"], m.group(4), signed=True)
        check(cc, fname, "repeat-lines", "%s minus %s, elsewhere" % (m.group(1), m.group(2)),
              p["el"], m.group(5), signed=True)
        k += 1
    return arms, keys


arms13, keys13 = compare_file("compare_ec.out", "ALL11")
arms12, keys12 = compare_file("compare_ec_cut.out", None)
split_file("split_ec_lead.out")
arms6, keys6 = pairs_file("pairs_ec.out", None)
cut_pairs = {"jit_p50_d_cut-jit_p50_ctl", "jit_p50_d_cut-jit_p50", "jit_p50_ctl-jit_p50"}
pairs_file("pairs_ec_cut.out", cut_pairs)

# ---- part B: three files outside the five named ones that print more of my numbers ----------
pairs_file("pairs_ec_ctl.out", None, named=False)
split_file("split_ec.out", named=False)


def stall_file(fname):
    L = read(fname)
    m = re.match(r"^(\d+) returning turns shared by (.*)", L[0])
    arms = [x.strip() for x in m.group(2).split(",")]
    keys = shared(arms)
    sname = None
    for s_ in SETS:
        if SH[s_] == keys:
            sname = s_
    print("%s: arms %s; my shared keys %d; same keys as my set: %s" % (fname, arms, len(keys), sname))
    base = "B" if sname else "X"
    check(base, fname, "head", "shared turns (%s)" % sname, len(keys), m.group(1))
    i = find(L, "per arm: mean TTFT = read stall")
    k = i + 1
    while L[k].strip():
        f = L[k].split()
        c = base if (sname and f[0] in SETS[sname]) else "X"
        check(c, fname, "per-arm", "%s mean TTFT" % f[0], mean([KM[f[0]][q]["ttft"] for q in keys]), f[1])
        k += 1
    i = find(L, "by stretch of the oracle's send order")
    k = i + 1
    cnt = {}
    while L[k].strip():
        m = re.match(r"^\s+(\d+)-\s*(\d+)\s+(\d+) \|(.*)", L[k])
        lo, hi = int(m.group(1)), int(m.group(2))
        cnt[(lo, hi)] = int(m.group(3))
        sk = [q for q in keys if q in POS and lo <= POS[q] < hi]
        check("X", fname, "stretches", "turns in stretch %d-%d" % (lo, hi), len(sk), m.group(3))
        for cell in m.group(4).split("|"):
            f = cell.split()
            check("X", fname, "stretches", "%s mean TTFT stretch %d-%d" % (f[0], lo, hi),
                  mean([KM[f[0]][q]["ttft"] for q in sk]), f[1])
        k += 1
    check(base, fname, "stretches", "turns in load episodes (two printed episode stretches)",
          sum(1 for q in keys if in_ep(q)), str(cnt[(5500, 7000)] + cnt[(8750, 11000)]))
    check(base, fname, "stretches", "turns elsewhere (three other printed stretches)",
          sum(1 for q in keys if not in_ep(q)), str(cnt[(0, 3000)] + cnt[(3000, 5500)] + cnt[(7000, 8750)]))
    i = find(L, "against jit_exact, paired by turn, seconds per shared turn")
    k = i + 1
    while k < len(L) and L[k].strip():
        f = L[k].split()
        check("X", fname, "paired-vs-jit_exact", "%s minus jit_exact" % f[0],
              paired(f[0], "jit_exact", keys)["all"], f[1], signed=True)
        k += 1


stall_file("stall_ec_ctl.out")

# relation of the 6-arm set of pairs_ec.out to LEAD5
l5 = set(SH["LEAD5"])
k6 = set(keys6)
print("\npairs_ec.out set against LEAD5: LEAD5 %d, six-arm %d, in both %d, LEAD5 only %d "
      "(episodes %d, elsewhere %d)" % (len(l5), len(k6), len(l5 & k6), len(l5 - k6),
                                         sum(1 for q in l5 - k6 if in_ep(q)),
                                         sum(1 for q in l5 - k6 if not in_ep(q))))

# ---------------------------------------------------------------- report
print()
by = {}
for r in ROWS:
    b = by.setdefault((r["cat"], r["file"], r["block"]), [0, 0])
    b[0] += 1
    b[1] += 1 if r["ok"] else 0
tot = {}
for (cat, fn, blk), (n, ok) in sorted(by.items()):
    print("  %-2s %-20s %-20s compared %4d  matched %4d" % (cat, fn, blk, n, ok))
    t = tot.setdefault(cat, [0, 0])
    t[0] += n
    t[1] += ok
    t2 = tot.setdefault((cat, fn), [0, 0])
    t2[0] += n
    t2[1] += ok
print()
for key in sorted(tot, key=str):
    print("  total %-28s compared %4d  matched %4d  not %d" % (str(key), tot[key][0], tot[key][1],
                                                              tot[key][0] - tot[key][1]))
print("\nrows that do not match:")
for r in ROWS:
    if not r["ok"]:
        print("  %-2s %-20s %-14s %-70s mine %10s (raw %.6f) theirs %10s"
              % (r["cat"], r["file"], r["block"], r["what"], r["mine"], r["raw"], r["theirs"]))
with open(os.path.join(HERE, "r2_item6_compare.json"), "w") as f:
    json.dump({"totals": {str(k): v for k, v in tot.items()}, "rows": ROWS}, f, indent=1)
print("\nwrote r2_item6_compare.json with %d rows" % len(ROWS))
