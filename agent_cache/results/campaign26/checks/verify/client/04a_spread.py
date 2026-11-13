#!/usr/bin/env python3
"""Item 5, part a: how large is the run-to-run spread, and how concentrated are the means.

Uses parsed.pkl only.  Same definitions as 03_compute.py.
"""
import json
import math
import os
import pickle
import random

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "parsed.pkl"), "rb") as f:
    P = pickle.load(f)

LEAD5 = ["jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend"]
ARMSETS = {
    "ALL11": ["baseline", "base_rep", "hintfree", "jit_p50", "jit_exact", "jit_p50_ctl",
              "hintfree_ctl", "ec_r13", "nofc", "ec_r1", "nofc_noend"],
    "LEAD5": LEAD5,
    "SEVEN": LEAD5 + ["jit_p50_ctl", "hintfree_ctl"],
    "CUT7": ["jit_p50", "jit_p50_ctl", "jit_p50_d_cut", "hintfree_ctl", "nofc_noend", "ec_r13", "jit_exact"],
}
EPISODES = [(5500, 6999), (8750, 10999)]
RES = {}


def pct(v, q):
    s = sorted(v)
    return s[min(len(s) - 1, int(q * len(s)))]


def mean(v):
    return math.fsum(v) / len(v) if v else float("nan")


def with_ttft_sorted(turns):
    v = [t for t in turns if t.get("ttft") is not None]
    v.sort(key=lambda t: t["t_send"])
    return v


def kept_map(turns):
    v = with_ttft_sorted(turns)
    cut = v[-1]["t_send"] - 300.0
    return {(t["conv"], t["turn"]): t for t in v if t["t_send"] < cut}


KM = {a: kept_map(P[a]["turns"]) for a in P}
POS = {(t["conv"], t["turn"]): i for i, t in enumerate(with_ttft_sorted(P["ORDER"]["turns"]))}


def in_episode(k):
    p = POS.get(k)
    return p is not None and any(lo <= p <= hi for lo, hi in EPISODES)


def shared_of(arms):
    return sorted(set.intersection(*[{k for k in KM[a] if k[1] > 0} for a in arms]))


SH = {s: shared_of(a) for s, a in ARMSETS.items()}


def boot_ci(first, second, keys, seed=1234, B=2000):
    """Paired mean difference with two resampling schemes: sessions, and blocks of consecutive turns."""
    ma, mb = KM[first], KM[second]
    d = {k: mb[k]["ttft"] - ma[k]["ttft"] for k in keys}
    point = mean(list(d.values()))
    rng = random.Random(seed)
    out = {"point": point, "n": len(keys)}
    # by session
    byc = {}
    for k, x in d.items():
        s = byc.setdefault(k[0], [0.0, 0])
        s[0] += x
        s[1] += 1
    cl = list(byc.values())
    sims = []
    for _ in range(B):
        tot = 0.0
        n = 0
        for _i in range(len(cl)):
            s = cl[rng.randrange(len(cl))]
            tot += s[0]
            n += s[1]
        sims.append(tot / n)
    sims.sort()
    out["session_boot_ci95"] = [sims[int(0.025 * B)], sims[int(0.975 * B) - 1]]
    out["sessions"] = len(cl)
    # by blocks of consecutive turns in the first arm's send order
    okeys = sorted(keys, key=lambda k: ma[k]["t_send"])
    for blk in (250, 1000):
        blocks = []
        for i in range(0, len(okeys), blk):
            part = okeys[i:i + blk]
            blocks.append((math.fsum(d[k] for k in part), len(part)))
        sims = []
        for _ in range(B):
            tot = 0.0
            n = 0
            for _i in range(len(blocks)):
                s = blocks[rng.randrange(len(blocks))]
                tot += s[0]
                n += s[1]
            sims.append(tot / n)
        sims.sort()
        out["block%d_boot_ci95" % blk] = [sims[int(0.025 * B)], sims[int(0.975 * B) - 1]]
        out["blocks%d" % blk] = len(blocks)
    vals = sorted(d.values())
    out["median_paired_diff"] = pct(vals, 0.5)
    out["share_second_slower"] = sum(1 for x in vals if x > 0) / len(vals)
    out["sum_diff_s"] = math.fsum(vals)
    # how few turns carry the whole difference
    absd = sorted(vals, key=lambda x: -abs(x))
    out["largest_abs_single_turn_diff_s"] = absd[0]
    top = math.fsum(absd[:50])
    out["sum_of_50_largest_abs_diffs_signed_s"] = top
    out["sum_abs_diff_s"] = math.fsum(abs(x) for x in vals)
    out["sum_abs_top1pct_share"] = math.fsum(abs(x) for x in absd[:max(1, len(absd) // 100)]) / math.fsum(abs(x) for x in vals)
    return out


print("=== A. repeat pairs (same configuration run twice), second minus first")
print("on the ALL11 shared turns (6715) and on each pair's own shared turns")
pairs = [("baseline", "base_rep"), ("hintfree", "hintfree_ctl"), ("jit_p50", "jit_p50_ctl")]
RES["repeat_pairs"] = {}
for first, second in pairs:
    for scope, keys in (("ALL11", SH["ALL11"]), ("pairwise", shared_of([first, second]))):
        epi = [k for k in keys if in_episode(k)]
        els = [k for k in keys if not in_episode(k)]
        m1 = mean([KM[first][k]["ttft"] for k in keys])
        m2 = mean([KM[second][k]["ttft"] for k in keys])
        de = mean([KM[second][k]["ttft"] - KM[first][k]["ttft"] for k in epi])
        do = mean([KM[second][k]["ttft"] - KM[first][k]["ttft"] for k in els])
        n10a = sum(1 for k in keys if KM[first][k]["ttft"] > 10)
        n10b = sum(1 for k in keys if KM[second][k]["ttft"] > 10)
        b = boot_ci(first, second, keys)
        r = {"n": len(keys), "mean_first": m1, "mean_second": m2, "diff": m2 - m1, "rel_diff": (m2 - m1) / m1,
             "diff_episodes": de, "n_episodes": len(epi), "diff_elsewhere": do, "n_elsewhere": len(els),
             "over10_first": n10a, "over10_second": n10b, "boot": b}
        RES["repeat_pairs"]["%s-%s@%s" % (second, first, scope)] = r
        print("  %-24s %-8s n=%5d  %.4f -> %.4f  diff=%+.4f (%+.1f%%)  episodes %+.4f (n=%d)  elsewhere %+.4f (n=%d)"
              "  >10s %d -> %d" % ("%s-%s" % (second, first), scope, len(keys), m1, m2, m2 - m1, 100 * (m2 - m1) / m1,
                                   de, len(epi), do, len(els), n10a, n10b))
        print("      session-bootstrap 95%% [%+.3f, %+.3f]; 250-turn-block [%+.3f, %+.3f]; 1000-turn-block [%+.3f, %+.3f]"
              % (b["session_boot_ci95"][0], b["session_boot_ci95"][1], b["block250_boot_ci95"][0],
                 b["block250_boot_ci95"][1], b["block1000_boot_ci95"][0], b["block1000_boot_ci95"][1]))

print()
print("=== B. LEAD5 pairs and CUT7 pairs: resampling intervals for the paired mean difference (second minus first)")
RES["pairs_boot"] = {}
jobs = []
arms = ARMSETS["LEAD5"]
for i in range(len(arms)):
    for j in range(i + 1, len(arms)):
        jobs.append(("LEAD5", arms[i], arms[j]))
jobs += [("CUT7", "jit_p50_ctl", "jit_p50_d_cut"), ("CUT7", "jit_p50", "jit_p50_d_cut"), ("CUT7", "jit_p50", "jit_p50_ctl")]
print("%-6s %-28s %8s %19s %19s %19s %8s %7s %9s %9s" % (
    "set", "pair", "diff", "session 95%", "block250 95%", "block1000 95%", "med_d", "P(>0)", "max|d|", "top1%abs"))
for sname, first, second in jobs:
    b = boot_ci(first, second, SH[sname])
    RES["pairs_boot"]["%s:%s-%s" % (sname, second, first)] = b
    print("%-6s %-28s %+8.4f [%+7.3f, %+7.3f] [%+7.3f, %+7.3f] [%+7.3f, %+7.3f] %+8.4f %7.3f %+9.2f %9.3f" % (
        sname, "%s-%s" % (second, first), b["point"], b["session_boot_ci95"][0], b["session_boot_ci95"][1],
        b["block250_boot_ci95"][0], b["block250_boot_ci95"][1], b["block1000_boot_ci95"][0],
        b["block1000_boot_ci95"][1], b["median_paired_diff"], b["share_second_slower"],
        b["largest_abs_single_turn_diff_s"], b["sum_abs_top1pct_share"]))

print()
print("=== C. how concentrated is each arm's mean (shared turns of each set)")
RES["concentration"] = {}
for sname, arms in ARMSETS.items():
    keys = SH[sname]
    print("-- %s (n=%d)" % (sname, len(keys)))
    print("   %-14s %8s %9s %9s %9s %7s %7s %10s %12s" % (
        "arm", "mean", "top1%sum", ">10s_sum", "mean<=10", ">10s", "bursts", "burst_turns", "biggest_burst"))
    for a in arms:
        v = sorted((KM[a][k]["ttft"] for k in keys), reverse=True)
        tot = math.fsum(v)
        top1 = math.fsum(v[:max(1, len(v) // 100)])
        over = [x for x in v if x > 10]
        under = [x for x in v if x <= 10]
        # bursts: slow turns (>10 s) grouped when their send times are within 60 s of the previous slow turn
        slow = sorted((KM[a][k]["t_send"], KM[a][k]["ttft"]) for k in keys if KM[a][k]["ttft"] > 10)
        bursts = []
        for ts, x in slow:
            if bursts and ts - bursts[-1][-1][0] <= 60.0:
                bursts[-1].append((ts, x))
            else:
                bursts.append([(ts, x)])
        big = max((len(b) for b in bursts), default=0)
        r = {"mean": tot / len(v), "top1pct_share_of_sum": top1 / tot, "over10_share_of_sum": math.fsum(over) / tot,
             "mean_of_turns_le_10s": mean(under), "over10": len(over), "bursts_60s": len(bursts),
             "biggest_burst_turns": big,
             "burst_spans_min": [[round(b[0][0] / 60, 1), round(b[-1][0] / 60, 1), len(b)] for b in bursts]}
        RES["concentration"]["%s:%s" % (sname, a)] = r
        print("   %-14s %8.4f %9.3f %9.3f %9.4f %7d %7d %10d %12d" % (
            a, tot / len(v), top1 / tot, math.fsum(over) / tot, mean(under), len(over), len(bursts), len(over), big))
    if sname == "LEAD5":
        for a in arms:
            print("   bursts of %s (start min, end min, turns): %s" % (
                a, RES["concentration"]["%s:%s" % (sname, a)]["burst_spans_min"]))

with open(os.path.join(HERE, "item5a_spread.json"), "w") as f:
    json.dump(RES, f, indent=1, sort_keys=True)
print()
print("wrote item5a_spread.json")
