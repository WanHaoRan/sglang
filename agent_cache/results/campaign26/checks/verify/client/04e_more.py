#!/usr/bin/env python3
"""Item 5, part e: remaining small checks (episode coverage of each set, admission order, first turns included)."""
import json
import math
import os
import pickle

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
ORDER_SORTED = with_ttft_sorted(P["ORDER"]["turns"])
POS = {(t["conv"], t["turn"]): i for i, t in enumerate(ORDER_SORTED)}


def epi_no(k):
    p = POS.get(k)
    if p is None:
        return None
    for i, (lo, hi) in enumerate(EPISODES):
        if lo <= p <= hi:
            return i + 1
    return 0


def shared_of(arms):
    return sorted(set.intersection(*[{k for k in KM[a] if k[1] > 0} for a in arms]))


SH = {s: shared_of(a) for s, a in ARMSETS.items()}

print("=== 1. which part of the two load episodes each shared set holds")
n_ep = {1: sum(1 for k in POS if epi_no(k) == 1 and k[1] > 0), 2: sum(1 for k in POS if epi_no(k) == 2 and k[1] > 0)}
print("   order run: returning turns in episode 1: %d, in episode 2 (8750..10318, the run ends there): %d" % (n_ep[1], n_ep[2]))
RES["episode_coverage"] = {"order_run_returning_in_episode1": n_ep[1], "order_run_returning_in_episode2": n_ep[2]}
for s, keys in SH.items():
    c = {0: 0, 1: 0, 2: 0, None: 0}
    for k in keys:
        c[epi_no(k)] += 1
    mxp = max((POS[k] for k in keys if k in POS))
    RES["episode_coverage"][s] = {"elsewhere_with_position": c[0], "episode1": c[1], "episode2": c[2], "absent": c[None],
                                  "max_position": mxp, "share_of_episode1": c[1] / n_ep[1], "share_of_episode2": c[2] / n_ep[2]}
    print("   %-6s n=%5d  elsewhere(with position) %5d  episode1 %4d (%.2f of it)  episode2 %4d (%.2f of it)  absent %4d  max position %d" % (
        s, len(keys), c[0], c[1], c[1] / n_ep[1], c[2], c[2] / n_ep[2], c[None], mxp))

print()
print("=== 2. per set and arm: mean TTFT by episode (episode 1, episode 2 separately)")
RES["by_episode"] = {}
for s, arms in ARMSETS.items():
    keys = SH[s]
    e1 = [k for k in keys if epi_no(k) == 1]
    e2 = [k for k in keys if epi_no(k) == 2]
    print("-- %s: episode1 n=%d, episode2 n=%d" % (s, len(e1), len(e2)))
    for a in arms:
        m1 = mean([KM[a][k]["ttft"] for k in e1])
        m2 = mean([KM[a][k]["ttft"] for k in e2]) if e2 else float("nan")
        o1 = sum(1 for k in e1 if KM[a][k]["ttft"] > 10)
        o2 = sum(1 for k in e2 if KM[a][k]["ttft"] > 10)
        RES["by_episode"]["%s:%s" % (s, a)] = {"episode1_mean": m1, "episode2_mean": m2, "episode1_over10": o1, "episode2_over10": o2}
        print("   %-14s episode1 %8.4f (>10s %4d)   episode2 %8.4f (>10s %4d)" % (a, m1, o1, m2, o2))

print()
print("=== 3. sessions are admitted in the same order in every run?")
ref = [r["conv"] for r in sorted(P["jit_exact"]["conv_start"], key=lambda r: (r["start_s"], r["_line"]))]
for a, d in P.items():
    seq = [r["conv"] for r in sorted(d["conv_start"], key=lambda r: (r["start_s"], r["_line"]))]
    n = min(len(seq), len(ref))
    same = sum(1 for i in range(n) if seq[i] == ref[i])
    asc = sum(1 for i in range(1, len(seq)) if seq[i] > seq[i - 1])
    RES.setdefault("admission", {})[a] = {"sessions": len(seq), "same_as_jit_exact_prefix": same, "compared": n,
                                          "ascending_steps": asc}
    print("   %-14s sessions %3d  same conv at the same admission rank as jit_exact: %d of %d  ascending conv ids: %d of %d steps" % (
        a, len(seq), same, n, asc, len(seq) - 1))

print()
print("=== 4. LEAD5 with first turns included (all keys kept in all five arms)")
keys_all = sorted(set.intersection(*[set(KM[a]) for a in LEAD5]))
print("   keys %d (returning %d, first %d)" % (len(keys_all), sum(1 for k in keys_all if k[1] > 0), sum(1 for k in keys_all if k[1] == 0)))
RES["lead5_with_first"] = {"keys": len(keys_all)}
for a in LEAD5:
    m = mean([KM[a][k]["ttft"] for k in keys_all])
    RES["lead5_with_first"][a] = m
    print("   %-12s mean TTFT %.4f" % (a, m))

print()
print("=== 5. the repeat pairs on the part of the run that LEAD5 and SEVEN cover")
for first, second in (("jit_p50", "jit_p50_ctl"), ("hintfree", "hintfree_ctl"), ("baseline", "base_rep")):
    keys = shared_of([first, second])
    e1 = [k for k in keys if epi_no(k) == 1]
    e2 = [k for k in keys if epi_no(k) == 2]
    el = [k for k in keys if epi_no(k) == 0]
    f = lambda ks: (mean([KM[second][k]["ttft"] - KM[first][k]["ttft"] for k in ks]) if ks else float("nan"))
    g = lambda a, ks: (mean([KM[a][k]["ttft"] for k in ks]) if ks else float("nan"))
    RES.setdefault("repeat_by_episode", {})["%s-%s" % (second, first)] = {
        "n": len(keys), "episode1": {"n": len(e1), "first": g(first, e1), "second": g(second, e1), "diff": f(e1)},
        "episode2": {"n": len(e2), "first": g(first, e2), "second": g(second, e2), "diff": f(e2)},
        "elsewhere": {"n": len(el), "first": g(first, el), "second": g(second, el), "diff": f(el)}}
    print("   %-22s n=%5d  episode1 (n=%4d) %7.3f -> %7.3f (%+.3f)   episode2 (n=%4d) %7.3f -> %7.3f (%+.3f)   elsewhere (n=%4d) %6.3f -> %6.3f (%+.3f)" % (
        "%s-%s" % (second, first), len(keys), len(e1), g(first, e1), g(second, e1), f(e1), len(e2), g(first, e2),
        g(second, e2), f(e2), len(el), g(first, el), g(second, el), f(el)))

print()
print("=== 6. the largest single stalls of each LEAD5 arm: 5-minute bin (own clock) with the most seconds of TTFT over 10 s")
for a in LEAD5:
    keys = SH["LEAD5"]
    b = {}
    for k in keys:
        t = KM[a][k]
        if t["ttft"] > 10:
            b.setdefault(int(t["t_send"] // 300) * 5, []).append(t["ttft"])
    tot = math.fsum(KM[a][k]["ttft"] for k in keys)
    rows = sorted(((math.fsum(v), m, len(v)) for m, v in b.items()), reverse=True)
    RES.setdefault("lead5_stall_bins", {})[a] = [{"minute": m, "turns_over_10s": n, "sum_s": s, "share_of_arm_sum": s / tot} for s, m, n in rows]
    print("   %-12s total %.0f s; " % (a, tot) + "; ".join("min %d: %d turns, %.0f s (%.1f%%)" % (m, n, s, 100 * s / tot) for s, m, n in rows[:4]))

with open(os.path.join(HERE, "item5e_more.json"), "w") as f:
    json.dump(RES, f, indent=1, sort_keys=True)
print()
print("wrote item5e_more.json")
