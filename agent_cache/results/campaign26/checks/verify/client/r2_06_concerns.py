#!/usr/bin/env python3
"""r2_06_concerns.py -- item 5: what makes a shared-turn comparison unfair or a number unreliable.

Fresh code on r2_parsed.pkl (+ r2_parsed_extra.pkl for the two arms that landed later:
jit_p50_d and ec_r13_rep, which are NOT in the task's list and are used only as repeats).
Output: stdout and r2_item5.json.

Run as:  nice -n 19 python3 -I -B r2_06_concerns.py
"""
import json
import math
import os
import pickle
import random

HERE = os.path.dirname(os.path.abspath(__file__))
SETS = {
    "ALL11": ["baseline", "base_rep", "hintfree", "jit_p50", "jit_exact", "jit_p50_ctl",
              "hintfree_ctl", "ec_r13", "nofc", "ec_r1", "nofc_noend"],
    "LEAD5": ["jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend"],
    "SEVEN": ["jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend", "jit_p50_ctl", "hintfree_ctl"],
    "CUT7": ["jit_p50", "jit_p50_ctl", "jit_p50_d_cut", "hintfree_ctl", "nofc_noend", "ec_r13",
             "jit_exact"],
}
ARMS12 = ["baseline", "base_rep", "hintfree", "hintfree_ctl", "jit_p50", "jit_p50_ctl",
          "jit_p50_d_cut", "jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend"]
EXTRA = ["jit_p50_d", "ec_r13_rep"]
EPISODES = [(5500, 6999), (8750, 10999)]
RES = {}


def pct(v, q):
    s = sorted(v)
    return s[min(len(s) - 1, int(q * len(s)))]


def mean(v):
    return math.fsum(v) / len(v) if v else float("nan")


with open(os.path.join(HERE, "r2_parsed.pkl"), "rb") as f:
    P = pickle.load(f)
with open(os.path.join(HERE, "r2_parsed_extra.pkl"), "rb") as f:
    P.update(pickle.load(f))

TURNS, OK, KEPT, KM, THR = {}, {}, {}, {}, {}
for a, d in P.items():
    TURNS[a] = [r for r in d["records"] if r.get("kind") == "turn"]
    t = sorted((r for r in TURNS[a] if r.get("ttft") is not None), key=lambda r: r["t_send"])
    OK[a] = t
    THR[a] = t[-1]["t_send"] - 300.0
    KEPT[a] = [r for r in t if r["t_send"] < THR[a]]
    KM[a] = {(r["conv"], r["turn"]): r for r in KEPT[a]}
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


SH = {s: shared(a) for s, a in SETS.items()}


def boot(d_by_key, order_time, B=2000, seed=20261112):
    """Resampling intervals of a paired mean difference.
    d_by_key: key -> difference; order_time: key -> send time (s) used to form time blocks.
    Returns the 5/95 and 2.5/97.5 percentiles for three schemes."""
    rng = random.Random(seed)
    out = {}
    keys = list(d_by_key)

    def run(groups):
        g = [(math.fsum(v), len(v)) for v in groups if v]
        sims = []
        n = len(g)
        for _ in range(B):
            tot = cnt = 0.0
            for _i in range(n):
                s = g[rng.randrange(n)]
                tot += s[0]
                cnt += s[1]
            sims.append(tot / cnt)
        sims.sort()
        return {"n_groups": n, "p05": sims[int(0.05 * B)], "p95": sims[int(0.95 * B) - 1],
                "p025": sims[int(0.025 * B)], "p975": sims[int(0.975 * B) - 1]}
    bys = {}
    for k in keys:
        bys.setdefault(k[0], []).append(d_by_key[k])
    out["session"] = run(list(bys.values()))
    byt = {}
    for k in keys:
        byt.setdefault(int(order_time[k] // 300.0), []).append(d_by_key[k])
    out["time_5min"] = run(list(byt.values()))
    byt = {}
    for k in keys:
        byt.setdefault(int(order_time[k] // 900.0), []).append(d_by_key[k])
    out["time_15min"] = run(list(byt.values()))
    return out


def pair_report(second, first, keys, do_boot=True):
    d = {k: KM[second][k]["ttft"] - KM[first][k]["ttft"] for k in keys}
    vals = [d[k] for k in keys]
    ep = [d[k] for k in keys if in_ep(k)]
    el = [d[k] for k in keys if not in_ep(k)]
    m1 = mean([KM[first][k]["ttft"] for k in keys])
    m2 = mean([KM[second][k]["ttft"] for k in keys])
    r = {"n": len(keys), "mean_first": m1, "mean_second": m2, "diff": mean(vals),
         "rel_diff_to_first": mean(vals) / m1, "diff_episodes": mean(ep), "n_episodes": len(ep),
         "diff_elsewhere": mean(el), "n_elsewhere": len(el),
         "median_diff": pct(vals, 0.5),
         "over10_first": sum(1 for k in keys if KM[first][k]["ttft"] > 10),
         "over10_second": sum(1 for k in keys if KM[second][k]["ttft"] > 10),
         "p99_first": pct([KM[first][k]["ttft"] for k in keys], 0.99),
         "p99_second": pct([KM[second][k]["ttft"] for k in keys], 0.99)}
    # how concentrated the summed difference is, in 5-minute bins of the first arm's clock
    bins = {}
    for k in keys:
        b = int(KM[first][k]["t_send"] // 300.0) * 5
        bins[b] = bins.get(b, 0.0) + d[k]
    tot = math.fsum(vals)
    top = sorted(bins.items(), key=lambda kv: -abs(kv[1]))[:2]
    r["sum_diff_s"] = tot
    r["top_two_5min_bins"] = [[b, v] for b, v in top]
    r["share_of_sum_in_top_two_bins"] = (top[0][1] + top[1][1]) / tot if tot else float("nan")
    if do_boot:
        r["boot"] = boot(d, {k: KM[first][k]["t_send"] for k in keys})
    return r


def show_pair(tag, second, first, keys, r):
    b = r.get("boot")
    print("  %-34s n=%5d  %.4f -> %.4f  diff %+.4f (%+.1f %%)  episodes %+.4f (n=%d)  elsewhere "
          "%+.4f (n=%d)  >10s %d -> %d  p99 %.2f -> %.2f"
          % ("%s: %s - %s" % (tag, second, first), r["n"], r["mean_first"], r["mean_second"],
             r["diff"], 100 * r["rel_diff_to_first"], r["diff_episodes"], r["n_episodes"],
             r["diff_elsewhere"], r["n_elsewhere"], r["over10_first"], r["over10_second"],
             r["p99_first"], r["p99_second"]))
    if b:
        print("        intervals of the paired mean: sessions 5-95 %% [%+.3f, %+.3f]; 5-min time blocks "
              "5-95 %% [%+.3f, %+.3f], 2.5-97.5 %% [%+.3f, %+.3f]; 15-min blocks 5-95 %% [%+.3f, %+.3f]"
              % (b["session"]["p05"], b["session"]["p95"], b["time_5min"]["p05"],
                 b["time_5min"]["p95"], b["time_5min"]["p025"], b["time_5min"]["p975"],
                 b["time_15min"]["p05"], b["time_15min"]["p95"]))
    print("        summed difference %+.1f s; two largest 5-min bins (first arm's minute, s): %s = "
          "%.0f %% of the sum; median paired difference %+.4f"
          % (r["sum_diff_s"], ["%d: %+.1f" % (b_, v) for b_, v in r["top_two_5min_bins"]],
             100 * r["share_of_sum_in_top_two_bins"], r["median_diff"]))


# =============================================================================== S1 repeats
print("=== S1. the same configuration run again (later minus earlier), paired on shared turns")
RES["repeats"] = {}
k13 = shared(SETS["ALL11"] + EXTRA)
assert k13 == SH["ALL11"], "13-arm keys differ from ALL11"
k6 = shared(SETS["LEAD5"] + ["ec_r13_rep"])
jobs = [
    ("ALL11", "base_rep", "baseline", SH["ALL11"]),
    ("ALL11", "hintfree_ctl", "hintfree", SH["ALL11"]),
    ("own", "hintfree_ctl", "hintfree", shared(["hintfree", "hintfree_ctl"])),
    ("ALL11", "jit_p50_ctl", "jit_p50", SH["ALL11"]),
    ("own", "jit_p50_ctl", "jit_p50", shared(["jit_p50", "jit_p50_ctl"])),
    ("ALL11", "jit_p50_d", "jit_p50_ctl", SH["ALL11"]),
    ("ALL11", "jit_p50_d", "jit_p50", SH["ALL11"]),
    ("own", "jit_p50_d", "jit_p50_ctl", shared(["jit_p50_ctl", "jit_p50_d"])),
    ("own", "jit_p50_d", "jit_p50", shared(["jit_p50", "jit_p50_d"])),
    ("own", "jit_p50_d", "jit_p50_d_cut", shared(["jit_p50_d_cut", "jit_p50_d"])),
    ("CUT7", "jit_p50_d_cut", "jit_p50_ctl", SH["CUT7"]),
    ("CUT7", "jit_p50_d_cut", "jit_p50", SH["CUT7"]),
    ("CUT7", "jit_p50_ctl", "jit_p50", SH["CUT7"]),
    ("ALL11", "ec_r13_rep", "ec_r13", SH["ALL11"]),
    ("LEAD5+rep", "ec_r13_rep", "ec_r13", k6),
    ("own", "ec_r13_rep", "ec_r13", shared(["ec_r13", "ec_r13_rep"])),
]
for tag, second, first, keys in jobs:
    r = pair_report(second, first, keys)
    RES["repeats"]["%s:%s-%s" % (tag, second, first)] = r
    show_pair(tag, second, first, keys, r)

print("\n=== S1b. LEAD5 pairs (second minus first) with the same intervals; and the ec_r13 repeat as yardstick")
RES["lead5_pairs"] = {}
L5 = SETS["LEAD5"]
for i in range(5):
    for j in range(i + 1, 5):
        r = pair_report(L5[j], L5[i], SH["LEAD5"])
        RES["lead5_pairs"]["%s-%s" % (L5[j], L5[i])] = r
        show_pair("LEAD5", L5[j], L5[i], SH["LEAD5"], r)
rep = RES["repeats"]["LEAD5+rep:ec_r13_rep-ec_r13"]["diff"]
print("  yardstick: ec_r13_rep - ec_r13 on the %d turns shared with LEAD5 = %+.4f s" % (len(k6), rep))
n_below = sum(1 for r in RES["lead5_pairs"].values() if abs(r["diff"]) <= abs(rep))
print("  LEAD5 pairs whose |difference| is not larger than that: %d of 10" % n_below)
RES["lead5_pairs_not_larger_than_ec_r13_repeat"] = n_below
print("  means on the six-arm set (LEAD5 + ec_r13_rep, %d turns):" % len(k6))
RES["six_arm_means"] = {}
for a in L5 + ["ec_r13_rep"]:
    t = [KM[a][k]["ttft"] for k in k6]
    RES["six_arm_means"][a] = {"mean": mean(t), "over10": sum(1 for x in t if x > 10),
                               "p99": pct(t, 0.99),
                               "mean_ep": mean([KM[a][k]["ttft"] for k in k6 if in_ep(k)]),
                               "mean_el": mean([KM[a][k]["ttft"] for k in k6 if not in_ep(k)])}
    print("     %-11s mean %.4f  episodes %.4f  elsewhere %.4f  p99 %.2f  over 10 s %d"
          % (a, mean(t), RES["six_arm_means"][a]["mean_ep"], RES["six_arm_means"][a]["mean_el"],
             pct(t, 0.99), RES["six_arm_means"][a]["over10"]))

# =============================================================================== S2 item-3 sensitivity
print("\n=== S2. item 3 on the same 6715 turns with the two later arms, and how far a repeat moves the share")
RES["gain_share_13"] = {}
m = {a: mean([KM[a][k]["ttft"] for k in SH["ALL11"]]) for a in SETS["ALL11"] + EXTRA}
for a in SETS["ALL11"] + EXTRA:
    g1 = (m["baseline"] - m[a]) / (m["baseline"] - m["jit_exact"])
    g2 = (m["base_rep"] - m[a]) / (m["base_rep"] - m["jit_exact"])
    RES["gain_share_13"][a] = {"mean": m[a], "vs_baseline": g1, "vs_base_rep": g2}
    print("  %-13s mean %.4f  share vs baseline %.4f  vs base_rep %.4f" % (a, m[a], g1, g2))
lead4 = ["ec_r13", "ec_r1", "nofc", "nofc_noend"]
sp = [RES["gain_share_13"][a]["vs_baseline"] for a in lead4]
print("  spread of the four campaign-26 leading arms (vs baseline): %.4f to %.4f = %.4f"
      % (min(sp), max(sp), max(sp) - min(sp)))
print("  ec_r13 against its repeat: %.4f against %.4f = %.4f"
      % (RES["gain_share_13"]["ec_r13"]["vs_baseline"], RES["gain_share_13"]["ec_r13_rep"]["vs_baseline"],
         RES["gain_share_13"]["ec_r13"]["vs_baseline"] - RES["gain_share_13"]["ec_r13_rep"]["vs_baseline"]))
print("  jit_p50 three runs (jit_p50, jit_p50_ctl, jit_p50_d): %.4f %.4f %.4f"
      % tuple(RES["gain_share_13"][a]["vs_baseline"] for a in ("jit_p50", "jit_p50_ctl", "jit_p50_d")))
print("  hint-free two runs: %.4f %.4f" % tuple(RES["gain_share_13"][a]["vs_baseline"]
                                               for a in ("hintfree", "hintfree_ctl")))
RES["gain_share_spread_lead4_vs_baseline"] = max(sp) - min(sp)

# =============================================================================== S3 coverage
print("\n=== S3. which of an arm's kept returning turns a shared set holds, and the mean TTFT inside and outside")
RES["coverage"] = {}
for s, arms in SETS.items():
    keys = set(SH[s])
    maxpos = max((POS[k] for k in keys if k in POS), default=None)
    n1 = sum(1 for k in keys if k in POS and 5500 <= POS[k] <= 6999)
    n2 = sum(1 for k in keys if k in POS and 8750 <= POS[k] <= 10999)
    nabs = sum(1 for k in keys if k not in POS)
    print("-- %s (%d shared): episode 1 turns %d, episode 2 turns %d, no order position %d, "
          "largest order position %s" % (s, len(keys), n1, n2, nabs, maxpos))
    RES["coverage"][s] = {"n": len(keys), "episode1": n1, "episode2": n2, "absent": nabs,
                          "max_position": maxpos, "arms": {}}
    for a in arms:
        ret = [r for r in KEPT[a] if r["turn"] > 0]
        ins = [r["ttft"] for r in ret if (r["conv"], r["turn"]) in keys]
        out = [r["ttft"] for r in ret if (r["conv"], r["turn"]) not in keys]
        last_in = max(r["t_send"] for r in ret if (r["conv"], r["turn"]) in keys) / 60.0
        RES["coverage"][s]["arms"][a] = {
            "kept_returning": len(ret), "share_inside": len(ins) / len(ret),
            "mean_all": mean([r["ttft"] for r in ret]), "mean_inside": mean(ins),
            "mean_outside": mean(out), "n_outside": len(out),
            "over10_outside": sum(1 for x in out if x > 10), "last_shared_send_min": last_in}
        print("   %-14s kept returning %5d  inside %.3f  mean all %.4f  inside %.4f  outside %s "
              "(n=%d, >10 s %d)  last shared send at minute %.1f"
              % (a, len(ret), len(ins) / len(ret), mean([r["ttft"] for r in ret]), mean(ins),
                 ("%.4f" % mean(out)) if out else "-", len(out),
                 sum(1 for x in out if x > 10), last_in))
print("   order run: returning turns at positions 5500-6999: %d; at 8750-10999: %d; its last position %d"
      % (sum(1 for k, p in POS.items() if k[1] > 0 and 5500 <= p <= 6999),
         sum(1 for k, p in POS.items() if k[1] > 0 and 8750 <= p <= 10999), len(POS) - 1))

# =============================================================================== S4 same key, other time
print("\n=== S4. the same key is sent at a different time in each arm: t_send(arm) - t_send(reference), shared turns")
RES["send_time_shift"] = {}
for s, ref in (("ALL11", "jit_exact"), ("SEVEN", "jit_exact"), ("LEAD5", "jit_exact"),
               ("CUT7", "jit_p50_ctl")):
    for a in SETS[s]:
        if a == ref:
            continue
        d = [KM[a][k]["t_send"] - KM[ref][k]["t_send"] for k in SH[s]]
        ad = [abs(x) for x in d]
        r = {"mean": mean(d), "median": pct(d, 0.5), "p90_abs": pct(ad, 0.9), "max_abs": max(ad),
             "share_over_60s": sum(1 for x in ad if x > 60) / len(ad),
             "share_over_600s": sum(1 for x in ad if x > 600) / len(ad)}
        RES["send_time_shift"]["%s:%s-%s" % (s, a, ref)] = r
        print("  %-6s %-13s - %-11s mean %+8.1f s  median %+8.1f  p90|d| %7.1f  max|d| %7.1f  "
              "share |d|>60 s %.3f  >600 s %.3f"
              % (s, a, ref, r["mean"], r["median"], r["p90_abs"], r["max_abs"],
                 r["share_over_60s"], r["share_over_600s"]))
print("  closed loop: turn records per minute of the run and sessions started")
RES["closed_loop"] = {}
for a in ARMS12 + EXTRA + ["ORDER"]:
    dur = OK[a][-1]["t_send"] / 60.0
    ncs = P[a]["kinds"].get("conv_start", 0)
    RES["closed_loop"][a] = {"turn_records": len(TURNS[a]), "last_send_min": dur,
                             "turns_per_min": len(TURNS[a]) / dur, "sessions_started": ncs}
    print("     %-14s turns %5d over %.1f min = %.2f per min; sessions started %d"
          % (a, len(TURNS[a]), dur, len(TURNS[a]) / dur, ncs))

# =============================================================================== S5 absent keys
print("\n=== S5. LEAD5 shared turns with no position in the order run (counted as 'elsewhere')")
keys = SH["LEAD5"]
absent = [k for k in keys if k not in POS]
elsepos = [k for k in keys if k in POS and not in_ep(k)]
e1 = [k for k in keys if k in POS and 5500 <= POS[k] <= 6999]
e2 = [k for k in keys if k in POS and 8750 <= POS[k] <= 10999]
RES["lead5_absent"] = {"absent": len(absent), "elsewhere_with_position": len(elsepos),
                       "episode1": len(e1), "episode2": len(e2), "arms": {}, "pairs": {}}
print("  absent %d, elsewhere with a position %d, episode 1 %d, episode 2 %d"
      % (len(absent), len(elsepos), len(e1), len(e2)))
for a in L5:
    r = {"mean_absent": mean([KM[a][k]["ttft"] for k in absent]),
         "mean_elsewhere_with_position": mean([KM[a][k]["ttft"] for k in elsepos]),
         "mean_episode1": mean([KM[a][k]["ttft"] for k in e1]),
         "mean_episode2": mean([KM[a][k]["ttft"] for k in e2]),
         "over10_episode1": sum(1 for k in e1 if KM[a][k]["ttft"] > 10),
         "over10_episode2": sum(1 for k in e2 if KM[a][k]["ttft"] > 10),
         "absent_first_send_min": min(KM[a][k]["t_send"] for k in absent) / 60.0,
         "absent_last_send_min": max(KM[a][k]["t_send"] for k in absent) / 60.0,
         "episode2_first_send_min": min(KM[a][k]["t_send"] for k in e2) / 60.0,
         "episode1_first_send_min": min(KM[a][k]["t_send"] for k in e1) / 60.0,
         "episode1_last_send_min": max(KM[a][k]["t_send"] for k in e1) / 60.0}
    RES["lead5_absent"]["arms"][a] = r
    print("   %-11s mean TTFT: absent %.4f | elsewhere with position %.4f | episode 1 %.4f (>10 s %d) "
          "| episode 2 %.4f (>10 s %d); absent keys sent minute %.1f-%.1f; episode 1 sent %.1f-%.1f; "
          "episode 2 from %.1f"
          % (a, r["mean_absent"], r["mean_elsewhere_with_position"], r["mean_episode1"],
             r["over10_episode1"], r["mean_episode2"], r["over10_episode2"],
             r["absent_first_send_min"], r["absent_last_send_min"], r["episode1_first_send_min"],
             r["episode1_last_send_min"], r["episode2_first_send_min"]))
for i in range(5):
    for j in range(i + 1, 5):
        a, b = L5[i], L5[j]
        r = {"absent": mean([KM[b][k]["ttft"] - KM[a][k]["ttft"] for k in absent]),
             "elsewhere_with_position": mean([KM[b][k]["ttft"] - KM[a][k]["ttft"] for k in elsepos])}
        RES["lead5_absent"]["pairs"]["%s-%s" % (b, a)] = r
        print("   %-24s paired: absent %+.4f | elsewhere with position %+.4f"
              % ("%s-%s" % (b, a), r["absent"], r["elsewhere_with_position"]))
t_ord = [r["t_send"] for r in OK["ORDER"]]
print("  order run by its own clock: position 5500 at minute %.1f, 6999 at %.1f, 8750 at %.1f, last (%d) at %.1f"
      % (t_ord[5500] / 60, t_ord[6999] / 60, t_ord[8750] / 60, len(t_ord) - 1, t_ord[-1] / 60))

# =============================================================================== S6 tails
print("\n=== S6. how few turns carry the means (shared turns of each set)")
RES["tails"] = {}
for s, arms in SETS.items():
    keys = SH[s]
    print("-- %s (n=%d): mean | share of summed TTFT in turns over 10 s | in the slowest 1 %% | mean "
          "of the turns at or under 10 s | turns over 10 s, in how many bursts (60 s apart), largest "
          "burst, its minutes" % (s, len(keys)))
    for a in arms:
        v = [KM[a][k]["ttft"] for k in keys]
        tot = math.fsum(v)
        srt = sorted(v, reverse=True)
        over = [x for x in v if x > 10]
        slow = sorted((KM[a][k]["t_send"], KM[a][k]["ttft"]) for k in keys if KM[a][k]["ttft"] > 10)
        bursts = []
        for ts, x in slow:
            if bursts and ts - bursts[-1][-1][0] <= 60.0:
                bursts[-1].append((ts, x))
            else:
                bursts.append([(ts, x)])
        big = max(bursts, key=len) if bursts else []
        r = {"mean": tot / len(v), "share_sum_over10": math.fsum(over) / tot,
             "share_sum_top1pct": math.fsum(srt[:len(srt) // 100]) / tot,
             "mean_le10": mean([x for x in v if x <= 10]), "over10": len(over),
             "bursts": len(bursts), "largest_burst_turns": len(big),
             "largest_burst_from_min": big[0][0] / 60.0 if big else None,
             "largest_burst_to_min": big[-1][0] / 60.0 if big else None,
             "largest_burst_sum_s": math.fsum(x for _, x in big),
             "largest_burst_share_of_sum": math.fsum(x for _, x in big) / tot}
        RES["tails"]["%s:%s" % (s, a)] = r
        print("   %-14s %.4f | %.3f | %.3f | %.4f | %4d in %2d bursts, largest %3d turns (minute "
              "%s-%s, %.0f s = %.1f %% of the arm's summed TTFT)"
              % (a, r["mean"], r["share_sum_over10"], r["share_sum_top1pct"], r["mean_le10"],
                 r["over10"], r["bursts"], r["largest_burst_turns"],
                 "%.1f" % r["largest_burst_from_min"] if big else "-",
                 "%.1f" % r["largest_burst_to_min"] if big else "-", r["largest_burst_sum_s"],
                 100 * r["largest_burst_share_of_sum"]))
print("  LEAD5 order by mean: %s" % sorted(L5, key=lambda a: RES["tails"]["LEAD5:" + a]["mean"]))
print("  LEAD5 order by mean of the turns at or under 10 s: %s"
      % sorted(L5, key=lambda a: RES["tails"]["LEAD5:" + a]["mean_le10"]))
# the same means with one window of each arm's own clock left out
keys = SH["LEAD5"]
print("  LEAD5 means when the keys that ANY of the five arms sent in its own minutes 85-92 are left out:")
drop = set(k for k in keys if any(85 * 60 <= KM[a][k]["t_send"] < 92 * 60 for a in L5))
keep = [k for k in keys if k not in drop]
RES["lead5_without_min85_92"] = {"dropped": len(drop), "kept": len(keep), "arms": {}}
for a in L5:
    RES["lead5_without_min85_92"]["arms"][a] = mean([KM[a][k]["ttft"] for k in keep])
print("     dropped %d keys, kept %d: %s" % (len(drop), len(keep), "  ".join(
    "%s %.4f" % (a, RES["lead5_without_min85_92"]["arms"][a]) for a in L5)))

# =============================================================================== S7 first turns
print("\n=== S7. first turns (left out of every comparison): TTFT on the sessions whose first turn is kept in every arm")
RES["first_turns"] = {}
for s, arms in SETS.items():
    c = None
    for a in arms:
        ks = set(k for k in KM[a] if k[1] == 0)
        c = ks if c is None else c & ks
    c = sorted(c)
    for a in arms:
        v = [KM[a][k]["ttft"] for k in c]
        RES["first_turns"]["%s:%s" % (s, a)] = {"n": len(c), "mean": mean(v), "median": pct(v, 0.5),
                                               "p90": pct(v, 0.9), "over10": sum(1 for x in v if x > 10)}
    print("-- %s (%d sessions): %s" % (s, len(c), "  ".join(
        "%s %.3f" % (a, RES["first_turns"]["%s:%s" % (s, a)]["mean"]) for a in arms)))
allk = None
for a in L5:
    ks = set(KM[a])
    allk = ks if allk is None else allk & ks
RES["lead5_with_first_turns"] = {"n": len(allk),
                                 "arms": {a: mean([KM[a][k]["ttft"] for k in allk]) for a in L5}}
print("  LEAD5 with first turns included (%d keys): %s" % (len(allk), "  ".join(
    "%s %.4f" % (a, RES["lead5_with_first_turns"]["arms"][a]) for a in L5)))

# =============================================================================== S8 pace
print("\n=== S8. client-side speed control: ms per output token after the first, (latency - ttft)/(completion_tokens - 1)")
RES["pace"] = {}
print("   run            start (UTC)        min 0-10: n  median   mean | min 10-25: n  median | "
      "first turns min 0-10: n, summed TTFT s | first request of the run (conv 0 turn 0) TTFT s")
import time as _time
for a in ARMS12 + EXTRA + ["ORDER"]:
    run = [r for r in P[a]["records"] if r.get("kind") == "run"][0]

    def paces(lo, hi):
        return [1000.0 * (r["latency"] - r["ttft"]) / (r["completion_tokens"] - 1) for r in OK[a]
                if lo <= r["t_send"] < hi and (r.get("completion_tokens") or 0) > 1]
    p1, p2 = paces(0, 600), paces(600, 1500)
    f0 = [r for r in OK[a] if r["turn"] == 0 and r["t_send"] < 600]
    c0 = [r for r in OK[a] if r["conv"] == 0 and r["turn"] == 0]
    RES["pace"][a] = {"t_start": run["t_start"], "n_0_10": len(p1), "median_0_10": pct(p1, 0.5),
                      "mean_0_10": mean(p1), "n_10_25": len(p2), "median_10_25": pct(p2, 0.5),
                      "first_turns_0_10": len(f0), "first_turns_0_10_sum_ttft": math.fsum(r["ttft"] for r in f0),
                      "conv0_turn0_ttft": c0[0]["ttft"] if c0 else None}
    print("   %-14s %s   %4d  %6.3f  %6.3f | %4d  %6.3f | %3d  %6.2f | %.3f"
          % (a, _time.strftime("%m-%d %H:%M", _time.gmtime(run["t_start"])), len(p1), pct(p1, 0.5),
             mean(p1), len(p2), pct(p2, 0.5), len(f0), math.fsum(r["ttft"] for r in f0),
             c0[0]["ttft"] if c0 else float("nan")))
med = [RES["pace"][a]["median_0_10"] for a in ARMS12 + EXTRA]
print("   median pace in minutes 0-10 over the 14 arms: %.3f to %.3f ms (%.1f %% apart)"
      % (min(med), max(med), 100 * (max(med) / min(med) - 1)))
RES["pace_spread_0_10"] = {"min": min(med), "max": max(med)}

# =============================================================================== S9 censoring
print("\n=== S9. turns in flight when a run stopped: sessions whose next turn was due before the 300 s cut but has no record")
gap = {}
for a in P:
    for r in TURNS[a]:
        gap[(r["conv"], r["turn"])] = r.get("gap_slept")
RES["censoring"] = {}
for a in ARMS12 + EXTRA + ["ORDER"]:
    last = {}
    for r in TURNS[a]:
        if r["conv"] not in last or r["turn"] > last[r["conv"]]["turn"]:
            last[r["conv"]] = r
    ended = set(r["conv"] for r in P[a]["records"] if r.get("kind") == "conv_end")
    open_ = [c for c in last if c not in ended]
    started = set(r["conv"] for r in P[a]["records"] if r.get("kind") == "conv_start")
    no_turn = [c for c in started if c not in last]
    due_missing = []
    unknown = 0
    for c in open_:
        r = last[c]
        g = gap.get((c, r["turn"] + 1))
        if g is None:
            unknown += 1
            continue
        due = r["t_send"] + r["latency"] + g
        if due < THR[a]:
            due_missing.append((c, r["turn"] + 1, round(due, 1)))
    RES["censoring"][a] = {"sessions_open_at_end": len(open_), "started_without_any_turn": len(no_turn),
                           "next_gap_unknown": unknown, "due_before_cut_without_record": len(due_missing),
                           "max_ttft": max(r["ttft"] for r in OK[a]),
                           "max_latency": max(r["latency"] for r in OK[a]),
                           "kept_turns_with_latency_over_300s": sum(1 for r in KEPT[a] if r["latency"] > 300)}
    print("   %-14s open at end %2d (no turn at all %d, next gap unknown %d); due before the cut with no "
          "record %d; max TTFT %.1f s, max latency %.0f s, kept turns with latency over 300 s: %d"
          % (a, len(open_), len(no_turn), unknown, len(due_missing), RES["censoring"][a]["max_ttft"],
             RES["censoring"][a]["max_latency"], RES["censoring"][a]["kept_turns_with_latency_over_300s"]))

# =============================================================================== S10 client args
print("\n=== S10. client arguments per run (the 'run' record): fields that are not the same in every run")
allargs = {}
for a in ARMS12 + EXTRA + ["ORDER"]:
    run = [r for r in P[a]["records"] if r.get("kind") == "run"][0]
    for k, v in (run.get("args") or {}).items():
        if k in ("output", "tag"):
            continue
        allargs.setdefault(k, {})[a] = v
    for k in ("model", "users", "turns"):
        allargs.setdefault("run." + k, {})[a] = run.get(k)
RES["client_args_differing"] = {}
for k, d in sorted(allargs.items()):
    vals = set(json.dumps(d.get(a, "<absent>"), sort_keys=True) for a in ARMS12 + EXTRA + ["ORDER"])
    if len(vals) > 1:
        RES["client_args_differing"][k] = {a: d.get(a, "<absent>") for a in ARMS12 + EXTRA + ["ORDER"]}
        print("   %-18s %s" % (k, "  ".join("%s=%s" % (a, d.get(a, "<absent>")) for a in ARMS12 + EXTRA + ["ORDER"])))
same = [k for k, d in sorted(allargs.items()) if k not in RES["client_args_differing"]]
print("   identical in all 15 runs: %s" % ", ".join("%s=%s" % (k, allargs[k]["baseline"]) for k in same))

# =============================================================================== S11 prompt drift
print("\n=== S11. prompt_tokens against the trace value (replay_prompt_tokens), per run")
RES["prompt_vs_trace"] = {}
for a in ARMS12 + EXTRA + ["ORDER"]:
    d = [(r["prompt_tokens"] - r["replay_prompt_tokens"], r["conv"], r["turn"]) for r in TURNS[a]
         if r.get("prompt_tokens") is not None and r.get("replay_prompt_tokens") is not None]
    n0 = sum(1 for x in d if x[0] == 0)
    n8 = sum(1 for x in d if 0 < abs(x[0]) <= 8)
    n64 = sum(1 for x in d if 8 < abs(x[0]) <= 64)
    nbig = sum(1 for x in d if abs(x[0]) > 64)
    worst = min(d)
    RES["prompt_vs_trace"][a] = {"n": len(d), "equal": n0, "abs_1_8": n8, "abs_9_64": n64,
                                 "abs_over_64": nbig, "worst": list(worst),
                                 "convs_with_abs_over_64": sorted(set(x[1] for x in d if abs(x[0]) > 64))}
    print("   %-14s n %5d  equal %5d  |d| 1-8 %4d  9-64 %3d  over 64 %2d (convs %s)  worst %s"
          % (a, len(d), n0, n8, n64, nbig, RES["prompt_vs_trace"][a]["convs_with_abs_over_64"], worst))

with open(os.path.join(HERE, "r2_item5.json"), "w") as f:
    json.dump(RES, f, indent=1, sort_keys=True)
print("\nwrote r2_item5.json")
