#!/usr/bin/env python3
"""Item 5, part b: what the shared-turn restriction and the episode split leave out or mix up."""
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

print("=== 1. each arm: all its kept returning turns against the part inside each shared set")
RES["coverage"] = {}
for sname, arms in ARMSETS.items():
    sh = set(SH[sname])
    print("-- %s (shared %d)" % (sname, len(sh)))
    print("   %-14s %8s %8s %9s %9s %9s %8s %9s %10s %10s" % (
        "arm", "kept_ret", "covered", "mean_all", "mean_shr", "mean_rest", "n_rest", ">10s_rest", "last_shr_min", "p50_shr_min"))
    for a in arms:
        own = [k for k in KM[a] if k[1] > 0]
        rest = [k for k in own if k not in sh]
        m_all = mean([KM[a][k]["ttft"] for k in own])
        m_sh = mean([KM[a][k]["ttft"] for k in sh])
        m_rest = mean([KM[a][k]["ttft"] for k in rest]) if rest else float("nan")
        o_rest = sum(1 for k in rest if KM[a][k]["ttft"] > 10)
        sends = sorted(KM[a][k]["t_send"] for k in sh)
        r = {"kept_returning": len(own), "covered_share": len(sh) / len(own), "mean_all_kept_returning": m_all,
             "mean_shared": m_sh, "mean_unshared": m_rest, "n_unshared": len(rest), "over10_unshared": o_rest,
             "last_shared_send_min": sends[-1] / 60, "median_shared_send_min": pct(sends, 0.5) / 60}
        RES["coverage"]["%s:%s" % (sname, a)] = r
        print("   %-14s %8d %8.3f %9.4f %9.4f %9.4f %8d %9d %10.1f %10.1f" % (
            a, len(own), len(sh) / len(own), m_all, m_sh, m_rest, len(rest), o_rest, sends[-1] / 60, pct(sends, 0.5) / 60))

print()
print("=== 2. when is the same key sent?  t_send(arm) - t_send(reference) on shared turns, seconds")
RES["alignment"] = {}
for sname, ref in (("ALL11", "jit_exact"), ("LEAD5", "jit_exact"), ("SEVEN", "jit_exact"), ("CUT7", "jit_p50_ctl")):
    print("-- %s, reference %s" % (sname, ref))
    for a in ARMSETS[sname]:
        if a == ref:
            continue
        d = [KM[a][k]["t_send"] - KM[ref][k]["t_send"] for k in SH[sname]]
        ad = [abs(x) for x in d]
        r = {"mean": mean(d), "p50": pct(d, 0.5), "p90_abs": pct(ad, 0.9), "max_abs": max(ad),
             "share_abs_gt_60s": sum(1 for x in ad if x > 60) / len(ad),
             "share_abs_gt_600s": sum(1 for x in ad if x > 600) / len(ad)}
        RES["alignment"]["%s:%s-%s" % (sname, a, ref)] = r
        print("   %-14s mean %+9.1f  median %+9.1f  p90|d| %8.1f  max|d| %8.1f  share |d|>60s %.3f  >600s %.3f" % (
            a, r["mean"], r["p50"], r["p90_abs"], r["max_abs"], r["share_abs_gt_60s"], r["share_abs_gt_600s"]))

print()
print("=== 3. LEAD5 shared keys absent from the order run (counted as 'elsewhere' by the definition)")
keys = SH["LEAD5"]
miss = [k for k in keys if k not in POS]
els_in = [k for k in keys if k in POS and epi_no(k) == 0]
ep1 = [k for k in keys if epi_no(k) == 1]
ep2 = [k for k in keys if epi_no(k) == 2]
print("   counts: absent %d, elsewhere-with-position %d, episode1 %d, episode2 %d (order run has %d turns, last position %d)"
      % (len(miss), len(els_in), len(ep1), len(ep2), len(ORDER_SORTED), len(ORDER_SORTED) - 1))
RES["lead5_absent"] = {"absent": len(miss), "elsewhere_with_position": len(els_in), "episode1": len(ep1), "episode2": len(ep2)}
print("   %-12s %10s %10s %10s %10s %10s %9s %9s %12s %12s" % (
    "arm", "absent", "else_pos", "episode1", "episode2", "else_def", ">10s_abs", ">10s_else", "abs_send_min", "abs_send_max"))
for a in LEAD5:
    f = lambda ks: mean([KM[a][k]["ttft"] for k in ks])
    o_abs = sum(1 for k in miss if KM[a][k]["ttft"] > 10)
    o_els = sum(1 for k in els_in if KM[a][k]["ttft"] > 10)
    s = [KM[a][k]["t_send"] for k in miss]
    r = {"mean_absent": f(miss), "mean_elsewhere_with_position": f(els_in), "mean_episode1": f(ep1),
         "mean_episode2": f(ep2), "mean_elsewhere_as_defined": f(miss + els_in), "over10_absent": o_abs,
         "over10_elsewhere_with_position": o_els, "absent_send_min_first": min(s) / 60, "absent_send_min_last": max(s) / 60,
         "absent_send_min_p10": pct(s, 0.1) / 60}
    RES["lead5_absent"][a] = r
    print("   %-12s %10.4f %10.4f %10.4f %10.4f %10.4f %9d %9d %12.1f %12.1f" % (
        a, f(miss), f(els_in), f(ep1), f(ep2), f(miss + els_in), o_abs, o_els, min(s) / 60, max(s) / 60))
print("   paired differences (second minus first) on the three parts: absent / elsewhere-with-position / episodes")
for i in range(len(LEAD5)):
    for j in range(i + 1, len(LEAD5)):
        a, b = LEAD5[i], LEAD5[j]
        g = lambda ks: mean([KM[b][k]["ttft"] - KM[a][k]["ttft"] for k in ks])
        RES["lead5_absent"]["%s-%s" % (b, a)] = {"absent": g(miss), "elsewhere_with_position": g(els_in), "episodes": g(ep1 + ep2)}
        print("     %-24s %+9.4f %+9.4f %+9.4f" % ("%s-%s" % (b, a), g(miss), g(els_in), g(ep1 + ep2)))
print("   positions of the order run by its own clock: last turn sent at %.1f min; position 8750 at %.1f min; 10318 at %.1f min"
      % (ORDER_SORTED[-1]["t_send"] / 60, ORDER_SORTED[8750]["t_send"] / 60, ORDER_SORTED[-1]["t_send"] / 60))

print()
print("=== 4. LEAD5 shared turns in 5-minute bins of jit_exact's send time (same keys in every column)")
bins = {}
for k in keys:
    b = int(KM["jit_exact"][k]["t_send"] // 300)
    bins.setdefault(b, []).append(k)
print("   %5s %6s %6s %6s %6s " % ("min", "n", "epi1", "epi2", "absent") + " ".join("%11s" % a for a in LEAD5))
RES["lead5_bins"] = {}
for b in sorted(bins):
    ks = bins[b]
    row = [mean([KM[a][k]["ttft"] for k in ks]) for a in LEAD5]
    RES["lead5_bins"][str(b * 5)] = {"n": len(ks), "means": dict(zip(LEAD5, row)),
                                     "epi1": sum(1 for k in ks if epi_no(k) == 1),
                                     "epi2": sum(1 for k in ks if epi_no(k) == 2),
                                     "absent": sum(1 for k in ks if k not in POS)}
    print("   %5d %6d %6d %6d %6d " % (b * 5, len(ks), sum(1 for k in ks if epi_no(k) == 1),
                                      sum(1 for k in ks if epi_no(k) == 2), sum(1 for k in ks if k not in POS))
          + " ".join("%11.3f" % x for x in row))
# share of each LEAD5 pair's summed difference that comes from the two worst bins
print("   share of each pair's summed difference carried by 5-minute bins 60 and 85-90 (jit_exact clock):")
for i in range(len(LEAD5)):
    for j in range(i + 1, len(LEAD5)):
        a, b = LEAD5[i], LEAD5[j]
        tot = math.fsum(KM[b][k]["ttft"] - KM[a][k]["ttft"] for k in keys)
        part = {}
        for nm, bl in (("60-65", [12]), ("85-95", [17, 18])):
            part[nm] = math.fsum(KM[b][k]["ttft"] - KM[a][k]["ttft"] for bb in bl for k in bins.get(bb, []))
        rest = tot - sum(part.values())
        RES.setdefault("lead5_bin_share", {})["%s-%s" % (b, a)] = {"total_s": tot, "bins": part, "rest_s": rest}
        print("     %-24s total %+8.1f s   min60-65 %+8.1f   min85-95 %+8.1f   all other bins %+8.1f" % (
            "%s-%s" % (b, a), tot, part["60-65"], part["85-95"], rest))

print()
print("=== 5. first turns (turn 0), which the comparison leaves out: TTFT on the convs whose first turn is kept in every arm")
RES["first_turns"] = {}
for sname, arms in ARMSETS.items():
    convs = sorted(set.intersection(*[{k[0] for k in KM[a] if k[1] == 0} for a in arms]))
    print("-- %s (%d convs)" % (sname, len(convs)))
    for a in arms:
        v = [KM[a][(c, 0)]["ttft"] for c in convs]
        r = {"n": len(v), "mean": mean(v), "median": pct(v, 0.5), "p90": pct(v, 0.9), "max": max(v),
             "over10": sum(1 for x in v if x > 10)}
        RES["first_turns"]["%s:%s" % (sname, a)] = r
        print("   %-14s mean %8.3f  median %7.3f  p90 %8.3f  max %8.2f  >10s %4d" % (
            a, r["mean"], r["median"], r["p90"], r["max"], r["over10"]))

print()
print("=== 6. closed loop: sessions started, wait for a slot before the first turn, turns done")
RES["closed_loop"] = {}
print("   %-14s %7s %9s %9s %9s %9s %8s %10s %9s" % (
    "run", "started", "q_mean_s", "q_p50_s", "q_max_s", "q>60s", "turns", "turns/min", "first_q>1s_min"))
for a, d in P.items():
    q = [r.get("queued_s") or 0.0 for r in d["conv_start"]]
    starts = sorted((r["start_s"], r.get("queued_s") or 0.0) for r in d["conv_start"])
    firstq = next((s for s, w in starts if w > 1.0), None)
    v = with_ttft_sorted(d["turns"])
    dur = v[-1]["t_send"] / 60
    r = {"sessions_started": len(q), "queued_mean_s": mean(q), "queued_median_s": pct(q, 0.5), "queued_max_s": max(q),
         "sessions_queued_over_60s": sum(1 for x in q if x > 60), "turn_records": len(v), "turns_per_min": len(v) / dur,
         "first_session_queued_over_1s_at_min": (firstq / 60) if firstq is not None else None}
    RES["closed_loop"][a] = r
    print("   %-14s %7d %9.1f %9.1f %9.1f %9d %8d %10.2f %9s" % (
        a, len(q), r["queued_mean_s"], r["queued_median_s"], r["queued_max_s"], r["sessions_queued_over_60s"],
        len(v), r["turns_per_min"], ("%.1f" % (firstq / 60)) if firstq is not None else "-"))
# the 60-second 'users' samples: how often the cap of 44 live sessions binds, and how many wait
print("   'users' samples (one per minute): share of samples with live == cap, mean waiting, max waiting")
for a, d in P.items():
    u = d["other"].get("users", [])
    if not u:
        continue
    capd = sum(1 for r in u if r["live"] >= r["cap"])
    r = {"samples": len(u), "share_at_cap": capd / len(u), "mean_waiting": mean([r["waiting"] for r in u]),
         "max_waiting": max(r["waiting"] for r in u), "mean_in_flight": mean([r["in_flight"] for r in u])}
    RES["closed_loop"][a].update({"users_" + k: v for k, v in r.items()})
    print("   %-14s samples %3d  at cap %.3f  mean waiting %6.1f  max waiting %4d  mean in_flight %5.1f" % (
        a, r["samples"], r["share_at_cap"], r["mean_waiting"], r["max_waiting"], r["mean_in_flight"]))

print()
print("=== 7. CUT7 shared turns by episode, and by 5-minute bin of jit_p50_ctl's send time")
keys = SH["CUT7"]
c = {0: 0, 1: 0, 2: 0, None: 0}
for k in keys:
    c[epi_no(k)] += 1
print("   elsewhere %d, episode1 (5500-6999) %d, episode2 (8750-10999) %d, absent %d; largest position %d"
      % (c[0], c[1], c[2], c[None], max(POS[k] for k in keys)))
RES["cut7"] = {"elsewhere": c[0], "episode1": c[1], "episode2": c[2], "absent": c[None],
               "max_position": max(POS[k] for k in keys)}
bins = {}
for k in keys:
    bins.setdefault(int(KM["jit_p50_ctl"][k]["t_send"] // 300), []).append(k)
trio = ["jit_p50", "jit_p50_ctl", "jit_p50_d_cut"]
print("   %5s %6s %6s " % ("min", "n", "epi1") + " ".join("%14s" % a for a in trio))
RES["cut7_bins"] = {}
for b in sorted(bins):
    ks = bins[b]
    row = [mean([KM[a][k]["ttft"] for k in ks]) for a in trio]
    RES["cut7_bins"][str(b * 5)] = {"n": len(ks), "means": dict(zip(trio, row))}
    print("   %5d %6d %6d " % (b * 5, len(ks), sum(1 for k in ks if epi_no(k) == 1)) + " ".join("%14.3f" % x for x in row))

with open(os.path.join(HERE, "item5b_shared_bias.json"), "w") as f:
    json.dump(RES, f, indent=1, sort_keys=True)
print()
print("wrote item5b_shared_bias.json")
