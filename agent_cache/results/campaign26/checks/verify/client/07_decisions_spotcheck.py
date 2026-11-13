#!/usr/bin/env python3
"""Spot-check of client-side figures quoted in DECISIONS.md on shared sets other than the task's four."""
import json
import math
import os
import pickle

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "parsed.pkl"), "rb") as f:
    P = pickle.load(f)
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


def shared_of(arms):
    return sorted(set.intersection(*[{k for k in KM[a] if k[1] > 0} for a in arms]))


def table(name, arms, quoted):
    keys = shared_of(arms)
    print("-- %s: %d shared returning turns (%s)" % (name, len(keys), ", ".join(arms)))
    RES[name] = {"shared": len(keys)}
    for a in arms:
        v = [KM[a][k]["ttft"] for k in keys]
        r = {"mean": mean(v), "p90": pct(v, 0.9), "p99": pct(v, 0.99), "over10": sum(1 for x in v if x > 10)}
        RES[name][a] = r
        print("   %-13s mean %.4f  p90 %.4f  p99 %.4f  >10s %d" % (a, r["mean"], r["p90"], r["p99"], r["over10"]))
    print("   DECISIONS.md quotes: %s" % quoted)
    return keys


k = table("set_9660", ["jit_p50", "jit_p50_ctl", "hintfree", "jit_exact"],
          "9,660 turns: jit_p50_ctl 1.46 against jit_p50 1.60 (hint-free 1.75, JIT exact 0.72)")
e1 = [x for x in k if x in POS and 5500 <= POS[x] < 7000]
print("   positions 5500-7000 (%d turns): jit_p50_ctl %.4f against jit_p50 %.4f   (quoted 3.92 against 5.24)" % (
    len(e1), mean([KM["jit_p50_ctl"][x]["ttft"] for x in e1]), mean([KM["jit_p50"][x]["ttft"] for x in e1])))
RES["set_9660"]["ep1"] = {"n": len(e1), "jit_p50_ctl": mean([KM["jit_p50_ctl"][x]["ttft"] for x in e1]),
                          "jit_p50": mean([KM["jit_p50"][x]["ttft"] for x in e1])}
table("trio_11299", ["jit_exact", "ec_r13", "nofc"],
      "11,299 turns: 0.70 / 0.76 / 0.88; p90, p99 0.86, 10.4 | 1.30, 9.2 | 1.51, 13.3; over 10 s 118 / 97 / 197")
k = table("set_9703", ["nofc_noend", "nofc", "ec_r13", "hintfree_ctl"],
          "9,703 turns: hint-free control 1.02 s behind nofc_noend; ec_r13 0.79 against nofc_noend 0.93")
d = mean([KM["hintfree_ctl"][x]["ttft"] - KM["nofc_noend"][x]["ttft"] for x in k])
print("   hintfree_ctl minus nofc_noend, paired: %+.4f" % d)
RES["set_9703"]["hintfree_ctl-nofc_noend"] = d
k = table("pair_11295", ["nofc", "nofc_noend"], "11,295 turns: nofc_noend 0.028 s behind nofc")
d = mean([KM["nofc_noend"][x]["ttft"] - KM["nofc"][x]["ttft"] for x in k])
print("   nofc_noend minus nofc, paired: %+.4f" % d)
RES["pair_11295"]["nofc_noend-nofc"] = d

print()
print("=== differences that DECISIONS.md states, against the unrounded ones (ALL11 shared turns unless said)")
keys = shared_of(["baseline", "base_rep", "hintfree", "jit_p50", "jit_exact", "jit_p50_ctl", "hintfree_ctl", "ec_r13", "nofc", "ec_r1", "nofc_noend"])
mu = {a: mean([KM[a][x]["ttft"] for x in keys]) for a in KM if a != "ORDER" and all(x in KM[a] for x in keys)}
rows = [("ec_r13 below the better JIT p50 run", mu["jit_p50_ctl"] - mu["ec_r13"], "0.69"),
        ("ec_r13 below the better hint-free run", mu["hintfree"] - mu["ec_r13"], "0.95"),
        ("nofc behind ec_r13 (first set)", mu["nofc"] - mu["ec_r13"], "0.03"),
        ("ec_r1 behind ec_r13 (first set)", mu["ec_r1"] - mu["ec_r13"], "0.01"),
        ("nofc_noend behind ec_r13 (first set)", mu["nofc_noend"] - mu["ec_r13"], "0.05"),
        ("nofc_noend behind jit_exact (first set)", mu["nofc_noend"] - mu["jit_exact"], "0.20"),
        ("JIT p50 slower than nofc, smallest", mu["jit_p50_ctl"] - mu["nofc"], "0.66"),
        ("JIT p50 slower than nofc, largest", mu["jit_p50"] - mu["nofc"], "0.91")]
RES["stated_differences"] = {}
for what, val, quoted in rows:
    ok = ("%.2f" % val) == quoted
    RES["stated_differences"][what] = {"unrounded": val, "quoted": quoted, "equal_at_2_decimals": ok}
    print("   %-45s unrounded %.4f  quoted %s  %s" % (what, val, quoted, "same" if ok else "differs at the second decimal"))

print()
print("=== 'all of it in the load episodes': the part of each repeat pair's move that lies outside the episodes (ALL11 turns)")


def in_episode(k):
    p = POS.get(k)
    return p is not None and any(lo <= p <= hi for lo, hi in EPISODES)


for first, second in (("jit_p50", "jit_p50_ctl"), ("hintfree", "hintfree_ctl"), ("baseline", "base_rep")):
    tot = math.fsum(KM[second][x]["ttft"] - KM[first][x]["ttft"] for x in keys)
    out = math.fsum(KM[second][x]["ttft"] - KM[first][x]["ttft"] for x in keys if not in_episode(x))
    RES["outside_share:%s-%s" % (second, first)] = {"move_per_turn": tot / len(keys), "outside_part_per_turn": out / len(keys),
                                                    "outside_share": out / tot}
    print("   %-24s move %+.4f s per turn, of it outside the episodes %+.4f (%.1f %%)" % (
        "%s-%s" % (second, first), tot / len(keys), out / len(keys), 100 * out / tot))

print()
print("=== 'covers the first load episode': the cut run and episode 1")
kd = KM["jit_p50_d_cut"]
ep1_all = [x for x in POS if 5500 <= POS[x] < 7000 and x[1] > 0]
have = [x for x in ep1_all if x in kd]
own = with_ttft_sorted(P["jit_p50_d_cut"]["turns"])
kept_n = len(kd)
print("   order-run episode 1 returning turns: %d; kept in jit_p50_d_cut: %d (%.3f); its kept turns: %d, so own-order positions"
      " 5500-%d of the 5500-6999 stretch" % (len(ep1_all), len(have), len(have) / len(ep1_all), kept_n, kept_n - 1))
last = sorted(kd.values(), key=lambda t: t["t_send"])[-300:]
print("   its last 300 kept turns (sent minutes %.1f-%.1f): mean TTFT %.3f s, turns over 10 s: %d" % (
    last[0]["t_send"] / 60, last[-1]["t_send"] / 60, mean([t["ttft"] for t in last]), sum(1 for t in last if t["ttft"] > 10)))
RES["cut_episode1"] = {"episode1_returning": len(ep1_all), "kept_in_cut": len(have), "share": len(have) / len(ep1_all),
                       "cut_kept_turns": kept_n, "last300_mean_ttft": mean([t["ttft"] for t in last]),
                       "last300_over10": sum(1 for t in last if t["ttft"] > 10)}

with open(os.path.join(HERE, "item6_decisions_spotcheck.json"), "w") as f:
    json.dump(RES, f, indent=1, sort_keys=True)
print()
print("wrote item6_decisions_spotcheck.json")
