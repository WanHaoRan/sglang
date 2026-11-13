#!/usr/bin/env python3
"""r2_11_intervals.py -- reproduce the bracketed [5 %, 95 %] intervals of the pairs files.

Written after items 1-4 were frozen and after reading the interval() function of
frozen/oracle_scripts_staging/c26_ec/pairs_ec.py: blocks of 500 consecutive positions of the
order run (keys with no position share one block), 2000 draws, random.Random(0), rng.choices,
percentiles by sorted(v)[min(len(v)-1, int(q*len(v)))].  The turns come from my own parser.

Run as:  nice -n 19 python3 -I -B r2_11_intervals.py
"""
import json
import os
import pickle
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP = os.path.join(HERE, "r2_out_snapshot")
BLOCK, DRAWS = 500, 2000
EPISODES = ((5500, 7000), (8750, 11000))


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


with open(os.path.join(HERE, "r2_parsed.pkl"), "rb") as f:
    P = pickle.load(f)
with open(os.path.join(HERE, "r2_parsed_extra.pkl"), "rb") as f:
    P.update(pickle.load(f))
KM, OK = {}, {}
for a, d in P.items():
    t = sorted((r for r in d["records"] if r.get("kind") == "turn" and r.get("ttft") is not None),
               key=lambda r: r["t_send"])
    OK[a] = t
    thr = t[-1]["t_send"] - 300.0
    KM[a] = {(r["conv"], r["turn"]): r for r in t if r["t_send"] < thr}
rank = {(r["conv"], r["turn"]): i for i, r in enumerate(OK["ORDER"])}


def in_episode(k):
    return any(lo <= rank.get(k, -1) < hi for lo, hi in EPISODES)


def interval(a, b, ks):
    blocks = {}
    for k in ks:
        s = blocks.setdefault(rank.get(k, 1 << 30) // BLOCK, [0.0, 0])
        s[0] += KM[a][k]["ttft"] - KM[b][k]["ttft"]
        s[1] += 1
    cells = list(blocks.values())
    rng = random.Random(0)
    draws = []
    for _ in range(DRAWS):
        pick = rng.choices(cells, k=len(cells))
        draws.append(sum(c[0] for c in pick) / sum(c[1] for c in pick))
    return sum(c[0] for c in cells) / sum(c[1] for c in cells), pct(draws, .05), pct(draws, .95), len(cells)


rows = []
summary = {}
for fname in ("pairs_ec.out", "pairs_ec_cut.out", "pairs_ec_ctl.out"):
    L = open(os.path.join(SNAP, fname)).read().split("\n")
    arms = [x.strip() for x in re.match(r"^\d+ returning turns shared by (.*)", L[0]).group(1).split(",")]
    keys = None
    for a in arms:
        ks = set(k for k in KM[a] if k[1] > 0)
        keys = ks if keys is None else keys & ks
    keys = sorted(keys, key=lambda k: rank.get(k, 1 << 30))
    parts = {"all shared returning turns:": keys,
             "the two load episodes": [k for k in keys if in_episode(k)],
             "the other positions:": [k for k in keys if not in_episode(k)]}
    n = ok = 0
    excl = 0
    for start, ks in parts.items():
        i = next(k for k, s in enumerate(L) if s.startswith(start))
        cols = L[i + 1].split()
        k = i + 2
        while L[k].strip():
            row = L[k].split()[0]
            ents = re.findall(r"([+-]\d+\.\d+) \[\s*([+-][\d.]+),\s*([+-][\d.]+)\]", L[k])
            for col, (val, lo, hi) in zip(cols, ents):
                d, mlo, mhi, nb = interval(row, col, ks)
                mine = ("%+5.2f" % mlo, "%+5.2f" % mhi)
                good = (mine[0].strip() == lo.strip() or float(mine[0]) == float(lo)) and \
                       (mine[1].strip() == hi.strip() or float(mine[1]) == float(hi))
                n += 2
                ok += (1 if float(mine[0]) == float(lo) else 0) + (1 if float(mine[1]) == float(hi) else 0)
                rows.append({"file": fname, "section": start, "pair": "%s-%s" % (row, col),
                             "printed": [val, lo, hi], "mine": [round(d, 4), mine[0], mine[1]],
                             "blocks": nb, "equal": good})
            k += 1
    summary[fname] = {"interval_bounds_compared": n, "equal": ok, "arms": arms, "keys": len(keys)}
    print("%-18s %2d arms, %5d keys: interval bounds compared %3d, equal %3d"
          % (fname, len(arms), len(keys), n, ok))
bad = [r for r in rows if not r["equal"]]
print("cells whose bounds differ:", len(bad))
for r in bad[:20]:
    print("   ", r)

# same-configuration repeats on the ALL11 keys, as printed by pairs_ec_ctl.out
print("\nrepeat pairs on the keys of pairs_ec_ctl.out (ALL11 keys), 5-95 % of this block bootstrap:")
for r in rows:
    if r["file"] == "pairs_ec_ctl.out" and r["section"].startswith("all") and r["pair"] in (
            "base_rep-baseline", "hintfree_ctl-hintfree", "jit_p50_ctl-jit_p50", "ec_r13_rep-ec_r13",
            "jit_p50_d-jit_p50_ctl", "jit_p50_d-jit_p50"):
        lo, hi = float(r["printed"][1]), float(r["printed"][2])
        print("   %-24s printed %s [%s, %s]  blocks %d  excludes zero: %s"
              % (r["pair"], r["printed"][0], r["printed"][1], r["printed"][2], r["blocks"],
                 lo > 0 or hi < 0))
print("\nLEAD5 arms on the six-arm keys of pairs_ec.out: does the printed interval exclude zero?")
for r in rows:
    if r["file"] == "pairs_ec.out" and r["section"].startswith("all"):
        lo, hi = float(r["printed"][1]), float(r["printed"][2])
        print("   %-24s printed %s [%s, %s]  excludes zero: %s"
              % (r["pair"], r["printed"][0], r["printed"][1], r["printed"][2], lo > 0 or hi < 0))
json.dump({"summary": summary, "rows": rows}, open(os.path.join(HERE, "r2_intervals.json"), "w"), indent=1)
