"""Read-only: within-run block bootstrap (250 oracle positions, 2,000 draws) of paired TTFT at 128 users, first 2 h.

Usage: python3 -I -B boot128.py <results_dir>
"""
import collections
import json
import os
import random
import sys

R = sys.argv[1]
ARMS = {
    "baseline": "20261018_103544_three_tier_wc_NAT160",
    "oracle": "20261018_154254_three_tier_wc_oracle_v2_nu_NAT160",
    "old_p50": "20261031_204526_three_tier_wc_oracle_v2_nu_NAT160",
    "new_exact": "20261102_002657_three_tier_wc_oracle_v2_nu_owned_NAT160",
    "new_p50": "20261102_023031_three_tier_wc_oracle_v2_nu_owned_NAT160",
}


def load(arm):
    d = os.path.join(R, ARMS[arm])
    sub = [x for x in os.listdir(d) if x.startswith("client_")][0]
    turns = {}
    with open(os.path.join(d, sub, "client.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r.get("kind") == "turn" and r.get("ttft") is not None and r["t_send"] <= 7200.0:
                turns[(r["conv"], r["turn"])] = r
    return turns


T = {a: load(a) for a in ARMS}
order = sorted(T["oracle"].values(), key=lambda r: r["t_send"])
pos = {(r["conv"], r["turn"]): i for i, r in enumerate(order)}
ret = {a: {k for k, r in t.items() if k[1] > 0} for a, t in T.items()}
s5 = sorted(set.intersection(*ret.values()))
s4 = sorted(set.intersection(*(ret[a] for a in ("oracle", "old_p50", "new_exact", "new_p50"))))
print(f"five-arm returning turns, first 2 h: {len(s5)}; four hinted arms: {len(s4)}")
for a in ARMS:
    print(f"  {a:9s} mean TTFT on {len(s5)}: {sum(T[a][k]['ttft'] for k in s5) / len(s5):.2f} s")


def boot(ks, a, b):
    random.seed(1)
    blocks = collections.defaultdict(list)
    for k in ks:
        blocks[pos[k] // 250].append(T[a][k]["ttft"] - T[b][k]["ttft"])
    bl = list(blocks.values())
    stats = []
    for _ in range(2000):
        draw = [random.choice(bl) for _ in bl]
        stats.append(sum(sum(x) for x in draw) / sum(len(x) for x in draw))
    stats.sort()
    return sum(sum(x) for x in bl) / len(ks), stats[50], stats[1949], len(bl)


for ks, lab in ((s4, f"{len(s4):,}"), (s5, f"{len(s5):,}")):
    for a, b in (("new_p50", "new_exact"), ("old_p50", "oracle"), ("new_exact", "oracle"), ("new_p50", "old_p50")):
        m, lo, hi, nb = boot(ks, a, b)
        print(f"{a} - {b} on {lab}: mean {m:+.3f} s, 95 % within-run interval {lo:+.3f} to {hi:+.3f} s, {nb} blocks")
