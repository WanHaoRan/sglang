"""Where the load sits in the order run's positions, what the keys absent from the order run are, and what share of each
paired difference falls outside the two episode ranges.
  nice -n 19 python3 -I -B positions.py"""
import collections, os, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *

rk = rank()
N_ORDER = len(rk)
print(f"order run: {N_ORDER} positions, so the ranges 8750-11000 and anything past {N_ORDER - 1} hold no position; EPISODES upper bound 11000 is {11000 - N_ORDER} past the last position")
order_turns = arm("ORDER", margin=0)["every"]
print(f"order run sends: position 5500 at minute {order_turns[5500]['t_send'] / 60:.1f}, 7000 at {order_turns[7000]['t_send'] / 60:.1f}, 8750 at {order_turns[8750]['t_send'] / 60:.1f}, last at {order_turns[-1]['t_send'] / 60:.1f}")

for names, label in ((LEAD5, "LEAD5"), (SEVEN, "SEVEN"), (ALL11, "ALL11")):
    keys = shared(names)
    present = [k for k in keys if k in rk]
    absent = [k for k in keys if k not in rk]
    print(f"\n== {label}: {len(keys)} shared returning turns; {len(absent)} absent from the order run ({100 * len(absent) / len(keys):.1f} %)")
    print("   mean TTFT by 250 order positions (turns in bin): " + " ".join(f"{a}" for a in names))
    for lo in range(0, N_ORDER, 250):
        ks = [k for k in present if lo <= rk[k] < lo + 250]
        if ks:
            flag = "E" if any(a <= lo < b for a, b in EPISODES) else " "
            print(f"   {lo:5d} {flag} {len(ks):4d} | " + " ".join(f"{mean(arm(a)['by'][k]['ttft'] for k in ks):6.2f}" for a in names))
    if absent:
        print(f"   absent {len(absent):4d} | " + " ".join(f"{mean(arm(a)['by'][k]['ttft'] for k in absent):6.2f}" for a in names))
        for a in names:
            d = arm(a)
            own = {(r["conv"], r["turn"]): i for i, r in enumerate(d["turns"])}
            pos = sorted(own[k] for k in absent)
            ts = sorted(d["by"][k]["t_send"] / 60 for k in absent)
            v = [d["by"][k]["ttft"] for k in absent]
            print(f"     {a:12s} absent keys: own positions {pos[0]}..{pos[-1]} (median {pos[len(pos) // 2]}), own minutes {ts[0]:.1f}..{ts[-1]:.1f} (median {ts[len(ts) // 2]:.1f}), mean TTFT {mean(v):.3f}, p90 {pct(v, .9):.2f}, over 10 s {sum(x > 10 for x in v)}, "
                  f"over 2 s {sum(x > 2 for x in v)}; present-and-outside-episodes mean {mean(d['by'][k]['ttft'] for k in present if not in_episode(k)):.3f}")
        # absent keys in bins of the first arm's own order
        ref = arm(names[0])
        own = {(r["conv"], r["turn"]): i for i, r in enumerate(ref["turns"])}
        print(f"   absent keys by 250 positions of {names[0]}'s own order: " + " ".join(names))
        for lo in range(min(own[k] for k in absent) // 250 * 250, max(own[k] for k in absent) + 1, 250):
            ks = [k for k in absent if lo <= own[k] < lo + 250]
            if ks:
                print(f"     {lo:5d} {len(ks):4d} | " + " ".join(f"{mean(arm(a)['by'][k]['ttft'] for k in ks):6.2f}" for a in names))
    print("   share of each paired difference that falls outside the episode ranges (row minus column: total s/turn, outside part s/turn, outside share)")
    ep = [k for k in keys if in_episode(k)]
    calm = [k for k in keys if not in_episode(k)]
    for i, a in enumerate(names):
        for b in names[:i]:
            tot = sum(arm(a)["by"][k]["ttft"] - arm(b)["by"][k]["ttft"] for k in keys)
            out = sum(arm(a)["by"][k]["ttft"] - arm(b)["by"][k]["ttft"] for k in calm)
            if label != "ALL11" or (a, b) in (("base_rep", "baseline"), ("hintfree_ctl", "hintfree"), ("jit_p50_ctl", "jit_p50"), ("nofc_noend", "jit_exact"), ("ec_r13", "jit_exact")):
                print(f"     {a:12s} - {b:12s} {tot / len(keys):+7.3f}  outside {out / len(keys):+7.3f}  = {100 * out / tot if abs(tot) > 1e-9 else float('nan'):6.1f} %   (per outside turn {out / len(calm):+.3f}, per episode turn {(tot - out) / len(ep):+.3f})")
