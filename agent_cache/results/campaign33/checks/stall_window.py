"""Paired TTFT differences with and without the requests sent in a window, for an arm whose server was drained from
outside (campaign 33 DECISIONS.md, "Outside requests"). Stdlib only. On the host, from agent_cache/results:
  python3 -I -B campaign33/checks/stall_window.py compare_c33_designs oracle_v2 400 460 jit_exact ec_full hicache
  python3 -I -B campaign33/checks/stall_window.py compare_c32_designs ec_exact 2716 3099 ec_full jit_exact
Reads the figure folder's turns_<arm>.csv. A request pairs across arms by (conv, turn) and counts when sent before
6,600 s in both; a pair is left out when either side was sent inside [START, END) s. Interval: 95 %, resampling convs,
2,000 draws, seed 0 (compare_arms.py resamples users by play number, so its intervals differ slightly)."""

import csv
import os
import random
import sys

CUT = 6600.0
DRAWS = 2000


def load(designs, arm):
    with open(os.path.join(designs, f"turns_{arm}.csv")) as f:
        return {(r["conv"], r["turn"]): r for r in csv.DictReader(f)}


def pairs(a, b, skip):
    out = []
    for key, ra in a.items():
        rb = b.get(key)
        if rb is None:
            continue
        ta, tb = float(ra["t0"]), float(rb["t0"])
        if ta >= CUT or tb >= CUT:
            continue
        if skip and (skip[0] <= ta < skip[1] or skip[0] <= tb < skip[1]):
            continue
        out.append((key[0], float(rb["ttft"]) - float(ra["ttft"])))
    return out


def interval(diffs):
    by_conv = {}
    for conv, d in diffs:
        by_conv.setdefault(conv, []).append(d)
    convs = list(by_conv)
    rnd = random.Random(0)
    means = []
    for _ in range(DRAWS):
        total = count = 0
        for conv in (rnd.choice(convs) for _ in convs):
            total += sum(by_conv[conv])
            count += len(by_conv[conv])
        means.append(total / count)
    means.sort()
    return means[int(0.025 * DRAWS)], means[int(0.975 * DRAWS)]


def main():
    designs, target, start, end, *refs = sys.argv[1:]
    window = (float(start), float(end))
    rows = {arm: load(designs, arm) for arm in [target, *refs]}
    print(f"{designs}: {target} against {', '.join(refs)}; window [{window[0]:.0f}, {window[1]:.0f}) s")
    for ref in refs:
        for skip in (None, window):
            p = pairs(rows[ref], rows[target], skip)
            mean = sum(d for _, d in p) / len(p)
            lo, hi = interval(p)
            label = "all" if skip is None else "without sends in the window"
            print(f"  {ref:>9} -> {target}: {label:28s} n={len(p):6d}  {mean:+.3f} s [{lo:+.3f}, {hi:+.3f}]")
    inside = [
        (key, float(r["ttft"]) - float(rows[refs[0]][key]["ttft"]))
        for key, r in rows[target].items()
        if key in rows[refs[0]] and window[0] <= float(r["t0"]) < window[1]
    ]
    print(
        f"  sent in the window by {target}: {len(inside)} requests, {sum(d for _, d in inside):.0f} s of TTFT"
        f" more than {refs[0]} on the same requests"
    )


if __name__ == "__main__":
    main()
