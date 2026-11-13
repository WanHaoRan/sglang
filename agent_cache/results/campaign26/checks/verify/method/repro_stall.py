"""Independent recomputation of stall_ec.py's tables (stall_ec.out, stall_ec_ctl.out, stall_ec_noend.out) and of
split_ec.py's read-class split, from the cache. Also splits the stall of the 'read was the last wait' class by what the
turn was doing during it (inside its own reads or not), which stall_ec.py does not print.
  nice -n 19 python3 -I -B repro_stall.py"""
import collections, os, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *

LANDED, SLACK, BOUND = 0.05, 0.1, 0.1
rk = rank()


def stalls(a):
    """Per key: stall, of it inside own reads, queue wait, reads ending after admission + slack, time parked after arrival."""
    d = arm(a)
    out = {}
    for k, arrive in d["entry"].items():
        wait = d["queue"][k]
        admit = arrive + wait
        mine = [(end, ms) for end, ms, _ in d["ios"].get(k, ()) if end <= admit + SLACK]
        last = max((end for end, _ in mine), default=arrive)
        stall = min(max(last - arrive, 0.0), wait)
        inside = sum(max(0.0, min(end, admit) - max(end - ms, arrive)) for end, ms in mine)
        out[k] = (stall, min(inside, stall), wait, len(d["ios"].get(k, ())) - len(mine))
    return out


def table(names, label):
    D = {a: arm(a) for a in names}
    S = {a: stalls(a) for a in names}
    keys = sorted(set.intersection(*(set(k for k in D[a]["by"] if k[1] > 0 and k in S[a]) for a in names)), key=lambda k: (rk.get(k, 1 << 30), k))
    N = len(keys)
    print(f"\n== {label}: {N} returning turns shared by {', '.join(names)}")
    print("   arm: TTFT = stall + rest + after | reading turns, landed by arrival, stall per late reading turn, inside | late>adm | last-wait class s/turn (turns), of it inside own reads | GPU-also class s/turn (turns), wait after landing")
    for a in names:
        s = [S[a][k] for k in keys]
        t = mean(D[a]["by"][k]["ttft"] for k in keys)
        stall, wait = mean(x[0] for x in s), mean(x[2] for x in s)
        reading = [S[a][k] for k in keys if D[a]["pages"][k] > 0]
        late = [x for x in reading if x[0] >= LANDED]
        last = [x for x in late if x[2] - x[0] < BOUND]
        both = [x for x in late if x[2] - x[0] >= BOUND]
        print(f"   {a:13s} {t:6.3f} = {stall:5.3f} + {wait - stall:5.3f} + {t - wait:5.3f} | {len(reading):5d} {len(reading) - len(late):5d} {mean(x[0] for x in late):5.2f} {mean(x[1] for x in late):5.2f} | {sum(x[3] for x in s):3d} | "
              f"{sum(x[0] for x in last) / N:5.3f} ({len(last):4d}) inside {sum(x[1] for x in last) / N:5.3f} = {100 * sum(x[1] for x in last) / max(1e-9, sum(x[0] for x in last)):4.0f} % | "
              f"{sum(x[0] for x in both) / N:5.3f} ({len(both):4d}) {sum(x[2] - x[0] for x in both) / N:5.3f}")
    for lo, hi in STRETCHES:
        ks = [k for k in keys if lo <= rk.get(k, -1) < hi]
        if ks:
            print(f"   {lo:5d}-{hi:5d} {len(ks):5d} | " + " | ".join(f"{a} {mean(D[a]['by'][k]['ttft'] for k in ks):5.2f} {mean(S[a][k][0] for k in ks):5.3f} {mean(S[a][k][2] - S[a][k][0] for k in ks):5.3f}" for a in names))
    ref = names[0]
    for a in names[1:]:
        dd = [(D[a]["by"][k]["ttft"] - D[ref]["by"][k]["ttft"], S[a][k][0] - S[ref][k][0], (S[a][k][2] - S[a][k][0]) - (S[ref][k][2] - S[ref][k][0])) for k in keys]
        total, stall, rest = (mean(x[i] for x in dd) for i in range(3))
        print(f"   against {ref}: {a:13s} {total:+6.3f} | {stall:+6.3f} | {rest:+6.3f} | {total - stall - rest:+6.3f}")
    return keys


def split(names, label):
    D = {a: arm(a) for a in names}
    keys = shared(names)
    N = len(keys)
    ref = names[0]
    print(f"\n== split {label}: {N} turns; against {ref}: total | read in both | only here | only in ref | neither ; recomputed per turn; slowest 1 % share")
    for a in names:
        v = sorted(D[a]["by"][k]["ttft"] for k in keys)
        rd = [k for k in keys if D[a]["pages"][k] > 0]
        nr = [k for k in keys if not D[a]["pages"][k] > 0]
        line = (f"   {a:13s} mean {mean(v):.3f} p99.9 {pct(v, .999):.1f} max {v[-1]:.1f} over2 {sum(x > 2 for x in v)} slowest1% {100 * sum(v[-max(1, N // 100):]) / sum(v):.1f} % | read {len(rd)} ttft {mean(D[a]['by'][k]['ttft'] for k in rd):.2f} "
                f"rec {mean(rec(D[a]['by'][k]) for k in rd):.0f} | no read {len(nr)} ttft {mean(D[a]['by'][k]['ttft'] for k in nr):.2f} rec {mean(rec(D[a]['by'][k]) for k in nr):.0f} | rec all {mean(rec(D[a]['by'][k]) for k in keys):.1f}")
        if a != ref:
            cells = []
            for here, there in ((True, True), (True, False), (False, True), (False, False)):
                ks = [k for k in keys if (D[a]["pages"][k] > 0) == here and (D[ref]["pages"][k] > 0) == there]
                cells.append(f"{len(ks):5d} {sum(D[a]['by'][k]['ttft'] - D[ref]['by'][k]['ttft'] for k in ks) / N:+6.3f}")
            line += f" || {mean(D[a]['by'][k]['ttft'] - D[ref]['by'][k]['ttft'] for k in keys):+6.3f} | " + " | ".join(cells)
        print(line)
    for a in names[1:]:
        cells = []
        for lo, hi in STRETCHES:
            ks = [k for k in keys if lo <= rk.get(k, -1) < hi]
            if ks:
                cells.append(f"{lo}-{hi}: {len(ks)} {sum(D[a]['by'][k]['ttft'] - D[ref]['by'][k]['ttft'] for k in ks) / N:+6.3f}")
        print(f"   {a:13s} by stretch: " + "  ".join(cells) + f"  absent: {sum(D[a]['by'][k]['ttft'] - D[ref]['by'][k]['ttft'] for k in keys if k not in rk) / N:+6.3f}")
    for a in names:
        ks = [k for k in keys if D[a]["staged"][k]]
        if ks:
            print(f"   staged in {a}: {len(ks)}: " + "  ".join(f"{b} {mean(D[b]['by'][k]['ttft'] for k in ks):.2f} (read {sum(D[b]['pages'][k] > 0 for k in ks)})" for b in names))


table(LEAD5, "stall_ec.out")
table(SEVEN, "stall_ec_ctl.out")
table(["nofc", "nofc_noend"], "stall_ec_noend.out")
split(LEAD5, "split_ec_lead.out")
split(["nofc_noend", "nofc", "ec_r13", "hintfree_ctl"], "split_ec_noend.out")
split(["jit_exact", "ec_r13", "nofc"], "split_ec_trio.out")
split(["jit_exact", "ec_r13", "nofc", "jit_p50_ctl", "hintfree_ctl"], "split_ec.out")
