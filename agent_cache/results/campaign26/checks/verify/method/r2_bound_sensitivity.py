"""stall_ec.py splits the read stall by BOUND = 0.1 s: a turn admitted within BOUND of its last read's end 'waited for
nothing but its read'. How the split moves with BOUND, and how soon after a read's end the scheduler can admit at all
(the delay from the s2h_io line to the 'HiCache prefetch success' line, which is when the scheduler sees the read).
From the cache; stall definitions as stall_ec.py.
  nice -n 19 python3 -I -B r2_bound_sensitivity.py"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import arm, shared, LEAD5, SEVEN, mean, pct

LANDED, SLACK = 0.05, 0.1
BOUNDS = (0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0)


def table(names, label):
    keys = shared(names)
    N = len(keys)
    print(f"\n== {label}: {N} shared returning turns")
    print("   arm: read stall s/turn | stall of the class 'admitted within BOUND of landing' (turns) for BOUND = " + ", ".join(str(b) for b in BOUNDS)
          + " | late reading turns: admission minus last read end: median, p75, p90 s")
    for a in names:
        d = arm(a)
        rows = []
        for k in keys:
            if not d["pages"][k] > 0:
                continue
            arrive, wait = d["entry"][k], d["queue"][k]
            admit = arrive + wait
            mine = [t for t, ms, _ in d["ios"][k] if t <= admit + SLACK]
            if not mine:
                continue
            stall = min(max(max(mine) - arrive, 0.0), wait)
            if stall >= LANDED:
                rows.append((stall, wait - stall))
        after = sorted(r[1] for r in rows)
        cells = " | ".join(f"{sum(s for s, w in rows if w < b) / N:.3f} ({sum(1 for s, w in rows if w < b):4d})" for b in BOUNDS)
        print(f"   {a:12s} {sum(s for s, _ in rows) / N:.3f} | {cells} | {pct(after, .5):.3f} {pct(after, .75):.3f} {pct(after, .9):.2f}")


table(LEAD5, "LEAD5 (stall_ec.out)")
table(SEVEN, "SEVEN (stall_ec_ctl.out)")

print("\nscheduler's view of a finished read: delay from each s2h_io line to the 'HiCache prefetch success' line of the same request (whole log)")
for a in LEAD5 + ["jit_p50_ctl", "hintfree_ctl"]:
    d = arm(a)
    succ = {}
    for t, body in d["server"]["misc"].get("succ", []):
        m = re.search(r"req=\S+?-c(\d+)-t(\d+) ", body)
        succ.setdefault((int(m[1]), int(m[2])), []).append(t)
    delays = []
    for k, v in d["ios"].items():
        ss = sorted(succ.get(k, []))
        for t, ms, _ in sorted(v):
            nxt = [s for s in ss if s >= t - 0.002]
            if nxt:
                delays.append(nxt[0] - t)
    print(f"   {a:12s} reads {len(delays):5d}: median {pct(delays, .5) * 1000:4.0f} ms, p75 {pct(delays, .75) * 1000:4.0f}, p90 {pct(delays, .9) * 1000:4.0f}, over 0.1 s {100 * sum(x > 0.1 for x in delays) / len(delays):4.1f} %, over 0.5 s {100 * sum(x > 0.5 for x in delays) / len(delays):4.1f} %")
