"""Where in the queue do reads start? Campaign 34 server logs (stdlib only).

For each planned turn: its position at arrival (ntp_arrive), the gate decisions (ntp_gate), the
re-issued reads while queued (ntp_refetch issued=1: position and wait left), re-evictions (ntp_reevict)
and the admission (ntp_admit wait_ms). Usage: queued_evictions.py <server.log> [t0_min t1_min]
"""
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime

log = sys.argv[1]
t0_min = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
t1_min = float(sys.argv[3]) if len(sys.argv) > 3 else 110.0
evt = re.compile(r"^\[(\S+ \S+)\] HICACHE_EVT (\w+) rid=(\S+) (.*)$")


def fields(rest):
    return dict(kv.split("=", 1) for kv in rest.split() if "=" in kv)


def ts(stamp):
    return datetime.strptime(stamp[:26], "%Y-%m-%d %H:%M:%S.%f").timestamp()


first = None
arrive, gates, refetch, reevict, admit, hold = {}, defaultdict(list), defaultdict(list), Counter(), {}, {}
with open(log, errors="replace") as fh:
    for line in fh:
        if first is None:
            m0 = re.match(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\]", line)
            if m0:
                first = ts(m0.group(1))
        m = evt.match(line)
        if not m:
            continue
        stamp, kind, rid, rest = m.groups()
        if kind not in ("ntp_arrive", "ntp_gate", "ntp_refetch", "ntp_reevict", "ntp_admit", "ntp_hold"):
            continue
        t = ts(stamp)
        f = fields(rest)
        if kind == "ntp_arrive":
            arrive[rid] = (t, int(f["pos"]), int(f["qlen"]))
        elif kind == "ntp_gate":
            gates[rid].append((t, f["decision"], int(f["matched"]), int(f["chain"])))
        elif kind == "ntp_refetch":
            if f.get("issued") == "1":
                refetch[rid].append((t, int(f["pos"]), int(f["resid_ms"]) / 1000 if f["resid_ms"] != "na" else None, int(f["shrink"])))
        elif kind == "ntp_reevict":
            reevict[rid] += 1
        elif kind == "ntp_admit":
            admit[rid] = (t, int(f["wait_ms"]) / 1000 if f["wait_ms"] != "na" else None)
        elif kind == "ntp_hold":
            hold[rid] = (t, int(f["pos"]))


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    return xs[min(len(xs) - 1, int(q * len(xs)))]


sel = [rid for rid, (t, pos, qlen) in arrive.items() if t0_min * 60 <= t - first < t1_min * 60 and rid in admit]
print(f"log {log}\nwindow minutes {t0_min}-{t1_min}: planned turns arrived and admitted {len(sel)}")
waits = [admit[r][1] for r in sel if admit[r][1] is not None]
print(f"queue wait s: p10 {pct(waits,.1):.1f} p50 {pct(waits,.5):.1f} p90 {pct(waits,.9):.1f}; position at arrival p50 {pct([arrive[r][1] for r in sel],.5)}")
first_gate = Counter()
missing_at_arrival = []
for r in sel:
    g = [x for x in gates.get(r, ()) if abs(x[0] - arrive[r][0]) < 0.5]
    if g:
        first_gate[g[0][1]] += 1
        if g[0][1] in ("read", "reread", "retry"):
            missing_at_arrival.append(g[0][3] - g[0][2])
    else:
        first_gate["none_at_arrival"] += 1
print("gate decision at arrival:", dict(first_gate))
print(f"  missing tokens at arrival when read: p50 {pct(missing_at_arrival,.5)} p90 {pct(missing_at_arrival,.9)} n {len(missing_at_arrival)}")
later = [(r, x) for r in sel for x in refetch.get(r, ()) if x[0] - arrive[r][0] >= 0.5]
print(f"re-issued reads while queued (>=0.5 s after arrival): {len(later)} over {len({r for r, _ in later})} turns")
if later:
    pos = [x[1] for _, x in later]
    resid = [x[2] for _, x in later if x[2] is not None]
    shrink = [x[3] for _, x in later]
    frac = [(x[0] - arrive[r][0]) / max(1e-9, admit[r][0] - arrive[r][0]) for r, x in later]
    print(f"  position when issued: p10 {pct(pos,.1)} p50 {pct(pos,.5)} p90 {pct(pos,.9)}; share at position <= 1: {sum(p <= 1 for p in pos)/len(pos):.2f}, <= 5: {sum(p <= 5 for p in pos)/len(pos):.2f}")
    print(f"  wait left when issued s: p10 {pct(resid,.1):.1f} p50 {pct(resid,.5):.1f} p90 {pct(resid,.9):.1f}")
    print(f"  tokens missing: p50 {pct(shrink,.5)} p90 {pct(shrink,.9)} total M {sum(shrink)/1e6:.2f}")
    print(f"  fraction of the turn's own queue wait elapsed when issued: p10 {pct(frac,.1):.2f} p50 {pct(frac,.5):.2f} p90 {pct(frac,.9):.2f}")
    byq = defaultdict(list)
    for r, x in later:
        byq[min(4, int(4 * min(0.999, (x[0] - arrive[r][0]) / max(1e-9, admit[r][0] - arrive[r][0]))))].append(x[3])
    print("  by quarter of the wait elapsed (count, M tokens):", {q: (len(v), round(sum(v) / 1e6, 2)) for q, v in sorted(byq.items())})
print(f"turns with a re-eviction event: {sum(1 for r in sel if reevict.get(r))}; holds logged: {sum(1 for r in sel if r in hold)}")
caps = sum(1 for r in sel for g in gates.get(r, ()) if g[1] == "capped")
print(f"gate 'capped' decisions: {caps} over {len({r for r in sel if any(g[1]=='capped' for g in gates.get(r, ()))})} turns")
