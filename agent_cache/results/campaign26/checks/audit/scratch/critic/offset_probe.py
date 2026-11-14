"""The engine's delivery-offset estimate over a run, read from ntp_plan lines (pred_arr_ms - gap_ms), and the share of
arrivals whose measured offset the estimate folds in (0-5 s). Read-only. Usage: python3 -I -B offset_probe.py label=run ..."""
import re, statistics as st, sys

PLAN = re.compile(r"HICACHE_EVT ntp_plan .* gap_ms=(-?\d+) .*pred_arr_ms=(-?\d+)")
ARR = re.compile(r"HICACHE_EVT ntp_arrive .* delta_ms=(-?\d+)")
q = lambda v, p: sorted(v)[min(len(v) - 1, int(p * len(v)))]
for name, run in (a.split("=") for a in sys.argv[1:]):
    d, deltas = [], []
    with open(f"/home/wanhr/sglang/agent_cache/results/{run}/server.log", errors="replace") as f:
        for line in f:
            if "HICACHE_EVT ntp_" not in line:
                continue
            m = PLAN.search(line)
            if m:
                d.append((int(m.group(2)) - int(m.group(1))) / 1000)
                continue
            m = ARR.search(line)
            if m:
                deltas.append(int(m.group(1)) / 1000)
    inr = [x for x in deltas if 0 <= x <= 5]
    print(f"{name:16s} plans {len(d):6d} offset estimate s: p10 {q(d, .1):.3f} p50 {q(d, .5):.3f} p90 {q(d, .9):.3f} max {max(d):.3f} mean {st.mean(d):.3f} | "
          f"arrivals {len(deltas):6d}, measured offset in 0-5 s (folded in) {len(inr):6d} ({100 * len(inr) / len(deltas):.0f} %), their median {st.median(inr):.3f} s; "
          f"below 0: {sum(x < 0 for x in deltas)}, above 5 s: {sum(x > 5 for x in deltas)}")
