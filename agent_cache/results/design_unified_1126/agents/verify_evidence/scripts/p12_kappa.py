"""ev_prices 1c: parked reads and parked reads past their latest start (time averages by window).
A read is late at t when it is parked and t > D - own read time, D = the turn's GPU turn (runtab.py)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib, runtab
for arm in ("ec_full", "jit_exact", "ec_exact", "jit_p50"):
    tab = runtab.build(34, arm); reqs, reads, stall = tab["reqs"], tab["reads"], tab["stall"]
    out = []
    for lo, hi in [(0, 28), (28, 70), (70, 90), (90, 110)]:
        a, b = lo * 60, hi * 60
        parked = late = busy = 0.0
        for r in reads:
            if r["disp"] is None or r.get("dead"): continue
            p0, p1 = max(r["park"], a), min(r["disp"], b)
            if p1 > p0: parked += p1 - p0
            if r["io_end"] is not None:
                i0, i1 = max(r["io_end"] - r["io_ms"], a), min(r["io_end"], b)
                if i1 > i0: busy += i1 - i0
            s = stall.get(r["rid"])
            if s is None or r["io_ms"] is None: continue
            ls = s["D"] - r["io_ms"]
            l0, l1 = max(r["park"], ls, a), min(r["disp"], b)
            if l1 > l0: late += l1 - l0
        out.append(f"{lo}-{hi}: parked {parked/(b-a):5.1f}, late {late/(b-a):5.2f}, read thread busy {100*busy/(b-a):3.0f} %")
    print(f"c34 {arm}: " + " | ".join(out))
