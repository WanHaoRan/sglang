"""Does a later reader change the request mix?  Per arm and window: users in session (lanes 'live'), in flight,
requests sent, mean recorded pause before the request (wait_s), share of requests after a human pause (no call)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib, runtab
from vlib import mean
tabs = {a: runtab.build(34, a) for a in ("jit_p50", "ec_full", "jit_exact", "ec_exact")}
for lo, hi in [(28, 60), (60, 85), (85, 110)]:
    print(f"=== minutes {lo}-{hi}")
    for a, tab in tabs.items():
        R = [q for q in tab["reqs"].values() if lo * 60 <= q["t_send"] < hi * 60]
        n = len(R); X = n / ((hi - lo) * 60.0)
        live = mean(l["live"] for l in tab["lanes"] if lo * 60 <= l["t"] < hi * 60)
        infl = mean(l["in_flight"] for l in tab["lanes"] if lo * 60 <= l["t"] < hi * 60)
        ttft = mean(q["ttft"] for q in R); dec = mean(q["latency"] - q["ttft"] for q in R)
        ws = mean((q["wait_s"] or 0.0) for q in R if q["k"] >= 1)
        ret = [q for q in R if q["k"] >= 1]
        long_ = sum(1 for q in ret if (q["wait_s"] or 0) >= 10) / len(ret)
        print(f"   {a:9s}: n {n:5d} ({X*60:5.1f}/min)  users in session {live:6.1f}  in flight {infl:5.1f}  pausing {live-infl:5.1f}  TTFT {ttft:6.2f}  decode {dec:5.2f}  "
              f"N/X - TTFT - decode = {live/X - ttft - dec:6.2f} s;  recorded pause before returning requests: mean {ws:5.2f} s, >= 10 s: {100*long_:4.1f} %")
