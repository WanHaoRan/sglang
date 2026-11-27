"""ev_prices 3b check: does a long prefill at light load hold back the turns that arrive during it?
Long prefill = a request with >= 12 K uncached prompt tokens, from its admission to its first token.  Turns with
no read of their own that enter during it: their queue wait, against (a) turns entering outside any such
interval (baseline) and (b) the serial prediction (wait until that prefill's first token)."""
import sys, os, bisect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib, runtab
from vlib import mean, pct
tot_n = 0; extra = []; serial = []; basel = []
for arm in ("ec_full", "jit_exact", "ec_exact", "jit_p50"):
    tab = runtab.build(34, arm); reqs = tab["reqs"]
    lo, hi = 8 * 60, 28 * 60
    longs = sorted((q["adm"], q["t_send"] + q["ttft"]) for q in reqs.values() if lo <= q["adm"] < hi and q["prompt"] - q["cached"] >= 12000)
    starts = [x[0] for x in longs]
    inside = []; outside = []
    for rid, q in reqs.items():
        if not (lo <= q["entry"] < hi) or q["reads"]: continue
        j = bisect.bisect_right(starts, q["entry"]) - 1
        hit = None
        while j >= 0 and starts[j] > q["entry"] - 5.0:
            if longs[j][0] < q["entry"] < longs[j][1]:
                hit = longs[j]; break
            j -= 1
        if hit: inside.append((q["queue"], hit[1] - q["entry"]))
        else: outside.append(q["queue"])
    b = mean(outside)
    tot_n += len(inside); extra += [x[0] - b for x in inside]; serial += [x[1] for x in inside]; basel.append(b)
    print(f"c34 {arm}: long prefills {len(longs)}; turns entering during one {len(inside)}: queue wait mean {mean(x[0] for x in inside):.2f} s (median {pct([x[0] for x in inside],.5):.2f}); outside {len(outside)}: mean {b:.3f} s; serial prediction mean {mean(x[1] for x in inside):.2f} s")
print(f"all four arms: {tot_n} turns; wait above the baseline {mean(extra):.2f} s; serial prediction {mean(serial):.2f} s")
