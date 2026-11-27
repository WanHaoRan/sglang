"""At the start of each fix-3 hold: whose writes were pending? A write of a request still running holds nothing that
eviction could take (its chain is on the GPU: mirror). The bound is the chains of finished requests' pending writes
(x 0.74-0.79 by the census calibration)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import PAGE, load_trees
from model import load
from write_held import Pending, q
trees = load_trees()
for label in sys.argv[2:]:
    m = load(sys.argv[1], label, trees)
    pd = Pending(m)
    rows = []
    for t, state, pending, parked, ppages in m.holds:
        if state != "on" or t > 7200: continue
        pend = pd.at(t + 0.001)
        run_tok = sum(o["tokens"] for o in pend if o["rid"] and pd.fin.get(o["rid"], 1e18) > t)
        big = max((o["tokens"] for o in pend), default=0)
        s = pd.summary(t + 0.001)
        rows.append((t, pending, run_tok, s["bound"], s["sessions"], len(pend), s["oldest_age"]))
    n = len(rows)
    print(f"{sys.argv[1]} {label}: {n} holds before the deadline; pending tokens at the start: median {q([r[1] for r in rows], .5) / 1e3:.0f} K; "
          f"of them written for a request still running (nothing evictable held): median share {q([r[2] / max(r[1], 1) for r in rows], .5):.0%}, "
          f"holds where that share is over 80 %: {sum(1 for r in rows if r[2] / max(r[1], 1) > 0.8)}")
    print(f"   chains of finished requests under pending writes (bound on what the writes hold): median {q([r[3] for r in rows], .5) / 1e3:.0f} K, "
          f"p90 {q([r[3] for r in rows], .9) / 1e3:.0f} K, max {max(r[3] for r in rows) / 1e3:.0f} K; holds with the bound under 50 K: "
          f"{sum(1 for r in rows if r[3] < 50000)}; under 100 K: {sum(1 for r in rows if r[3] < 100000)}; pending ops median {q([r[5] for r in rows], .5)}, "
          f"oldest op median {q([r[6] for r in rows], .5):.1f} s")
