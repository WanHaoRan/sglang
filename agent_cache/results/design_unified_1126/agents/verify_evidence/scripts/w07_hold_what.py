"""What was pending when each write hold switched on: the share that is one GPU->host copy submitted within the
second before (a cold prompt written at the end of its prefill, nodes > 1 or tokens >= 8,192)."""
import sys, os, bisect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib
from vlib import kv, pct
for arm in ("ec_full", "jit_exact", "jit_p50", "ec_exact"):
    ev = vlib.load(34, arm)
    d2h = [(t, int(kv(r)["tokens"]), int(kv(r)["nodes"])) for t, r in ev["d2h_submit"]]
    ts = [x[0] for x in d2h]
    pre = [(t, r) for t, r in ev["Prefill"]]
    shares = []; pend = []
    for t, r in ev["srs_write_hold"]:
        d = kv(r)
        if d["state"] != "on" or t >= 7200: continue
        p = int(d["pending_write_tokens"])
        j = bisect.bisect_right(ts, t) - 1
        best = 0
        while j >= 0 and ts[j] > t - 1.0:
            best = max(best, d2h[j][1]); j -= 1
        shares.append(min(best, p) / p); pend.append(p)
    print(f"c34 {arm}: holds {len(shares)}; pending at the start median {pct(pend,.5)/1e3:.1f} K; share that is one copy submitted in the second before: median {100*pct(shares,.5):.0f} %, holds with over 80 %: {sum(1 for s in shares if s > 0.8)}")
