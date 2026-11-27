"""ev_write 2 (join-free check): from the engine's own ec_start lines, how soon after a turn's commit its chain
was first seen missing 256+ tokens from memory; and the engine's logged price (ev_prices 1d)."""
import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib
from vlib import kv, pct
for arm in ("ec_full", "ec_exact"):
    ev = vlib.load(34, arm)
    first = {}
    for t, r in ev["ec_start"]:
        d = kv(r)
        if d["rid"] not in first:
            first[d["rid"]] = (t, float(d["idle_ms"]) / 1e3, int(d["tokens"]))
    sub = [v for v in first.values() if v[0] >= 1800]
    w5 = [v for v in sub if v[1] <= 5.0]; w1 = [v for v in sub if v[1] <= 1.0]
    print(f"c34 {arm}: plans with an ec_start line {len(first)}; from minute 30: {len(sub)}; first seen missing within 5 s of the commit {len(w5)} ({100*len(w5)/len(sub):.0f} %), within 1 s {len(w1)}; "
          f"tokens missing at that first sight (within 5 s): median {pct([v[2] for v in w5],.5)/1e3:.1f} K")
    prices = collections.Counter(kv(r)["price"] for t, r in ev["ec_price"])
    print(f"   ec_price updates {len(ev['ec_price'])}: {dict(prices.most_common(4))}")
