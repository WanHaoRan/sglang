"""ev_prices 1d: inputs of the 'index of the marginal holder' estimate: host turnover (host-only tokens over the
host eviction rate), readers' stall per token read, c, by window; hazard of return at the turnover age from the
run's own gaps (returning requests' recorded pause before them, x1)."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib, runtab
from vlib import kv, pct, mean
for arm in ("ec_full", "jit_exact"):
    ev = vlib.load(34, arm); tab = runtab.build(34, arm)
    reqs, reads, stall = tab["reqs"], tab["reads"], tab["stall"]
    cs = [(t, kv(r)) for t, r in ev["host_census"]]
    evh = [(t, int(kv(r)["tokens"])) for t, r in ev["evict_host"]]
    print(f"c34 {arm}")
    for lo, hi in [(0, 8), (8, 28), (28, 60), (60, 90), (90, 110)]:
        a, b = lo * 60, hi * 60
        ho = [int(d["host_only"]) for t, d in cs if a <= t < b]
        fr = [int(d["free"]) for t, d in cs if a <= t < b]
        rate = sum(x[1] for x in evh if a <= x[0] < b) / (b - a)
        turn = pct(ho, .5) / rate if rate > 0 else float("inf")
        rd = [rid for rid, q in reqs.items() if a <= q["t_send"] < b and rid in stall]
        st = sum(stall[r]["stall"] for r in rd); fre = sum(stall[r]["free"] for r in rd)
        tok = sum(reads[i]["pages"] * 64 for r in rd for i in reqs[r]["reads"] if not reads[i].get("dead"))
        c = fre / st if st > 0 else float("nan")
        # hazard of return at age ~turn among returning requests sent in the window (live sessions only): P(turn <= G < turn+dt | G >= turn)/dt
        g = [q["wait_s"] or 0.0 for q in reqs.values() if q["k"] >= 1 and a <= q["t_send"] < b]
        if turn < 1e9:
            lo_a, hi_a = turn * 0.75, turn * 1.25
            risk = sum(1 for x in g if x >= lo_a); evn = sum(1 for x in g if lo_a <= x < hi_a)
            hz = -math.log(1 - evn / risk) / (hi_a - lo_a) if risk > evn > 0 else float("nan")
        else:
            hz = float("nan")
        lam = c * (st / tok if tok else float("nan")) * hz
        print(f"   minutes {lo}-{hi}: free median {pct(fr,.5)/1e3:.0f} K, host-only {pct(ho,.5)/1e3:.0f} K, host eviction {rate/1e3:.2f} K tok/s -> turnover {turn:.0f} s; "
              f"stall per token read {st/tok*1e5 if tok else float('nan'):.2f}e-5 s; c {c:.2f}; hazard at that age {hz:.4f}/s -> lambda = c x stall/token x hazard = {lam:.2e}")
