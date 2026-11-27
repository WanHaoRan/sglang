"""ev_prices 2a: the closed-loop identity, campaign 32 against 34, by window of send time.
   N users in session = X * (TTFT + decode + pause), X = requests per second.  g = 1/X.
Everything from client records (finished requests only) plus the lanes records (live users, in flight) and
the Decode batch lines (#running-req)."""
import sys, os, bisect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib, runtab
from vlib import mean, pct

def window(camp, arm, lo, hi):
    tab = runtab.build(camp, arm)
    reqs = tab["reqs"]
    ev = vlib.load(camp, arm)
    R = [q for q in reqs.values() if lo <= q["t_send"] < hi and q["ttft"] is not None]
    n = len(R)
    X = n / (hi - lo)
    ttft = mean(q["ttft"] for q in R)
    dec = mean(q["latency"] - q["ttft"] for q in R)
    # completions in the window (a second throughput estimate)
    done = sum(1 for q in reqs.values() if lo <= q["t_send"] + q["latency"] < hi)
    live = mean(l["live"] for l in tab["lanes"] if lo <= l["t"] < hi)
    infl = mean(l["in_flight"] for l in tab["lanes"] if lo <= l["t"] < hi)
    B = mean([r for t, k, r, qn, u, _, _ in vlib.batches(ev) if k == "D" and lo <= t < hi])
    Q = mean([qn for t, k, r, qn, u, _, _ in vlib.batches(ev) if lo <= t < hi])
    # recomputed history, tokens per request
    lost = sum(max(q["ideal"] - q["cached"], 0) for q in R) / n
    newt = sum(q["prompt"] - q["cached"] for q in R) / n
    outt = sum(q["out"] for q in R) / n
    # pause per request by the identity
    pause = live / X - ttft - dec
    return dict(n=n, X=X, g=1 / X, ttft=ttft, dec=dec, done=done, live=live, infl=infl, B=B, Q=Q, lost=lost, newt=newt, outt=outt, pause=pause)

for lo, hi in [(5400, 6600), (6000, 6600), (5100, 6600)]:
    print(f"=== requests sent in [{lo}, {hi}) s")
    for arm in ("ec_full", "jit_exact", "ec_exact"):
        a = window(32, arm, lo, hi); b = window(34, arm, lo, hi)
        for nm, w in (("c32", a), ("c34", b)):
            print(f"  {nm} {arm:9s}: reqs {w['n']:5d} ({w['X']*60:6.1f}/min, done {w['done']})  g=1/X {w['g']:.3f} s  TTFT {w['ttft']:6.2f}  decode {w['dec']:6.2f}  users live {w['live']:6.1f} in flight {w['infl']:5.1f} "
                  f"running B {w['B']:4.1f} queue {w['Q']:5.1f}  pause/req (N/X-TTFT-decode) {w['pause']:6.2f}  lost history {w['lost']:6.0f} tok/req  uncached {w['newt']:6.0f} tok/req  out {w['outt']:.0f}")
        dT = a["ttft"] - b["ttft"]; dg = a["g"] - b["g"]
        print(f"     c32 - c34: dTTFT {dT:+.2f} s  dg {dg:+.3f} s  ratio {dT/dg:.1f} TTFT-s per GPU-s;  N*dg {b['live']*dg:.1f} = dTTFT {dT:.1f} + d decode {a['dec']-b['dec']:+.1f} + d pause {a['pause']-b['pause']:+.1f} "
              f"(+ dN*g {(a['live']-b['live'])*a['g']:+.2f});  (dTTFT + d pause)/dg = {(dT + a['pause'] - b['pause'])/dg:.1f};  N-B = {b['live']-b['B']:.1f};  B*g: c32 {a['B']*a['g']:.2f} c34 {b['B']*b['g']:.2f};"
              f"  d lost history {a['lost']-b['lost']:+.0f} tok/req -> at 0.043 ms {(a['lost']-b['lost'])*0.043e-3:.3f} s, d uncached {(a['newt']-b['newt']):+.0f} tok -> {(a['newt']-b['newt'])*0.043e-3:.3f} s")
