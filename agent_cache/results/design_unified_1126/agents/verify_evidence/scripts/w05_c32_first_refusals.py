"""ev_write 3c: campaign 32's first refused reads (srs_dispatch launched=0) and the write queue at that moment."""
import sys, os, bisect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib
from vlib import kv, pct
for arm in ("ec_full", "jit_exact", "jit_p50"):
    ev = vlib.load(32, arm)
    ref = [t for t, r in ev["srs_dispatch"] if kv(r).get("launched") == "0"]
    sub = {}; 
    for t, r in ev["h2s_submit"]:
        d = kv(r); sub[d["op"]] = (t, int(d["tokens"]))
    done = {}
    for t, r in ev["h2s_done"]:
        d = kv(r); done[d["op"]] = t
    short = [t for t, r in ev["evict_host"] if int(kv(r)["tokens"]) < int(kv(r)["requested"])]
    print(f"c32 {arm}: refused dispatches {len(ref)}; first at {ref[0]:.0f} s; first short host eviction at {short[0]:.0f} s")
    # bursts: refusals separated by more than 60 s start a new burst
    bursts = []
    for t in ref:
        if not bursts or t - bursts[-1][-1] > 60: bursts.append([t])
        else: bursts[-1].append(t)
    for b in bursts[:4]:
        t0 = b[0]
        pend = [(st, tk) for op, (st, tk) in sub.items() if st <= t0 and done.get(op, 1e18) > t0]
        if pend:
            print(f"   burst at {t0:.0f} s ({len(b)} refusals, to {b[-1]:.0f} s): write ops pending {len(pend)}, tokens {sum(x[1] for x in pend)/1e3:.0f} K, oldest {t0 - min(x[0] for x in pend):.1f} s, median op {pct([x[1] for x in pend],.5):.0f} tokens")
        else:
            print(f"   burst at {t0:.0f} s ({len(b)} refusals): no write pending")
    # the same quantities over minutes 60-76.5 in general
    samp = []
    ops = sorted((st, done.get(op, 1e18), tk) for op, (st, tk) in sub.items())
    for t in range(3600, 4590, 5):
        pend = [(st, tk) for st, dn, tk in ops if st <= t < dn]
        samp.append((len(pend), sum(x[1] for x in pend), (t - min(x[0] for x in pend)) if pend else 0.0))
    print(f"   minutes 60-76.5 (every 5 s): pending ops p90 {pct([s[0] for s in samp],.9):.0f}, tokens p90 {pct([s[1] for s in samp],.9)/1e3:.0f} K, oldest age p90 {pct([s[2] for s in samp],.9):.1f} s")
ev = vlib.load(34, "ec_full")
sub = {}; done = {}
for t, r in ev["h2s_submit"]:
    d = kv(r); sub[d["op"]] = (t, int(d["tokens"]))
for t, r in ev["h2s_done"]:
    d = kv(r); done[d["op"]] = t
ops = sorted((st, done.get(op, 1e18), tk) for op, (st, tk) in sub.items())
samp = []
for t in range(3600, 4590, 5):
    pend = [(st, tk) for st, dn, tk in ops if st <= t < dn]
    samp.append((len(pend), sum(x[1] for x in pend), (t - min(x[0] for x in pend)) if pend else 0.0))
print(f"c34 ec_full minutes 60-76.5: pending ops p90 {pct([s[0] for s in samp],.9):.0f}, tokens p90 {pct([s[1] for s in samp],.9)/1e3:.0f} K, oldest age p90 {pct([s[2] for s in samp],.9):.1f} s")
