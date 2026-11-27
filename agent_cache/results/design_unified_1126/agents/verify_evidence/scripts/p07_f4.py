"""ev_prices 3a/3b (F4): readers in the first 40 minutes, the GPU's state at their arrival, their stall, what a
recompute on the idle-prefill fit would cost them, and the arrivals during that recompute."""
import sys, os, bisect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib, runtab
from vlib import mean, pct
A, B1, B2 = 0.0349, 0.0120e-3, 1.46e-6     # my idle-prefill fit (p06_costs.py): s, s/token, s per token per K position
def recompute(cached_mem, L):
    """extra prefill seconds for L history tokens sitting at positions [cached_mem, cached_mem+L)"""
    a, b = cached_mem, cached_mem + L
    return B1 * L + B2 * (b * b - a * a) / 2000.0
for camp, arm in [(34, "ec_full"), (34, "jit_exact"), (34, "ec_exact"), (34, "jit_p50")]:
    tab = runtab.build(camp, arm)
    reqs, reads, T, U, NR, stall = tab["reqs"], tab["reads"], tab["T"], tab["U"], tab["NR"], tab["stall"]
    ents = sorted(q["entry"] for q in reqs.values())
    ret = [rid for rid, q in reqs.items() if q["k"] >= 1 and q["t_send"] < 2400]
    rd = [rid for rid in ret if rid in stall]
    idle = []; q13 = []; q4 = []; idle_full = []
    for rid in rd:
        q = reqs[rid]
        j = bisect.bisect_left(T, q["entry"]) - 1
        nr = NR[j] if j >= 0 else 0; u = U[j] if j >= 0 else 0.0
        if nr == 0 and u < 0.90: idle.append(rid)
        elif nr == 0: idle_full.append(rid)
        elif nr <= 3: q13.append(rid)
        else: q4.append(rid)
    staged = sum(1 for rid in ret if any(reads[i]["staged"] for i in reqs[rid]["reads"]) and rid not in stall)
    print(f"=== c{camp} {arm}: first 40 minutes: returning {len(ret)}; read from the SSD after arrival {len(rd)} ({100*len(rd)/len(ret):.2f} %), staged-only {staged}")
    print(f"   arrived to: no ready turn & pool < 0.90: {len(idle)} ({100*len(idle)/len(ret):.2f} % of returning); no ready turn, pool full: {len(idle_full)}; 1-3 ready: {len(q13)}; 4+: {len(q4)}")
    for nm, grp in (("idle", idle), ("idle, pool full", idle_full), ("queue", q13 + q4)):
        if not grp: continue
        st = [stall[r]["stall"] for r in grp]
        tok = [sum(reads[i]["pages"] * 64 for i in reqs[r]["reads"] if not reads[i].get("dead")) for r in grp]
        print(f"     {nm:16s} n {len(grp):3d}: stall mean {mean(st):.2f} s median {pct(st,.5):.2f}; tokens read mean {mean(tok)/1e3:.1f} K")
    # own gain and the arrivals during the recompute, idle arrivals only
    own = 0.0; ext = 0.0; narr = []; faster = 0; rcs = []
    for r in idle:
        q = reqs[r]
        L = q["sto"] if q["sto"] else sum(reads[i]["pages"] * 64 for i in q["reads"])
        mem = q["dev"] + q["host"] - 0  # tokens served from memory sit before the read span (approximation)
        rc = recompute(max(q["cached"] - L, 0), L)
        rcs.append(rc)
        s = stall[r]["stall"]
        own += s - rc
        if rc < s: faster += 1
        # turns that enter during [entry, entry+rc): each waits the rest of the recompute (serial prefill)
        a = bisect.bisect_right(ents, q["entry"]); b = bisect.bisect_left(ents, q["entry"] + rc)
        narr.append(b - a)
        ext += sum(q["entry"] + rc - ents[i] for i in range(a, b))
    if idle:
        n = len(idle)
        print(f"   idle arrivals: recompute {mean(rcs):.2f} s against stall {mean(stall[r]['stall'] for r in idle):.2f} s; own saving {own/n:+.2f} s per request ({faster} of {n} faster); "
              f"arrivals during the recompute {mean(narr):.2f} per request, their added wait {ext/n:.2f} s per request (serial model)")
        print(f"   if every idle arrival recomputed: own {own:+.1f} TTFT-s, others {-ext:+.1f}, net {own-ext:+.1f} TTFT-s in 40 minutes = {(own-ext)/len(ret):+.4f} s per returning request; own only: {own/len(ret):+.4f} s")
