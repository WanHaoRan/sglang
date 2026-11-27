"""Queue rebuild checks, c by regime, readers' stall and the pre-arrival ceiling min(gap, stall) (ev_fetch 2c,
ev_prices 1a/1b/1e), with my own rules (runtab.py)."""
import sys, os, bisect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib, runtab
from vlib import kv, pct, mean

def tavg(T, V, lo, hi):
    i = bisect.bisect_left(T, lo); tot = 0.0; prev = lo
    v = V[i - 1] if i > 0 else 0
    while i < len(T) and T[i] < hi:
        tot += v * (T[i] - prev); prev = T[i]; v = V[i]; i += 1
    tot += v * (hi - prev)
    return tot / (hi - lo)

arms = [(34, "ec_full"), (34, "jit_exact"), (34, "ec_exact"), (34, "jit_p50")]
if len(sys.argv) > 1:
    arms = [(int(a.split(":")[0]), a.split(":")[1]) for a in sys.argv[1:]]
for camp, arm in arms:
    tab = runtab.build(camp, arm)
    ev = vlib.load(camp, arm)
    reqs, reads, T, U, M, NR, NQ, stall = (tab[k] for k in ("reqs", "reads", "T", "U", "M", "NR", "NQ", "stall"))
    print(f"===== c{camp} {arm}")
    # (1) rebuilt queue length against the logged #queue-req on batch lines
    ok = near = n = 0
    for t, kind, running, queue, usage, _, _ in vlib.batches(ev):
        if t < 0 or t > 7200: continue
        j = bisect.bisect_right(T, t) - 1
        if j < 0: continue
        n += 1
        # the batch line itself is in T; take the value just at it
        if NQ[j] == queue: ok += 1
        if abs(NQ[j] - queue) <= 1: near += 1
    print(f"  rebuilt waiting count == logged #queue-req on {100*ok/n:.1f} % of {n} batch lines (within 1: {100*near/n:.1f} %)")
    # (2) position at arrival against ntp_arrive pos
    ent = sorted((q["entry"], rid) for rid, q in reqs.items())
    okp = nearp = npos = 0
    for t, r in ev.get("ntp_arrive", []):
        d = kv(r); q = reqs.get(d["rid"])
        if q is None: continue
        # my position: waiting turns with an earlier entry at this turn's entry
        pos = sum(1 for rid2, q2 in ()) if False else None
        npos += 1
    # cheap position: count of turns with entry < e and adm > e
    adm_sorted = sorted(q["adm"] for q in reqs.values()); ent_sorted = sorted(q["entry"] for q in reqs.values())
    diffs = []
    for t, r in ev.get("ntp_arrive", []):
        d = kv(r); q = reqs.get(d["rid"])
        if q is None: continue
        e = q["entry"]
        mypos = bisect.bisect_left(ent_sorted, e) - bisect.bisect_right(adm_sorted, e)
        diffs.append(mypos - int(d["pos"]))
    if diffs:
        print(f"  position at arrival == ntp_arrive pos for {100*sum(1 for x in diffs if x==0)/len(diffs):.1f} % of {len(diffs)} planned turns (within 1: {100*sum(1 for x in diffs if abs(x)<=1)/len(diffs):.1f} %)")
    # (3) by regime
    wins = [(0, 8), (8, 28), (28, 70), (70, 90), (90, 110), (0, 30), (30, 70), (70, 110)]
    adm_list = sorted((q["adm"], rid) for rid, q in reqs.items())
    print("  minutes | adm | c_adm (no ready turn left after the admission) | c_adm & pool<0.90 | queue: ready, all (time avg) | users running B | readers % of returning | stall s/returning | c_stall | ceiling min(gap,stall) s/returning | c-weighted")
    dec = [(t, running) for t, kind, running, queue, usage, _, _ in vlib.batches(ev) if kind == "D"]
    for lo, hi in wins:
        a = lo * 60; b = hi * 60
        nadm = cfree = croom = 0
        for t, rid in adm_list:
            if a <= t < b:
                j = bisect.bisect_right(T, t) - 1
                nadm += 1
                if NR[j] == 0:
                    cfree += 1
                    if U[j] < 0.90: croom += 1
        ret = [q for q in reqs.values() if q["k"] >= 1 and a <= q["t_send"] < b]
        rd = [(rid, q) for rid, q in reqs.items() if q["k"] >= 1 and a <= q["t_send"] < b and rid in stall]
        st = sum(stall[rid]["stall"] for rid, q in rd)
        fr = sum(stall[rid]["free"] for rid, q in rd)
        ceil = sum(min(q["wait_s"] or 0.0, stall[rid]["stall"]) for rid, q in rd)
        ceil_c = sum(min(q["wait_s"] or 0.0, stall[rid]["stall"]) * (stall[rid]["free"] / stall[rid]["stall"] if stall[rid]["stall"] > 0 else 0) for rid, q in rd)
        B = mean([r for t, r in dec if a <= t < b])
        print(f"  {lo:3d}-{hi:<3d} | {nadm:5d} | {cfree/max(nadm,1):.3f} | {croom/max(nadm,1):.3f} | {tavg(T,NR,a,b):5.1f} {tavg(T,NQ,a,b):5.1f} | {B:4.1f} | {100*len(rd)/max(len(ret),1):5.1f} % ({len(rd)} of {len(ret)}) | {st/max(len(ret),1):6.3f} | {fr/max(st,1e-9):.3f} | {ceil/max(len(ret),1):.4f} | {ceil_c/max(len(ret),1):.4f}")
