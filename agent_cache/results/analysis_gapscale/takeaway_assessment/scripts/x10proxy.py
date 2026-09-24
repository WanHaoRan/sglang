import sys, os, bisect
sys.path.insert(0, os.path.dirname(__file__))
from arrival_loss import load, pct
for scale, arm in [("x10","three_tier_to"),("x10","three_tier_wc"),("x1","three_tier_to")]:
    run, ev, evict, turns, t0 = load(scale, arm)
    et=[t for t,_,_ in evict]; cum=[0]
    for _,tok,_ in evict: cum.append(cum[-1]+tok)
    rows=[]
    for r in turns:
        if r["turn"]<1: continue
        e=ev.get(f"cmp_{arm}-c{r['conv']}-t{r['turn']}",{})
        if "pf_tok" not in e: continue
        A=r["prompt_tokens"]-1-e["pf_tok"]; C=r.get("cached_tokens") or 0
        lost=max(0,A+e.get("loaded",0)-C)
        ft=e["pf_t"]+r["ttft"]   # first-token time (upper bound on admission)
        i,j=bisect.bisect_left(et,e["pf_t"]),bisect.bisect_right(et,ft)
        rows.append((r["ttft"],lost>=1024,cum[j]-cum[i],r["t_send"]))
    print(scale,arm)
    s=[]
    for lo,hi in [(0,10),(10,30),(30,60),(60,120),(120,1e9)]:
        g=[x for x in rows if lo<=x[0]<hi]
        if g: s.append(f"ttft[{lo},{hi if hi<1e9 else 'inf'}): lost {sum(x[1] for x in g)}/{len(g)}")
    print("  ","  ".join(s))
    s=[]
    for lo,hi in [(0,1e5),(1e5,3e5),(3e5,5e5),(5e5,1e6),(1e6,1e12)]:
        g=[x for x in rows if lo<=x[2]<hi]
        if g: s.append(f"evicted(arrival..first token)[{lo:.0e},{hi:.0e}): lost {sum(x[1] for x in g)}/{len(g)}")
    print("  ","  ".join(s))
    early=[x for x in rows if x[3]<300]; late=[x for x in rows if x[3]>=300]
    print(f"   t_send<300s: lost {sum(x[1] for x in early)}/{len(early)};  t_send>=300s: lost {sum(x[1] for x in late)}/{len(late)}")
