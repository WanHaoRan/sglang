import sys, os, statistics as st
sys.path.insert(0, os.path.dirname(__file__))
from arrival_loss import load, pct
for arm in ["three_tier_to","three_tier_wc"]:
    run, ev, evict, turns, t0 = load("x70", arm)
    allt=[]; l3=[]; hidden=[]
    for r in turns:
        if r["turn"]<1 or r.get("ttft") is None: continue
        e=ev.get(f"cmp_{arm}-c{r['conv']}-t{r['turn']}",{})
        allt.append(r["ttft"])
        if e.get("loaded",0)>0 and "ld_t" in e:
            l3.append(r["ttft"]); hidden.append(min(r["ttft"], e["ld_t"]-e["pf_t"]))
    m=st.mean(allt)
    print(f"x70 {arm}: returning n={len(allt)} mean TTFT {m:.2f}s p50 {pct(allt,50):.2f}; L3-loaded n={len(l3)} ({100*len(l3)/len(allt):.1f}%) mean TTFT {st.mean(l3):.2f}s; sum of exposed L3 time / n_all = {sum(hidden)/len(allt):.2f}s -> mean TTFT if fully hidden <= {m-sum(hidden)/len(allt):.2f}s")
