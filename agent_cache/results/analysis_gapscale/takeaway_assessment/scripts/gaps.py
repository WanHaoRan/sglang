import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from arrival_loss import load, pct
for scale, arm in [("x70","three_tier_to"),("x70","three_tier_wc"),("x10","three_tier_to"),("x1","three_tier_to")]:
    run, ev, evict, turns, t0 = load(scale, arm)
    l3, no = [], []
    for r in turns:
        if r["turn"] < 1: continue
        e = ev.get(f"cmp_{arm}-c{r['conv']}-t{r['turn']}", {})
        if "pf_tok" not in e: continue
        (l3 if e.get("loaded",0) > 0 else no).append(r["gap_slept"])
    allg = l3 + no
    print(f"{scale} {arm}: gap_slept all p50 {pct(allg,50):.1f}s p90 {pct(allg,90):.1f}; L3-loaded turns n={len(l3)} gap p10 {pct(l3,10):.1f} p50 {pct(l3,50):.1f} p90 {pct(l3,90):.1f}; others p50 {pct(no,50):.1f}")
    # share of returns whose gap exceeded 60s / 120s and were L3-loaded
    for th in (60, 120, 240):
        g = [x for x in allg if x >= th]
        gl = [x for x in l3 if x >= th]
        print(f"   gap>={th}s: {len(g)} turns, L3-loaded {len(gl)}")
