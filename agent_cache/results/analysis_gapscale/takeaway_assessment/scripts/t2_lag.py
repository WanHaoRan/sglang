import pickle, statistics as st, bisect
D = pickle.load(open("t2.pkl","rb"))
for k, run in D.items():
    t0 = run["t0"]; ev = run["ev"]
    iv = [(r["t_send"], r["t_send"]+r["latency"]) for r in run["turns"] if "t_send" in r and r.get("latency")]
    starts = sorted(a for a,b in iv); ends = sorted(b for a,b in iv)
    bins = {}
    for r in run["turns"]:
        if r["turn"]==0 or "t_send" not in r: continue
        rid=f"cmp_{k[1]}-c{r['conv']}-t{r['turn']}"; e=ev.get(rid,{})
        if not e.get("prefetch_start"): continue
        s=r["t_send"]
        inflight = bisect.bisect_left(starts, s) - bisect.bisect_right(ends, s)  # sent before s and not yet finished
        lag = e["prefetch_start"][0][0]-(t0+s)
        b = min(inflight//10*10, 120)
        bins.setdefault(b, []).append(lag)
    print(k)
    for b in sorted(bins):
        v=bins[b]; print(f"  inflight_before_send {b:3d}-{b+9:3d}: n={len(v):4d} lag p50={st.median(v):6.2f}s mean={st.mean(v):6.2f}s share>1s={sum(x>1 for x in v)/len(v):.2f}")
