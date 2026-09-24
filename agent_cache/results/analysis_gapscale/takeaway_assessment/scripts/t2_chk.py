import pickle, statistics as st
D = pickle.load(open("t2.pkl","rb"))
for k, run in D.items():
    t0 = run["t0"]; ev = run["ev"]; d1=[]; d2=[]; d3=[]
    for r in run["turns"]:
        if r["turn"]==0: continue
        rid=f"cmp_{k[1]}-c{r['conv']}-t{r['turn']}"; e=ev.get(rid,{})
        send=t0+r["t_send"]
        if e.get("prefetch_start"): d1.append(e["prefetch_start"][0][0]-send)
        if e.get("rts"):
            d2.append(e["rts"][-1][1]["entry"]-send)
            if e.get("prefetch_start"): d3.append(e["rts"][-1][1]["entry"]-e["prefetch_start"][0][0])
    q=lambda v: [round(x,3) for x in (st.quantiles(v,n=20)[0], st.median(v), st.quantiles(v,n=20)[-1], max(v))] if v else None
    print(k, "ps-send p5/p50/p95/max", q(d1), "entry-send", q(d2), "entry-ps", q(d3))
