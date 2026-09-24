import pickle, csv, statistics as st, bisect
D = pickle.load(open("t2.pkl","rb"))
rows = list(csv.DictReader(open("t2_turns.csv")))
for key in [("x1","three_tier_to"),("x1","three_tier_wc")]:
    run = D[key]; t0 = run["t0"]; ev = run["ev"]
    # all turns incl turn 0, from ReqTimeStats
    iv = []
    for r in run["turns"]:
        rid=f"cmp_{key[1]}-c{r['conv']}-t{r['turn']}"; e=ev.get(rid,{})
        if not e.get("rts") or not e.get("prefetch_start"): continue
        x=e["rts"][-1][1]; P=int(r["prompt_tokens"]); T=int(e["prefetch_start"][0][1]["tokens"])
        iv.append((x["entry"], x["entry"]+x["queue"], P-1-T, P, x["entry"]+x["queue"]+x["forward"]))
    B = run["glob"]["batch"]; bt=[b[0] for b in B]
    print(key)
    print("  t(min)  queued_n  queued_private_ctx(tok)  queued_total_ctx  running_n(server)  device_token_usage  running_ctx_sum(tok)")
    for m in list(range(2, 140, 8)):
        t = t0 + 60*m
        q = [x for x in iv if x[0] <= t < x[1]]
        run_ = [x for x in iv if x[1] <= t < x[4]]
        i = bisect.bisect_right(bt, t)-1
        b = B[i] if i>=0 else None
        print(f"  {m:5d}  {len(q):8d}  {sum(max(0,x[2]-7808) for x in q):>12,}  {sum(x[2] for x in q):>12,}   {b[5] if b else '-':>6}   {b[4] if b else '-':>6}   {sum(x[3] for x in run_):>12,} (n={len(run_)})")
    # steady state throughput: minutes 15-115
    a=t0+15*60; z=t0+115*60
    adm=[x for x in iv if a<=x[1]<z]
    print(f"  steady min15-115: admissions/s={len(adm)/(z-a):.3f}, mean P at admission={st.mean(x[3] for x in adm):,.0f}, P*rate={len(adm)/(z-a)*st.mean(x[3] for x in adm):,.0f} tok/s")
    nt=sum(b[2] for b in B if a<=b[0]<z); print(f"  prefill new-token rate min15-115 = {nt/(z-a):,.0f} tok/s; batches={sum(1 for b in B if a<=b[0]<z)}")
