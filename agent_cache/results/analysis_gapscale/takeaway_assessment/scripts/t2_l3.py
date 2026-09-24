import pickle, statistics as st
D = pickle.load(open("t2.pkl","rb"))
for key, run in D.items():
    t0=run["t0"]; ev=run["ev"]; out=[]
    for r in run["turns"]:
        if r["turn"]==0: continue
        rid=f"cmp_{key[1]}-c{r['conv']}-t{r['turn']}"; e=ev.get(rid,{})
        if not e.get("s2h_io"): continue
        a=e["prefetch_start"][0][0]; io=e["s2h_io"][0][0]; pf=e["pf_result"][0][0] if e.get("pf_result") else None
        loaded=e["pf_result"][0][1]["loaded"] if e.get("pf_result") else 0
        adm = e["rts"][-1][1]["entry"]+e["rts"][-1][1]["queue"] if e.get("rts") else (e["load_back_init"][-1][0] if e.get("load_back_init") else None)
        P=int(r["prompt_tokens"]); C=int(r["cached_tokens"]); T=int(e["prefetch_start"][0][1]["tokens"]); m=P-1-T
        lost = (m+loaded-C) > 64
        pages=int(e["s2h_io"][0][1]["pages"]); ms=float(e["s2h_io"][0][1]["ms"])
        out.append(dict(rid=rid, t=a-t0, io=io-a, pf=(pf-a) if pf else None, adm=(adm-a) if adm else None, P=P, m=m, loaded=loaded, C=C, lost=lost, gibs=pages*64*98304/2**30/(ms/1000) if ms>0 else None, ttft=r["ttft"], lag=a-(t0+r["t_send"])))
    L=[o for o in out if o["lost"]]; K=[o for o in out if not o["lost"]]
    print(key, f"L3 reads={len(out)} lost_after_prefetch={len(L)} kept={len(K)}; s2h_io GiB/s p50={st.median([o['gibs'] for o in out if o['gibs']]):.2f}")
    print("   io_done-arrival p50 %.2fs; prefetch-inserted(success log)-arrival: lost p50 %s / kept p50 %s" % (st.median([o['io'] for o in out]), st.median([o['pf'] for o in L]) if L else None, st.median([o['pf'] for o in K]) if K else None))
    for o in sorted(out, key=lambda o:o["t"]):
        print(f"   {o['rid']:32s} arr@{o['t']:7.0f}s io+{o['io']:5.2f} inserted+{o['pf']:7.2f} adm+{o['adm'] if o['adm'] is None else round(o['adm'],2)} P={o['P']} m_arr={o['m']} loaded={o['loaded']} C={o['C']} lost={o['lost']} ttft={o['ttft']:.1f} prelag={o['lag']:.1f}")
