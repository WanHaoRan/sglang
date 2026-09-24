import csv, pickle, statistics as st
D = pickle.load(open("t2.pkl","rb"))
rows = list(csv.DictReader(open("t2_turns.csv")))
f=lambda x: None if x in ("",None) else float(x)
def pct(v,p): v=sorted(v); return v[min(len(v)-1,int(p/100*len(v)))]
for scale in ("x10","x1"):
    for arm in ("three_tier_to","three_tier_wc"):
        R=[r for r in rows if r["scale"]==scale and r["arm"]==arm]
        for r in R:
            for k in ("P","C","m_arr","delta","queue","ttft","entry","t_send","pf_loaded","io_pages"): r[k]=f(r[k])
        full_prev=[r for r in R if r["delta"] is not None and r["m_arr"] >= r["P"]-r["delta"]-64]
        rec=[r for r in R if r["C"] < 0.5*r["P"]]
        rec_prev=[r for r in rec if r["delta"] is not None and r["m_arr"] >= r["P"]-r["delta"]-64]
        print(f"{scale} {arm}: n={len(R)}; whole previous context (P-delta, within 64) matched in L1+L2 at arrival: {len(full_prev)} ({len(full_prev)/len(R)*100:.1f}%); recomputed={len(rec)}, of which whole prev ctx matched at arrival: {len(rec_prev)} ({len(rec_prev)/len(rec)*100:.1f}%)")
        print(f"   abs tokens: m_arr p50={pct([r['m_arr'] for r in R],50):,.0f}, C p50={pct([r['C'] for r in R],50):,.0f}; among recomputed: m_arr p50={pct([r['m_arr'] for r in rec],50):,.0f} C p50={pct([r['C'] for r in rec],50):,.0f} P p50={pct([r['P'] for r in rec],50):,.0f}")
        L3=[r for r in R if r["io_pages"] and r["io_pages"]>0]
        lostL3=[r for r in L3 if r["m_arr"]+r["pf_loaded"]-r["C"]>64]
        print(f"   L3 reads={len(L3)} lost={len(lostL3)} tokens read from SSD then discarded={sum(r['pf_loaded'] for r in lostL3):,.0f} of {sum(r['pf_loaded'] for r in L3):,.0f}")
        if scale=="x1":
            run=D[(scale,arm)]; t0=run["t0"]
            lag=[r["entry"]-(t0+r["t_send"]) for r in R if r["entry"]]
            q=[r["queue"] for r in R if r["queue"] is not None]
            print(f"   mean ttft={st.mean(r['ttft'] for r in R):.1f}s = pre-arrival lag {st.mean(lag):.1f}s + server queue {st.mean(q):.1f}s + rest {st.mean(r['ttft'] for r in R)-st.mean(lag)-st.mean(q):.1f}s")
