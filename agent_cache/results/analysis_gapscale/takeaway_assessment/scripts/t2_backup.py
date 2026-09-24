import pickle
D = pickle.load(open("t2.pkl","rb"))
for k, run in D.items():
    g = run["glob"]
    eh = g["evict_host"]
    short = [x for x in eh if x[1] < x[2]]
    newtok = sum(b[2] for b in g["batch"]); cachedtok = sum(b[3] for b in g["batch"])
    d2h = sum(x[1] for x in g["d2h_done"]); h2s = sum(x[1] for x in g["h2s_done"])
    outs = sum(int(r.get("completion_tokens") or 0) for r in run["turns"])
    print(k, f"evict_host events={len(eh)} short(tokens<requested)={len(short)} short_deficit={sum(x[2]-x[1] for x in short):,}; prefill new-token sum={newtok:,} cached-token sum={cachedtok:,}; completion tokens={outs:,}; d2h_done tokens={d2h:,}; h2s_done tokens={h2s:,}")
