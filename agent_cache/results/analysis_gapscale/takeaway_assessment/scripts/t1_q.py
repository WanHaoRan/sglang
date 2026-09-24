import pickle, numpy as np
S = "/tmp/claude-1001/-home-wanhr-sglang/454133c8-c6f1-4d1f-8d97-1e8cda557a0d/scratchpad/assess/"
D = pickle.load(open(S + "t1.pkl", "rb")); R = pickle.load(open(S + "t1_rows.pkl", "rb")); SIM = pickle.load(open(S + "t1_sim.pkl", "rb"))
for arm in ("three_tier_to", "three_tier_wc"):
    jobs, _ = SIM[arm]; sh = [x for x in R[arm] if x["tier"] == "storage"]
    tot = sum(x["io_end"] - x["ps"] for x in sh)
    print("=====", arm)
    for qc, (lo, hi) in {"q0": (0, 0), "q1-4": (1, 4), "q5-9": (5, 9), "q10+": (10, 999)}.items():
        s = [x for x in sh if lo <= x["qdepth"] <= hi]
        f = sum(x["io_end"] - x["ps"] for x in s)
        # storage reads pending (released, not finished) at this request's arrival (excluding itself)
        pend = [sum(1 for j in jobs if j["sq"] < x["ps"] < j["io_end"]) for x in s]
        print("  %-5s storage hits %3d (%.0f%%) | share of total fetch-window time %.1f%% | mean fetch window %.2f s | mean own IO %.2f s | mean #other storage reads pending at arrival %.1f | mean qdepth %.1f" % (
            qc, len(s), 100 * len(s) / len(sh), 100 * f / tot, f / max(len(s), 1), np.mean([x["io_ms"] / 1000 for x in s]), np.mean(pend), np.mean([x["qdepth"] for x in s])))
    q = np.array([x["qdepth"] for x in sh]); p = np.array([sum(1 for j in jobs if j["sq"] < x["ps"] < j["io_end"]) for x in sh])
    print("  corr(qdepth at arrival, #storage reads pending at arrival) = %.2f; mean qdepth %.2f vs mean pending reads %.2f" % (np.corrcoef(q, p)[0, 1], q.mean(), p.mean()))
    # final-turn share: end-of-reply events not followed by another turn
    turns = [r for r in D[arm]["turns"] if "t_send" in r]
    convs = {}
    for r in turns: convs.setdefault(r["conv"], []).append(r["turn"])
    print("  end-of-reply events %d, of which final turns %d (%.1f%%)" % (len(turns), len(convs), 100 * len(convs) / len(turns)))
