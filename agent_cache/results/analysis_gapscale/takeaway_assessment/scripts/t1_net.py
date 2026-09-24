import pickle, numpy as np, bisect
S = "/tmp/claude-1001/-home-wanhr-sglang/454133c8-c6f1-4d1f-8d97-1e8cda557a0d/scratchpad/assess/"
D = pickle.load(open(S + "t1.pkl", "rb")); R = pickle.load(open(S + "t1_rows.pkl", "rb")); SIM = pickle.load(open(S + "t1_sim.pkl", "rb"))
POOL = 1627648
for arm in ("three_tier_to", "three_tier_wc"):
    A = D[arm]
    E = sorted((t, int(k["tokens"])) for t, k in A["g"]["evict_host"]); et = np.array([t for t, _ in E]); ec = np.cumsum([v for _, v in E])
    def V(lo, hi):
        if hi <= lo: return 0
        a = bisect.bisect_right(et, lo); b = bisect.bisect_right(et, hi)
        return int((ec[b - 1] if b > 0 else 0) - (ec[a - 1] if a > 0 else 0))
    rr = [x for x in R[arm] if x["tier"] in ("host", "storage") and x["end_prev"] and x["send"] > et[0]]
    vv = np.array([V(x["end_prev"], x["send"]) / POOL for x in rr]); e_ = np.array([x["tier"] == "storage" for x in rr])
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 99]; P = [e_[(vv >= lo) & (vv < hi)].mean() for lo, hi in zip(bins, bins[1:])]
    pev = lambda x: P[min(bisect.bisect_right(bins, x) - 1, len(P) - 1)]
    jobs, res = SIM[arm]; H = [j for j in jobs if j["hit"]]; F = np.array([j["io_end"] - j["ps"] for j in H])
    print("=====", arm)
    for name, v in res.items():
        if v[1] is None: continue
        tt, resid, exp_ = v
        rev = np.array([pev(V(c, a) / POOL) if r > 0 else 0.0 for r, c, a in resid])
        hid = F - np.minimum(F, exp_)
        net = ((1 - rev) * hid).sum() / F.sum()
        # TTFT with re-evicted ones falling back to their actual (reactive) TTFT
        tt = np.array(tt); ttb = np.array([j["ttft"] for j in H])
        tt_exp = (1 - rev) * tt + rev * ttb
        print("  %-24s hidden %5.1f%% | expected re-evicted %4.1f%% | net hidden %5.1f%% | expected TTFT mean %.2f (all-hidden %.2f, baseline %.2f)" % (name, 100 * hid.sum() / F.sum(), 100 * rev.mean(), 100 * net, tt_exp.mean(), tt.mean(), ttb.mean()))
