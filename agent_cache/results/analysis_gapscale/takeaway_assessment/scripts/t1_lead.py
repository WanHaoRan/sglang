import pickle, numpy as np, bisect
exec(open("/tmp/claude-1001/-home-wanhr-sglang/454133c8-c6f1-4d1f-8d97-1e8cda557a0d/scratchpad/assess/t1_sim.py").read().split("out = {}")[0])
POOL = 1627648
for arm in ("three_tier_to", "three_tier_wc"):
    A = D[arm]; rows = {f"cmp_{arm}-c{x['conv']}-t{x['turn']}": x for x in R[arm]}; ev = A["ev"]; t0 = A["t0"]
    E = sorted((t, int(k["tokens"])) for t, k in A["g"]["evict_host"]); et = np.array([t for t, _ in E]); ec = np.cumsum([v for _, v in E])
    def V(lo, hi):
        if hi <= lo: return 0
        a = bisect.bisect_right(et, lo); b = bisect.bisect_right(et, hi)
        return int((ec[b - 1] if b > 0 else 0) - (ec[a - 1] if a > 0 else 0))
    # empirical survival curve from the actual run
    rr = [x for x in R[arm] if x["tier"] in ("host", "storage") and x["end_prev"] and x["send"] > et[0]]
    vv = np.array([V(x["end_prev"], x["send"]) / POOL for x in rr]); e_ = np.array([x["tier"] == "storage" for x in rr])
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 99]; P = [e_[(vv >= lo) & (vv < hi)].mean() for lo, hi in zip(bins, bins[1:])]
    pev = lambda x: P[min(bisect.bisect_right(bins, x) - 1, len(P) - 1)]
    jobs = []
    for t, k in A["g"]["s2h_io"]:
        rid = k["rid"]; e = ev[rid]; x = rows.get(rid)
        j = dict(rid=rid, dur=float(k["ms"]) / 1000, pages=int(k["pages"]), io_end=t, sq=e["s2h_query"][0][0], ps=e["prefetch_start"][0][0], hit=bool(x and x["tier"] == "storage"))
        if j["hit"]: j.update(gap=x["gap"], end_prev=x["end_prev"], ttft=x["ttft"])
        jobs.append(j)
    base = sum(j["io_end"] - j["ps"] for j in jobs if j["hit"])
    peak = lambda j: t0 + 1200 <= j["ps"] < t0 + 1500
    print("=====", arm, " fixed lead L before arrival (not before end of reply), EDF on arrival deadline")
    for L in (1, 2, 5, 10, 20, 30, 60, 120, 1e9):
        for j in jobs:
            if j["hit"]: j["rel"] = max(j["end_prev"], j["ps"] - L); j["dl"] = j["ps"]
            else: j["rel"] = j["sq"]; j["dl"] = j["ps"]
        sim = simulate(jobs, lambda j, t: j["dl"])
        H = [(j, s) for j, s in zip(jobs, sim) if j["hit"]]
        exp_ = np.array([max(0, s - j["ps"]) for j, s in H]); tt = np.array([j["ttft"] - (j["io_end"] - j["ps"]) + max(0, s - j["ps"]) for j, s in H])
        rev = np.array([pev(V(s, j["ps"]) / POOL) if s < j["ps"] else 0.0 for j, s in H]); Fb = np.array([j["io_end"] - j["ps"] for j, s in H]); net = ((1 - rev) * (Fb - np.minimum(Fb, exp_))).sum() / Fb.sum(); xb = sum(r * j["pages"] * 64 * 98304 for r, (j, s) in zip(rev, H)) / 2**30
        pk = np.array([peak(j) for j, s in H])
        print("  net(hidden & not re-evicted) %5.1f%%, extra re-read %.0f GiB" % (100 * net, xb)); print("  L=%-6s hidden %5.1f%% | TTFT' mean %.2f p50 %.2f p90 %.2f | expected re-evicted before arrival %4.1f%% | peak phase (+1200-1500 s, n=%d): hidden %5.1f%%, re-evicted %4.1f%%" % (
            "ASAP" if L > 1e8 else f"{L:g}s", 100 * (1 - exp_.sum() / base), tt.mean(), np.median(tt), np.percentile(tt, 90), 100 * rev.mean(), pk.sum(),
            100 * (1 - exp_[pk].sum() / sum(j["io_end"] - j["ps"] for j, s in H if peak(j))), 100 * rev[pk].mean()))
