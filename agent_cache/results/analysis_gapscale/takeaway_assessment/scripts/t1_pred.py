import pickle, numpy as np, bisect
exec(open("/tmp/claude-1001/-home-wanhr-sglang/454133c8-c6f1-4d1f-8d97-1e8cda557a0d/scratchpad/assess/t1_sim.py").read().split("out = {}")[0])
POOL = 1627648
for arm in ("three_tier_to", "three_tier_wc"):
    A = D[arm]; ev = A["ev"]
    turns = {(r["conv"], r["turn"]): r for r in A["turns"] if "t_send" in r and r.get("ttft")}
    # predictors
    pred = {}
    for (c, t), r in turns.items():
        if t < 2: continue
        prev = [turns[(c, u)]["gap_slept"] for u in range(1, t) if (c, u) in turns]
        if not prev: continue
        pred[(c, t)] = dict(prev=prev[-1], med=float(np.median(prev)), g=r["gap_slept"])
    rows = {f"cmp_{arm}-c{x['conv']}-t{x['turn']}": x for x in R[arm]}
    shk = {(x["conv"], x["turn"]) for x in R[arm] if x["tier"] == "storage"}
    print("=====", arm)
    allg = np.array([turns[k]["gap_slept"] for k in turns if k[1] > 0])
    print(" all return gaps (s): p10 %.1f p50 %.1f p90 %.1f; share at 1200 s cap %.1f%%; share with gap in [49,53] s: %.1f%%" % (np.percentile(allg, 10), np.median(allg), np.percentile(allg, 90), 100 * (allg >= 1199.9).mean(), 100 * ((allg >= 49) & (allg <= 53)).mean()))
    for name in ("prev", "med"):
        for sub, keys in (("all returns t>=2", list(pred)), ("storage-hit turns", [k for k in pred if k in shk])):
            e = np.array([pred[k][name] / pred[k]["g"] - 1 for k in keys if pred[k]["g"] > 0])
            print("  predictor=%-4s %-18s n %4d: rel.err p10 %+.0f%% p50 %+.0f%% p90 %+.0f%% | |e|<=25%%: %.1f%% | under-est (e<0): %.1f%% | over-est >+25%%: %.1f%%" % (
                name, sub, len(e), 100 * np.percentile(e, 10), 100 * np.median(e), 100 * np.percentile(e, 90), 100 * (np.abs(e) <= 0.25).mean(), 100 * (e < 0).mean(), 100 * (e > 0.25).mean()))
    # policy simulation with prev-gap predictor
    E = sorted((t, int(k["tokens"])) for t, k in A["g"]["evict_host"]); et = np.array([t for t, _ in E]); ec = np.cumsum([v for _, v in E])
    def V(lo, hi):
        if hi <= lo: return 0
        a = bisect.bisect_right(et, lo); b = bisect.bisect_right(et, hi)
        return int((ec[b - 1] if b > 0 else 0) - (ec[a - 1] if a > 0 else 0))
    rr = [x for x in R[arm] if x["tier"] in ("host", "storage") and x["end_prev"] and x["send"] > et[0]]
    vv = np.array([V(x["end_prev"], x["send"]) / POOL for x in rr]); e_ = np.array([x["tier"] == "storage" for x in rr])
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 99]; P = [e_[(vv >= lo) & (vv < hi)].mean() for lo, hi in zip(bins, bins[1:])]
    pev = lambda x: P[min(bisect.bisect_right(bins, x) - 1, len(P) - 1)]
    jobs = []
    for t, k in A["g"]["s2h_io"]:
        rid = k["rid"]; e = ev[rid]; x = rows.get(rid)
        j = dict(rid=rid, dur=float(k["ms"]) / 1000, pages=int(k["pages"]), io_end=t, sq=e["s2h_query"][0][0], ps=e["prefetch_start"][0][0], hit=bool(x and x["tier"] == "storage"))
        if j["hit"]: j.update(gap=x["gap"], end_prev=x["end_prev"], ttft=x["ttft"], key=(x["conv"], x["turn"]))
        jobs.append(j)
    base = sum(j["io_end"] - j["ps"] for j in jobs if j["hit"])
    for name in ("prev", "med"):
        for sfac in (0.0, 0.25, 0.5):
            L = 10.0
            for j in jobs:
                if j["hit"] and j["key"] in pred:
                    gh = pred[j["key"]][name] * (1 - sfac); rel = j["end_prev"] + max(0.0, gh - L)
                    if rel >= j["ps"]: rel = j["sq"]
                    j["rel"] = rel; j["dl"] = min(j["end_prev"] + gh, j["ps"]) if rel < j["ps"] else j["ps"]
                else: j["rel"] = j["sq"]; j["dl"] = j["ps"]
            sim = simulate(jobs, lambda j, t: (min(j["dl"], j["ps"]) if t >= j["ps"] else j["dl"]))
            H = [(j, s) for j, s in zip(jobs, sim) if j["hit"]]
            exp_ = np.array([max(0, s - j["ps"]) for j, s in H]); tt = np.array([j["ttft"] - (j["io_end"] - j["ps"]) + max(0, s - j["ps"]) for j, s in H])
            rev = np.array([pev(V(s, j["ps"]) / POOL) if s < j["ps"] else 0.0 for j, s in H]); Fb = np.array([j["io_end"] - j["ps"] for j, s in H]); net = ((1 - rev) * (Fb - np.minimum(Fb, exp_))).sum() / Fb.sum(); xb = sum(r * j["pages"] * 64 * 98304 for r, (j, s) in zip(rev, H)) / 2**30; early = np.array([s < j["ps"] - 1e-9 for j, s in H])
            print("  net(hidden & not re-evicted) %5.1f%%, extra re-read %.0f GiB" % (100 * net, xb)); print("  policy: start at end_of_reply + max(0, %s_gap*(1-%.2f) - 10s): hidden %5.1f%% | TTFT' mean %.2f p50 %.2f p90 %.2f | fetched before arrival %.1f%% | expected re-evicted %.1f%% (of all hits)" % (
                name, sfac, 100 * (1 - exp_.sum() / base), tt.mean(), np.median(tt), np.percentile(tt, 90), 100 * early.mean(), 100 * rev.mean()))
