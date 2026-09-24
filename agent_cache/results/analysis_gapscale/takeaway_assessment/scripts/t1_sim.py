import pickle, numpy as np, bisect, heapq
S = "/tmp/claude-1001/-home-wanhr-sglang/454133c8-c6f1-4d1f-8d97-1e8cda557a0d/scratchpad/assess/"
D = pickle.load(open(S + "t1.pkl", "rb")); R = pickle.load(open(S + "t1_rows.pkl", "rb"))
POOL = 1627648; BPT = 98304; GiB = 2**30; BW = 1.88 * GiB
def q(a, p): return float(np.percentile(a, p)) if len(a) else float('nan')
def summ(a): a = np.asarray(a, float); return "mean %.2f p50 %.2f p90 %.2f p99 %.2f" % (a.mean(), q(a,50), q(a,90), q(a,99))

def simulate(jobs, policy):
    """single non-preemptive read thread. jobs: dict with rel (release), dur, key(t) -> priority."""
    n = len(jobs); order = sorted(range(n), key=lambda i: jobs[i]["rel"]); done = [None] * n
    t = 0.0; k = 0; ready = []
    while k < n or ready:
        if not ready:
            t = max(t, jobs[order[k]]["rel"])
        while k < n and jobs[order[k]]["rel"] <= t + 1e-12:
            ready.append(order[k]); k += 1
        # pick by priority evaluated at time t
        best = min(ready, key=lambda i: policy(jobs[i], t))
        ready.remove(best)
        t = t + jobs[best]["dur"]; done[best] = t
    return done

out = {}
for arm in ("three_tier_to", "three_tier_wc"):
    A = D[arm]; rows = {f"cmp_{arm}-c{x['conv']}-t{x['turn']}": x for x in R[arm]}
    ev = A["ev"]
    jobs = []
    for t, k in A["g"]["s2h_io"]:
        rid = k["rid"]; e = ev[rid]; x = rows.get(rid)
        j = dict(rid=rid, dur=float(k["ms"]) / 1000, pages=int(k["pages"]), io_end=t, sq=e["s2h_query"][0][0], ps=e["prefetch_start"][0][0])
        j["hit"] = bool(x and x["tier"] == "storage")
        if j["hit"]:
            j.update(gap=x["gap"], end_prev=x["end_prev"], send=x["send"], ttft=x["ttft"], lb=x["lb"], queue=x["queue"])
        jobs.append(j)
    hits = [j for j in jobs if j["hit"]]
    # validation: FIFO with release = sq
    for j in jobs: j["rel"] = j["sq"]
    sim = simulate(jobs, lambda j, t: j["rel"])
    err = np.array([s - j["io_end"] for s, j in zip(sim, jobs)])
    print("=====", arm, "jobs", len(jobs), "hits", len(hits))
    print(" validation FIFO(release=s2h_query): sim io_end - actual: p50 %.3f p90 %.3f max|.| %.3f; mean fetch window sim %.3f vs actual %.3f" % (
        q(err, 50), q(err, 90), np.abs(err).max(), np.mean([s - j["ps"] for s, j in zip(sim, jobs) if j["hit"]]), np.mean([j["io_end"] - j["ps"] for j in hits])))
    # reference: time from fetch completion to TTFT
    def ttft_cf(j, c):
        # c = simulated completion; arrival at scheduler = ps; exposed = max(0, c - ps)
        return j["ttft"] - (j["io_end"] - j["ps"]) + max(0.0, c - j["ps"])
    res = {}
    base = [ttft_cf(j, s) for s, j in zip(sim, jobs) if j["hit"]]
    res["baseline(sim)"] = (base, None)
    tokens = np.array([j["pages"] * 64 for j in jobs])
    Fhat_bw = {id(j): j["pages"] * 64 * BPT / BW for j in jobs}   # estimate from size and 1.88 GiB/s
    scen = [("oracle lead=own IO", "oracle", 0.0, 0.0), ("oracle lead=own IO+2s", "oracle", 0.0, 2.0), ("ASAP (end of reply)", "asap", 0.0, 0.0)]
    for e in (-0.5, -0.25, 0.25, 0.5):
        for m in (0.0, 2.0, 10.0):
            scen.append((f"pred e={e:+.0%} m={m:g}s", "pred", e, m))
    for name, kind, e, m in scen:
        for j in jobs:
            if not j["hit"]:
                j["rel"] = j["sq"]; j["dl"] = j["ps"]; continue
            a = j["ps"]
            if kind == "oracle":
                rel = max(j["end_prev"], a - Fhat_bw[id(j)] - m); dl = a
            elif kind == "asap":
                rel = j["end_prev"]; dl = a
            else:
                ghat = j["gap"] * (1 + e)
                rel = j["end_prev"] + max(0.0, ghat - Fhat_bw[id(j)] - m); dl = j["end_prev"] + ghat
                if rel >= a: rel = j["sq"]; dl = a     # too late: reactive fetch at arrival
            j["rel"] = rel; j["dl"] = dl
        # EDF: deadline = predicted deadline, but a request that has actually arrived gets deadline min(dl, ps)
        pol = lambda j, t: (min(j["dl"], j["ps"]) if t >= j["ps"] else j["dl"])
        sim2 = simulate(jobs, pol)
        tt = [ttft_cf(j, s) for s, j in zip(sim2, jobs) if j["hit"]]
        exp_ = np.array([max(0.0, s - j["ps"]) for s, j in zip(sim2, jobs) if j["hit"]])
        resid = [(max(0.0, j["ps"] - s), s, j["ps"]) for s, j in zip(sim2, jobs) if j["hit"]]
        res[name] = (tt, resid, exp_)
    base_exp = np.array([j["io_end"] - j["ps"] for j in hits])
    print(" actual TTFT", summ([j["ttft"] for j in hits]), "; actual fetch window total %.0f s" % base_exp.sum())
    for name, v in res.items():
        tt = v[0]
        if v[1] is None:
            print(" %-26s TTFT' %s" % (name, summ(tt))); continue
        resid = np.array([r[0] for r in v[1]]); exp_ = v[2]
        print(" %-26s TTFT' %s | fetch wait hidden %.1f%% | fully hidden turns %.1f%% | residency p50 %.1f p90 %.1f max %.1f s" % (
            name, summ(tt), 100 * (1 - exp_.sum() / base_exp.sum()), 100 * (exp_ <= 1e-6).mean(), q(resid, 50), q(resid, 90), resid.max()))
    out[arm] = (jobs, res)
pickle.dump(out, open(S + "t1_sim.pkl", "wb"))
