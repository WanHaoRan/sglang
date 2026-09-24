import pickle, numpy as np
exec(open("/tmp/claude-1001/-home-wanhr-sglang/454133c8-c6f1-4d1f-8d97-1e8cda557a0d/scratchpad/assess/t1_sim.py").read().split("out = {}")[0])
POOL = 1627648
for arm in ("three_tier_to", "three_tier_wc"):
    A = D[arm]; rows = {f"cmp_{arm}-c{x['conv']}-t{x['turn']}": x for x in R[arm]}; ev = A["ev"]; t0 = A["t0"]
    jobs = []
    for t, k in A["g"]["s2h_io"]:
        rid = k["rid"]; e = ev[rid]; x = rows.get(rid)
        j = dict(rid=rid, dur=float(k["ms"]) / 1000, pages=int(k["pages"]), io_end=t, sq=e["s2h_query"][0][0], ps=e["prefetch_start"][0][0], hit=bool(x and x["tier"] == "storage"))
        if j["hit"]: j.update(gap=x["gap"], end_prev=x["end_prev"])
        jobs.append(j)
    for L in (10, 20):
        for j in jobs:
            j["rel"] = max(j["end_prev"], j["ps"] - L) if j["hit"] else j["sq"]; j["dl"] = j["ps"]
        sim = simulate(jobs, lambda j, t: j["dl"])
        lo, hi = t0 + 900, t0 + 2100
        area = sum(j["pages"] * 64 * max(0, min(hi, j["ps"]) - max(lo, s)) for j, s in zip(jobs, sim) if j["hit"] and s < j["ps"])
        # peak parked
        evs = sorted([(s, j["pages"] * 64) for j, s in zip(jobs, sim) if j["hit"] and s < j["ps"]] + [(j["ps"], -j["pages"] * 64) for j, s in zip(jobs, sim) if j["hit"] and s < j["ps"]])
        cur = 0; pk = 0
        for t, d in evs: cur += d; pk = max(pk, cur)
        print(arm, "L=%ds: tokens parked in L2 awaiting arrival, busy phase +900-2100 s time-avg %.0fK (%.1f%% of pool), peak %.0fK (%.1f%%)" % (L, area / (hi - lo) / 1e3, 100 * area / (hi - lo) / POOL, pk / 1e3, 100 * pk / POOL))
