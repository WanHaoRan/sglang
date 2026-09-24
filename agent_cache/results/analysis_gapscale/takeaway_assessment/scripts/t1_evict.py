import pickle, numpy as np, bisect
S = "/tmp/claude-1001/-home-wanhr-sglang/454133c8-c6f1-4d1f-8d97-1e8cda557a0d/scratchpad/assess/"
D = pickle.load(open(S + "t1.pkl", "rb")); R = pickle.load(open(S + "t1_rows.pkl", "rb")); SIM = pickle.load(open(S + "t1_sim.pkl", "rb"))
POOL = 1627648
def q(a, p): return float(np.percentile(a, p)) if len(a) else float('nan')
for arm in ("three_tier_to", "three_tier_wc"):
    A = D[arm]; t0 = A["t0"]
    E = sorted((t, int(k["tokens"])) for t, k in A["g"]["evict_host"])
    et = np.array([t for t, _ in E]); ec = np.cumsum([v for _, v in E])
    def V(lo, hi):  # host tokens evicted in (lo, hi]
        if hi <= lo: return 0
        a = bisect.bisect_right(et, lo); b = bisect.bisect_right(et, hi)
        return int((ec[b - 1] if b > 0 else 0) - (ec[a - 1] if a > 0 else 0))
    def turnover(t):  # smallest D such that V(t-D, t) >= POOL
        b = bisect.bisect_right(et, t)
        if b == 0 or ec[b - 1] < POOL: return float('inf')
        target = ec[b - 1] - POOL; a = bisect.bisect_right(ec, target)   # first index with cum > target
        return t - et[a]
    print("=====", arm, "evict_host events", len(E), "total host tokens evicted %.2fM (%.1f pools); first at +%.0f s, last at +%.0f s, run end +%.0f s" % (ec[-1] / 1e6, ec[-1] / POOL, et[0] - t0, et[-1] - t0, max(x["send"] for x in R[arm]) - t0))
    # eviction rate over time (per 5 min)
    for lo in range(0, 6000, 600):
        v = V(t0 + lo, t0 + lo + 600)
        if v: print("   +%4d-%4d s: evicted %.2fM tokens, %.0f tok/s -> turnover %.0f s" % (lo, lo + 600, v / 1e6, v / 600, POOL / (v / 600)))
    rows = R[arm]
    # validation of the LRU survival criterion on the actual run: V over the gap for host vs storage returns
    for tier in ("host", "storage", "device"):
        vv = np.array([V(x["end_prev"], x["send"]) / POOL for x in rows if x["tier"] == tier and x["end_prev"] and x["send"] > et[0]])
        if len(vv): print("  actual run, %-7s returns arriving after first host eviction: n %4d  evicted-during-gap / pool: p10 %.2f p50 %.2f p90 %.2f; share >=1.0: %.1f%%" % (tier, len(vv), q(vv, 10), q(vv, 50), q(vv, 90), 100 * (vv >= 1).mean()))
    sh = [x for x in rows if x["tier"] == "storage"]
    T = np.array([turnover(x["send"]) for x in sh]); gap = np.array([x["gap"] for x in sh])
    print("  L2 turnover time at each storage hit's arrival (s): p10 %.0f p50 %.0f p90 %.0f min %.0f; inf(no full turnover yet) %d" % (q(T[np.isfinite(T)], 10), q(T[np.isfinite(T)], 50), q(T[np.isfinite(T)], 90), T.min(), (~np.isfinite(T)).sum()))
    print("  gap / turnover for storage hits: p10 %.2f p50 %.2f p90 %.2f" % (q(gap / T, 10), q(gap / T, 50), q(gap / T, 90)))
    fetch = np.array([x["io_end"] - x["ps"] for x in sh])
    need = gap - fetch
    print("  ASAP residency need (gap - fetch window): p50 %.0f p90 %.0f s; share with need > turnover: %.1f%%" % (q(need, 50), q(need, 90), 100 * (need > T).mean()))
    jobs, res = SIM[arm]
    for name in ("oracle lead=own IO+2s", "ASAP (end of reply)", "pred e=-50% m=2s", "pred e=-25% m=2s", "pred e=-25% m=10s"):
        resid = res[name][1]
        v = np.array([V(c, a) / POOL for r, c, a in resid]); rr = np.array([r for r, c, a in resid])
        Tn = np.array([turnover(a) for r, c, a in resid])
        print("  %-24s residency p50 %6.1f p90 %6.1f s | host evicted during residency / pool p50 %.2f p90 %.2f | share >= 1 pool (surely evicted under LRU) %.1f%% | share >= 0.5 pool %.1f%% | residency > turnover %.1f%%" % (
            name, q(rr, 50), q(rr, 90), q(v, 50), q(v, 90), 100 * (v >= 1).mean(), 100 * (v >= 0.5).mean(), 100 * (rr > Tn).mean()))
    # extra L2 occupancy held by early prefetches (time-average & peak)
    for name in ("ASAP (end of reply)", "pred e=-50% m=2s", "pred e=-25% m=2s"):
        resid = res[name][1]; hits = [j for j in jobs if j["hit"]]
        evs = []
        for (r, c, a), j in zip(resid, hits):
            if r > 0: evs += [(c, j["pages"] * 64), (a, -j["pages"] * 64)]
        evs.sort(); cur = 0; peak = 0; area = 0; lt = evs[0][0]
        for t, d in evs:
            area += cur * (t - lt); lt = t; cur += d; peak = max(peak, cur)
        span = evs[-1][0] - evs[0][0]
        print("  %-24s early-prefetched tokens parked in L2 before arrival: time-avg %.0fK (%.1f%% of pool), peak %.0fK (%.1f%% of pool)" % (name, area / span / 1e3, 100 * area / span / POOL, peak / 1e3, 100 * peak / POOL))
print("\n#### empirical survival: returns (host or storage tier) by host-evicted-during-gap / pool")
for arm in ("three_tier_to", "three_tier_wc"):
    A = D[arm]; E = sorted((t, int(k["tokens"])) for t, k in A["g"]["evict_host"])
    et = np.array([t for t, _ in E]); ec = np.cumsum([v for _, v in E])
    def V(lo, hi):
        if hi <= lo: return 0
        a = bisect.bisect_right(et, lo); b = bisect.bisect_right(et, hi)
        return int((ec[b - 1] if b > 0 else 0) - (ec[a - 1] if a > 0 else 0))
    rows = [x for x in R[arm] if x["tier"] in ("host", "storage") and x["end_prev"] and x["send"] > et[0]]
    v = np.array([V(x["end_prev"], x["send"]) / POOL for x in rows]); ev_ = np.array([x["tier"] == "storage" for x in rows])
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 99]
    print(" ", arm, " ".join("[%.1f,%.1f): %d/%d=%.0f%%" % (lo, hi, ev_[(v >= lo) & (v < hi)].sum(), ((v >= lo) & (v < hi)).sum(), 100 * ev_[(v >= lo) & (v < hi)].mean() if ((v >= lo) & (v < hi)).sum() else float('nan')) for lo, hi in zip(bins, bins[1:])))
    P = {}
    for lo, hi in zip(bins, bins[1:]):
        m = (v >= lo) & (v < hi); P[(lo, hi)] = ev_[m].mean() if m.sum() else 1.0
    def pev(x):
        for (lo, hi), p in P.items():
            if lo <= x < hi: return p
        return 1.0
    jobs, res = SIM[arm]
    for name in ("oracle lead=own IO+2s", "ASAP (end of reply)", "pred e=-50% m=2s", "pred e=-25% m=2s", "pred e=-25% m=10s"):
        resid = res[name][1]
        pe = np.array([pev(V(c, a) / POOL) for r, c, a in resid])
        print("   %-24s expected share of early-prefetched contexts evicted again before arrival (empirical survival curve): %.1f%%" % (name, 100 * pe.mean()))
