import pickle, numpy as np
S = "/tmp/claude-1001/-home-wanhr-sglang/454133c8-c6f1-4d1f-8d97-1e8cda557a0d/scratchpad/assess/"
D = pickle.load(open(S + "t1.pkl", "rb")); R = pickle.load(open(S + "t1_rows.pkl", "rb")); SIM = pickle.load(open(S + "t1_sim.pkl", "rb"))
BPT = 98304; GiB = 2**30; CEIL = 1.88
def q(a, p): return float(np.percentile(a, p)) if len(a) else float('nan')
for arm in ("three_tier_to", "three_tier_wc"):
    A = D[arm]; t0 = A["t0"]; jobs, res = SIM[arm]
    hits = [j for j in jobs if j["hit"]]
    T0 = min(j["ps"] for j in jobs) - 1; T1 = max(j["io_end"] for j in jobs) + 1
    dt = 0.1; grid = np.arange(T0, T1, dt); n = len(grid)
    def add_rate(arr, lo, hi, bytes_):
        if hi <= lo: return
        a = int((lo - T0) / dt); b = int((hi - T0) / dt)
        a = max(a, 0); b = min(max(b, a + 1), n)
        arr[a:b] += bytes_ / ((b - a) * dt)
    # actual read throughput over time (bytes spread over each IO interval)
    act = np.zeros(n)
    for j in jobs: add_rate(act, j["io_end"] - j["dur"], j["io_end"], j["pages"] * 64 * BPT)
    # write throughput (h2s_io intervals)
    wr = np.zeros(n)
    for t, k in A["g"]["h2s_io"]:
        if T0 <= t <= T1: add_rate(wr, t - float(k["ms"]) / 1000, t, int(k["pages"]) * 64 * BPT)
    # concurrency of fetch windows ps->io_end
    conc = np.zeros(n)
    for j in jobs:
        a = int((j["ps"] - T0) / dt); b = int((j["io_end"] - T0) / dt); conc[max(a, 0):b] += 1
    def win(x, W):  # moving average over W seconds
        k = int(W / dt); c = np.cumsum(np.r_[0, x]); return (c[k:] - c[:-k]) / k
    tot_bytes = sum(j["pages"] * 64 * BPT for j in jobs)
    print("=====", arm, "reads %d, %.0f GiB over %.0f s (%.2f GiB/s average over read span); writes during same span %.0f GiB" % (len(jobs), tot_bytes / GiB, T1 - T0, tot_bytes / GiB / (T1 - T0), wr.sum() * dt / GiB))
    print(" fetch windows in flight (ps->io_end): max %d; time-share with >=1 in flight %.1f%%; mean when >=1: %.2f; time-share with >=3: %.1f%%" % (conc.max(), 100 * (conc >= 1).mean(), conc[conc >= 1].mean(), 100 * (conc >= 3).mean()))
    # reactive offered demand: bytes released at s2h_query must be served ASAP; demand in window W = bytes released in window / W
    for W in (10, 30, 60, 300):
        rel = np.zeros(n)
        for j in jobs:
            i = int((j["sq"] - T0) / dt); rel[min(max(i, 0), n - 1)] += j["pages"] * 64 * BPT / dt
        d = win(rel, W) / GiB; a_ = win(act, W) / GiB; w_ = win(wr, W) / GiB
        print("  W=%3ds: reactive offered read demand peak %.2f GiB/s, p99 %.2f, time-share > %.2f GiB/s: %.1f%% | achieved read tput peak %.2f | writes peak %.2f GiB/s, reads+writes peak %.2f" % (
            W, d.max(), q(d, 99), CEIL, 100 * (d > CEIL).mean(), a_.max(), w_.max(), (win(act + wr, W) / GiB).max()))
    # spread scenarios: each hit's bytes spread uniformly over [end_prev, arrival] or last fraction of gap
    for name, frac in (("spread over whole gap", 1.0), ("spread over last 50% of gap", 0.5), ("spread over last 25% of gap", 0.25), ("spread over last 10% of gap", 0.10)):
        sp = np.zeros(n)
        for j in jobs:
            B = j["pages"] * 64 * BPT
            if j["hit"]: add_rate(sp, j["ps"] - frac * j["gap"], j["ps"], B)
            else: add_rate(sp, j["io_end"] - j["dur"], j["io_end"], B)   # wasted reads kept as they happened
        spg = sp / GiB
        for W in (1, 10, 60):
            ww = win(sp, W) / GiB if W > dt else spg
            print("  %-28s W=%2ds: peak demand %.2f GiB/s, p99 %.2f, p50(when>0) %.2f, time-share > %.2f GiB/s: %.2f%%" % (name, W, ww.max(), q(ww, 99), q(ww[ww > 0], 50), CEIL, 100 * (ww > CEIL).mean()))
    # simulated read-thread busy share in the scenarios (utilisation unchanged, only timing)
