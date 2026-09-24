import pickle, numpy as np
S = "/tmp/claude-1001/-home-wanhr-sglang/454133c8-c6f1-4d1f-8d97-1e8cda557a0d/scratchpad/assess/"
D = pickle.load(open(S + "t1.pkl", "rb")); R = pickle.load(open(S + "t1_rows.pkl", "rb"))
def q(a, p): return float(np.percentile(a, p)) if len(a) else float('nan')
def summ(a): a = np.asarray(a, float); return "n %d mean %.3f p10 %.3f p50 %.3f p90 %.3f p99 %.3f max %.3f" % (len(a), a.mean(), q(a,10), q(a,50), q(a,90), q(a,99), a.max()) if len(a) else "n 0"
for arm in ("three_tier_to", "three_tier_wc"):
    A = D[arm]
    io = sorted([(t - float(k["ms"]) / 1000, t, int(k["pages"]), k["rid"]) for t, k in A["g"]["s2h_io"]])
    ov = sum(1 for a, b in zip(io, io[1:]) if b[0] < a[1] - 0.002)
    print("=====", arm, "s2h_io ops", len(io), "overlapping consecutive IO intervals", ov)
    zero = [x for x in io if x[2] == 0]; print(" zero-page s2h_io", len(zero))
    span = io[-1][1] - io[0][0]; busy = sum(b - a for a, b, _, _ in io)
    print(" read span %.0f s, IO busy %.0f s (%.1f%%), bytes %.1f GiB" % (span, busy, 100 * busy / span, sum(p for *_, p, _ in io) * 64 * 98304 / 2**30))
    # h2s writes
    w = sorted([(t - float(k["ms"]) / 1000, t, int(k["pages"])) for t, k in A["g"]["h2s_io"]])
    wov = sum(1 for a, b in zip(w, w[1:]) if b[0] < a[1] - 0.002)
    wbusy = sum(b - a for a, b, _ in w)
    print(" h2s writes", len(w), "overlapping", wov, "busy %.0f s, bytes %.1f GiB" % (wbusy, sum(p for *_, p in w) * 64 * 98304 / 2**30))
    # for each storage hit: in [sq, io_start], how much time the read thread was busy with other ops, and the write thread
    sh = [x for x in R[arm] if x["tier"] == "storage"]
    ib = np.array([[a, b] for a, b, _, _ in io]); wb = np.array([[a, b] for a, b, _ in w])
    def busy_in(arr, lo, hi):
        if hi <= lo: return 0.0
        s = np.clip(np.minimum(arr[:, 1], hi) - np.maximum(arr[:, 0], lo), 0, None); return float(s.sum())
    rb, wbz, wt = [], [], []
    for x in sh:
        lo, hi = x["sq"], x["io_start"]; wt.append(hi - lo)
        rb.append(busy_in(ib, lo, hi)); wbz.append(busy_in(wb, lo, hi))
    wt = np.array(wt); rb = np.array(rb); wbz = np.array(wbz)
    print(" wait sq->io_start total %.0f s; read-thread busy with other reads in that wait %.0f s (%.1f%%); write-thread busy (may overlap) %.0f s" % (wt.sum(), rb.sum(), 100 * rb.sum() / wt.sum(), wbz.sum()))
    big = wt > 1.0
    print(" for waits >1 s (n %d): read-busy share %.1f%%" % (big.sum(), 100 * rb[big].sum() / wt[big].sum()))
    # number of reads ahead in queue at sq time: reads with sq < mine and io_end > my sq
    ahead = []
    sqs = {x["conv"], } 
    allops = [(x["sq"], x["io_start"], x["io_end"]) for x in sh]
    for x in sh:
        ahead.append(sum(1 for s, a, b in allops if s < x["sq"] and b > x["sq"]))
    print(" storage reads pending/in-flight at my query time:", summ(ahead))
