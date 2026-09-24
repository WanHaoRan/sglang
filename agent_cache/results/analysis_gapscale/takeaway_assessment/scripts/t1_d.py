import pickle, numpy as np
S = "/tmp/claude-1001/-home-wanhr-sglang/454133c8-c6f1-4d1f-8d97-1e8cda557a0d/scratchpad/assess/"
R = pickle.load(open(S + "t1_rows.pkl", "rb"))
def med(a): return float(np.median(a)) if len(a) else float('nan')
for arm in ("three_tier_to", "three_tier_wc"):
    rows = R[arm]
    sh = [x for x in rows if x["tier"] == "storage"]; hh = [x for x in rows if x["tier"] == "host"]
    print("=====", arm, "(1) storage-hit vs host-hit TTFT by 4K prompt bin  [bin: n_sto med/mean | n_host med/mean | excess med, mean | per-turn fetch-in-queue med/mean]")
    tot_ex = []; 
    for b in range(0, 56000, 4000):
        s = [x for x in sh if b <= x["prompt"] < b + 4000]; h = [x for x in hh if b <= x["prompt"] < b + 4000]
        if not s: continue
        fq = [min(x["queue"], x["io_end"] - x["send"]) for x in s]
        print(" %2d-%2dK: sto n %3d TTFT med %6.2f mean %6.2f | host n %3d med %6.2f mean %6.2f | excess med %6.2f mean %6.2f | fetch-in-queue med %6.2f mean %6.2f" % (
            b // 1000, (b + 4000) // 1000, len(s), med([x["ttft"] for x in s]), np.mean([x["ttft"] for x in s]), len(h), med([x["ttft"] for x in h]) if h else np.nan,
            np.mean([x["ttft"] for x in h]) if h else np.nan, med([x["ttft"] for x in s]) - (med([x["ttft"] for x in h]) if h else np.nan),
            np.mean([x["ttft"] for x in s]) - (np.mean([x["ttft"] for x in h]) if h else np.nan), med(fq), np.mean(fq)))
        if h:
            hm = np.mean([x["ttft"] for x in h]); hmed = med([x["ttft"] for x in h])
            for x in s: tot_ex.append((x["ttft"] - hm, x["ttft"] - hmed, x["ttft"]))
    ex = np.array(tot_ex)
    print(" pooled (bin-matched): storage TTFT mean %.2f, excess over bin host mean: mean %.2f (%.0f%% of storage TTFT); excess over bin host median: median %.2f" % (ex[:, 2].mean(), ex[:, 0].mean(), 100 * ex[:, 0].mean() / ex[:, 2].mean(), np.median(ex[:, 1])))
    # control for load: qdepth class
    print(" matched by prompt bin x qdepth class (0, 1-4, 5-9, >=10):")
    ex2 = []
    for qc, (lo, hi) in {"q0": (0, 0), "q1-4": (1, 4), "q5-9": (5, 9), "q10+": (10, 999)}.items():
        s_all = [x for x in sh if lo <= x["qdepth"] <= hi]; h_all = [x for x in hh if lo <= x["qdepth"] <= hi]
        exq = []
        for b in range(0, 56000, 4000):
            s = [x for x in s_all if b <= x["prompt"] < b + 4000]; h = [x for x in h_all if b <= x["prompt"] < b + 4000]
            if s and len(h) >= 3:
                hm = np.mean([x["ttft"] for x in h])
                for x in s: exq.append((x["ttft"], x["ttft"] - hm, min(x["queue"], x["io_end"] - x["send"])))
        if exq:
            e = np.array(exq); ex2 += exq
            print("   %-5s sto n %3d (matched %3d) host n %4d: sto TTFT mean %.2f  excess mean %.2f  fetch-in-queue mean %.2f  storage TTFT med %.2f" % (qc, len(s_all), len(e), len(h_all), e[:, 0].mean(), e[:, 1].mean(), e[:, 2].mean(), np.median(e[:, 0])))
    e = np.array(ex2); print("   all matched n %d: sto TTFT mean %.2f  excess mean %.2f (%.0f%%)  fetch-in-queue mean %.2f" % (len(e), e[:, 0].mean(), e[:, 1].mean(), 100 * e[:, 1].mean() / e[:, 0].mean(), e[:, 2].mean()))
    # share of storage hit TTFT: io-only vs queued-behind-other-reads
    io = np.array([x["io_ms"] / 1000 for x in sh]); wait = np.array([x["io_start"] - x["sq"] for x in sh]); tt = np.array([x["ttft"] for x in sh])
    pre = np.array([x["sq"] - x["send"] for x in sh]); post = np.array([x["lb"] - x["io_end"] for x in sh]); a2f = np.array([x["adm2ft"] for x in sh])
    print(" storage TTFT mean decomposition: send->query %.3f + wait-for-read-thread %.3f + own IO %.3f + io_end->admission %.3f + admission->first token %.3f = %.3f (TTFT mean %.3f)" % (pre.mean(), wait.mean(), io.mean(), post.mean(), a2f.mean(), pre.mean() + wait.mean() + io.mean() + post.mean() + a2f.mean(), tt.mean()))
    print(" median fetch window / median own IO: %.2f / %.2f" % (np.median([x["io_end"] - x["ps"] for x in sh]), np.median(io)))
    # host-hit queue & ttft overall
    print(" host hits: n %d TTFT med %.3f mean %.3f; queue(lb-send) med %.3f mean %.3f; qdepth mean %.2f" % (len(hh), med([x["ttft"] for x in hh]), np.mean([x["ttft"] for x in hh]), med([x["queue"] for x in hh if "queue" in x]), np.mean([x["queue"] for x in hh if "queue" in x]), np.mean([x["qdepth"] for x in hh])))
