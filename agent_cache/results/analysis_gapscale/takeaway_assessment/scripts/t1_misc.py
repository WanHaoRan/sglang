import pickle, numpy as np
S = "/tmp/claude-1001/-home-wanhr-sglang/454133c8-c6f1-4d1f-8d97-1e8cda557a0d/scratchpad/assess/"
D = pickle.load(open(S + "t1.pkl", "rb")); R = pickle.load(open(S + "t1_rows.pkl", "rb")); SIM = pickle.load(open(S + "t1_sim.pkl", "rb"))
for arm in ("three_tier_to", "three_tier_wc", "hbm_host"):
    T = [r for r in D[arm]["turns"] if "t_send" in r and r.get("ttft")]
    ret = [r for r in T if r["turn"] > 0]
    tt = np.array([r["ttft"] for r in T]); tr = np.array([r["ttft"] for r in ret])
    print(arm, "all turns n %d mean TTFT %.3f p50 %.3f | returns n %d mean %.3f p50 %.3f p90 %.3f" % (len(T), tt.mean(), np.median(tt), len(ret), tr.mean(), np.median(tr), np.percentile(tr, 90)))
    if arm in SIM:
        jobs, res = SIM[arm]; H = [j for j in jobs if j["hit"]]
        saved = sum(j["io_end"] - j["ps"] for j in H)
        print("   removing all storage-hit fetch windows (%.0f s) would lower the mean over returns by %.3f s -> %.3f; over all turns by %.3f -> %.3f" % (saved, saved / len(ret), tr.mean() - saved / len(ret), saved / len(T), tt.mean() - saved / len(T)))
        w = [j for j in jobs if not j["hit"]]
        if w:
            rows = {f"cmp_{arm}-c{x['conv']}-t{x['turn']}": x for x in R[arm]}
            ww = [rows[j["rid"]] for j in w if j["rid"] in rows]
            print("   reads not used (timeout fired before read completed): %d, %.0f GiB; those turns: TTFT mean %.2f, cached mean %.0f of prompt %.0f; read wait (s2h_query->io start) mean %.2f s; own IO mean %.2f s" % (
                len(w), sum(j["pages"] for j in w) * 64 * 98304 / 2**30, np.mean([x["ttft"] for x in ww]), np.mean([x["cached"] for x in ww]), np.mean([x["prompt"] for x in ww]),
                np.mean([j["io_end"] - j["dur"] - j["sq"] for j in w]), np.mean([j["dur"] for j in w])))
