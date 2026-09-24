import pickle, statistics as st, numpy as np
D = pickle.load(open("/tmp/claude-1001/-home-wanhr-sglang/454133c8-c6f1-4d1f-8d97-1e8cda557a0d/scratchpad/assess/t1.pkl", "rb"))
def q(a, p): return float(np.percentile(a, p)) if len(a) else float('nan')
for arm in ("three_tier_to", "three_tier_wc"):
    A = D[arm]; t0 = A["t0"]; ev = A["ev"]
    # clock check: prefetch_start - send for all turns
    dl = []
    for r in A["turns"]:
        rid = f"cmp_{arm}-c{r['conv']}-t{r['turn']}"
        e = ev.get(rid, {})
        if "prefetch_start" in e: dl.append(e["prefetch_start"][0][0] - (t0 + r["t_send"]))
    print(arm, "prefetch_start - send: n", len(dl), "min %.3f p1 %.3f p50 %.3f p99 %.3f max %.3f" % (min(dl), q(dl,1), q(dl,50), q(dl,99), max(dl)))
    # gap semantics: t_send(t) - (t_send(t-1)+latency(t-1)) vs gap_slept
    bykey = {(r["conv"], r["turn"]): r for r in A["turns"] if "t_send" in r and r.get("ttft")}
    diffs = []
    for (c, t), r in bykey.items():
        p = bykey.get((c, t - 1))
        if p and t > 0: diffs.append(r["t_send"] - (p["t_send"] + p["latency"]) - r["gap_slept"])
    print(" send - prev_end - gap_slept: p1 %.3f p50 %.3f p99 %.3f max %.3f" % (q(diffs,1), q(diffs,50), q(diffs,99), max(diffs)))
    sh = [r for r in A["turns"] if (r.get("cached_details") or {}).get("storage", 0) > 0]
    ret = [r for r in A["turns"] if r["turn"] > 0]
    print(" returns", len(ret), "storage-hit turns", len(sh), "%.1f%%" % (100 * len(sh) / len(ret)))
    # event coverage
    cov = {k: sum(1 for r in sh if k in ev.get(f"cmp_{arm}-c{r['conv']}-t{r['turn']}", {})) for k in ("prefetch_start", "s2h_query", "s2h_io", "pf_success", "load_back_init")}
    print(" coverage", cov)
    multi = sum(1 for r in sh if len(ev.get(f"cmp_{arm}-c{r['conv']}-t{r['turn']}", {}).get("prefetch_start", [])) > 1)
    print(" storage-hit turns with >1 prefetch_start", multi)
