"""Verifier's check of three unread concerns of the client-side recomputation (C6, C11, C12). Read-only."""
import json, os, statistics as st
RES = "/home/wanhr/sglang/agent_cache/results"
man = {f[0]: (f[1], f[2]) for f in (l.split() for l in open(os.path.join(RES, "campaign26", "manifest.txt"))) if f}
runs = {"jit_exact": ("20261101_202142_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70", None), "baseline": ("20261101_082229_three_tier_wc_custom_L1nat_H70", None),
        "base_rep": ("20261105_075438_three_tier_wc_custom_L1nat_H70", None)}
for a in ("ec_r13", "ec_r1", "nofc", "nofc_noend", "jit_p50_ctl", "hintfree_ctl", "jit_exact_d", "nofc_noend_rep", "ec_full"):
    runs[a] = man[a]
import glob
def read(run, cl):
    p = os.path.join(RES, run, cl, "client.jsonl") if cl else sorted(glob.glob(os.path.join(RES, run, "client_*", "client.jsonl")))[-1]
    t = [r for r in (json.loads(l) for l in open(p) if l.strip()) if r.get("kind") == "turn" and r.get("ttft") is not None]
    last = max(r["t_send"] for r in t)
    return {(r["conv"], r["turn"]): r for r in t if r["t_send"] < last - 300}
D = {a: read(*runs[a]) for a in runs}
L5 = ["jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend"]
first = set.intersection(*(set(k for k in D[a] if k[1] == 0) for a in L5))
print("C11: sessions whose first turn is kept in all five:", len(first), {a: round(st.mean(D[a][k]["ttft"] for k in first), 3) for a in L5})
allk = set.intersection(*(set(D[a]) for a in L5))
print("     with first turns included:", len(allk), {a: round(st.mean(D[a][k]["ttft"] for k in allk), 4) for a in L5})
rec = lambda r: max(r.get("reply_reprefilled") or 0, 0)
print("C12: recomputed tokens per returning turn sent in minutes 0-10 (mean, median):")
for a in D:
    v = [rec(r) for k, r in D[a].items() if k[1] > 0 and r["t_send"] < 600]
    print(f"     {a:15s} n {len(v):4d} mean {st.mean(v):5.1f} median {st.median(v):4.0f}")
print("C6: how much later an arm sends a shared returning key than jit_exact (mean s, median s, share over 600 s):")
for a in ("jit_p50_ctl", "hintfree_ctl", "baseline", "base_rep", "ec_r13", "nofc", "nofc_noend", "jit_exact_d", "ec_full"):
    ks = [k for k in D[a] if k[1] > 0 and k in D["jit_exact"]]
    d = [D[a][k]["t_send"] - D["jit_exact"][k]["t_send"] for k in ks]
    print(f"     {a:15s} n {len(ks):5d} mean {st.mean(d):7.1f} median {st.median(d):7.1f} over600 {100 * sum(x > 600 for x in d) / len(d):5.1f} %")
