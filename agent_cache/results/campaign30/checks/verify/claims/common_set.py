"""The requests every arm sent before the cut (recorded with a ttft): where they fall in time, and a few paired means."""
import json, os, sys, statistics as st
base = "/home/wanhr/sglang/agent_cache/results"
CUT = 6600.0
arms = {}
for line in open(sys.argv[1]):
    if not line.strip():
        continue
    label, run, prog = line.split()[:3]
    d = {}
    for l in open(os.path.join(base, run, prog, "client.jsonl")):
        r = json.loads(l)
        if r["kind"] == "req" and r["phase"] == "profile" and r["t_send"] < CUT and r.get("ttft") is not None:
            d[(r["play"], r["stream"], r["k"])] = r
    arms[label] = d
common = set.intersection(*(set(d) for d in arms.values()))
print("common requests", len(common), "users", len({k[0] for k in common}))
ref = arms["ec_exact"]
ts = sorted(ref[k]["t_send"] for k in common)
for a, b in ((0, 600), (600, 1800), (1800, 3600), (3600, 6600)):
    n = sum(1 for t in ts if a <= t < b)
    print(f"  ec_exact send time {a//60}-{b//60} min: {n} ({100*n/len(ts):.1f} %)")
print("  median send time in ec_exact (s):", round(st.median(ts), 1))
for lab in ("continuum", "sglang_vanilla", "vllm_vanilla"):
    tsl = sorted(arms[lab][k]["t_send"] for k in common)
    print(f"  median send time in {lab} (s): {st.median(tsl):.1f}; share sent after 30 min: {100*sum(1 for t in tsl if t>=1800)/len(tsl):.1f} %")
print("first requests (s0,k0) in common set:", sum(1 for k in common if k[1] == 0 and k[2] == 0))
for lab, d in arms.items():
    v = [d[k]["ttft"] for k in common]
    print(f"  {lab:18s} mean {st.mean(v):7.2f} p50 {st.median(v):7.2f}")
def pair(a, b):
    ks = set(arms[a]) & set(arms[b])
    da = [arms[b][k]["ttft"] - arms[a][k]["ttft"] for k in ks]
    fa = sum(1 for k in ks if arms[b][k]["ttft"] < arms[a][k]["ttft"])
    print(f"pair {a} -> {b}: n {len(ks)}, A mean {st.mean(arms[a][k]['ttft'] for k in ks):.2f}, B mean {st.mean(arms[b][k]['ttft'] for k in ks):.2f}, B-A {st.mean(da):+.2f}, B faster {100*fa/len(ks):.1f} %")
for a, b in (("sglang_vanilla", "continuum"), ("vllm_vanilla", "continuum"), ("sglang_vanilla", "vllm_vanilla"),
             ("vllm_lmcache", "continuum_lmcache"), ("tokencake", "continuum"), ("sglang_vanilla", "tokencake")):
    pair(a, b)
