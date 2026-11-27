"""Slot wait (first send - arrival) of each baseline's own users, against ec_full's and ec_exact's on the same users,
with compare_arms.py's definitions (users whose first request was answered and sent before the cut)."""
import json, os, random, sys, statistics as st
base = "/home/wanhr/sglang/agent_cache/results"
CUT = 6600.0
def arrival_times(seed, rate, n):
    rng, t, out = random.Random(f"{seed}:arrivals"), 0.0, []
    for _ in range(n):
        out.append(t)
        t += rng.expovariate(rate)
    return out
arr = arrival_times(0, 0.1, 2000)
starts = {}
for line in open(sys.argv[1]):
    if not line.strip():
        continue
    label, run, prog = line.split()[:3]
    s = {}
    for l in open(os.path.join(base, run, prog, "client.jsonl")):
        r = json.loads(l)
        if r["kind"] == "req" and r["phase"] == "profile" and r["t_send"] < CUT and r.get("ttft") is not None and r["stream"] == 0 and r["k"] == 0:
            s[r["play"]] = r["t_send"]
    starts[label] = s
for b in ("vllm_lmcache", "continuum_lmcache", "tokencake", "continuum", "sglang_vanilla", "vllm_vanilla"):
    users = starts[b]
    mb = st.mean(t - arr[p] for p, t in users.items())
    out = [f"{b:18s} own users {len(users):3d}: slot wait mean {mb:7.1f}"]
    for a in ("ec_full", "ec_exact"):
        common = [p for p in users if p in starts[a]]
        out.append(f"{a} on them ({len(common)}) {st.mean(starts[a][p] - arr[p] for p in common):6.1f}")
    print(" | ".join(out))
