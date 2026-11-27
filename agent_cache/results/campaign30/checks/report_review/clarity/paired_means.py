"""Ours' (ec_full) and each baseline's mean TTFT on the requests both answered before 6,600 s, to 3 decimals;
plus ours' GPU cache in average conversations, and Continuum late-bin composition."""
import json, os, statistics
AC = "/home/wanhr/sglang/agent_cache"
CUT = 6600.0
man = os.path.join(AC, "results/campaign30/checks/vs_c34/manifest.txt")
arms = {}
for line in open(man):
    if not line.strip():
        continue
    label, run, prog = line.split()[:3]
    rs = []
    with open(os.path.join(AC, "results", run, prog, "client.jsonl")) as f:
        for l in f:
            if not l.strip():
                continue
            r = json.loads(l)
            if r.get("kind") == "req" and r.get("phase") == "profile" and not r.get("error") and r.get("ttft") is not None and r["t_send"] < CUT:
                rs.append(r)
    arms[label] = rs
ours = {(r["play"], r["stream"], r["k"]): r["ttft"] for r in arms["ec_full"]}
for b in ("vllm_lmcache", "hicache", "continuum_lmcache", "tokencake", "continuum", "sglang_vanilla", "vllm_vanilla", "ec_exact"):
    bt = {(r["play"], r["stream"], r["k"]): r["ttft"] for r in arms[b]}
    keys = ours.keys() & bt.keys()
    print(f"{b:18s} n={len(keys):6d} ours={statistics.fmean(ours[k] for k in keys):.5f} base={statistics.fmean(bt[k] for k in keys):.3f}")
# Continuum late bins: first vs returning in 5-min bins from minute 60
for a in ("continuum", "continuum_lmcache", "ec_full"):
    print(a)
    for m in range(60, 110, 5):
        rs = [r for r in arms[a] if m * 60 <= r["t_send"] < (m + 5) * 60]
        fi = [r for r in rs if r["stream"] == 0 and r["k"] == 0]
        print(f"  min {m:3d}: n={len(rs):4d} first={len(fi):3d} mean={statistics.fmean(r['ttft'] for r in rs):7.2f}")
