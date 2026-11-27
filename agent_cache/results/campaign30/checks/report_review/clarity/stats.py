"""Clarity review helper: history-recomputed shares, paired medians/percentiles vs ours (ec_full), first-request shares.
Read-only over the client records named in campaign30/checks/vs_c34/manifest.txt."""
import json, os, sys, statistics
AC = "/home/wanhr/sglang/agent_cache"
CUT = 6600.0
man = os.path.join(AC, "results/campaign30/checks/vs_c34/manifest.txt")
arms = {}
for line in open(man):
    if not line.strip():
        continue
    label, run, prog = line.split()[:3]
    reqs = []
    with open(os.path.join(AC, "results", run, prog, "client.jsonl")) as f:
        for l in f:
            if not l.strip():
                continue
            r = json.loads(l)
            if r.get("kind") == "req" and r.get("phase") == "profile" and not r.get("error") and r.get("ttft") is not None and r["t_send"] < CUT:
                reqs.append(r)
    arms[label] = reqs

def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))] if v else float("nan")

def cached_of(r):
    d = r.get("cached_details") or {}
    if d:
        return sum(int(d.get(k) or 0) for k in ("device", "host", "storage"))
    return int(r.get("cached_tokens") or 0)

print("arm | n | prompt recomputed % | history recomputed % (1-cached/ideal) | new share of prompt % | first-request share % | mean prompt tok | mean ideal tok")
for a, rs in arms.items():
    prompt = sum(r["prompt_tokens"] or 0 for r in rs)
    cached = sum(cached_of(r) for r in rs)
    ideal = sum(r.get("ideal_cached") or 0 for r in rs)
    first = sum(1 for r in rs if r["stream"] == 0 and r["k"] == 0)
    print(f"{a:18s} | {len(rs):6d} | {100*(1-cached/prompt):6.2f} | {100*(1-cached/ideal):6.2f} | {100*(1-ideal/prompt):6.2f} | {100*first/len(rs):5.2f} | {prompt/len(rs):9.0f} | {ideal/len(rs):9.0f}")

ours = {(r["play"], r["stream"], r["k"]): r["ttft"] for r in arms["ec_full"]}
print("\npaired vs ec_full on the requests both answered before the cut")
print("B | n | ours mean p50 p90 p99 | B mean p50 p90 p99 | B faster %")
for b, rs in arms.items():
    if b == "ec_full":
        continue
    bt = {(r["play"], r["stream"], r["k"]): r["ttft"] for r in rs}
    keys = ours.keys() & bt.keys()
    o = [ours[k] for k in keys]; v = [bt[k] for k in keys]
    faster = sum(1 for k in keys if bt[k] < ours[k]) / len(keys)
    print(f"{b:18s} | {len(keys):6d} | {statistics.fmean(o):6.2f} {q(o,.5):6.2f} {q(o,.9):6.2f} {q(o,.99):7.2f} | {statistics.fmean(v):6.2f} {q(v,.5):6.2f} {q(v,.9):6.2f} {q(v,.99):7.2f} | {100*faster:5.1f}")

print("\nown requests: mean p50 p90 p99, first-request mean, returning mean, share of requests answered under 1 s")
for a, rs in arms.items():
    t = [r["ttft"] for r in rs]
    fi = [r["ttft"] for r in rs if r["stream"] == 0 and r["k"] == 0]
    rt = [r["ttft"] for r in rs if not (r["stream"] == 0 and r["k"] == 0)]
    print(f"{a:18s} | {statistics.fmean(t):6.2f} {q(t,.5):6.2f} {q(t,.9):6.2f} {q(t,.99):8.2f} | first {statistics.fmean(fi):8.2f} (n={len(fi)}) | returning {statistics.fmean(rt):6.2f} | <1s {100*sum(1 for x in t if x<1)/len(t):5.1f}%")

# share of each arm's requests sent in the first 30 min
print("\nshare of own requests sent in minutes 0-30 / 60-110")
for a, rs in arms.items():
    e = sum(1 for r in rs if r["t_send"] < 1800) / len(rs)
    l = sum(1 for r in rs if r["t_send"] >= 3600) / len(rs)
    print(f"{a:18s} | {100*e:5.1f}% | {100*l:5.1f}%")
