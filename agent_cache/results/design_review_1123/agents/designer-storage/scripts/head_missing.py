"""Requests after the first SSD cleaner pass whose every history lookup found 0 and started at the
session's first page (cleaner_hits3.py's zero_head class), with: the age of their session (first
send to this send), and whether the session's first request shared a prefix with earlier sessions
(ideal_cached of turn 0 > 0), i.e. its head pages were written by another, older session.
Usage: python head_missing.py client.jsonl events.txt"""
import json, re, sys, statistics as st
from collections import defaultdict
from datetime import datetime, timezone
def f64(x): return (int(x) // 64) * 64
t_start = None; reqs = {}; first = {}
for line in open(sys.argv[1]):
    if '"kind":"run"' in line and t_start is None: t_start = json.loads(line)["t_start"]; continue
    if '"kind":"req"' not in line: continue
    r = json.loads(line)
    m = re.match(r"(.*-c\d+)-t(\d+)$", r["rid"]); conv, k = m[1], int(m[2])
    if k == 0: first[conv] = (r["t_send"], r.get("ideal_cached") or 0, r.get("prompt_tokens") or 0)
    if r.get("ttft") is None or k < 1: continue
    reqs[r["rid"]] = dict(conv=conv, t=r["t_send"], ideal=r.get("ideal_cached") or 0, cached=r.get("cached_tokens") or 0,
                          prompt=r.get("prompt_sent") or r.get("prompt_tokens") or 0)
ts = re.compile(r"^\[(\S+ \S+)\]"); kv = re.compile(r"(\w+)=([^\s]+)")
def run_s(line):
    return datetime.strptime(ts.match(line)[1], "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp() - t_start
look = defaultdict(list); passes = []
for line in open(sys.argv[2]):
    if "NIXL L3 cleanup" in line: passes.append(run_s(line)); continue
    d = dict(kv.findall(line))
    if "s2h_query" in line: look[d["rid"]].append((run_s(line), int(d["tokens_req"]), int(d["storage_hit"])))
p0 = passes[0]
rows = []
for rid, r in reqs.items():
    if not (p0 <= r["t"] < 6600) or r["ideal"] - r["cached"] < 1024: continue
    new = max(0, f64(r["prompt"]) - f64(r["ideal"]))
    hl = [(t, q - new, h, f64(r["prompt"]) - q) for (t, q, h) in look.get(rid, []) if t >= r["t"] - 0.5 and q - new >= 1024]
    if not hl or max(x[2] for x in hl) > 0 or min(x[3] for x in hl) > 0: continue
    t0, ideal0, prompt0 = first.get(r["conv"], (None, 0, 0))
    rows.append((rid, (r["t"] - t0) / 60 if t0 is not None else None, ideal0, prompt0, r["ideal"]))
print(f"first pass at {p0/60:.1f} min; zero_head requests sent before 6,600 s: {len(rows)}")
ages = [x[1] for x in rows if x[1] is not None]
shared = [x for x in rows if x[2] >= 64]
print(f"session age at the send: p10 {sorted(ages)[len(ages)//10]:.1f}, p50 {st.median(ages):.1f}, p90 {sorted(ages)[9*len(ages)//10]:.1f}, max {max(ages):.1f} min")
print(f"session's first request shared >= 64 tokens with earlier sessions: {len(shared)} of {len(rows)}; "
      f"shared share of that first prompt, median {st.median([x[2]/max(1,x[3]) for x in shared]) if shared else 0:.0%}")
print(f"sessions younger than the first pass time minus 75 min (so their own first write came after minute {p0/60-75:.0f}): "
      f"{sum(1 for x in rows if x[1] is not None and x[1] < 75)} of {len(rows)}")
