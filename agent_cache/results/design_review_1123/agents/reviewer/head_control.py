"""Reviewer control for the shared-head explanation of post-cleaner zero hits (read-only).
Among returning requests sent between the first cleaner pass and 6,600 s whose storage lookup started at the
session's first page (nothing of the prompt in GPU or host memory) and asked >= 1,024 history tokens, compare the
share whose session's first request shared >= 64 tokens with older sessions (ideal_cached of turn 0), for lookups
that found 0 against lookups that found something. Also the base rate over all conversations.
Usage: python3 head_control.py client.jsonl events.txt"""
import json, re, sys
from collections import defaultdict
from datetime import datetime, timezone
def f64(x): return (int(x) // 64) * 64
t_start = None; reqs = {}; first = {}
for line in open(sys.argv[1]):
    if '"kind":"run"' in line and t_start is None:
        t_start = json.loads(line)["t_start"]; continue
    if '"kind":"req"' not in line: continue
    r = json.loads(line)
    m = re.match(r"(.*-c\d+)-t(\d+)$", r["rid"]); conv, k = m[1], int(m[2])
    if k == 0: first[conv] = (r["t_send"], r.get("ideal_cached") or 0)
    if k < 1: continue
    reqs[r["rid"]] = dict(conv=conv, t=r["t_send"], ideal=r.get("ideal_cached") or 0,
                          prompt=r.get("prompt_sent") or r.get("prompt_tokens") or 0)
ts = re.compile(r"^\[(\S+ \S+)\]"); kv = re.compile(r"(\w+)=([^\s]+)")
def run_s(line):
    return datetime.strptime(ts.match(line)[1], "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp() - t_start
look = defaultdict(list); passes = []
for line in open(sys.argv[2]):
    if "NIXL L3 cleanup" in line: passes.append(run_s(line)); continue
    if "s2h_query" in line:
        d = dict(kv.findall(line)); look[d["rid"]].append((run_s(line), int(d["tokens_req"]), int(d["storage_hit"])))
p0 = passes[0]
grp = {"zero": [], "hit": []}
for rid, r in reqs.items():
    if not (p0 <= r["t"] < 6600): continue
    new = max(0, f64(r["prompt"]) - f64(r["ideal"]))
    hl = [(q - new, h) for (t, q, h) in look.get(rid, [])
          if t >= r["t"] - 0.5 and q - new >= 1024 and f64(r["prompt"]) - q == 0]
    if not hl: continue
    shared = first.get(r["conv"], (None, 0))[1] >= 64
    grp["zero" if max(h for _, h in hl) == 0 else "hit"].append(shared)
base = [v[1] >= 64 for v in first.values()]
print(f"first pass {p0/60:.1f} min | base rate, all conversations: {sum(base)}/{len(base)} = {sum(base)/len(base):.0%} shared head")
for g, xs in grp.items():
    print(f"  page-0 history lookups after the pass, {g:4s}: {len(xs)} requests, shared head {sum(xs)} = {sum(xs)/max(1,len(xs)):.0%}")
