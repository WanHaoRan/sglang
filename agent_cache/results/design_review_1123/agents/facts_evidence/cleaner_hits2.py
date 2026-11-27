#!/usr/bin/env python3
"""Campaign 34: what returning requests found on the SSD around the cleaner passes.

Usage: cleaner_hits2.py <label> <client.jsonl> <events file (grep of server.log)>

Same inputs and definitions as cleaner_hits.py, but over every returning request (k >= 1)
whose ideal_cached >= 1,024, and a request "lost history" when ideal_cached - cached_tokens
>= 1,024 (no 5 % rule). "again" totals use max(0, ideal - cached) over all returning
requests, as campaign34/checks/fix_minutes.py does, so the per-minute column can be checked
against fix_minutes_table_<arm>.out.

Lookup classes (best history lookup of the request; history part of a lookup is
tokens_req - new, new = floor64(prompt) - floor64(ideal); matched at the lookup =
floor64(prompt) - tokens_req):
  zero_head: all history lookups found 0 and the best one started at matched = 0
             (the session's first page is missing on the SSD)
  zero_mid:  all found 0, lookup started after an in-memory prefix (matched > 0)
  part:      best found > 0 and < hist_req - 64 (contiguous run stops partway)
  full:      best found >= hist_req - 64
  nolook:    no lookup asked for >= 1,024 history tokens
"""
import json
import re
import sys
from collections import defaultdict, Counter
from datetime import datetime, timezone

label, client_path, evt_path = sys.argv[1:4]


def f64(x):
    return (int(x) // 64) * 64


t_start = None
reqs = {}
with open(client_path) as fh:
    for line in fh:
        if '"kind":"run"' in line and t_start is None:
            t_start = json.loads(line)["t_start"]
            continue
        if '"kind":"req"' not in line:
            continue
        r = json.loads(line)
        if r.get("ttft") is None or (r.get("k") or 0) < 1:
            continue
        cd = r.get("cached_details") or {}
        reqs[r["rid"]] = dict(
            t=r["t_send"],
            ideal=r.get("ideal_cached") or 0,
            cached=r.get("cached_tokens") or 0,
            storage=cd.get("storage") or 0,
            prompt=r.get("prompt_sent") or r.get("prompt_tokens") or 0,
            ttft=r["ttft"],
            wait=r.get("wait_s"),
        )

ts_re = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\]")
kv_re = re.compile(r"(\w+)=([^\s]+)")


def run_s(line):
    m = ts_re.match(line)
    d = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
    return d.timestamp() - t_start


look = defaultdict(list)
refused = Counter()
capped = set()
passes = []
with open(evt_path) as fh:
    for line in fh:
        if "NIXL L3 cleanup" in line:
            passes.append(run_s(line))
            continue
        kv = dict(kv_re.findall(line))
        rid = kv.get("rid")
        if rid is None:
            continue
        if "s2h_query" in line:
            look[rid].append((run_s(line), int(kv["tokens_req"]), int(kv["storage_hit"])))
        elif "srs_dispatch" in line and kv.get("launched") == "0":
            refused[rid] += 1
        elif ("ntp_reevict" in line and kv.get("capped") == "1") or ("ntp_gate" in line and kv.get("decision") == "capped"):
            capped.add(rid)

first = passes[0]


def classify(rid, r):
    new = max(0, f64(r["prompt"]) - f64(r["ideal"]))
    hl = []
    for (t, q, h) in look.get(rid, []):
        hist = q - new
        if hist >= 1024:
            hl.append((t, hist, h, f64(r["prompt"]) - q))
    if not hl:
        return "nolook", None
    best = max(hl, key=lambda x: (x[2], -x[3]))
    if best[2] == 0:
        head = min(x[3] for x in hl) <= 0
        return ("zero_head" if head else "zero_mid"), best
    if best[2] < best[1] - 64:
        return "part", best
    return "full", best


def since_pass(t):
    prev = [p for p in passes if p <= t]
    return t - prev[-1] if prev else None


CL = ("zero_head", "zero_mid", "part", "full", "nolook")
print(f"{label}: cleaner passes at run s {[round(p) for p in passes]} (minutes {[round(p/60,1) for p in passes]})")


def window(name, lo, hi, minhist=1024):
    sel = {rid: r for rid, r in reqs.items() if lo <= r["t"] < hi and r["ideal"] >= minhist}
    again = sum(max(0, r["ideal"] - r["cached"]) for r in sel.values())
    hist = sum(r["ideal"] for r in sel.values())
    lost = {rid: r for rid, r in sel.items() if r["ideal"] - r["cached"] >= 1024}
    cls = defaultdict(lambda: [0, 0])
    for rid, r in lost.items():
        c, _ = classify(rid, r)
        cls[c][0] += 1
        cls[c][1] += r["ideal"] - r["cached"]
    hl_all = Counter(classify(rid, r)[0] for rid, r in sel.items())
    lost_long = sum(1 for r in lost.values() if r["ideal"] >= 16384)
    capped_lost = sum(1 for rid in lost if rid in capped)
    ref_lost = sum(1 for rid in lost if refused[rid])
    mins = (hi - lo) / 60
    print(f"\n  [{name}] sent {lo:.0f}-{hi:.0f} s ({mins:.1f} min): returning {len(sel)}, history {hist/1e6:.2f} M; computed again {again/1e6:.2f} M ({again/1e3/mins:.1f} K a minute, {100*again/max(1,hist):.2f} % of history)")
    print(f"    requests that lost >= 1,024 history tokens: {len(lost)} ({lost_long} with history >= 16 K); of them capped {capped_lost}, with a refused dispatch {ref_lost}")
    for c in CL:
        print(f"      {c:9s}: {cls[c][0]:5d} requests, {cls[c][1]/1e6:6.2f} M lost")
    print(f"    best history lookup over all {len(sel)} returning requests: " + ", ".join(f"{c} {hl_all[c]}" for c in CL))


window("minute 60 to first pass", 3600, first)
span = 6600 - first
window(f"equal span before the pass ({span/60:.0f} min)", first - span, first)
window("first pass to minute 110", first, 6600)

print("\n  per minute (send time), all returning requests: again K | lost>=1,024: zero_head zero_mid part full nolook | passes")
for m in range(78, 111):
    lo, hi = m * 60, (m + 1) * 60
    sel = {rid: r for rid, r in reqs.items() if lo <= r["t"] < hi}
    again = sum(max(0, r["ideal"] - r["cached"]) for r in sel.values())
    cnt = Counter()
    for rid, r in sel.items():
        if r["ideal"] - r["cached"] >= 1024 and r["ideal"] >= 1024:
            cnt[classify(rid, r)[0]] += 1
    mark = " ".join(f"pass@{p/60:.1f}" for p in passes if lo <= p < hi)
    print(f"    {m:3d} | {again/1e3:6.0f} | " + " ".join(f"{cnt[c]:4d}" for c in CL) + f" | {mark}")

# time since the most recent pass for lost requests after the first pass, zero_head class
buckets = Counter()
tok = Counter()
for rid, r in reqs.items():
    if first <= r["t"] < 6600 and r["ideal"] - r["cached"] >= 1024:
        c, _ = classify(rid, r)
        s = since_pass(r["t"])
        b = "0-2 min" if s < 120 else "2-5 min" if s < 300 else "5-10 min" if s < 600 else ">10 min"
        buckets[(c, b)] += 1
        tok[(c, b)] += r["ideal"] - r["cached"]
print("\n  lost requests after the first pass by time since the latest pass (count, M tokens):")
for c in CL:
    print(f"    {c:9s}: " + "; ".join(f"{b} {buckets[(c,b)]} {tok[(c,b)]/1e6:.2f}" for b in ("0-2 min", "2-5 min", "5-10 min", ">10 min")))

# history size and idle gap of zero_head requests after the pass
zh = [r for rid, r in reqs.items() if first <= r["t"] < 6600 and r["ideal"] - r["cached"] >= 1024 and classify(rid, r)[0] == "zero_head"]
if zh:
    hs = sorted(r["ideal"] for r in zh)
    ws = sorted(r["wait"] for r in zh if r["wait"] is not None)
    print(f"\n  zero_head after the pass: {len(zh)} requests, history median {hs[len(hs)//2]} (min {hs[0]}, max {hs[-1]}); wait before the request median {ws[len(ws)//2]:.1f} s (min {ws[0]:.1f}, max {ws[-1]:.1f}); TTFT mean {sum(r['ttft'] for r in zh)/len(zh):.1f} s")
