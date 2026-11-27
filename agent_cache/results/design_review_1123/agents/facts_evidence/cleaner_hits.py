#!/usr/bin/env python3
"""Campaign 34: storage hits of returning requests before and after the first SSD cleaner pass.

Usage: cleaner_hits.py <label> <client.jsonl> <events file (grep of server.log)>

Per returning request (k >= 1) from the client record: ideal_cached (history an unbounded
cache would hold), cached_tokens, cached_details.storage, prompt_sent, t_send.
Per request from the server events: s2h_query (tokens_req, storage_hit), srs_dispatch
launched, s2h_io tokens, ntp_reevict capped, ntp_gate decisions.

A lookup asks the SSD for the prompt beyond the in-memory match, so tokens_req includes the
new text, which is never stored. Its history part is hist_req = tokens_req - new, with
new = floor64(prompt_sent) - floor64(ideal_cached). Only lookups with hist_req >= 1,024 are
"history lookups".

Lost history = ideal_cached - cached_tokens; a request "lost history" when that is >= 1,024
tokens and >= 5 % of ideal_cached (campaign 32's definition).
Classes of a lost-history request, by its best history lookup:
  nolook: no history lookup
  zero:   every history lookup found 0 (first page after the in-memory match missing)
  part:   best found > 0 but < hist_req - 64 (the contiguous run stops partway)
  full:   best found >= hist_req - 64 (the SSD held it; lost elsewhere: refused, cut, capped)
"""
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone

label, client_path, evt_path = sys.argv[1:4]
LONG = 16384


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
        if r.get("ttft") is None:
            continue
        cd = r.get("cached_details") or {}
        reqs[r["rid"]] = dict(
            k=r.get("k", 0),
            t=r["t_send"],
            ideal=r.get("ideal_cached") or 0,
            cached=r.get("cached_tokens") or 0,
            storage=cd.get("storage") or 0,
            host=cd.get("host") or 0,
            device=cd.get("device") or 0,
            prompt=r.get("prompt_sent") or r.get("prompt_tokens") or 0,
            ttft=r["ttft"],
        )

ts_re = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\]")


def run_s(line):
    m = ts_re.match(line)
    dt = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
    return dt.timestamp() - t_start


kv_re = re.compile(r"(\w+)=([^\s]+)")
look = defaultdict(list)  # rid -> [(run_s, tokens_req, hit)]
disp = defaultdict(list)  # rid -> [launched]
s2h = defaultdict(int)  # rid -> tokens landed
capped = set()
gate_capped = set()
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
        elif "srs_dispatch" in line:
            disp[rid].append(int(kv["launched"]))
        elif "s2h_io" in line:
            s2h[rid] += int(kv["tokens"])
        elif "ntp_reevict" in line and kv.get("capped") == "1":
            capped.add(rid)
        elif "ntp_gate" in line and kv.get("decision") == "capped":
            gate_capped.add(rid)

first = passes[0]
print(f"{label}: t_start {t_start:.3f}; cleaner passes at run s {[round(p) for p in passes]}; first {first:.0f} s (minute {first/60:.1f})")
print(f"  returning requests with a record: {sum(1 for r in reqs.values() if r['k'] >= 1)}")


def classify(rid, r):
    new = max(0, f64(r["prompt"]) - f64(r["ideal"]))
    hist_looks = [(t, q - new, h) for (t, q, h) in look.get(rid, []) if q - new >= 1024]
    if not hist_looks:
        return "nolook", None
    best = max(hist_looks, key=lambda x: x[2])
    if best[2] == 0:
        return "zero", best
    if best[2] < best[1] - 64:
        return "part", best
    return "full", best


windows = [("min 60 to pass", 3600, first), ("pass to min 110", first, 6600)]
# equal-length windows either side of the pass
span = 6600 - first
windows.append((f"{span/60:.0f} min before pass", first - span, first))

for wname, lo, hi in windows:
    sel = [(rid, r) for rid, r in reqs.items() if r["k"] >= 1 and lo <= r["t"] < hi and r["ideal"] >= LONG]
    n = len(sel)
    lost = [(rid, r) for rid, r in sel if (r["ideal"] - r["cached"]) >= 1024 and (r["ideal"] - r["cached"]) >= 0.05 * r["ideal"]]
    lost_tok = sum(r["ideal"] - r["cached"] for _, r in lost)
    hist_tok = sum(r["ideal"] for _, r in sel)
    cls = defaultdict(lambda: [0, 0])
    for rid, r in lost:
        c, _ = classify(rid, r)
        cls[c][0] += 1
        cls[c][1] += r["ideal"] - r["cached"]
    # all history lookups of long-history returning requests in the window
    allc = defaultdict(int)
    nlook_req = 0
    for rid, r in sel:
        c, best = classify(rid, r)
        if c == "nolook":
            continue
        nlook_req += 1
        # outcome of the best lookup regardless of whether history was lost
        allc[c] += 1
    zero_lost_capped = sum(1 for rid, r in lost if classify(rid, r)[0] == "zero" and (rid in capped or rid in gate_capped))
    print(f"\n  [{wname}] sent {lo:.0f}-{hi:.0f} s ({(hi-lo)/60:.1f} min); returning requests with history >= {LONG}: {n} ({n/((hi-lo)/60):.1f}/min), history {hist_tok/1e6:.2f} M")
    print(f"    lost history (>=1,024 and >=5 %): {len(lost)} requests ({100*len(lost)/max(n,1):.1f} %), {lost_tok/1e6:.2f} M tokens ({100*lost_tok/max(hist_tok,1):.1f} % of history)")
    for c in ("zero", "part", "full", "nolook"):
        print(f"      {c:6s}: {cls[c][0]:5d} requests, {cls[c][1]/1e6:6.2f} M tokens")
    print(f"      of 'zero', capped after the lookups: {zero_lost_capped}")
    print(f"    every long-history request with a history lookup ({nlook_req}), best lookup: zero {allc['zero']}, part {allc['part']}, full {allc['full']}")

# 5-minute bins of send time, minutes 70 to 110
print("\n  by 5-minute send bin (long-history returning requests): n | lost n, M tokens | zero n, M | part n, M | full n | nolook n")
for b0 in range(70, 110, 5):
    lo, hi = b0 * 60, (b0 + 5) * 60
    sel = [(rid, r) for rid, r in reqs.items() if r["k"] >= 1 and lo <= r["t"] < hi and r["ideal"] >= LONG]
    lost = [(rid, r) for rid, r in sel if (r["ideal"] - r["cached"]) >= 1024 and (r["ideal"] - r["cached"]) >= 0.05 * r["ideal"]]
    cls = defaultdict(lambda: [0, 0])
    for rid, r in lost:
        c, _ = classify(rid, r)
        cls[c][0] += 1
        cls[c][1] += r["ideal"] - r["cached"]
    lt = sum(r["ideal"] - r["cached"] for _, r in lost)
    mark = " <- first pass" if lo <= first < hi else ""
    print(f"    {b0:3d}-{b0+5:3d}: {len(sel):4d} | {len(lost):4d} {lt/1e6:5.2f} | {cls['zero'][0]:4d} {cls['zero'][1]/1e6:5.2f} | {cls['part'][0]:4d} {cls['part'][1]/1e6:5.2f} | {cls['full'][0]:4d} | {cls['nolook'][0]:4d}{mark}")

# examples after the pass
ex = []
for rid, r in reqs.items():
    if r["k"] >= 1 and first <= r["t"] < 6600 and r["ideal"] >= LONG and (r["ideal"] - r["cached"]) >= 0.5 * r["ideal"]:
        c, best = classify(rid, r)
        if c in ("zero", "part"):
            ex.append((c, rid, round(r["t"]), r["ideal"], r["cached"], best))
ex.sort(key=lambda x: x[2])
print("\n  examples after the pass (class, rid, t_send, ideal, cached, best history lookup (run s, hist_req, hit)):")
for c in ("zero", "part"):
    for e in [e for e in ex if e[0] == c][:4]:
        print("   ", e)
