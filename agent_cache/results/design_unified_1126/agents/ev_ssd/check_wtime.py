"""Delay from the client's prefill end (t_send + ttft) / request end (t_send + latency) to the logged SSD write.
Output nodes are linked exactly (nue_register rid -> node -> h2s_submit -> h2s_io); prompt nodes of at most one
chunk are matched to the nearest unmatched h2s_submit of the same size.  Usage: python3 -I -B check_wtime.py <arm>"""
import sys, collections, bisect
sys.path.insert(0, "/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/ev_ssd")
from c34lib import *
arm = sys.argv[1]
t_start, reqs, plays, args = load_client(arm)
ev = load_events(arm)
sub = ev["h2s_submit"]; io = {d["op"]: d for d in ev["h2s_io"]}
by_node = collections.defaultdict(list)
for d in sub:
    by_node[d["node"]].append(d)
def q(xs, p):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(p * len(xs)))]
# output nodes
reg = collections.defaultdict(list)
for d in ev["nue_register"]:
    reg[d["rid"]].append(d)
byrid = {r["rid"]: r for r in reqs}
d_sub, d_io, n_out, n_miss, tok_mis = [], [], 0, 0, 0
for rid, r in byrid.items():
    if r.get("ttft") is None: continue
    P, C = r["prompt_tokens"], r["completion_tokens"]
    wo = ((P + C - 1) // 64) * 64 - (P // 64) * 64
    if wo <= 0: continue
    n_out += 1
    t_end = t_start + r["t_send"] + r["latency"]
    cand = [s for g in reg.get(rid, []) for s in by_node.get(g["node"], []) if abs(s["t"] - t_end) < 5]
    if not cand:
        n_miss += 1; continue
    s = min(cand, key=lambda s: abs(s["t"] - t_end))
    if s["tokens"] != wo: tok_mis += 1
    d_sub.append(s["t"] - t_end); d_io.append(io[s["op"]]["t"] - t_end)
print(f"{arm}: output nodes expected {n_out}, linked {len(d_sub)}, not linked {n_miss}, size differs {tok_mis}")
for nm, xs in (("submit - client end", d_sub), ("io done - client end", d_io)):
    print(f"   {nm}: p1 {q(xs,.01):.3f} p10 {q(xs,.1):.3f} p50 {q(xs,.5):.3f} p90 {q(xs,.9):.3f} p99 {q(xs,.99):.3f} max {max(xs):.3f}")
# prompt nodes, single chunk
times = [d["t"] for d in sub]
used = set()
dp_sub, dp_io, n_p, n_pm = [], [], 0, 0
late = collections.defaultdict(list)
for r in sorted((r for r in reqs if r.get("ttft") is not None), key=lambda r: r["t_send"] + r["ttft"]):
    P, c = r["prompt_tokens"], r["cached_tokens"] or 0
    wp = (P // 64) * 64 - c
    if wp <= 0 or wp > 8192: continue
    n_p += 1
    tp = t_start + r["t_send"] + r["ttft"]
    lo = bisect.bisect_left(times, tp - 3.0); hi = bisect.bisect_right(times, tp + 30.0)
    best = None
    for i in range(lo, hi):
        if i in used or sub[i]["tokens"] != wp: continue
        if best is None or abs(times[i] - tp) < abs(times[best] - tp): best = i
    if best is None:
        n_pm += 1; continue
    used.add(best)
    dp_sub.append(times[best] - tp); dp_io.append(io[sub[best]["op"]]["t"] - tp)
    late[int(r["t_send"] // 1800)].append(io[sub[best]["op"]]["t"] - tp)
print(f"   prompt nodes of one chunk: {n_p}, matched {len(dp_sub)}, unmatched {n_pm}")
for nm, xs in (("submit - client prefill end", dp_sub), ("io done - client prefill end", dp_io)):
    print(f"   {nm}: p1 {q(xs,.01):.3f} p10 {q(xs,.1):.3f} p50 {q(xs,.5):.3f} p90 {q(xs,.9):.3f} p99 {q(xs,.99):.3f} max {max(xs):.3f}")
for k in sorted(late):
    xs = late[k]
    print(f"   prompt io done - client prefill end, send minute {30*k}-{30*k+30}: p50 {q(xs,.5):.3f} p90 {q(xs,.9):.3f} p99 {q(xs,.99):.3f}")
