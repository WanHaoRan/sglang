"""First look at the lookups: staged (before the request's arrival) or not, history or new text.
Usage: python3 -I -B look1.py <arm>"""
import sys, collections
sys.path.insert(0, "/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/ev_ssd")
from c34lib import *
arm = sys.argv[1]
t_start, reqs, plays, args = load_client(arm)
ev = load_events(arm)
byrid = {r["rid"]: r for r in reqs}
entry = {d["rid"]: d["entry"] for d in ev["req_stats"]}
cnt = collections.Counter(); ex = collections.defaultdict(list)
for d in ev["s2h_query"]:
    r = byrid.get(d["rid"])
    if r is None:
        cnt["no client record"] += 1; continue
    e = entry.get(d["rid"])
    staged = (d["t"] < e - 0.001) if e is not None else None
    P = r["prompt_tokens"]
    ql = (P - 1) // 64
    n = d["tokens_req"] // 64
    m = ql - n
    H = r["ideal_cached"] // 64
    if staged:
        k = "staged"
    elif m >= H:
        k = "arrival: new text only"
    elif m < 0:
        k = "arrival: m<0"
    else:
        k = "arrival: history"
    cnt[k] += 1
    cnt[k + (" hit>0" if d["storage_hit"] > 0 else " hit=0")] += 1
    if len(ex[k]) < 3:
        ex[k].append((d["rid"], round(d["t"] - t_start, 2), d["tokens_req"], d["storage_hit"], P, r["ideal_cached"], r["cached_tokens"], round(r["t_send"], 2)))
for k in sorted(cnt):
    print(f"{k:40s} {cnt[k]}")
for k, v in ex.items():
    print(k, v)
# offset between client send and server entry
offs = sorted(entry[r["rid"]] - (t_start + r["t_send"]) for r in reqs if r["rid"] in entry)
print("server entry - client send: p1 %.3f p50 %.3f p99 %.3f max %.3f" % (offs[len(offs)//100], offs[len(offs)//2], offs[99*len(offs)//100], offs[-1]))
