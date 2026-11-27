"""How lookups, parks, reads and read results pair up per request; delays from lookup to read.
Usage: python3 -I -B look2.py <arm>"""
import sys, collections
sys.path.insert(0, "/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/ev_ssd")
from c34lib import *
arm = sys.argv[1]
t_start, reqs, plays, args = load_client(arm)
ev = load_events(arm)
per = collections.defaultdict(list)
for name in ("s2h_query", "srs_park", "srs_dispatch", "s2h_io", "prefetch_ok", "read_fail", "prefetch_start", "ntp_gate", "shrank"):
    for d in ev.get(name, []):
        per[d["rid"]].append((d["t"], name, d))
cnt = collections.Counter(); delays = collections.defaultdict(list)
seqs = collections.Counter()
for rid, lst in per.items():
    lst.sort(key=lambda x: x[0])
    seq = []
    last_q = None
    for t, name, d in lst:
        if name == "s2h_query":
            last_q = (t, d)
            seq.append("Q+" if d["storage_hit"] > 0 else "Q0")
            if 0 < d["storage_hit"] < 256: cnt["hit below 256"] += 1
        elif name == "srs_park":
            seq.append("P")
            if last_q is None: cnt["park without query"] += 1
            elif d["pages"] * 64 != last_q[1]["storage_hit"]:
                cnt["park pages != hit"] += 1
        elif name == "s2h_io":
            seq.append("R")
            if last_q is not None:
                delays[int((t - t_start) // 1800)].append(t - last_q[0])
        elif name == "prefetch_ok":
            seq.append("K")
            if last_q is not None and d["completed"] != last_q[1]["storage_hit"]:
                cnt["completed != hit"] += 1
        elif name == "read_fail":
            seq.append("F")
    s = "".join(x if len(x) == 1 else "(" + x + ")" for x in seq)
    seqs[s] += 1
print(arm, dict(cnt))
for s, n in seqs.most_common(25):
    print(f"  {n:6d}  {s}")
for k in sorted(delays):
    xs = sorted(delays[k])
    print(f"  lookup -> read done, minute {30*k}-{30*k+30}: n {len(xs)} p50 {xs[len(xs)//2]:.2f} p90 {xs[9*len(xs)//10]:.2f} max {xs[-1]:.2f}")
