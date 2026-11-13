"""checks/latency_lead.out and latency_all.out recomputed from the cache, with the latency of turns that could be cut
by the end of a run looked at: turns still running when the client's last send happened, and the share of each arm's
kept turns whose reply ended after the kept cutoff.
  nice -n 19 python3 -I -B r2_latency.py"""
import os, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import arm, shared, mean

for names in (["nofc_noend", "nofc", "ec_r13", "ec_r13_rep", "ec_r1", "jit_exact"],
              ["nofc_noend", "baseline", "base_rep", "hintfree_ctl", "jit_p50_ctl", "jit_p50_d", "ec_r13", "ec_r13_rep", "jit_exact"]):
    keys = shared(names)
    print(f"\n{len(keys)} returning turns shared by {', '.join(names)}")
    print("   arm: mean TTFT, mean latency, median latency, mean latency - TTFT, completion tokens per turn | own kept turns, completion tokens (M) | paired against the first arm: TTFT, latency, latency - TTFT | "
          "latency null among shared turns | median output tokens per second after the first token")
    ref = arm(names[0])["by"]
    for a in names:
        d = arm(a)
        r = [d["by"][k] for k in keys]
        own = d["turns"]
        dd = [(d["by"][k]["ttft"] - ref[k]["ttft"], d["by"][k]["latency"] - ref[k]["latency"]) for k in keys]
        rate = [x["completion_tokens"] / (x["latency"] - x["ttft"]) for x in r if x["completion_tokens"] and x["latency"] - x["ttft"] > 0.05]
        print(f"   {a:12s} {mean(x['ttft'] for x in r):6.2f} {mean(x['latency'] for x in r):7.2f} {st.median(x['latency'] for x in r):7.2f} {mean(x['latency'] - x['ttft'] for x in r):7.2f} "
              f"{mean((x['completion_tokens'] or 0) for x in r):6.0f} | {len(own):6d} {sum((t['completion_tokens'] or 0) for t in own) / 1e6:6.2f} | "
              f"{mean(x[0] for x in dd):+7.3f} {mean(x[1] for x in dd):+7.3f} {mean(x[1] - x[0] for x in dd):+7.3f} | {sum(x['latency'] is None for x in r)} | {st.median(rate):6.1f}")
