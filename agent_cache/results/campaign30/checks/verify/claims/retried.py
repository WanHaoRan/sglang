"""Requests the client retried (t_send and ttft then describe only the second attempt), per arm."""
import json, os, sys
base = "/home/wanhr/sglang/agent_cache/results"
for line in open(sys.argv[1]):
    if not line.strip():
        continue
    label, run, prog = line.split()[:3]
    reqs = [r for r in map(json.loads, open(os.path.join(base, run, prog, "client.jsonl"))) if r["kind"] == "req"]
    ret = [r for r in reqs if r.get("retried")]
    errs = [r for r in reqs if r.get("error")]
    print(f"{label:18s} retried {len(ret):3d}; errors {len(errs)}; retried with t_send>=6600: {sum(1 for r in ret if r['t_send']>=6600)}")
    for r in ret:
        print(f"    {r['rid']:28s} play {r['play']:4d} s{r['stream']} k{r['k']:3d} t_send {r['t_send']:8.1f} ttft {r.get('ttft')} err {r.get('error')}")
