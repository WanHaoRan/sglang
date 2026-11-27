"""First (stream 0, k 0) vs returning requests by send-time window, from raw client.jsonl, for every arm in a manifest.
Also: subagent stream first requests (stream>0, k 0), users let in per window (lanes 'plays'), cancelled line."""
import json, os, sys
base = "/home/wanhr/sglang/agent_cache/results"
man = sys.argv[1]
W = ((0, 600), (600, 1800), (1800, 3600), (3600, 6600))
m = lambda v: sum(v) / len(v) if v else float("nan")
for line in open(man):
    if not line.strip():
        continue
    label, run, prog = line.split()[:3]
    lanes, reqs, canc = {}, [], None
    for l in open(os.path.join(base, run, prog, "client.jsonl")):
        r = json.loads(l)
        if r["kind"] == "lanes":
            lanes[round(r["t"])] = r
        elif r["kind"] == "req" and r["phase"] == "profile":
            reqs.append(r)
        elif r["kind"] == "cancelled":
            canc = r
    ok = [r for r in reqs if r.get("ttft") is not None]
    print(f"== {label}  cancelled={canc}")
    for a, b in W:
        f = [r["ttft"] for r in ok if a <= r["t_send"] < b and r["stream"] == 0 and r["k"] == 0]
        sub = [r["ttft"] for r in ok if a <= r["t_send"] < b and r["stream"] > 0 and r["k"] == 0]
        g = [r["ttft"] for r in ok if a <= r["t_send"] < b and not (r["stream"] == 0 and r["k"] == 0)]
        g2 = [r["ttft"] for r in ok if a <= r["t_send"] < b and r["k"] > 0]
        pa = lanes.get(a, {}).get("plays", 0) if a else 0
        pb = lanes.get(b, {}).get("plays")
        let_in = None if pb is None else pb - pa
        print(f"  {a//60:3d}-{b//60:3d}: first n={len(f):4d} mean={m(f):8.1f} | returning(all non-first) n={len(g):5d} mean={m(g):7.2f} | k>0 n={len(g2):5d} mean={m(g2):7.2f} | sub-stream firsts n={len(sub):3d} mean={m(sub):7.2f} | users let in in window={let_in}")
    started = lanes[6600]["plays"]
    firsts = {r["play"] for r in reqs if r["stream"] == 0 and r["k"] == 0 and r["t_send"] < 6600}
    print(f"  let in by 6600: {started}; recorded first (incl. errors): {len(firsts)}; without: {started-len(firsts)}; in flight at 6600: {lanes[6600]['in_flight']}")
