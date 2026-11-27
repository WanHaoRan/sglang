"""Requests sent before a cut that the client never recorded (still in flight when it stopped at 7,230 s, cancelled).
Per arm: in flight at the cut (the client's 'lanes' line at that second) minus the recorded requests sent before the
cut that ended after it. Usage, from agent_cache/results:
  python3 -I -B campaign30/checks/missing_at_cut.py <comparison folder> [cut s ...]"""
import json, os, sys

folder, cuts = sys.argv[1], [float(c) for c in sys.argv[2:]] or [6600.0]
for line in open(os.path.join(folder, "manifest.txt")):
    if not line.strip():
        continue
    label, run, prog = line.split()[:3]
    lanes, reqs = {}, []
    for l in open(os.path.join(run, prog, "client.jsonl")):
        r = json.loads(l)
        if r["kind"] == "lanes":
            lanes[round(r["t"])] = r["in_flight"]
        elif r["kind"] == "req" and r["phase"] == "profile":
            reqs.append(r)
    out = []
    for cut in cuts:
        fl = lanes.get(round(cut))
        ended_after = sum(1 for r in reqs if r["t_send"] < cut and r["t_send"] + (r.get("latency") or 0) > cut)
        sent = sum(1 for r in reqs if r["t_send"] < cut and r.get("ttft") is not None)
        miss = None if fl is None else fl - ended_after
        out.append(f"cut {cut:.0f}: in flight {fl}, recorded ending later {ended_after}, never recorded {miss} (recorded before the cut {sent})")
    print(f"{label:18s} " + " | ".join(out))
