"""Share of the earlier conversation (ideal_cached: what an unbounded cache would hold) that each arm recomputed,
aggregate (1 - cached/ideal) and clamped per request (sum max(0, ideal - cached) / sum ideal). Read-only."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
for line in open(os.path.join(RESULTS, "campaign30", "checks", "vs_c34", "manifest.txt")):
    label, run, prog = line.split()
    p = c = i = lost = over = 0
    n = 0
    for l in open(os.path.join(RESULTS, run, prog, "client.jsonl")):
        r = json.loads(l)
        if r["kind"] != "req" or r["phase"] != "profile" or r.get("ttft") is None or r.get("error") or r["t_send"] >= 6600:
            continue
        d = r.get("cached_details") or {}
        cached = sum(int(d.get(k) or 0) for k in ("device", "host", "storage")) if d else int(r.get("cached_tokens") or 0)
        ideal = int(r.get("ideal_cached") or 0)
        p += r["prompt_tokens"]; c += cached; i += ideal; lost += max(0, ideal - cached); over += max(0, cached - ideal); n += 1
    print(f"{label:18s} n {n:6d} prompt {p/1e6:7.1f} M | recomputed of prompt {100*(1-c/p):5.1f} % | new text {100*(1-i/p):4.1f} % "
          f"| history recomputed: aggregate {100*(1-c/i):5.1f} %, clamped per request {100*lost/i:5.1f} % | cached beyond ideal {100*over/i:4.2f} %")
