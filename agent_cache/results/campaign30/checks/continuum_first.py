"""Continuum's queue: a user's first request (stream 0, index 0) against returning requests, by send time, and how many
users had been let in by a cut without any recorded first request (it was still waiting when the client stopped).
Usage, from agent_cache/results: python3 -I -B campaign30/checks/continuum_first.py <run>/<programs> [...]"""
import json, sys

WINDOWS = ((0, 600), (600, 1800), (1800, 3600), (3600, 6600))
for path in sys.argv[1:]:
    lanes, reqs = {}, []
    for l in open(f"{path}/client.jsonl"):
        r = json.loads(l)
        if r["kind"] == "lanes":
            lanes[round(r["t"])] = r
        elif r["kind"] == "req" and r["phase"] == "profile":
            reqs.append(r)
    ok = [r for r in reqs if r.get("ttft") is not None]
    print(path)
    for a, b in WINDOWS:
        f = [r["ttft"] for r in ok if a <= r["t_send"] < b and r["stream"] == 0 and r["k"] == 0]
        g = [r["ttft"] for r in ok if a <= r["t_send"] < b and not (r["stream"] == 0 and r["k"] == 0)]
        m = lambda v: sum(v) / len(v) if v else float("nan")
        print(f"  minutes {a // 60}-{b // 60}: first requests {len(f)} mean {m(f):.1f} s; returning {len(g)} mean {m(g):.2f} s")
    cut = 6600
    started = lanes[cut]["plays"]
    firsts = {r["play"] for r in reqs if r["stream"] == 0 and r["k"] == 0 and r["t_send"] < cut}
    print(f"  users let in by {cut} s: {started}; with a recorded first request (or its error): {len(firsts)}; "
          f"without: {started - len(firsts)}; in flight at {cut} s per the client: {lanes[cut]['in_flight']}")
