"""The Continuum arms' retried and failed requests in context: each one's play, with the requests before it.
nice -n 10 python3 -I -B retried.py > retried.out"""
import json
import os

RES = "/home/wanhr/sglang/agent_cache/results"
ARMS = (("continuum", "20261124_203658_continuum_continuum/programs_203842_continuum"),
        ("continuum_lmcache", "20261124_224027_continuum_continuum_lmcache/programs_224335_continuum_lmcache"))

for label, rel in ARMS:
    rows = [json.loads(line) for line in open(os.path.join(RES, rel, "client.jsonl")) if line.strip()]
    reqs = [r for r in rows if r["kind"] == "req" and r.get("phase") == "profile"]
    plays = {r["play"]: r for r in rows if r["kind"] == "play"}
    odd = [r for r in reqs if r.get("retried") or r.get("error")]
    print(f"== {label}: {len(odd)} retried or failed requests")
    for o in odd:
        p = o["play"]
        print(f" play {p} (play record: {plays.get(p)})")
        same = sorted((r for r in reqs if r["play"] == p and r["stream"] == o["stream"]), key=lambda r: r["k"])
        for r in same:
            if abs(r["k"] - o["k"]) > 2:
                continue
            end = r["t_send"] + (r.get("latency") or 0)
            print(f"   s{r['stream']} k{r['k']:3d} {r['rid']:28s} wait_s {r['wait_s']:8.3f} t_send {r['t_send']:9.3f} "
                  f"ttft {r.get('ttft')} latency {r.get('latency')} end {end:9.3f} retried {r.get('retried')} "
                  f"error {(r.get('error') or '')[:50]}")
        if o["k"] > 0:
            prev = [r for r in same if r["k"] == o["k"] - 1]
            if prev:
                pr = prev[0]
                due = pr["t_send"] + (pr.get("latency") or 0) + o["wait_s"]
                print(f"   -> previous ended {pr['t_send'] + pr['latency']:.3f}; this one due about {due:.3f}; "
                      f"recorded t_send {o['t_send']:.3f} (gap {o['t_send'] - due:.1f} s)")
        else:
            print(f"   -> first request; recorded t_send {o['t_send']:.3f}; minus the 3600 s socket timeout {o['t_send'] - 3600:.3f}")
