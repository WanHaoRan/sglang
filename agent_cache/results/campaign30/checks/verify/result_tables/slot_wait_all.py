"""Slot wait over every user let in by 6,600 s (the client's 'plays' count), not only users with a recorded first request.
A user's start: its play record if it has one, else its recorded first request's send (when not retried), else bounds from
the 60 s 'lanes' lines (plays start in arrival order). nice -n 10 python3 -I -B slot_wait_all.py > slot_wait_all.out"""
import collections
import json
import os
import random

RES = "/home/wanhr/sglang/agent_cache/results"
MAN = os.path.join(RES, "campaign30/checks/vs_c34/manifest.txt")
CUT = 6600.0


def arrivals(n):
    rng, t, out = random.Random("0:arrivals"), 0.0, []
    for _ in range(n):
        out.append(t)
        t += rng.expovariate(0.1)
    return out


arr = arrivals(2000)
for line in open(MAN):
    if not line.strip():
        continue
    label, run, prog = line.split()[:3]
    rows = [json.loads(x) for x in open(os.path.join(RES, run, prog, "client.jsonl")) if x.strip()]
    lanes = sorted((r["t"], r["plays"]) for r in rows if r["kind"] == "lanes")
    plays = {r["play"]: r for r in rows if r["kind"] == "play"}
    firsts = {r["play"]: r for r in rows if r["kind"] == "req" and r.get("phase") == "profile"
              and r["stream"] == 0 and r["k"] == 0}
    n_in = dict(lanes)[min(lanes, key=lambda x: abs(x[0] - CUT))[0]]
    lo_w, hi_w, how = [], [], collections.Counter()
    script_w = [f["t_send"] - arr[p] for p, f in firsts.items()
                if f["t_send"] < CUT and f.get("ttft") is not None and not f.get("error")]
    for p in range(n_in):
        if p in plays:
            s = plays[p]["start"]
            lo_w.append(s - arr[p]); hi_w.append(s - arr[p]); how["play record"] += 1
        elif p in firsts and not firsts[p].get("retried"):
            s = firsts[p]["t_send"]
            lo_w.append(s - arr[p]); hi_w.append(s - arr[p]); how["first request"] += 1
        else:
            prev = max([t for t, n in lanes if n <= p], default=0.0)
            nxt = min(t for t, n in lanes if n > p)
            lo_w.append(max(prev, arr[p]) - arr[p]); hi_w.append(nxt - arr[p]); how["lanes bounds"] += 1
    m = lambda v: sum(v) / len(v)
    print(f"{label:18s} let in by 6600 s: {n_in:4d} | script (recorded first requests before the cut): n {len(script_w):4d} "
          f"mean {m(script_w):7.1f} max {max(script_w):7.1f} | all let in: mean {m(lo_w):7.1f} to {m(hi_w):7.1f}, "
          f"max {max(lo_w):7.1f} to {max(hi_w):7.1f} | starts from {dict(how)}")
