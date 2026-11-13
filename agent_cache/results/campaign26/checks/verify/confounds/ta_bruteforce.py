"""Check of test (a)'s arithmetic: the load state at a turn's send recomputed by brute force (a scan over every session
and every request for each sampled turn) against the heap-based routine used by ta_load.py and lib2.load_state.
  nice -n 19 python3 -I -B ta_bruteforce.py"""
import os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import lib2 as L

rng = random.Random(0)
for a in C.LEAD5 + ["ec_r13_rep"]:
    d = C.load(a)
    turns = sorted(d["turns"], key=lambda t: t["t_send"])
    conv = {}
    for t in turns:
        conv.setdefault(t["conv"], []).append(t)
    for fix in (False, True):
        fast = L.load_state(d, fix)
        bad = 0
        sample = rng.sample(turns, 400)
        for t in sample:
            now = t["t_send"]
            n = ctx = fl = flp = 0
            for c, v in conv.items():
                end = v[-1]["t_send"] + (v[-1]["latency"] or 0.0)
                if fix:
                    ce = d["conv_end"].get(str(c))
                    if ce is None or ce.get("capped"):
                        end = float("inf")
                if v[0]["t_send"] <= now < end:
                    latest = [x for x in v if x["t_send"] <= now][-1]
                    n += 1
                    ctx += (latest["pt"] or 0) + (latest["ct"] or 0)
                for x in v:
                    if x["t_send"] <= now < x["t_send"] + (x["latency"] or 0.0):
                        fl += 1
                        flp += x["pt"] or 0
            if (n, ctx, fl, flp) != tuple(fast[(t["conv"], t["turn"])]):
                bad += 1
        print(f"{a:11s} censoring fix {str(fix):5s}: {len(sample)} sampled turns, {bad} differ from the heap-based routine")
# and the cached states written by ta_load.py on 2026-11-11 against the routine in lib2
import json
for name, fix in (("load_state.json", False), ("load_state_fix.json", True)):
    saved = json.load(open(os.path.join(C.CACHE, name)))
    for a in C.LEAD5:
        fast = L.load_state(C.load(a), fix)
        diff = sum(1 for k, v in fast.items() if tuple(saved[a][f"{k[0]}-{k[1]}"]) != tuple(v))
        print(f"{name:20s} {a:11s} {len(fast)} turns, {diff} differ from lib2.load_state")

print("\nwhere the two differ: turns of the run with the same send time to 0.1 ms as another turn (ties are counted in sort order by the routine, together by the scan)")
for a in C.LEAD5 + ["ec_r13_rep"]:
    ts = sorted(t["t_send"] for t in C.load(a)["turns"])
    ties = sum(1 for x, y in zip(ts, ts[1:]) if y - x < 1e-4)
    print(f"  {a:11s} {ties} of {len(ts)} sends within 0.1 ms of the previous send")
