"""Referee: does the simulated client's schedule (seed 0) equal the recorded one? For each arm of campaigns 34 and 32,
every recorded request (play, stream, k) is compared with the workload's: prompt tokens, output tokens, wait, gap
kind, key; every recorded play with its arrival; and the hint the recorded client sent with the table the simulator
would build from the same send order (first 1,200 s is the builder's check; here the whole run, replayed in the
recorded send order so that interleaving is not a difference)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import kvsim_trace as kt
import reflib

w = kt.Workload(seed=0)
req = {(r.play.pid, r.s, r.k): r for r in w.requests}
arr = {p.pid: p.arrival for p in w.plays}
for camp in ("campaign34", "campaign32"):
    for arm, path in reflib.manifest(camp).items():
        n = bad = miss = 0
        plays = bad_arr = 0
        hints = hint_same = hint_diff = 0
        hinter = kt.Hinter("dist", hint_no_call=True, deadline=7200.0)
        rows = []
        with open(path) as f:
            for line in f:
                if line.startswith('{"kind":"req"'):
                    rows.append(json.loads(line))
                elif line.startswith('{"kind":"play"'):
                    d = json.loads(line)
                    plays += 1
                    if d["play"] not in arr or abs(arr[d["play"]] - d["arrival_s"]) > 2e-3:
                        bad_arr += 1
        rows.sort(key=lambda d: d["t_send"])
        for d in rows:
            n += 1
            r = req.get((d["play"], d["stream"], d["k"]))
            if r is None:
                miss += 1
                continue
            gap = kt.GAP_NAMES[r.gap]
            if (r.prompt != d["prompt_sent"] or r.out != d["out_target"] or abs(r.wait - d["wait_s"]) > 1e-3
                    or gap != d.get("gap") or r.key != d.get("key") or r.play.tree["id"] != d["tree"]):
                bad += 1
            h = d.get("hint")
            if h is not None and "matched" in h and h.get("list_len") is not None:
                # replay the table in the recorded send order
                got = hinter.at_send(r, d["t_send"])
                hints += 1
                rec_p50 = h.get("point_s")
                if got is None or got.get("end"):
                    same = rec_p50 is None
                else:
                    same = rec_p50 is not None and abs(got["p50"] - rec_p50) <= 1e-6 * max(1.0, rec_p50) + 5e-4
                hint_same += same
                hint_diff += not same
        print(f"{camp}/{arm}: {n} recorded requests, {miss} not in the workload, {bad} differ (prompt, output, wait, gap kind, key or tree); "
              f"{plays} plays, {bad_arr} with another arrival time; table hints replayed in the recorded order: {hints}, "
              f"same p50 (or none in both) {hint_same}, different {hint_diff}", flush=True)
