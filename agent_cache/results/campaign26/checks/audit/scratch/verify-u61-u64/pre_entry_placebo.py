#!/usr/bin/env python3
"""Verifier's probe for U64, control: is the time above the per-token line tied to a prefill in progress at the
moment the request reaches the scheduler, or only to busy stretches? Same join as pre_entry_probe.py, with the
moment shifted by a few seconds (a busy stretch lasts minutes, a prefill step under a few seconds).

usage: pre_entry_placebo.py RUN_DIR [RUN_DIR ...]"""
import bisect, glob, json, os, re, statistics as st, sys

RTS = re.compile(r"ReqTimeStats\(rid=\S+?-c(\d+)-t(\d+), input_len=(\d+), cached_input_len=(\d+), output_len=(\d+).*"
                 r"queue_duration=([\d.]+)ms, forward_duration=([\d.]+)ms, entry_time=([\d.]+)")
A5 = {"20261112_045511": (0.0064, 1.81e-6), "20261111_130801": (0.0065, 1.63e-6)}  # p5 lines from pre_entry_probe.py

for run in sys.argv[1:]:
    a5, b5 = next(v for k, v in A5.items() if k in run)
    path = sorted(glob.glob(os.path.join(run, "client_*", "client.jsonl")))[-1]
    rows = [json.loads(l) for l in open(path) if l.strip()]
    t0 = next(r["t_start"] for r in rows if r["kind"] == "run")
    turns = {(r["conv"], r["turn"]): r for r in rows if r["kind"] == "turn" and r.get("ttft") is not None}
    srv = {}
    for line in open(os.path.join(run, "server.log"), errors="replace"):
        if "ReqTimeStats(" in line and (m := RTS.search(line)):
            srv[(int(m[1]), int(m[2]))] = dict(inp=int(m[3]), queue=float(m[6]) / 1000, entry=float(m[8]))
    allreq = []
    for k, r in turns.items():
        if k in srv:
            s = srv[k]
            send = t0 + r["t_send"]
            allreq.append(dict(k=k, send=send, exc=max(0.0, s["entry"] - send - a5 - b5 * s["inp"]), prompt=s["inp"],
                               fstart=s["entry"] + s["queue"], fend=send + r["ttft"], ret=k[1] > 0 and not r.get("mid_start")))
    rec = [x for x in allreq if x["ret"]]
    iv = sorted((x["fstart"], x["fend"], x["k"]) for x in allreq if x["fend"] > x["fstart"])
    starts = [i[0] for i in iv]
    longest = max(e - s for s, e, _ in iv)
    print(f"\n{os.path.basename(run)}: {len(rec)} returning turns; mean seconds above the per-token line, by whether another request's prefill")
    print("  was in progress at (the moment the request reaches the scheduler + shift): shift, turns inside, mean inside, mean outside")
    for shift in (0.0, 2.0, 5.0, -5.0, 30.0, -30.0):
        ins, outs = [], []
        for x in rec:
            t = x["send"] + a5 + b5 * x["prompt"] + shift
            i = bisect.bisect_right(starts, t) - 1
            hit = False
            while i >= 0 and starts[i] >= t - longest:
                s, e, k = iv[i]
                if k != x["k"] and e > t:
                    hit = True
                    break
                i -= 1
            (ins if hit else outs).append(x["exc"])
        print(f"    {shift:+6.1f} s: {len(ins):5d} ({len(ins) / len(rec):.0%})  {st.mean(ins):.3f} s  {st.mean(outs):.3f} s")
