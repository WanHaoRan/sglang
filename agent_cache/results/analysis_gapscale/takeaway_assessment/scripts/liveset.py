"""Live set of session-private KV over time from client.jsonl.

L(t) = 7,808 shared prompt + sum over sessions that have started and not yet
finished their last turn of (latest completed turn's prompt_tokens +
completion_tokens - 7,808), page-floored to 64.
Reports peak L, and the time L first exceeds H (inclusive capacity ~= H) and
H + D (exclusive capacity).
"""
import json, sys, glob, subprocess
from datetime import datetime, timezone

H, D, SYS, PG = 1_627_648, 668_160, 7_808, 64
R = "/home/wanhr/sglang/agent_cache/results/"

for run in sys.argv[1:]:
    path = glob.glob(R + run + "/client_*/client.jsonl")[0]
    turns, last_turn = [], {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["kind"] == "run":
                t_start = r["t_start"]
            elif r["kind"] == "conv_start":
                last_turn[r["conv"]] = r["turns"] - 1
            elif r["kind"] == "turn":
                turns.append(r)
    ev = []  # (time, conv, new_size or None for end)
    for r in turns:
        tdone = r["t_send"] + r["latency"]
        size = ((r["prompt_tokens"] + r["completion_tokens"]) // PG) * PG - SYS
        ev.append((r["t_send"], r["conv"], "start", None))
        ev.append((tdone, r["conv"], "size", size))
        if r["turn"] == last_turn.get(r["conv"]):
            ev.append((tdone + 1e-6, r["conv"], "end", None))
    ev.sort(key=lambda e: e[0])
    cur, active, L = {}, set(), SYS
    peak, t_h, t_hd, series = 0, None, None, []
    for t, c, kind, size in ev:
        if kind == "start":
            active.add(c)
        elif kind == "size" and c in active:
            L += size - cur.get(c, 0); cur[c] = size
        elif kind == "end":
            active.discard(c); L -= cur.pop(c, 0)
        series.append((t, L))
        if L > peak: peak, tpeak = L, t
        if t_h is None and L >= H: t_h = t
        if t_hd is None and L >= H + D: t_hd = t
    T = series[-1][0]
    above_h = sum(series[i+1][0]-series[i][0] for i in range(len(series)-1) if series[i][1] >= H)
    above_hd = sum(series[i+1][0]-series[i][0] for i in range(len(series)-1) if series[i][1] >= H + D)
    fmt = lambda x: "never" if x is None else f"{x/60:.1f} min"
    print(f"== {run}: t_start={datetime.fromtimestamp(t_start, timezone.utc):%Y-%m-%d %H:%M:%S} UTC, duration {T/60:.0f} min")
    print(f"  live set peak={peak:,} at {tpeak/60:.1f} min ({peak/H:.2f} H, {peak/(H+D):.2f} (H+D))")
    print(f"  L>=H first at {fmt(t_h)}, time above H={above_h/60:.0f} min ({above_h/T:.0%});"
          f" L>=H+D first at {fmt(t_hd)}, time above H+D={above_hd/60:.0f} min ({above_hd/T:.0%})")
