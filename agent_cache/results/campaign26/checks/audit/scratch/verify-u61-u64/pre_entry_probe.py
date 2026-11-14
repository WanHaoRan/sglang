#!/usr/bin/env python3
"""Verifier's probe for U64 (read-only, stdlib): what the logs alone can say about the time before the
scheduler has the request (floor.py's 'before entry' = server entry_time - client send time).

Per run: (1) quantiles of that time by prompt length, to see whether its lower envelope grows with length
(work done per request) or only its mean (waiting); (2) how much of it sits above the envelope; (3) whether
the part above the envelope goes with another request's prefill step being in progress (the scheduler receives
between steps) or with other requests sent just before (one tokenizer worker).

usage: pre_entry_probe.py RUN_DIR [RUN_DIR ...]   (absolute run folders)"""
import bisect, glob, json, os, re, statistics as st, sys

RTS = re.compile(r"ReqTimeStats\(rid=\S+?-c(\d+)-t(\d+), input_len=(\d+), cached_input_len=(\d+), output_len=(\d+).*"
                 r"queue_duration=([\d.]+)ms, forward_duration=([\d.]+)ms, entry_time=([\d.]+)")


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")


def fit(xs, ys, ws):
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    return my - b * mx, b


def corr(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sxx * syy) ** 0.5


for run in sys.argv[1:]:
    path = sorted(glob.glob(os.path.join(run, "client_*", "client.jsonl")))[-1]
    rows = [json.loads(l) for l in open(path) if l.strip()]
    t0 = next(r["t_start"] for r in rows if r["kind"] == "run")
    turns = {(r["conv"], r["turn"]): r for r in rows if r["kind"] == "turn" and r.get("ttft") is not None}
    srv = {}
    for line in open(os.path.join(run, "server.log"), errors="replace"):
        if "ReqTimeStats(" in line and (m := RTS.search(line)):
            srv[(int(m[1]), int(m[2]))] = dict(inp=int(m[3]), cached=int(m[4]), queue=float(m[6]) / 1000,
                                               fwd=float(m[7]) / 1000, entry=float(m[8]))
    allreq = []  # every turn with both logs, first turns included: they occupy the tokenizer and the GPU too
    for k, r in turns.items():
        if k not in srv:
            continue
        s = srv[k]
        send = t0 + r["t_send"]
        allreq.append(dict(k=k, send=send, pre=s["entry"] - send, prompt=s["inp"], new=s["inp"] - s["cached"],
                           fstart=s["entry"] + s["queue"], fend=send + r["ttft"], ret=k[1] > 0 and not r.get("mid_start")))
    rec = [x for x in allreq if x["ret"]]
    n = len(rec)
    tot = sum(x["pre"] for x in rec)
    print(f"\n{os.path.basename(run)}: {n} returning turns; before entry mean {tot / n:.3f} s, median {st.median(x['pre'] for x in rec):.3f}, "
          f"min {min(x['pre'] for x in rec):.3f}, p99 {q([x['pre'] for x in rec], .99):.3f}, max {max(x['pre'] for x in rec):.2f}; "
          f"negative {sum(1 for x in rec if x['pre'] < 0)}")
    print("  by prompt length (tokens): n, p5, p25, p50, p75, p95, mean  (s)")
    xs, y5, y50, ws = [], [], [], []
    for lo in range(0, 96000, 8000):
        v = [x for x in rec if lo <= x["prompt"] < lo + 8000]
        if len(v) < 30:
            continue
        p = [x["pre"] for x in v]
        print(f"    {lo:6d}-{lo + 8000:6d}: {len(v):5d}  {q(p, .05):.3f} {q(p, .25):.3f} {q(p, .5):.3f} {q(p, .75):.3f} {q(p, .95):.3f}  {st.mean(p):.3f}")
        xs.append(st.mean(x["prompt"] for x in v)); y5.append(q(p, .05)); y50.append(q(p, .5)); ws.append(len(v))
    a5, b5 = fit(xs, y5, ws)
    a50, b50 = fit(xs, y50, ws)
    print(f"  line through the bins' p5:     {a5 * 1000:.1f} ms + {b5 * 1e6:.2f} us per prompt token")
    print(f"  line through the bins' median: {a50 * 1000:.1f} ms + {b50 * 1e6:.2f} us per prompt token")
    floor = lambda L: a5 + b5 * L
    for x in allreq:
        x["exc"] = max(0.0, x["pre"] - floor(x["prompt"]))
    exc = sum(x["exc"] for x in rec)
    print(f"  seconds before entry: {tot:.0f} = {tot - exc:.0f} on or under the p5 line ({(tot - exc) / tot:.0%}) + {exc:.0f} above it ({exc / tot:.0%}); "
          f"mean above {exc / n:.3f} s, median {st.median(x['exc'] for x in rec):.3f}")
    # (3a) was another request's prefill step in progress when this one would have reached the scheduler?
    iv = sorted((x["fstart"], x["fend"], x["k"]) for x in allreq if x["fend"] > x["fstart"])
    starts = [i[0] for i in iv]
    longest = max(e - s for s, e, _ in iv)
    for x in rec:
        t = x["send"] + floor(x["prompt"])
        j = bisect.bisect_right(starts, t)
        blk, blk_big = 0.0, 0.0
        i = j - 1
        while i >= 0 and starts[i] >= t - longest:
            s, e, k = iv[i]
            if k != x["k"] and e > t:
                blk = max(blk, e - t)
            i -= 1
        x["blk"] = blk
    inside = [x for x in rec if x["blk"] > 0]
    outside = [x for x in rec if x["blk"] == 0]
    print(f"  reached the scheduler while another request's prefill was in progress (its admission to its first token): {len(inside)} turns ({len(inside) / n:.0%}); "
          f"mean above the line {st.mean(x['exc'] for x in inside):.3f} s against {st.mean(x['exc'] for x in outside):.3f} s for the other {len(outside)}; "
          f"they hold {sum(x['exc'] for x in inside) / exc:.0%} of the seconds above the line")
    print(f"    correlation of the seconds above the line with the time left in that prefill: all {corr([x['blk'] for x in rec], [x['exc'] for x in rec]):.2f}, "
          f"those turns {corr([x['blk'] for x in inside], [x['exc'] for x in inside]):.2f}")
    for lo, hi in ((0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 1e9)):
        v = [x for x in inside if lo <= x["blk"] < hi]
        if v:
            print(f"      time left {lo:4.2f}-{hi if hi < 1e8 else float('inf'):4.2f} s: {len(v):5d} turns, above the line mean {st.mean(x['exc'] for x in v):.3f} median {st.median(x['exc'] for x in v):.3f}")
    # (3b) other requests sent in the 0.15 s before this one (one tokenizer worker)
    sends = sorted(x["send"] for x in allreq)
    for x in rec:
        x["near"] = bisect.bisect_left(sends, x["send"]) - bisect.bisect_left(sends, x["send"] - 0.15)
    for lab, sel in (("none", lambda c: c == 0), ("one", lambda c: c == 1), ("two or more", lambda c: c >= 2)):
        v = [x for x in rec if sel(x["near"])]
        if v:
            vo = [x for x in v if x["blk"] == 0]
            print(f"  other requests sent in the 0.15 s before: {lab:12s} {len(v):5d} turns, above the line mean {st.mean(x['exc'] for x in v):.3f} s"
                  + (f"; of them with no prefill in progress {len(vo)}: {st.mean(x['exc'] for x in vo):.3f} s" if vo else ""))
