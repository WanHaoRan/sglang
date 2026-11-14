#!/usr/bin/env python3
"""Verifier's probe for U64, second cut (read-only, stdlib): the time before the scheduler has the request,
against what the same prompt length costs a turn that met no forward step of another request's prefill.

'Clean' turns: no other request between its admission and its first token at any moment from this turn's send to
its entry. Their mean by prompt length is the cost with nothing in the way; the rest of the run's seconds is waiting.

usage: pre_entry_clean.py RUN_DIR [RUN_DIR ...]"""
import bisect, glob, json, os, re, statistics as st, sys

RTS = re.compile(r"ReqTimeStats\(rid=\S+?-c(\d+)-t(\d+), input_len=(\d+), cached_input_len=(\d+), output_len=(\d+).*"
                 r"queue_duration=([\d.]+)ms, forward_duration=([\d.]+)ms, entry_time=([\d.]+)")


def fit(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


for run in sys.argv[1:]:
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
            allreq.append(dict(k=k, send=send, entry=s["entry"], pre=s["entry"] - send, prompt=s["inp"], ttft=r["ttft"],
                               fstart=s["entry"] + s["queue"], fend=send + r["ttft"], ret=k[1] > 0 and not r.get("mid_start")))
    rec = [x for x in allreq if x["ret"]]
    iv = sorted((x["fstart"], x["fend"], x["k"]) for x in allreq if x["fend"] > x["fstart"])
    starts = [i[0] for i in iv]
    longest = max(e - s for s, e, _ in iv)
    for x in rec:
        # any other request's prefill overlapping [send, entry]
        lo, hi = x["send"], x["entry"]
        i = bisect.bisect_right(starts, hi) - 1
        hit = False
        while i >= 0 and starts[i] >= lo - longest:
            s, e, k = iv[i]
            if k != x["k"] and e > lo and s < hi:
                hit = True
                break
            i -= 1
        x["clean"] = not hit
    clean = [x for x in rec if x["clean"]]
    a, b = fit([x["prompt"] for x in clean], [x["pre"] for x in clean])
    tot = sum(x["pre"] for x in rec)
    base = sum(a + b * x["prompt"] for x in rec)
    ttft = sum(x["ttft"] for x in rec)
    print(f"\n{os.path.basename(run)}: {len(rec)} returning turns, {len(clean)} ({len(clean) / len(rec):.0%}) met no other request's prefill between send and entry")
    print(f"  those turns: before entry mean {st.mean(x['pre'] for x in clean):.3f} s, median {st.median(x['pre'] for x in clean):.3f}; "
          f"least squares on prompt length: {a * 1000:.1f} ms + {b * 1e6:.2f} us per token")
    print(f"  the others:  before entry mean {st.mean(x['pre'] for x in rec if not x['clean']):.3f} s, median {st.median(x['pre'] for x in rec if not x['clean']):.3f}")
    print(f"  all turns: {tot:.0f} s before entry = {base:.0f} s at the clean turns' line ({base / tot:.0%}; {base / len(rec):.3f} s a turn; {base / ttft:.1%} of all TTFT seconds) "
          f"+ {tot - base:.0f} s more ({(tot - base) / tot:.0%}; {(tot - base) / len(rec):.3f} s a turn; {(tot - base) / ttft:.1%} of all TTFT seconds)")
    print(f"  mean TTFT {ttft / len(rec):.3f} s; before entry is {tot / ttft:.1%} of all TTFT seconds")
