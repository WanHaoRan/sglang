"""Lost history (ideal_cached - cached_tokens) per run, by time and size class; cleaner passes from the logs.
Client records only (finished requests); requests sent before 6,600 s unless stated."""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib, runtab
from vlib import mean, pct

CL = re.compile(r"deleted (\d+) groups / (\d+) files \(([0-9.]+) GiB\) in ([0-9.]+)s")
for camp, arm in [(34, "ec_full"), (34, "jit_exact"), (34, "ec_exact"), (34, "jit_p50"), (34, "hicache")]:
    tab = runtab.build(camp, arm); reqs = tab["reqs"]; ev = vlib.load(camp, arm)
    passes = []
    for t, r in ev.get("cleanup", []):
        m = CL.search(r)
        passes.append((t, int(m.group(1)), float(m.group(4))))
    R = [q for q in reqs.values() if q["t_send"] < 6600]
    lost = lambda q: max(q["ideal"] - q["cached"], 0)
    tot = sum(lost(q) for q in R)
    newtxt = sum(q["prompt"] - q["ideal"] for q in R)
    first = sum(q["prompt"] - q["ideal"] for q in R if q["k"] == 0 and q["stream"] == 0)
    print(f"=== c{camp} {arm}: requests sent before 6,600 s {len(R)}; mean TTFT {mean(q['ttft'] for q in R):.2f} s; passes (end time, groups, took s): {[(round(t), g, d) for t, g, d in passes]}")
    print(f"  new text (prompt - ideal) {newtxt/1e6:.2f} M (first requests of sessions {first/1e6:.2f} M); lost history {tot/1e6:.3f} M tokens = {tot*0.043e-3:.1f} GPU-s at 0.043 ms")
    small = sum(lost(q) for q in R if lost(q) < 256)
    print(f"  lost < 256 tokens a request (page rounding): {small/1e3:.0f} K over {sum(1 for q in R if 0 < lost(q) < 256)} requests; lost >= 256: {(tot-small)/1e6:.3f} M over {sum(1 for q in R if lost(q) >= 256)} requests")
    if passes:
        p1 = passes[0][0] - passes[0][2]   # scan start of the first pass
        pre = [q for q in R if q["t_send"] < p1]
        post = [q for q in R if q["t_send"] >= p1]
        print(f"  first pass starts at {p1:.0f} s. before it: {sum(1 for q in pre if lost(q) >= 256)} requests lost >= 256 tokens ({sum(lost(q) for q in pre if lost(q) >= 256)/1e3:.0f} K); {sum(1 for q in pre if lost(q) >= 1024)} lost >= 1,024")
        b256 = sum(lost(q) for q in post if lost(q) >= 256)
        print(f"  from the first pass to 6,600 s: {len(post)} requests; lost >= 256: {sum(1 for q in post if lost(q) >= 256)} requests, {b256/1e6:.3f} M; lost >= 1,024: {sum(1 for q in post if lost(q) >= 1024)} requests, {sum(lost(q) for q in post if lost(q) >= 1024)/1e6:.3f} M;"
              f" of them first requests of a session: {sum(1 for q in post if lost(q) >= 256 and q['k']==0 and q['stream']==0)} requests, {sum(lost(q) for q in post if lost(q) >= 256 and q['k']==0 and q['stream']==0)/1e3:.0f} K")
        # the pre-pass rate of >=256 losses per request as the 'nothing deleted' baseline for the same number of requests
        w0 = [q for q in R if p1 - 1337 <= q["t_send"] < p1]
        base = sum(lost(q) for q in w0 if lost(q) >= 256)
        print(f"  the {1337} s before the pass: {len(w0)} requests, lost >= 256: {sum(1 for q in w0 if lost(q) >= 256)} requests, {base/1e3:.0f} K  -> excess after the pass about {(b256-base)/1e6:.3f} M")
        m90 = [q for q in R if 5400 <= q["t_send"] < 6600]
        l90 = sum(lost(q) for q in m90 if lost(q) >= 256)
        n = len(m90)
        print(f"  minutes 90-110: {n} requests, mean TTFT {mean(q['ttft'] for q in m90):.2f} s, lost >= 256: {l90/1e6:.3f} M = {l90*0.043e-3:.1f} GPU-s;  by N g with N=128: {128*l90*0.043e-3/n:.2f} s; with N-B=108.3: {108.3*l90*0.043e-3/n:.2f} s per request")
        # TTFT of the losers against their window's mean
        losers = [q for q in m90 if lost(q) >= 1024]
        if losers:
            print(f"  minutes 90-110: TTFT of the {len(losers)} requests that lost >= 1,024 tokens: mean {mean(q['ttft'] for q in losers):.1f} s  (all {mean(q['ttft'] for q in m90):.1f} s)")
