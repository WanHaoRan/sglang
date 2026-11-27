"""ev_write 3b: what fix 3's write holds cost readers.  For each hold [on, off): timed reads (deadline != now)
parked during it, their parked seconds inside the hold, and the seconds the read thread sat idle inside the hold
while a timed read was parked.  Also what was pending at the hold's start: the pending ops and whether the
largest is the prompt write of a request still running (matched on admission time and uncached tokens)."""
import sys, os, bisect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib, runtab
from vlib import kv, pct
for arm in ("ec_full", "jit_exact", "jit_p50", "ec_exact"):
    ev = vlib.load(34, arm); tab = runtab.build(34, arm)
    reqs, reads = tab["reqs"], tab["reads"]
    holds = []; on = None
    for t, r in ev["srs_write_hold"]:
        d = kv(r)
        if d["state"] == "on": on = t
        elif on is not None: holds.append((on, t)); on = None
    holds = [h for h in holds if h[0] < 7200]
    io = sorted((r["io_end"] - r["io_ms"], r["io_end"]) for r in reads if r["io_end"] is not None)
    timed = [(r["park"], r["disp"] if r["disp"] is not None else 1e18, r["pages"]) for r in reads if r["deadline"] != "now"]
    alltimed_park = sum(min(b, 7200) - a for a, b, p in timed if b < 1e17 and a < 7200)
    n_across = set(); park_in = 0.0; idle_in = 0.0; tok_across = 0; reader_sec = 0.0
    for a, b in holds:
        # timed reads parked at some moment of the hold
        ps = [(max(pa, a), min(pb, b), pg, i) for i, (pa, pb, pg) in enumerate(timed) if pa < b and pb > a]
        for x0, x1, pg, i in ps:
            if i not in n_across: tok_across += pg * 64
            n_across.add(i); park_in += x1 - x0
        if not ps: continue
        # seconds inside the hold with a timed read parked and the read thread idle (0.01 s grid)
        t = a
        while t < b:
            parked = sum(1 for x0, x1, pg, i in ps if x0 <= t < x1)
            if parked:
                j = bisect.bisect_right([x[0] for x in io], t) - 1 if False else None
            t += 0.01
        # exact: union of parked spans minus io busy spans
        spans = sorted((x0, x1) for x0, x1, pg, i in ps)
        merged = []
        for s0, s1 in spans:
            if merged and s0 <= merged[-1][1]: merged[-1][1] = max(merged[-1][1], s1)
            else: merged.append([s0, s1])
        for s0, s1 in merged:
            busy = 0.0
            k = bisect.bisect_left(io, (s0, 0)) - 1
            k = max(k, 0)
            while k < len(io) and io[k][0] < s1:
                lo = max(io[k][0], s0); hi = min(io[k][1], s1)
                if hi > lo: busy += hi - lo
                k += 1
            idle_in += (s1 - s0) - busy
    # pending at hold start
    sub = {}; done = {}
    for t, r in ev["h2s_submit"]:
        d = kv(r); sub[d["op"]] = (t, int(d["tokens"]))
    for t, r in ev["h2s_done"]:
        d = kv(r); done[d["op"]] = t
    adm = sorted((q["adm"], rid) for rid, q in reqs.items())
    adm_t = [x[0] for x in adm]
    running_prompt = 0; shares = []
    for a, b in holds:
        pend = [(st, tk) for op, (st, tk) in sub.items() if st <= a + 1e-3 and done.get(op, 1e18) > a]
        tot = sum(x[1] for x in pend)
        got = 0
        for st, tk in pend:
            # a request admitted in the 6 s before the op's submit, still running at the hold's start, whose uncached prompt is this op
            j = bisect.bisect_right(adm_t, st)
            k = j - 1
            hit = False
            while k >= 0 and adm_t[k] > st - 6.0:
                q = reqs[adm[k][1]]
                unc = (q["prompt"] // 64) * 64 - q["cached"]
                if abs(unc - tk) <= 64 and q["fin"] > a:
                    hit = True; break
                k -= 1
            if hit: got += tk
        shares.append(got / tot if tot else 0.0)
    print(f"c34 {arm}: holds {len(holds)} ({sum(b-a for a,b in holds):.1f} s); timed reads parked across a hold {len(n_across)} ({tok_across/1e6:.1f} M tokens); their parked seconds inside holds {park_in:.0f} s "
          f"({100*park_in/alltimed_park:.1f} % of all timed-read parking {alltimed_park/1e3:.1f} K s); read thread idle inside holds with a timed read parked {idle_in:.1f} s")
    print(f"   pending tokens at the hold's start that are the prompt write of a request still running: median share {100*pct(shares,.5):.0f} %, holds with over 80 %: {sum(1 for s in shares if s > 0.8)} of {len(shares)}")
