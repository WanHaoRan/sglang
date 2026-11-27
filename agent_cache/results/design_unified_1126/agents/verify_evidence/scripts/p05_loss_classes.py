"""Lost history >= 256 tokens by cause, per request (requests sent before 6,600 s): my classification.
 first      first request of a session (k = 0, main stream)
 readfail   a read of the turn failed on a missing file ('failed to retrieve page')
 empty      some lookup started at or before the turn's final cached length and found less than the lost span
 nolookup   no lookup started at or before the cached length: capped (gate 'capped' / reevict capped=1) or other
"""
import sys, os, re, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib, runtab
from vlib import kv
for camp, arm in [(34, "ec_full"), (34, "jit_exact"), (34, "ec_exact"), (34, "jit_p50")]:
    tab = runtab.build(camp, arm); reqs = tab["reqs"]; ev = vlib.load(camp, arm)
    rs = vlib.req_stats(ev, camp, arm)
    p1 = min(t for t, r in ev["cleanup"]) - 12
    fail = collections.Counter()
    for t, r in ev.get("other", []):
        m = re.match(r"Prefetch operation (\S+) failed to retrieve page", r)
        if m: fail[m.group(1)] += 1
    capped = set()
    for t, r in ev.get("ntp_gate", []):
        d = kv(r)
        if d.get("decision") == "capped": capped.add(d["rid"])
    reev = set()
    for t, r in ev.get("ntp_reevict", []):
        d = kv(r); reev.add(d["rid"])
        if d.get("capped") == "1": capped.add(d["rid"])
    starts = collections.defaultdict(list)
    last = {}
    for name in ("prefetch_start", "s2h_query"):
        pass
    seq = sorted([(t, 0, r) for t, r in ev["prefetch_start"]] + [(t, 1, r) for t, r in ev["s2h_query"]])
    for t, k, r in seq:
        d = kv(r); rid = d["rid"]
        if k == 0:
            last[rid] = int(d["tokens"])
        else:
            q = rs.get(rid)
            if q is None or rid not in last: continue
            start = q["input"] - 1 - last[rid]
            starts[rid].append((t, start, int(d["tokens_req"]), int(d["storage_hit"])))
    cls = collections.Counter(); tok = collections.Counter(); n_fail_lines = sum(fail.values())
    for rid, q in reqs.items():
        if q["t_send"] >= 6600: continue
        lost = q["ideal"] - q["cached"]
        if lost < 256: continue
        when = "post" if q["t_send"] >= p1 else "pre"
        if q["k"] == 0 and q["stream"] == 0: c = "first"
        elif fail.get(rid): c = "readfail"
        else:
            lk = [x for x in starts.get(rid, []) if x[1] <= q["cached"] + 63]
            if lk and any(x[3] < min(x[2], lost) for x in lk): c = "empty"
            elif rid in capped: c = "nolookup_capped"
            elif rid in reev: c = "nolookup_reevicted"
            else: c = "other"
        cls[(when, c)] += 1; tok[(when, c)] += lost
    print(f"=== c{camp} {arm}: read-failure lines {n_fail_lines} over {len(fail)} turns")
    for key in sorted(cls):
        print(f"   {key[0]:4s} {key[1]:20s} {cls[key]:3d} requests {tok[key]/1e3:8.0f} K tokens")
    post = sum(v for k, v in tok.items() if k[0] == "post"); cl = sum(v for k, v in tok.items() if k[0] == "post" and k[1] in ("first", "readfail", "empty"))
    print(f"   after the first pass: {post/1e6:.3f} M lost, of which first + readfail + empty lookups {cl/1e6:.3f} M ({100*cl/post:.0f} %)")
