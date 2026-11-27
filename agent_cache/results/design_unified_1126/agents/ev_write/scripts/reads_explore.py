"""Explore: where each SSD read starts in its request's chain (page index) from prefetch_start / s2h_query."""
import sys, collections
sys.path.insert(0, __import__("os").path.dirname(__file__))
from lib import Run, PAGE, events, kv

camp, label = sys.argv[1], sys.argv[2]
run = Run(camp, label)
arrive = {}; plan_chain = {}; last_ps = {}; last_q = {}; park = {}; disp = {}
stats = collections.Counter(); examples = []
reads = []
for t, ev, rest in events(run.extract, {"REQSTAT", "ntp_plan", "prefetch_start", "s2h_query", "s2h_io", "srs_park", "srs_dispatch", "srs_pin", "ntp_stage", "ntp_arrive"}):
    d = kv(rest)
    rid = d.get("rid")
    if ev == "REQSTAT":
        arrive[rid] = float(d["entry"])
    elif ev == "ntp_arrive":
        arrive.setdefault(rid, t)
    elif ev == "ntp_plan":
        plan_chain[rid] = int(d["chain"])
    elif ev == "prefetch_start":
        last_ps[rid] = (t, int(d["tokens"]))
    elif ev == "s2h_query":
        last_q[rid] = (t, int(d["tokens_req"]), int(d["storage_hit"]), last_ps.get(rid))
    elif ev == "srs_park":
        park[rid] = (t, int(d["pages"]), d["deadline_ms"], last_q.get(rid))
    elif ev == "srs_dispatch":
        disp[rid] = (t, int(d["pages"]), int(d["waited_ms"]), int(d["launched"]), park.get(rid))
    elif ev == "s2h_io":
        reads.append((t, rid, int(d["pages"]), float(d["ms"]), disp.get(rid), last_q.get(rid)))
# second pass needs arrival times known for all (REQSTAT is logged at finish) -> evaluate now
for t, rid, pages, ms, dp, q in reads:
    r = run.reqs.get(rid)
    if r is None:
        stats["rid not in client"] += 1; continue
    if dp is None or dp[4] is None or dp[4][3] is None:
        stats["no park/query chain"] += 1; continue
    pk = dp[4]; qq = pk[3]; ps = qq[3]
    if ps is None:
        stats["no prefetch_start"] += 1; continue
    t_ps, tok = ps
    arr = arrive.get(rid)
    staged = arr is None or t_ps < arr - 0.002
    keylen = plan_chain.get(rid) if staged else r.P - 1
    if keylen is None:
        stats["staged without plan"] += 1; continue
    start = keylen - tok
    stats["staged" if staged else "arrived"] += 1
    if start % PAGE:
        stats["start not aligned (%s)" % ("staged" if staged else "arrived")] += 1
        if len(examples) < 10: examples.append((rid, staged, keylen, tok, start, qq[:3], pages, r.P, r.cached, r.dev, r.host, r.stor))
    if qq[2] // PAGE != pk[1] or pk[1] != dp[1] or dp[1] < pages:
        stats["hit/park/dispatch/io pages differ"] += 1
        if dp[1] > pages: stats["io pages < dispatched"] += 1
print(f"{camp} {label}: s2h_io {len(reads)}")
for k, v in sorted(stats.items()): print(f"  {k}: {v}")
for e in examples: print("  ex", e)
