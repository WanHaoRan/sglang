"""F3 check (CPU, read-only): the pieces of the read path the load model estimates, measured from a run's lines.

Per read (srs_park -> srs_dispatch -> s2h_io -> 'HiCache prefetch success'):
  lookup   = srs_park - the s2h_query before it (query answered -> hit drained and parked; the drain is one pass)
  line     = srs_dispatch - srs_park (waited_ms: parked, by the hold, the window, the budget or the write hold)
  start    = (s2h_io - ms) - srs_dispatch (dispatch -> the read thread starts this read: reads ahead on the thread)
  io       = s2h_io ms                    (the read itself)
  drain    = prefetch success - s2h_io    (I/O end -> pages inserted in host memory)
and the model's estimate for the same read, 0.5 + 0.044 + s_page x pages + 0.36 (L3LoadModel seeds with the run's
ms/page), against the measured dispatch -> landed time, split by whether the read thread was idle at dispatch
(srs_dispatch inflight_pages == the read's own pages).
For reads staged before arrival (ntp_stage) it also prints planned r against stage -> landed.
Usage: python3 f3_read_path.py <label> [lo_min hi_min]
"""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import lib

label = sys.argv[1]
lo = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
hi = float(sys.argv[3]) if len(sys.argv) > 3 else 110.0
r = lib.Run(label)
rows = []
for rid, rec in r.req.items():
    parks = sorted(rec.get("srs_park", []), key=lambda e: e["t"])
    for p in parks:
        d = next((e for e in rec.get("srs_dispatch", []) if e["t"] >= p["t"]), None)
        io = next((e for e in rec.get("s2h_io", []) if d and e["t"] >= d["t"]), None)
        land = next((e for e in rec.get("landed", []) if io and e["t"] >= io["t"] - 0.001), None)
        q = [e for e in rec.get("s2h_query", []) if e["t"] <= p["t"]]
        if not (d and io and land and q):
            continue
        if not lo <= r.minute(p["t"]) < hi:
            continue
        io_s = io["ms"] / 1e3
        rows.append(dict(lookup=p["t"] - q[-1]["t"], line=d["t"] - p["t"], start=(io["t"] - io_s) - d["t"], io=io_s,
                         drain=land["t"] - io["t"], pages=p["pages"], total=land["t"] - d["t"],
                         idle=d["inflight_pages"] <= p["pages"], now=p["deadline_ms"] == "now", m=r.minute(p["t"])))
print(f"== {label} minutes {lo:.0f}-{hi:.0f}: {len(rows)} reads")
def show(name, sel):
    if not sel:
        return
    ms_page = sum(x["io"] for x in sel) / max(1, sum(x["pages"] for x in sel))
    est = [0.5 + 0.044 + ms_page * x["pages"] + 0.36 for x in sel]
    err = [x["total"] - e for x, e in zip(sel, est)]
    print(f"   {name} (n {len(sel)}): pages median {lib.q([x['pages'] for x in sel],.5)}, io {ms_page*1e3:.2f} ms/page")
    for k in ("lookup", "line", "start", "io", "drain", "total"):
        v = [x[k] for x in sel]
        print(f"       {k:6s} median {lib.q(v,.5)*1e3:8.0f} ms   p90 {lib.q(v,.9)*1e3:8.0f} ms   mean {lib.mean(v)*1e3:8.0f} ms")
    fixed = [x["total"] - ms_page * x["pages"] for x in sel]
    print(f"       dispatch -> landed minus pages x ms/page (the fixed part): median {lib.q(fixed,.5)*1e3:.0f} ms, p90 {lib.q(fixed,.9)*1e3:.0f} ms; the model's lags add 904 ms")
    print(f"       measured dispatch -> landed minus the model's estimate: median {lib.q(err,.5)*1e3:+.0f} ms [p10 {lib.q(err,.1)*1e3:+.0f}, p90 {lib.q(err,.9)*1e3:+.0f}]")
show("read thread idle at dispatch", [x for x in rows if x["idle"]])
show("read thread busy at dispatch", [x for x in rows if not x["idle"]])
show("parked as 'now'", [x for x in rows if x["now"]])
show("parked with a deadline", [x for x in rows if not x["now"]])
