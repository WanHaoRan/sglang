"""Calibration helper: the lags of SSD reads from a server log (read-only, streaming).

  TZ=UTC python -I -B calib_read_lags.py <run>/server.log

Per read: dispatch to the start of its I/O (the read thread's own line), I/O time and ms per page, the end of the
I/O to the pin (pages usable), lookup issued to answered, answered to parked. Output used in CALIBRATION.md: lookup
0.02 s, landing 0.03 s (I/O fixed 13 ms plus 12 to 16 ms to the pin), 2.5 to 3.3 ms a page.
"""
import re, sys, collections, statistics as st
from datetime import datetime
path = sys.argv[1]
ts_re = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\] HICACHE_EVT (\w+) (.*)")
def ts(s): return datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").timestamp()
t0 = None
disp = {}; io = collections.defaultdict(list); pin = {}; park = {}; query = {}; pstart = {}
for line in open(path, errors="replace"):
    if "HICACHE_EVT" not in line: continue
    m = ts_re.match(line)
    if not m: continue
    t = ts(m.group(1)); ev = m.group(2); kv = dict(x.split("=", 1) for x in m.group(3).split() if "=" in x)
    if t0 is None: t0 = t
    rid = kv.get("rid")
    if ev == "srs_dispatch" and kv.get("launched") == "1": disp[rid] = (t, int(kv["pages"]), int(kv["waited_ms"]))
    elif ev == "s2h_io": io[rid].append((t, int(kv["ms"]), int(kv["pages"]), int(kv["tokens"])))
    elif ev == "srs_pin": pin.setdefault(rid, t)
    elif ev == "srs_park": park[rid] = t
    elif ev == "s2h_query": query[rid] = (t, int(kv["tokens_req"]), int(kv["storage_hit"]))
    elif ev == "prefetch_start": pstart[rid] = t
rows = []
for rid, (td, pages, waited) in disp.items():
    if rid not in io: continue
    ios = io[rid]
    t_land = max(x[0] for x in ios); ms_sum = sum(x[1] for x in ios)
    first = min(ios, key=lambda x: x[0])
    io_start = first[0] - first[1] / 1000.0
    rows.append((td - t0, pages, io_start - td, t_land - td, ms_sum / 1000.0, (pin.get(rid, t_land) - t_land), len(ios),
                 (park.get(rid, td) - query.get(rid, (td,))[0]) if rid in query else None,
                 (query[rid][0] - pstart[rid]) if rid in query and rid in pstart else None))
def med(xs): 
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else float('nan')
def q(xs, p):
    xs = sorted(x for x in xs if x is not None)
    return xs[int(p * (len(xs) - 1))] if xs else float('nan')
print("window(min) n pages_med | dispatch->io_start med p90 | dispatch->land med | io_s med | ms/page med | land->pin med p90 | chunks | query->park med | start->query med p90")
for lo, hi in ((0, 30), (30, 60), (60, 90), (90, 120)):
    w = [r for r in rows if lo * 60 <= r[0] < hi * 60]
    if not w: continue
    print(f"{lo}-{hi} {len(w)} {med([r[1] for r in w]):.0f} | {med([r[2] for r in w]):.3f} {q([r[2] for r in w], .9):.3f} | {med([r[3] for r in w]):.3f} | {med([r[4] for r in w]):.3f} | "
          f"{med([1000 * r[4] / r[1] for r in w]):.2f} | {med([r[5] for r in w]):.3f} {q([r[5] for r in w], .9):.3f} | {med([r[6] for r in w]):.0f} | {med([r[7] for r in w]):.3f} | {med([r[8] for r in w]):.3f} {q([r[8] for r in w], .9):.3f}")
# io time vs pages regression (single chunk reads)
import math
xs = [(r[1], r[4]) for r in rows if r[6] >= 1]
n = len(xs); sx = sum(x for x, _ in xs); sy = sum(y for _, y in xs); sxx = sum(x * x for x, _ in xs); sxy = sum(x * y for x, y in xs)
b = (n * sxy - sx * sy) / (n * sxx - sx * sx); a = (sy - b * sx) / n
print(f"io seconds = {a:.4f} + {b * 1000:.3f} ms x pages over {n} reads")
