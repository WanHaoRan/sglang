"""Reviewer check (read-only): the engine's logged queue-wait prediction (ntp_admit pred_wait_ql_ms) against the
measured wait (wait_ms), split by whether the request had its own SSD read (an s2h_io line for its rid at or before
its admission line) and binned two ways: by the PREDICTED wait (the fair binning) and by the MEASURED wait (the
binning that selects on the outcome). Window: requests admitted in [lo, hi) minutes after the first ntp_admit.
Usage: python3 admit_err.py <server.log> lo_min hi_min"""
import collections, datetime, re, sys
LINE = re.compile(r"^\[(\S+ \S+)\] HICACHE_EVT (ntp_admit|s2h_io) rid=(\S+)(.*)")
def ts(s):
    return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").timestamp()
lo, hi = float(sys.argv[2]) * 60, float(sys.argv[3]) * 60
admit, land = {}, collections.defaultdict(list)
t0 = None
for line in open(sys.argv[1], errors="replace"):
    if "HICACHE_EVT" not in line:
        continue
    m = LINE.match(line)
    if not m:
        continue
    t = ts(m[1])
    if m[2] == "s2h_io":
        land[m[3]].append(t)
        continue
    if t0 is None:
        t0 = t
    w = re.search(r"wait_ms=(\d+) pred_wait_ql_ms=(\d+)", m[4])
    if w and m[3] not in admit:
        admit[m[3]] = (t, int(w[1]) / 1e3, int(w[2]) / 1e3)
def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(p * len(xs))))] if xs else float("nan")
rows = []
for rid, (t, w, p) in admit.items():
    if not lo <= t - t0 < hi:
        continue
    reader = any(x <= t + 0.05 for x in land.get(rid, ()))
    rows.append((w, p, reader))
bins = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 30), (30, 60), (60, 1e9)]
print(f"{sys.argv[1].split('/')[-2][:15]} window {lo/60:.0f}-{hi/60:.0f} min: {len(rows)} admissions, "
      f"{sum(r[2] for r in rows)} with their own SSD read")
for key, name in ((1, "PREDICTED"), (0, "MEASURED")):
    print(f"  binned by {name} wait: err = measured - predicted (s), median [p10, p90] (n)")
    for b in bins:
        out = []
        for grp, lab in ((True, "readers"), (False, "non-readers")):
            e = [r[0] - r[1] for r in rows if r[2] == grp and b[0] <= r[key] < b[1]]
            out.append(f"{lab} {q(e,.5):+6.1f} [{q(e,.1):+6.1f},{q(e,.9):+6.1f}] (n {len(e):4d})")
        print(f"    {b[0]:>3}-{b[1] if b[1] < 1e9 else 'inf':>3} s | " + " | ".join(out))
for grp, lab in ((True, "readers"), (False, "non-readers")):
    e = [r[0] - r[1] for r in rows if r[2] == grp]
    print(f"  all {lab}: median err {q(e,.5):+.1f} s, p10 {q(e,.1):+.1f}, p90 {q(e,.9):+.1f} (n {len(e)})")
