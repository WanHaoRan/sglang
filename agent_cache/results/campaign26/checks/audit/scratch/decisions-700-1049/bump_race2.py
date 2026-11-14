"""Audit scratch (read-only): ec_bump events against the bumped turn's scheduler entry time (ReqTimeStats entry_time,
epoch s) and against its ntp_arrive line. A bump after the entry is the one-pass race of "Known and left"."""
import datetime as dt, re, sys
EVT = re.compile(r"^\[(.{23})\] HICACHE_EVT (ec_bump|ntp_arrive) rid=\S+?-c(\d+)-t(\d+)(.*)")
RTS = re.compile(r"ReqTimeStats\(rid=\S+?-c(\d+)-t(\d+),.*entry_time=([\d.]+)")
stamp = lambda s: dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp()
bumps, arrive, entry = [], {}, {}
for line in open(sys.argv[1], errors="replace"):
    m = EVT.match(line)
    if m:
        t, kind, key = stamp(m.group(1)), m.group(2), (int(m.group(3)), int(m.group(4)))
        if kind == "ec_bump":
            bumps.append((t, key))
        else:
            arrive.setdefault(key, t)
        continue
    r = RTS.search(line)
    if r:
        entry.setdefault((int(r.group(1)), int(r.group(2))), float(r.group(3)))
have = [(t, k) for t, k in bumps if k in entry]
print(f"bumps {len(bumps)}; with a scheduler entry time {len(have)}; without {len(bumps) - len(have)}")
d = [entry[k] - t for t, k in have]
for lo, hi in ((None, -0.1), (-0.1, 0.0), (0.0, 0.1), (0.1, 0.5), (0.5, 1.0), (1.0, None)):
    n = sum((lo is None or g > lo) and (hi is None or g <= hi) for g in d)
    print(f"  scheduler entry minus bump in ({lo}, {hi}] s: {n}")
after = sorted((entry[k] - t, k) for t, k in have if entry[k] <= t)
print("  bumps at or after the turn's scheduler entry:", len(after), [(round(g, 3), k) for g, k in after[:12]])
lag = sorted(arrive[k] - entry[k] for k in arrive if k in entry)
print(f"  ntp_arrive minus scheduler entry over {len(lag)} turns: p10 {lag[len(lag)//10]:.3f} p50 {lag[len(lag)//2]:.3f} p90 {lag[9*len(lag)//10]:.3f} max {lag[-1]:.3f} s")
