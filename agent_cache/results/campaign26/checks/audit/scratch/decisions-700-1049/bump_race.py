"""Audit scratch (read-only): in ec_full, how many ec_bump events hit a turn that had already arrived (ntp_arrive
before the bump), i.e. the one-pass race the record lists under "Known and left". Same event parsing as checks/bumps.py."""
import datetime as dt, re, sys
EVT = re.compile(r"^\[(.{23})\] HICACHE_EVT (ec_bump|ntp_arrive) rid=\S+?-c(\d+)-t(\d+)(.*)")
stamp = lambda s: dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp()
bumps, arrive = [], {}
for line in open(sys.argv[1], errors="replace"):
    m = EVT.match(line)
    if not m:
        continue
    t, kind, key = stamp(m.group(1)), m.group(2), (int(m.group(3)), int(m.group(4)))
    if kind == "ec_bump":
        bumps.append((t, key))
    else:
        arrive.setdefault(key, t)   # first arrival line of the turn
gaps = [arrive[k] - t for t, k in bumps if k in arrive]
never = sum(k not in arrive for _, k in bumps)
print(f"bumps {len(bumps)}; bumped turn never arrived {never}; with an arrival {len(gaps)}")
for lo, hi in ((None, 0.0), (0.0, 0.1), (0.1, 0.5), (0.5, 1.0), (1.0, 5.0), (5.0, None)):
    n = sum((lo is None or g > lo) and (hi is None or g <= hi) for g in gaps)
    print(f"  arrival minus bump in ({lo}, {hi}] s: {n}")
late = sorted(g for g in gaps if g <= 0)
print("  bumps at or after the turn's arrival (gap <= 0):", len(late), [round(g, 3) for g in late[:20]])
