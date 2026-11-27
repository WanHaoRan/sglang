"""Refinement of w_estimators.py (read-only, CPU only).

Bins the error by the PREDICTED wait (binning by the actual wait selects on the outcome), and adds
  P1r: (queued ahead that are not waiting on an SSD read + 0.5) / admissions per second (60 s)
A request j counts as waiting on a read at t if it has arrived and its last s2h_io landing (at or before its
admission) is after t. Log clock -> entry_time clock: median of (ntp_arrive line time - entry_time).
Usage: python3 w_estimators2.py <server.log> <rid prefix> t_from_min t_to_min
"""
import bisect, collections, re, statistics as st, sys, datetime

RX = re.compile(r"ReqTimeStats\(rid=(\S+?), input_len=(\d+), cached_input_len=(\d+), output_len=(\d+).*?"
                r"queue_duration=([0-9.]+)(ms|s), .*?entry_time=([0-9.]+)")
EV = re.compile(r"^\[(\S+ \S+)\] HICACHE_EVT (s2h_io|ntp_arrive) rid=(\S+)")

def stamp(s):
    return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").timestamp()

def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(p * len(xs))))] if xs else float("nan")

def summ(name, errs):
    if not errs:
        return f"{name}: none"
    ab = [abs(e) for e in errs]
    return (f"{name}: n {len(errs)} | err p10 {q(errs,.1):+.1f} p50 {q(errs,.5):+.1f} p90 {q(errs,.9):+.1f}"
            f" | |err| p50 {q(ab,.5):.1f} p90 {q(ab,.9):.1f}")

path, prefix = sys.argv[1], sys.argv[2]
lo, hi = float(sys.argv[3]) * 60, float(sys.argv[4]) * 60
reqs, lands, arrive_log = {}, collections.defaultdict(list), {}
with open(path, errors="replace") as f:
    for line in f:
        if "ReqTimeStats" in line:
            m = RX.search(line)
            if m and m[1].startswith(prefix + "-c"):
                w = float(m[5]) / (1000.0 if m[6] == "ms" else 1.0)
                reqs[m[1]] = (float(m[7]), w, int(m[2]))
        elif "HICACHE_EVT" in line:
            m = EV.match(line)
            if m:
                if m[2] == "s2h_io":
                    lands[m[3]].append(stamp(m[1]))
                else:
                    arrive_log.setdefault(m[3], stamp(m[1]))
offs = [arrive_log[r] - reqs[r][0] for r in arrive_log if r in reqs]
off = st.median(offs)
rows = []
for rid, (a, w, tok) in reqs.items():
    d = a + w
    ls = [t - off for t in lands.get(rid, ()) if t - off <= d + 0.1]
    rows.append((a, w, d, tok, max(ls) if ls else None))
rows.sort()
t0 = rows[0][0]
adm_t = sorted(r[2] for r in rows)

def rate(t, win=60.0):
    n = bisect.bisect_right(adm_t, t) - bisect.bisect_right(adm_t, t - win)
    return n / win, n

def ahead(i, t):
    n = nr = 0
    for j in range(i - 1, -1, -1):
        a, w, d, tok, land = rows[j]
        if d > t:
            n += 1
            if land is None or land <= t:
                nr += 1
        if t - a > 2000:
            break
    return n, nr

print(f"{path.split('/')[-2]} {prefix}: {len(rows)} requests, clock offset {off:.3f} s (spread p10 {q(offs,.1)-off:+.3f} p90 {q(offs,.9)-off:+.3f}); window {lo/60:.0f}-{hi/60:.0f} min")
pbins = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 40), (40, 1e9)]
for label, use_ready in (("P1 (all ahead)", False), ("P1r (ahead not reading)", True)):
    byb = {b: [] for b in pbins}
    own = {True: [], False: []}
    for i, (a, w, d, tok, land) in enumerate(rows):
        if not lo <= a - t0 < hi:
            continue
        r, n60 = rate(a)
        n, nr = ahead(i, a)
        k = nr if use_ready else n
        if n == 0 or n60 < 5:
            continue
        p = (k + 0.5) / r
        b = next(bb for bb in pbins if bb[0] <= p < bb[1])
        byb[b].append(w - p)
        own[land is not None].append(w - p)
    print(f"  at arrival, {label}, binned by the predicted wait:")
    for b in pbins:
        print("    " + summ(f"pred {b[0]}-{b[1] if b[1] < 1e9 else 'inf'} s", byb[b]))
    print("    " + summ("requests with their own SSD read", own[True]))
    print("    " + summ("requests without", own[False]))
    for h in (2.0, 5.0, 10.0):
        e = []
        for i, (a, w, d, tok, land) in enumerate(rows):
            t = d - h
            if t < a or not lo <= a - t0 < hi:
                continue
            r, n60 = rate(t)
            if n60 < 5:
                continue
            n, nr = ahead(i, t)
            e.append(h - ((nr if use_ready else n) + 0.5) / r)
        print("    " + summ(f"wait left {h:.0f} s before admission", e))
