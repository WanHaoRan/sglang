"""Reviewer check (read-only): is a reader's admission explained by max(GPU turn, its own landing)?
For requests with their own SSD read (s2h_io at or before the ntp_admit line), compare the measured wait with
  (a) the engine's position x rate prediction (pred_wait_ql_ms), and
  (b) max(pred, last landing - arrival), where arrival = admit line time - wait_ms.
Window: admissions [lo, hi) minutes after the first ntp_admit. Usage: python3 max_structure.py <server.log> lo hi"""
import collections, datetime, re, sys
LINE = re.compile(r"^\[(\S+ \S+)\] HICACHE_EVT (ntp_admit|s2h_io) rid=(\S+)(.*)")
def ts(s): return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").timestamp()
lo, hi = float(sys.argv[2]) * 60, float(sys.argv[3]) * 60
admit, land, t0 = {}, collections.defaultdict(list), None
for line in open(sys.argv[1], errors="replace"):
    if "HICACHE_EVT" not in line: continue
    m = LINE.match(line)
    if not m: continue
    t = ts(m[1])
    if m[2] == "s2h_io": land[m[3]].append(t); continue
    if t0 is None: t0 = t
    w = re.search(r"wait_ms=(\d+) pred_wait_ql_ms=(\d+)", m[4])
    if w and m[3] not in admit: admit[m[3]] = (t, int(w[1]) / 1e3, int(w[2]) / 1e3)
def q(xs, p):
    xs = sorted(xs); return xs[min(len(xs) - 1, max(0, int(p * len(xs))))] if xs else float("nan")
ea, eb = [], []
for rid, (t, w, p) in admit.items():
    if not lo <= t - t0 < hi: continue
    ls = [x for x in land.get(rid, ()) if x <= t + 0.05]
    if not ls: continue
    arr = t - w
    ea.append(w - p); eb.append(w - max(p, max(ls) - arr))
for name, e in (("position x rate", ea), ("max(position x rate, own landing)", eb)):
    ab = [abs(x) for x in e]
    print(f"  readers n {len(e)} | {name:34s}: err median {q(e,.5):+5.1f} s [p10 {q(e,.1):+5.1f}, p90 {q(e,.9):+5.1f}] | |err| p90 {q(ab,.9):4.1f} s")
