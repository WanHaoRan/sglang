"""Wait split by 10-minute window of arrival (c34_figures.wait_split's rule), from one server.log.
stall = last SSD landing by admission+0.1 s minus arrival, within [0, queue wait];
'read only' = landed within 0.1 s of admission; 'gpu also' = other stalls.
Usage: python split_windows.py server.log"""
import re, sys, collections, datetime as dt
READ = re.compile(r"^\[([\d\- :.]+)\] HICACHE_EVT s2h_io rid=\S+-c(\d+)-t(\d+) pages=(\d+) tokens=(\d+) ms=(\d+)")
ADMIT = re.compile(r"rid=\S+?-c(\d+)-t(\d+)[,)].*?queue_duration=([0-9.]+)(ms|s), .*?entry_time=([0-9.]+)")
def stamp(s):
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp()
arrival, queue, lands = {}, {}, collections.defaultdict(list)
for line in open(sys.argv[1], errors="replace"):
    if "s2h_io" in line:
        m = READ.match(line)
        if m: lands[(int(m[2]), int(m[3]))].append(stamp(m[1]))
    elif "queue_duration=" in line:
        m = ADMIT.search(line)
        if m:
            k = (int(m[1]), int(m[2]))
            arrival[k], queue[k] = float(m[5]), float(m[3]) / (1000.0 if m[4] == "ms" else 1.0)
t0 = min(arrival.values())
win = collections.defaultdict(lambda: collections.Counter())
for k, a in arrival.items():
    if k[1] == 0: continue   # returning requests only (turn > 0)
    w = int((a - t0) // 600) * 10
    qw = queue[k]
    ends = [t for t in lands.get(k, ()) if t <= a + qw + 0.1]
    stall = min(qw, max(0.0, max(ends) - a)) if ends else 0.0
    c = win[w]; c["n"] += 1; c["qw"] += qw
    if k in lands:
        c["reading"] += 1
        if stall < 0.05: c["landed"] += 1
        elif qw - stall < 0.1: c["read_only"] += stall; c["n_read_only"] += 1
        else: c["gpu_also"] += stall
print("win(min) | returning | mean queue s | own read then GPU s | read only s (n) | reading turns")
for w in sorted(win):
    c = win[w]; n = c["n"]
    print(f"{w:3d}-{w+10:<3d} | {n:5d} | {c['qw']/n:6.2f} | {c['gpu_also']/n:5.2f} | {c['read_only']/n:5.2f} ({c['n_read_only']:4d}) | {c['reading']:5d}")
