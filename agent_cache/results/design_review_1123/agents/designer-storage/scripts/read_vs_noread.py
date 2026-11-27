"""Per 10-minute window of arrival: queue wait of returning requests with an SSD read vs without,
and the read stall (arrival to last landing by admission). Usage: python read_vs_noread.py server.log"""
import re, sys, collections, statistics as st, datetime as dt
READ = re.compile(r"^\[([\d\- :.]+)\] HICACHE_EVT s2h_io rid=\S+-c(\d+)-t(\d+) pages=(\d+) tokens=(\d+) ms=(\d+)")
ADMIT = re.compile(r"rid=\S+?-c(\d+)-t(\d+)[,)].*?queue_duration=([0-9.]+)(ms|s), .*?entry_time=([0-9.]+)")
def stamp(s):
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp()
arrival, queue, lands, toks = {}, {}, collections.defaultdict(list), collections.Counter()
for line in open(sys.argv[1], errors="replace"):
    if "s2h_io" in line:
        m = READ.match(line)
        if m:
            k = (int(m[2]), int(m[3])); lands[k].append(stamp(m[1])); toks[k] += int(m[5])
    elif "queue_duration=" in line:
        m = ADMIT.search(line)
        if m:
            k = (int(m[1]), int(m[2]))
            arrival[k], queue[k] = float(m[5]), float(m[3]) / (1000.0 if m[4] == "ms" else 1.0)
t0 = min(arrival.values())
W = collections.defaultdict(lambda: {"rd": [], "no": [], "stall": [], "tok": []})
for k, a in arrival.items():
    if k[1] == 0: continue
    w = int((a - t0) // 600) * 10
    qw = queue[k]
    ends = [t for t in lands.get(k, ()) if t <= a + qw + 0.1]
    if ends:
        W[w]["rd"].append(qw); W[w]["stall"].append(min(qw, max(0.0, max(ends) - a))); W[w]["tok"].append(toks[k])
    else:
        W[w]["no"].append(qw)
def m(x): return st.mean(x) if x else float("nan")
def md(x): return st.median(x) if x else float("nan")
print("win | n read, mean/median queue s | n no read, mean/median queue s | read stall mean/median s | read K tokens mean")
for w in sorted(W):
    d = W[w]
    print(f"{w:3d} | {len(d['rd']):4d} {m(d['rd']):6.1f} {md(d['rd']):6.1f} | {len(d['no']):5d} {m(d['no']):6.1f} {md(d['no']):6.1f} | {m(d['stall']):6.1f} {md(d['stall']):6.1f} | {m(d['tok'])/1e3:5.1f}")
