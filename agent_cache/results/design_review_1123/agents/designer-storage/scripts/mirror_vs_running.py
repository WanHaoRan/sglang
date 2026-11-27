"""Census mirror vs the GPU tokens of running requests (#token on the nearest Decode batch line),
and host-only room vs the history of requests waiting in the queue at that moment.
Usage: python mirror_vs_running.py server.log client.jsonl"""
import re, sys, json, bisect, statistics as st, datetime as dt
def stamp(s): return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp()
ADMIT = re.compile(r"rid=\S+?-c(\d+)-t(\d+)[,)].*?queue_duration=([0-9.]+)(ms|s), .*?entry_time=([0-9.]+)")
dec_t, dec_tok, dec_q, cen = [], [], [], []
spans = []   # (arrival, admission, key)
for line in open(sys.argv[1], errors="replace"):
    if "Decode batch" in line:
        m = re.match(r"\[(\S+ \S+)\].*#token: (\d+),.*#queue-req: (\d+)", line)
        if m: dec_t.append(stamp(m[1])); dec_tok.append(int(m[2])); dec_q.append(int(m[3]))
    elif "host_census why=tick" in line:
        m = re.match(r"\[(\S+ \S+)\]", line); d = dict(re.findall(r"(\w+)=(-?\d+)", line))
        cen.append((stamp(m[1]), {k: int(v) for k, v in d.items()}))
    elif "queue_duration=" in line:
        m = ADMIT.search(line)
        if m:
            a = float(m[5]); q = float(m[3]) / (1000.0 if m[4] == "ms" else 1.0)
            spans.append((a, a + q, (int(m[1]), int(m[2]))))
hist = {}
for line in open(sys.argv[2]):
    if '"kind":"req"' not in line: continue
    r = json.loads(line); m = re.match(r".*-c(\d+)-t(\d+)$", r["rid"])
    hist[(int(m[1]), int(m[2]))] = r["ideal_cached"] or 0
t0 = min(s[0] for s in spans)
rows = []
for t, d in cen:
    i = bisect.bisect_left(dec_t, t)
    if i == 0 or i >= len(dec_t) or d["mirror"] == 0: continue
    queued = [hist.get(k, 0) for a, b, k in spans if a <= t < b and k[1] > 0]
    rows.append((t - t0, d["mirror"], dec_tok[i], d["host_only"], len(queued), sum(queued), d["evict_casc"] + d["evict_leaf"]))
print("window | mirror K | GPU tokens of running reqs K | host-only K | returning reqs queued | their history K | evictable K   (medians)")
for w0 in range(0, 7200, 600):
    sel = [r for r in rows if w0 <= r[0] < w0 + 600]
    if not sel: continue
    med = lambda j: st.median(r[j] for r in sel)
    print(f"{w0//60:3d}-{w0//60+10:<3d} | {med(1)/1e3:6.0f} | {med(2)/1e3:6.0f} | {med(3)/1e3:6.0f} | {med(4):4.0f} | {med(5)/1e3:6.0f} | {med(6)/1e3:6.0f}")
