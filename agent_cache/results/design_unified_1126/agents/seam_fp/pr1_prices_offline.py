"""PR1 check (CPU, read-only): what the proposed estimators of c, mu and kappa would have read in a recorded run.

The waiting queue is rebuilt from ReqTimeStats (entry_time, queue_duration) of every request; a queued request is
'ready' at time t when none of its own storage reads is under way (from its lookup that hit, s2h_query with
storage_hit >= 256, to that read's landing, 'HiCache prefetch success'; a staged read counts from its park).
  c      per admission: 1 when no other ready request is left waiting (requests admitted within 30 ms are the same
         pass), else 0; reported as the mean per window, over all admissions and over readers' admissions.
  mu     1 + ready requests waiting, sampled once a second (time average); also the whole queue.
  kappa  c x (parked reads past their latest start), sampled once a second. A parked read (srs_park .. srs_dispatch)
         is past its latest start when it was parked as 'now' or when park + deadline - own load <= t, own load =
         0.9 s + pages x 3.6 ms (the load model's lags and campaign 34's logged ms/page).
Also per window: host_census medians (free, pending_write, pinned, n_parked), and the device's share from s2h_io.
Usage: python3 pr1_prices_offline.py <label> [window_min]
"""
import bisect, collections, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import lib

label = sys.argv[1]; W = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
r = lib.Run(label)
TH = 256
reqs = []
for rid, rec in r.requests():
    rts = rec["rts"]; adm = rts["entry"] + rts["queue_s"]
    busy = []
    lands = sorted(e["t"] for e in rec.get("landed", []))
    for p in rec.get("srs_park", []):
        qs = [e["t"] for e in rec.get("s2h_query", []) if e["t"] <= p["t"]]
        start = max(qs) - 0.02 if qs else p["t"]
        end = next((t for t in lands if t >= p["t"]), None)
        if end is None:
            end = adm
        busy.append((start, end))
    reader = any(e["t"] <= adm + 0.05 for e in rec.get("s2h_io", []))
    reqs.append(dict(rid=rid, entry=rts["entry"], adm=adm, busy=busy, reader=reader))
reqs.sort(key=lambda x: x["entry"])
def is_ready(x, t):
    return not any(s <= t < e for s, e in x["busy"])

# sweep: active set by entry order
events = sorted([(x["adm"], i) for i, x in enumerate(reqs)])
c_all = collections.defaultdict(list); c_rd = collections.defaultdict(list)
entries = [x["entry"] for x in reqs]
adm_sorted = sorted(x["adm"] for x in reqs)
by_entry_adm = [(x["entry"], x["adm"], i) for i, x in enumerate(reqs)]
# For each admission, the queued set = entered before a, admitted after a + 0.03. Queue is short (<~130): scan a window.
max_wait = max(x["adm"] - x["entry"] for x in reqs)
for x in reqs:
    a = x["adm"]
    lo_i = bisect.bisect_left(entries, a - max_wait - 1.0)
    hi_i = bisect.bisect_right(entries, a)
    ready_left = 0
    for y in reqs[lo_i:hi_i]:
        if y is x or y["adm"] <= a + 0.03:
            continue
        if is_ready(y, a):
            ready_left += 1
            break
    w = int(r.minute(a) // W)
    c_all[w].append(1.0 if ready_left == 0 else 0.0)
    if x["reader"]:
        c_rd[w].append(1.0 if ready_left == 0 else 0.0)

# time sampling once a second
t_end = max(x["adm"] for x in reqs)
ready_ts = collections.defaultdict(list); queue_ts = collections.defaultdict(list)
late_ts = collections.defaultdict(list); parked_ts = collections.defaultdict(list)
parks = []
for rid, rec in r.requests():
    ds = sorted(rec.get("srs_dispatch", []), key=lambda e: e["t"])
    for p in rec.get("srs_park", []):
        d = next((e["t"] for e in ds if e["t"] >= p["t"]), None)
        if d is None:
            continue
        own = 0.9 + p["pages"] * 0.0036
        latest = p["t"] if p["deadline_ms"] == "now" else p["t"] + p["deadline_ms"] / 1e3 - own
        parks.append((p["t"], d, latest))
parks.sort()
pstarts = [p[0] for p in parks]
max_park = max((p[1] - p[0] for p in parks), default=0.0)
t = r.t0
while t < t_end:
    w = int(r.minute(t) // W)
    lo_i = bisect.bisect_left(entries, t - max_wait - 1.0); hi_i = bisect.bisect_right(entries, t)
    qn = rn = 0
    for y in reqs[lo_i:hi_i]:
        if y["adm"] > t:
            qn += 1
            if is_ready(y, t):
                rn += 1
    queue_ts[w].append(qn); ready_ts[w].append(rn)
    lo_p = bisect.bisect_left(pstarts, t - max_park - 1.0); hi_p = bisect.bisect_right(pstarts, t)
    np_ = nl = 0
    for ps, pd, latest in parks[lo_p:hi_p]:
        if pd > t:
            np_ += 1
            if latest <= t:
                nl += 1
    parked_ts[w].append(np_); late_ts[w].append(nl)
    t += 1.0

census = collections.defaultdict(list)
for e in r.kind["host_census"]:
    census[int(r.minute(e["t"]) // W)].append(e)
io = collections.defaultdict(float)
for e in r.kind["s2h_io"]:
    io[int(r.minute(e["t"]) // W)] += e["ms"] / 1e3
print(f"== {label}: per {W:.0f}-minute window")
print("   min    adm     c    c(readers) n_rd | mu=1+ready  queue | parked  late  kappa=c*late | free K  pend_w K pinned K | read thread busy")
for w in sorted(set(c_all) | set(ready_ts)):
    if w * W >= 115:
        continue
    ca = lib.mean(c_all[w]); cr = lib.mean(c_rd[w]) if c_rd[w] else float("nan")
    cs = census.get(w, [])
    med = lambda k: lib.q([e.get(k, 0) for e in cs], .5) / 1e3 if cs else float("nan")
    late = lib.mean(late_ts[w]); 
    print(f"   {w*W:3.0f}-{(w+1)*W:<3.0f} {len(c_all[w]):5d}  {ca:5.3f}  {cr:5.3f}  {len(c_rd[w]):5d} | {1+lib.mean(ready_ts[w]):7.1f}  {lib.mean(queue_ts[w]):6.1f} | {lib.mean(parked_ts[w]):6.1f} {late:5.1f}  {ca*late:8.3f}     | {med('free'):6.1f} {med('pending_write'):7.1f} {med('pinned'):7.1f}  | {io[w]/(W*60):5.2f}")
