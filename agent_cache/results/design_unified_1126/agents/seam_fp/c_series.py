"""Per-minute series of the proposed lateness weight c and of 1 + ready turns waiting, rebuilt from a run's records
(same definitions as pr1_prices_offline.py). Writes c_series_<label>.json: {"minute": [c, n_admissions, mu_ready, queue]}.
Usage: python3 c_series.py <label>"""
import bisect, collections, json, os, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import lib

label = sys.argv[1]
r = lib.Run(label)
reqs = []
for rid, rec in r.requests():
    rts = rec["rts"]; adm = rts["entry"] + rts["queue_s"]
    busy = []
    lands = sorted(e["t"] for e in rec.get("landed", []))
    for p in rec.get("srs_park", []):
        qs = [e["t"] for e in rec.get("s2h_query", []) if e["t"] <= p["t"]]
        start = max(qs) - 0.02 if qs else p["t"]
        end = next((t for t in lands if t >= p["t"]), None)
        busy.append((start, adm if end is None else end))
    reqs.append((rts["entry"], adm, busy))
reqs.sort()
entries = [x[0] for x in reqs]
max_wait = max(x[1] - x[0] for x in reqs)
def ready(x, t):
    return not any(s <= t < e for s, e in x[2])
c = collections.defaultdict(list)
for x in reqs:
    a = x[1]
    lo_i = bisect.bisect_left(entries, a - max_wait - 1.0); hi_i = bisect.bisect_right(entries, a)
    left = any(y is not x and y[1] > a + 0.03 and ready(y, a) for y in reqs[lo_i:hi_i])
    c[int(r.minute(a))].append(0.0 if left else 1.0)
mu = collections.defaultdict(list); qq = collections.defaultdict(list)
t = r.t0; t_end = max(x[1] for x in reqs)
while t < t_end:
    lo_i = bisect.bisect_left(entries, t - max_wait - 1.0); hi_i = bisect.bisect_right(entries, t)
    qn = rn = 0
    for y in reqs[lo_i:hi_i]:
        if y[1] > t:
            qn += 1; rn += ready(y, t)
    m = int(r.minute(t)); mu[m].append(1 + rn); qq[m].append(qn)
    t += 1.0
out = {str(m): [lib.mean(c[m]) if c[m] else None, len(c[m]), lib.mean(mu[m]), lib.mean(qq[m])] for m in sorted(mu)}
json.dump(out, open(os.path.join(lib.HERE, f"c_series_{label}.json"), "w"))
print(label, "minutes", len(out))
