"""PR1 check (CPU, read-only): how the offline c depends on its definition.
  c_any   a ready request is left waiting after the admission (same pass = within 30 ms), as pr1_prices_offline.py
  c_0.5s  ... and it has been ready (queued, no own read under way) for at least 0.5 s (NTP_GPU_BOUND_S), the test
          the engine's _queue_ahead() already applies under expected-cost timing
  c_tick  what tick() would compute: sampled at the next scheduler pass, here approximated by the admission time
          plus 20 ms, counting requests that entered before the admission
and an EWMA of the per-admission samples (alpha 1/16) against the window mean: the share of admissions at which the
EWMA is in (0.1, 0.9), i.e. how often a smoothed c is neither 0 nor 1.
Usage: python3 pr1_c_variants.py <label>"""
import bisect, collections, sys
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
def ready_since(x, t):
    """Seconds x has been ready at t (queued with no own read under way), or None."""
    if any(s <= t < e for s, e in x[2]):
        return None
    last_end = max([e for s, e in x[2] if e <= t], default=x[0])
    return t - max(x[0], last_end)
by_adm = sorted(reqs, key=lambda x: x[1])
out = collections.defaultdict(lambda: collections.defaultdict(list))
ew = None; mid = collections.defaultdict(list)
for x in by_adm:
    a = x[1]
    lo_i = bisect.bisect_left(entries, a - max_wait - 1.0); hi_i = bisect.bisect_right(entries, a)
    any_left = half_left = False
    for y in reqs[lo_i:hi_i]:
        if y is x or y[1] <= a + 0.03:
            continue
        rs = ready_since(y, a)
        if rs is None:
            continue
        any_left = True
        if rs >= 0.5:
            half_left = True
            break
    w = int(r.minute(a) // 10)
    out[w]["any"].append(0.0 if any_left else 1.0)
    out[w]["half"].append(0.0 if half_left else 1.0)
    s = 0.0 if half_left else 1.0
    ew = s if ew is None else ew + (s - ew) / 16
    mid[w].append(0.1 < ew < 0.9)
print(f"== {label}: c per 10-minute window")
print("   min      adm   c_any  c_0.5s  | share of admissions with the EWMA(1/16) of c_0.5s in (0.1, 0.9)")
for w in sorted(out):
    if w >= 11:
        continue
    print(f"   {w*10:3d}-{w*10+10:<3d} {len(out[w]['any']):5d}  {lib.mean(out[w]['any']):5.3f}  {lib.mean(out[w]['half']):5.3f}   | {lib.mean(mid[w]):5.3f}")
