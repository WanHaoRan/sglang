"""F1 check (CPU, read-only): the error of the engine's admission estimate, as an online tracker would see it.

Part A (what ntp_admit already logs): err = wait_ms - pred_wait_ql_ms per planned turn, for turns with no SSD read
of their own (the estimate of D, the GPU turn) and for readers, binned by the PREDICTED wait.
Part B (the hold's own decision point): the queue is rebuilt from ReqTimeStats (entry_time, queue_duration) of every
request. For a turn i at time t, position = requests that entered before i and are still queued, rate = admissions in
the last 60 s / 60 (300 s when the minute holds fewer than 5), predicted residual p(t) = (position + 0.5) / rate: the
engine's residual_wait(). For each threshold X the script takes the first time (0.25 s grid from the entry) at which
p(t) <= X, which is where a hold with load + margin = X would release, and scores err = (admission - t) - p(t).
Non-readers only: a reader's admission is the later of its GPU turn and its own landing.
Part C: landed reads: seconds from the last landing to admission (the earliness the hold bought) and the pinned
token-seconds; ntp_hold lines (first hold per request).
The margin a landing quantile q implies is -Q_q(err).
Usage: python3 f1_admission_error.py <label> lo_min hi_min
"""
import bisect, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import lib

label = sys.argv[1]; lo = float(sys.argv[2]); hi = float(sys.argv[3])
r = lib.Run(label)
QS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)

reqs = []
for rid, rec in r.requests():
    rts = rec["rts"]
    adm = rts["entry"] + rts["queue_s"]
    ios = [e for e in rec.get("s2h_io", []) if e["t"] <= adm + 0.05]
    na = rec.get("ntp_admit")
    pred = na[0]["pred_wait_ql_ms"] / 1e3 if na and isinstance(na[0].get("pred_wait_ql_ms"), int) else None
    reqs.append(dict(rid=rid, entry=rts["entry"], adm=adm, wait=rts["queue_s"], reader=bool(ios), pred=pred,
                     last_land=max((e["t"] for e in ios), default=None), rec=rec))
reqs.sort(key=lambda x: x["entry"])
adm_sorted = sorted(x["adm"] for x in reqs)

def rate_at(t):
    n60 = bisect.bisect_right(adm_sorted, t) - bisect.bisect_right(adm_sorted, t - 60.0)
    if n60 >= 5:
        return n60 / 60.0
    n300 = bisect.bisect_right(adm_sorted, t) - bisect.bisect_right(adm_sorted, t - 300.0)
    return n300 / 300.0 if n300 >= 5 else None

win = [x for x in reqs if lo <= r.minute(x["adm"]) < hi]
print(f"== {label} minutes {lo:.0f}-{hi:.0f}: {len(win)} admissions, {sum(x['reader'] for x in win)} readers")

def qline(e):
    return " ".join(f"{lib.q(e, p):+6.1f}" for p in QS)

print("  A. err = measured wait - pred_wait_ql (s) at arrival, quantiles " + " ".join(f"q{int(p*100):02d}" for p in QS))
bins = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 30), (30, 60), (60, 1e9)]
for grp, name in ((False, "non-readers"), (True, "readers")):
    for b in bins:
        e = [x["wait"] - x["pred"] for x in win if x["reader"] == grp and x["pred"] is not None and b[0] <= x["pred"] < b[1]]
        if len(e) >= 20:
            print(f"     {name:11s} pred {b[0]:>3}-{b[1] if b[1] < 1e9 else 'inf':>3} s (n {len(e):5d}): {qline(e)}")
    e = [x["wait"] - x["pred"] for x in win if x["reader"] == grp and x["pred"] is not None]
    print(f"     {name:11s} all                (n {len(e):5d}): {qline(e)}")
# readers: error of max(pred, own landing)
e = [x["adm"] - max(x["entry"] + x["pred"], x["last_land"]) for x in win if x["reader"] and x["pred"] is not None]
print(f"     readers, admission - max(arrival + pred, own last landing) (n {len(e)}): {qline(e)}")

# Part B: first crossing of the predicted residual
print("  B. non-readers: err = actual residual - predicted residual at the first time the prediction is <= X")
entries = [x["entry"] for x in reqs]
idx_lo = bisect.bisect_left(entries, r.t0 + lo * 60 - 600)
out = {X: [] for X in (2.0, 3.0, 5.0, 8.0, 12.0, 20.0)}
never = {X: 0 for X in out}
active = []  # (adm, entry) of requests still queued, kept sorted by entry (FCFS append)
j = 0
for i, x in enumerate(reqs):
    # queue at x's entry: earlier-entered requests not yet admitted
    active = [a for a in active if a > x["entry"]]
    ahead = sorted(active)
    active.append(x["adm"])
    if x["reader"] or not (lo <= r.minute(x["adm"]) < hi) or x["wait"] < 0.05:
        continue
    todo = sorted(out, reverse=True)
    t = x["entry"]
    while t < x["adm"] and todo:
        pos = len(ahead) - bisect.bisect_right(ahead, t)
        rt = rate_at(t)
        if rt:
            p = (pos + 0.5) / rt
            while todo and p <= todo[0]:
                out[todo.pop(0)].append((x["adm"] - t) - p)
        t += 0.25
    for X in todo:
        never[X] += 1
for X in sorted(out):
    e = out[X]
    print(f"     X = {X:4.1f} s (n {len(e):5d}, never crossed {never[X]:4d}): {qline(e)}   margin for q=0.1: {-lib.q(e,.1):+.1f} s, q=0.5: {-lib.q(e,.5):+.1f} s, q=0.9: {-lib.q(e,.9):+.1f} s")

# Part C: landed reads
slack, tok_s, tot_tok = [], 0.0, 0
for x in win:
    if not x["reader"]:
        continue
    pins = x["rec"].get("srs_pin", []); unp = x["rec"].get("srs_unpin", [])
    s = x["adm"] - x["last_land"]
    slack.append(s)
    for p in pins:
        end = min((u["t"] for u in unp if u["t"] >= p["t"]), default=x["adm"])
        tok_s += p["tokens"] * (end - p["t"]); tot_tok += p["tokens"]
span = (hi - lo) * 60
print(f"  C. readers: last landing -> admission (s): {qline(slack)}; pinned token-seconds {tok_s/1e6:.2f} M over {span:.0f} s = mean {tok_s/span/1e3:.1f} K tokens pinned; mean pin {tok_s/max(1,tot_tok):.2f} s")
holds = [h for h in r.kind["ntp_hold"] if lo <= r.minute(h["t"]) < hi]
if holds:
    print(f"     ntp_hold (first hold per request) n {len(holds)}: pos median {lib.q([h['pos'] for h in holds],.5)}, predicted wait median {lib.q([h['wait_ms'] for h in holds],.5)/1e3:.1f} s, "
          f"load estimate median {lib.q([h['load_ms'] for h in holds],.5)/1e3:.1f} s, hold median {lib.q([h['hold_ms'] for h in holds],.5)/1e3:.1f} s [p90 {lib.q([h['hold_ms'] for h in holds],.9)/1e3:.1f}]")
    by = {x["rid"]: x for x in reqs}
    e = []
    for h in holds:
        x = by.get(h["rid"])
        if x:
            e.append((x["adm"] - h["t"]) - h["wait_ms"] / 1e3)
    print(f"     held readers: (admission - hold line) - predicted residual (s): {qline(e)}")
