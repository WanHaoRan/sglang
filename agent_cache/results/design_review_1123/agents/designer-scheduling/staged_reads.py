"""Staged (pre-arrival) reads in one server log: the engine's read-time estimate at staging (ntp_due load_ms)
against stage -> landing (first s2h_io of the rid after staging), and landing against the turn's arrival
(ntp_arrive) and admission (ntp_admit line time). Read-only. Usage: python3 staged_reads.py <server.log>"""
import collections, datetime, re, sys

LINE = re.compile(r"^\[(\S+ \S+)\] HICACHE_EVT (ntp_due|ntp_stage|s2h_io|ntp_arrive|ntp_admit) rid=(\S+)(.*)")
first_prefill = None
due, stage, lands, arrive, admit = {}, {}, collections.defaultdict(list), {}, {}
def ts(s):
    return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").timestamp()
for line in open(sys.argv[1], errors="replace"):
    if first_prefill is None and "Prefill batch" in line:
        first_prefill = ts(line[1:24])
    if "HICACHE_EVT" not in line:
        continue
    m = LINE.match(line)
    if not m:
        continue
    t, kind, rid, rest = ts(m[1]), m[2], m[3], m[4]
    if kind == "ntp_due" and " case=1" in rest:
        due.setdefault(rid, (t, int(re.search(r"load_ms=(\d+)", rest)[1])))
    elif kind == "ntp_stage":
        stage.setdefault(rid, t)
    elif kind == "s2h_io":
        lands[rid].append(t)
    elif kind == "ntp_arrive":
        arrive.setdefault(rid, t)
    elif kind == "ntp_admit":
        admit.setdefault(rid, t)
def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")
win = collections.defaultdict(lambda: collections.defaultdict(list))
for rid, ts_ in stage.items():
    l = [x for x in lands.get(rid, ()) if x >= ts_]
    if not l or rid not in due:
        continue
    land = l[0]
    m = (ts_ - first_prefill) / 60
    w = "0-30" if m < 30 else "30-60" if m < 60 else "60-90" if m < 90 else "90+"
    d = win[w]
    d["pred_r"].append(due[rid][1] / 1000)
    d["actual_r"].append(land - ts_)
    if rid in arrive:
        d["land_minus_arrival"].append(land - arrive[rid])
    if rid in admit:
        d["admit_minus_land"].append(admit[rid] - land)
for w in ("0-30", "30-60", "60-90", "90+"):
    d = win[w]
    if not d["pred_r"]:
        continue
    print(f"min {w:>5}: staged reads {len(d['pred_r'])}", end="")
    for k in ("pred_r", "actual_r", "land_minus_arrival", "admit_minus_land"):
        xs = d[k]
        print(f" | {k} p10/p50/p90 {q(xs,.1):.1f}/{q(xs,.5):.1f}/{q(xs,.9):.1f}", end="")
    print()
