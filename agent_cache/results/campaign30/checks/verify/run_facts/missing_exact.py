"""Missing at the cut, at the exact time of the client's lanes line near the cut (not the rounded second), and with
error requests still in flight at that time counted as recorded. Read-only.
usage: missing_exact.py <manifest> [cut]   (from agent_cache/results)"""
import json, sys

man, cut = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 6600.0
for line in open(man):
    if not line.strip():
        continue
    label, run, prog = line.split()[:3]
    lanes, reqs = [], []
    for l in open(f"{run}/{prog}/client.jsonl"):
        r = json.loads(l)
        if r["kind"] == "lanes":
            lanes.append(r)
        elif r["kind"] == "req":
            reqs.append(r)
    ln = min(lanes, key=lambda x: abs(x["t"] - cut))
    T = ln["t"]
    ok = [r for r in reqs if r.get("latency") is not None]
    err = [r for r in reqs if r.get("error")]
    # recorded with a latency, in flight at T (sent before T, ended after T)
    fl_ok = [r for r in ok if r["t_send"] < T < r["t_send"] + r["latency"]]
    # sent in [cut, T): in flight at T but sent after the cut
    sent_between = [r for r in fl_ok if r["t_send"] >= cut]
    ended_between = [r for r in ok if r["t_send"] < cut and cut < r["t_send"] + r["latency"] <= T]
    # errors: errors counted by T per the lanes line; later errors were in flight at T if sent before T
    errs_after = len(err) - ln["errors"]
    fl_err = [r for r in err if r["t_send"] < T][:max(0, errs_after)] if errs_after > 0 else []
    miss_T = ln["in_flight"] - len(fl_ok) - len(fl_err)
    print(f"{label:18s} lanes t {T}: in flight {ln['in_flight']}; recorded in flight at T {len(fl_ok)} "
          f"(of them sent in [{cut:.0f}, T): {len(sent_between)}); recorded ending in ({cut:.0f}, T]: {len(ended_between)}; "
          f"error requests in flight at T {len(fl_err)}; never recorded, sent before T: {miss_T}")
