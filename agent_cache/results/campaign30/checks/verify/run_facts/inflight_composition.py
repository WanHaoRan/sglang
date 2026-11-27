"""What the client's in-flight count at the lanes line near 6,600 s is made of, per arm: recorded requests in flight
then, error requests still in flight then, re-sent requests whose first attempt was in flight then (their record holds
the re-send's t_send, so a cut on t_send drops them), and the rest (cancelled at the stop, no record). Also the users
admitted by then whose first request (stream 0, k 0) has no record at all. Read-only.
usage: inflight_composition.py <manifest> [cut]   (from agent_cache/results)"""
import json, sys

TIMEOUT = 3600.0   # aiohttp sock_read in replay_programs.py / replay_baseline.py
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
    rec_ok = [r for r in reqs if r.get("latency") is not None and r["t_send"] < T < r["t_send"] + r["latency"]]
    errs_after = len([r for r in reqs if r.get("error")]) - ln["errors"]
    err_in = [r for r in reqs if r.get("error") and r["t_send"] < T] if errs_after > 0 else []
    # re-sent: the record's t_send is the retry's; the first attempt was sent about TIMEOUT s earlier
    resent = [r for r in reqs if r.get("retried") and r["t_send"] - TIMEOUT < T <= r["t_send"]]
    rest = ln["in_flight"] - len(rec_ok) - len(err_in) - len(resent)
    admitted = ln["plays"]
    firsts_any = {r["play"] for r in reqs if r["stream"] == 0 and r["k"] == 0}
    no_first = [p for p in range(admitted) if p not in firsts_any]
    firsts_before = {r["play"] for r in reqs if r["stream"] == 0 and r["k"] == 0 and r["t_send"] < cut}
    print(f"{label:18s} T {T}: in flight {ln['in_flight']} = recorded {len(rec_ok)} + errors {len(err_in)} "
          f"{[e['rid'] for e in err_in]} + re-sent {len(resent)} {[x['rid'] + '@' + str(x['t_send']) for x in resent]} "
          f"+ cancelled with no record {rest}; users admitted by T {admitted}: no first-request record at all "
          f"{len(no_first)}, none sent before the cut (continuum_first.py's count) {admitted - len(firsts_before & set(range(admitted)))}")
