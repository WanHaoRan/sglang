"""Continuum's scheduler_timestamps: for each job (user), the last arrival with no departure (in flight at the end),
and whether it reached waiting_to_running before the client stopped. Also the error request's job. Read-only.
usage: continuum_sched.py <run dir> <programs dir> [job]   (from agent_cache/results)"""
import json, sys
run, prog = sys.argv[1], sys.argv[2]
job_err = sys.argv[3] if len(sys.argv) > 3 else None
t0 = json.loads(open(f"{run}/{prog}/client.jsonl").readline())["t_start"]
st = json.load(open(f"{run}/scheduler_timestamps"))
stop = t0 + 7230.0
cut = t0 + 6600.0
open_reqs = []
for job, evs in st.items():
    cur = None
    for e in evs:
        if "Request_arrival_time" in e:
            cur = {"job": job, "arr": e["Request_arrival_time"], "run": None, "dep": None}
        elif "waiting_to_running" in e and cur is not None:
            if cur["run"] is None:
                cur["run"] = e["waiting_to_running"]
        elif "Request_departure_time" in e and cur is not None:
            cur["dep"] = e["Request_departure_time"]
            cur = None
    if cur is not None:
        open_reqs.append(cur)
print(f"{run}: jobs {len(st)}; arrivals with no departure: {len(open_reqs)}")
bc = [r for r in open_reqs if r["arr"] < cut]
print(f"  of them arrived before the cut (6,600 s): {len(bc)}; reached running before the stop: "
      f"{sum(1 for r in bc if r['run'] is not None and r['run'] < stop)}")
for r in bc:
    if r["run"] is not None:
        print(f"    {r['job']} arrived {r['arr'] - t0:.1f} s, running at {r['run'] - t0:.1f} s")
if job_err:
    print(f"  job {job_err}: events")
    for e in st.get(job_err, []):
        print("   ", {k: (round(v - t0, 3) if isinstance(v, float) and v > 1e9 else v) for k, v in e.items()})
