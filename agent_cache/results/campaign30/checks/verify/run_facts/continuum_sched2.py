"""Continuum's scheduler_timestamps: requests that arrived before the cut and departed at or after the client's stop
(7,230 s; cancelled) or after 7,199 s; whether each reached waiting_to_running and when. Read-only.
usage: continuum_sched2.py <run dir> <programs dir>   (from agent_cache/results)"""
import json, sys, collections
run, prog = sys.argv[1], sys.argv[2]
t0 = json.loads(open(f"{run}/{prog}/client.jsonl").readline())["t_start"]
st = json.load(open(f"{run}/scheduler_timestamps"))
reqs = []
for job, evs in st.items():
    cur = None
    for e in evs:
        if "Request_arrival_time" in e:
            if cur is not None:
                reqs.append(cur)
            cur = {"job": job, "arr": e["Request_arrival_time"] - t0, "run": None, "dep": None, "n": len([r for r in reqs if r["job"] == job])}
        elif "waiting_to_running" in e and cur is not None and cur["run"] is None:
            cur["run"] = e["waiting_to_running"] - t0
        elif "Request_departure_time" in e and cur is not None:
            cur["dep"] = e["Request_departure_time"] - t0
            reqs.append(cur)
            cur = None
    if cur is not None:
        reqs.append(cur)
late = [r for r in reqs if r["arr"] < 6600 and (r["dep"] is None or r["dep"] >= 7229.0)]
print(f"{run}: requests {len(reqs)}; arrived before 6,600 s and departed at/after 7,229 s (or never): {len(late)}")
firsts = [r for r in late if r["n"] == 0]
print(f"  of them a job's first request: {len(firsts)}; reached running: {sum(1 for r in late if r['run'] is not None)}")
for r in late:
    if r["run"] is not None:
        print(f"    {r['job']} #{r['n']} arrived {r['arr']:.1f}, running {r['run']:.1f}, departed {r['dep']}")
deps = collections.Counter(round(r["dep"]) for r in late if r["dep"] is not None)
print(f"  departure seconds: {dict(sorted(deps.items())[:10])}")
mid = [r for r in reqs if r["arr"] < 6600 and r["dep"] is not None and 7199 <= r["dep"] < 7229]
print(f"  arrived before 6,600 s, departed in [7,199, 7,229): {[(r['job'], r['n'], round(r['arr'],1), r['run'], round(r['dep'],1)) for r in mid if r['run'] is None]}")
