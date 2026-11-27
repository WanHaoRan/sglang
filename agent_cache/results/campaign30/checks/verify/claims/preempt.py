"""Continuum's own scheduler record: evictions from the running queue (preemptions) during a job's first request
(from its first arrival to its first departure) against during its later requests."""
import json, os, sys
base = "/home/wanhr/sglang/agent_cache/results"
for run in sys.argv[1:]:
    d = json.load(open(os.path.join(base, run, "scheduler_timestamps")))
    first_ev = later_ev = 0
    jobs_first_preempted = 0
    for job, evs in d.items():
        seen_dep = False
        fe = 0
        for e in evs:
            for k in e:
                if k == "Request_departure_time":
                    seen_dep = True
                elif k == "Request_evicted_from_running_queue_time":
                    if seen_dep:
                        later_ev += 1
                    else:
                        fe += 1
        first_ev += fe
        jobs_first_preempted += fe > 0
    print(f"{run}: jobs {len(d)}; evictions during a job's first request {first_ev} (jobs with any: {jobs_first_preempted}); during later requests {later_ev}")
