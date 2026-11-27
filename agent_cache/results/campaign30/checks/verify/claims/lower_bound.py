"""Continuum arms: the requests in flight at the cut without a before-cut record, split by what the records do hold,
and the mean TTFT raised by counting them at a floor."""
import json, os
base = "/home/wanhr/sglang/agent_cache/results"
CUT = 6600
for label, run, prog, missing in (("continuum", "20261124_203658_continuum_continuum", "programs_203842_continuum", 72),
                                  ("continuum_lmcache", "20261124_224027_continuum_continuum_lmcache", "programs_224335_continuum_lmcache", 21)):
    rows = [json.loads(l) for l in open(os.path.join(base, run, prog, "client.jsonl"))]
    reqs = [r for r in rows if r["kind"] == "req" and r["phase"] == "profile"]
    lanes = {round(r["t"]): r for r in rows if r["kind"] == "lanes"}
    ok = [r for r in reqs if r["t_send"] < CUT and r.get("ttft") is not None]
    s, n = sum(r["ttft"] for r in ok), len(ok)
    inflight = lanes[CUT]["in_flight"]
    ending_later = sum(1 for r in reqs if r["t_send"] < CUT and r.get("latency") is not None and r["t_send"] + r["latency"] > CUT)
    err_inflight = sum(1 for r in reqs if r.get("error") and r["t_send"] < CUT)
    retried_after = [r for r in reqs if r.get("retried") and r["t_send"] >= CUT and r["t_send"] - 3600 < CUT]
    print(f"== {label}: n {n} mean {s/n:.3f}; in flight at cut {inflight}; recorded ending later {ending_later}; "
          f"error records sent before the cut {err_inflight}; retried records whose first attempt (t_send - 3600 s) was before the cut {len(retried_after)}")
    print(f"   truly without any record: {inflight - ending_later - err_inflight - len(retried_after)}")
    for floor in (630,):
        m2 = (s + missing * floor) / (n + missing)
        print(f"   counting {missing} at {floor} s: mean {m2:.3f} (raise {m2 - s/n:.3f})")
