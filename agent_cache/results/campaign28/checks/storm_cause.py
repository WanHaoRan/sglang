"""Campaign 28 or 29, an arm that recomputes history: do the requests that got nothing from the cache share a cause?
On the host, from agent_cache/:
  python3 -I -B results/campaign28/checks/storm_cause.py results/<server run> <programs dir> [from_s to_s] > results/campaign28/checks/storm_cause_<label>.out
Over the returning requests sent in [from_s, to_s) (default 3,600 to 6,600 s): how many got nothing, part or all of the
history an unbounded cache would hold, and for each group how many had reached the engine's cap on SSD lookups at
arrival ("prefetch reissue cap reached"), how many the just-in-time engine refused a read at its own cap of three
lookups ("ntp_gate ... decision=capped") and how many were served from SSD in the end. Stdlib only."""
import collections
import json
import re
import sys

run, programs = sys.argv[1], sys.argv[2]
lo, hi = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (3600.0, 6600.0)
capped, jit_capped, found = set(), set(), collections.Counter()
with open(f"{run}/server.log", errors="replace") as f:
    for line in f:
        if "reissue cap reached" in line:
            capped.add(re.search(r"req=(\S+)", line)[1])
        elif "decision=capped" in line:
            jit_capped.add(re.search(r"rid=(\S+)", line)[1])
        elif "s2h_query" in line and (m := re.search(r"rid=(\S+) tokens_req=\d+ storage_hit=(\d+)", line)):
            found[m[1]] += int(m[2])
with open(f"{run}/{programs}/client.jsonl") as f:
    reqs = [json.loads(line) for line in f if '"kind":"req"' in line]


def cached(r):
    d = r.get("cached_details") or {}
    return sum(int(d.get(k) or 0) for k in ("device", "host", "storage")) if d else int(r.get("cached_tokens") or 0)


back = [r for r in reqs if lo <= r["t_send"] < hi and r["k"] > 0 and r["ideal_cached"] > 0 and r.get("ttft") is not None]
groups = (("nothing (under 2 %)", lambda s: s < 0.02), ("part", lambda s: 0.02 <= s < 0.95), ("all (95 % or more)", lambda s: s >= 0.95))
ideal = sum(r["ideal_cached"] for r in back)
print(f"{run.split('/')[-1]}: {len(back)} returning requests sent in [{lo:.0f}, {hi:.0f}) s; of the history an unbounded cache holds, "
      f"{100 * sum(max(0, r['ideal_cached'] - cached(r)) for r in back) / ideal:.1f} % was computed again; {len(capped)} requests of the run reached the lookup cap, "
      f"{len(jit_capped)} the just-in-time read cap")
print(f"{'history delivered':22s} {'requests':>8s} {'TTFT mean s':>11s} {'lookup cap at arrival':>21s} {'JIT read cap':>12s} {'SSD had some at arrival':>23s} {'served from SSD':>15s}")
for name, test in groups:
    rs = [r for r in back if test(cached(r) / r["ideal_cached"])]
    if rs:
        print(f"{name:22s} {len(rs):8d} {sum(r['ttft'] for r in rs) / len(rs):11.1f} {sum(r['rid'] in capped for r in rs):21d} {sum(r['rid'] in jit_capped for r in rs):12d} "
              f"{sum(found[r['rid']] > 0 for r in rs):23d} {sum(int((r.get('cached_details') or {}).get('storage') or 0) > 0 for r in rs):15d}")
