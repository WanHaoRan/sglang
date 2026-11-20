"""Campaign 28 or 29, a just-in-time arm: how much of the history computed again belongs to requests the engine
refused a read at its cap of three SSD lookups ("ntp_gate ... decision=capped"). On the host, from agent_cache/:
  python3 -I -B results/campaign28/checks/cap_share.py results/<server run> <programs dir> [cut_s]
Over the returning requests sent before cut_s (default 6,600 s). Stdlib only."""
import json
import re
import sys

run, programs = sys.argv[1], sys.argv[2]
cut = float(sys.argv[3]) if len(sys.argv) > 3 else 6600.0
capped, decisions = set(), {}
with open(f"{run}/server.log", errors="replace") as f:
    for line in f:
        if "ntp_gate" in line and (m := re.search(r"decision=(\w+)", line)):
            decisions[m[1]] = decisions.get(m[1], 0) + 1
            if m[1] == "capped":
                capped.add(re.search(r"rid=(\S+)", line)[1])
with open(f"{run}/{programs}/client.jsonl") as f:
    reqs = [json.loads(line) for line in f if '"kind":"req"' in line]


def cached(r):
    d = r.get("cached_details") or {}
    return sum(int(d.get(k) or 0) for k in ("device", "host", "storage")) if d else int(r.get("cached_tokens") or 0)


back = [r for r in reqs if r["t_send"] < cut and r["k"] > 0 and r["ideal_cached"] > 0 and r.get("ttft") is not None]
again = {r["rid"]: max(0, r["ideal_cached"] - cached(r)) for r in back}
at_cap = [r for r in back if r["rid"] in capped]
ideal, total, by_cap = sum(r["ideal_cached"] for r in back), sum(again.values()), sum(again[r["rid"]] for r in at_cap)
print(f"{run.split('/')[-1]}: gate decisions of the run {dict(sorted(decisions.items()))}")
print(f"returning requests sent before {cut:.0f} s: {len(back)}; refused a read at the cap: {len(at_cap)} "
      f"({sum(again[r['rid']] > 0 for r in at_cap)} of them computed history again)")
print(f"history computed again: {total:,} of {ideal:,} tokens ({100 * total / ideal:.2f} %); by the requests at the cap: "
      f"{by_cap:,} ({100 * by_cap / max(1, total):.1f} % of it)")
if at_cap:
    ttft = sorted(r["ttft"] for r in at_cap)
    rest = [r["ttft"] for r in back if r["rid"] not in capped]
    print(f"TTFT mean: at the cap {sum(ttft) / len(ttft):.1f} s, the others {sum(rest) / len(rest):.1f} s")
