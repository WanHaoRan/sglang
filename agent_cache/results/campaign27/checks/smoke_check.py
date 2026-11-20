"""Correctness checks of replay_programs.py runs against the engine: one block per client run of a server run folder.
From agent_cache/:  python3 -I -B results/campaign27/checks/smoke_check.py results/<server run> [more runs]
Client side: errors, the server's prompt count against the count sent, output length against the recorded one, cached
tokens against what an infinite cache would hold, tiers, and which table entry each hint would pick.
Server side (the whole log of the run folder): tracebacks, warnings, hint events and the keys the engine picked."""
import collections
import glob
import json
import os
import re
import sys

PICK = re.compile(r"HICACHE_EVT ntp_pick rid=(\S+) key=(\S+) entry=(\S+)")
EVT = re.compile(r"HICACHE_EVT (\w+)")
BINS = ((0, 1), (1, 10), (10, 60), (60, 300), (300, float("inf")))


def q(values, p):
    v = sorted(values)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")


def client(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    run = next((r for r in rows if r["kind"] == "run"), None)
    if run is None:
        print(f"--- {os.path.basename(os.path.dirname(path))}: no run record yet")
        return set()
    reqs = [r for r in rows if r["kind"] == "req"]
    prof = [r for r in reqs if r["phase"] == "profile" and not r.get("error") and r.get("ttft") is not None]
    args = run["args"]
    print(f"--- {os.path.basename(os.path.dirname(path))}: {run['trace_header']['source']}, recipe {args['recipe']}, hints {args['hints']}, "
          f"max context {args['max_context']}, {run['trees']} trees")
    summary = next((r for r in rows if r["kind"] == "summary"), {})
    print(f"  requests {len(reqs)} (profile {len(prof)}), errors {sum(1 for r in reqs if r.get('error'))}, plays started {summary.get('plays')}, "
          f"drained {sum(1 for r in rows if r['kind'] == 'play')}, unsent at the end {summary.get('skipped')}")
    if not prof:
        return set()
    same = sum(1 for r in prof if r.get("prompt_tokens") == r["prompt_sent"])
    out_ok = sum(1 for r in prof if r.get("completion_tokens") == r["out_target"])
    prompt = sum(r["prompt_tokens"] for r in prof)
    tiers = collections.Counter()
    for r in prof:
        for k in ("device", "host", "storage"):
            tiers[k] += int((r.get("cached_details") or {}).get(k) or 0)
    cached, ideal = sum(tiers.values()), sum(r["ideal_cached"] for r in prof)
    print(f"  server's prompt count equals the count sent: {same} of {len(prof)}; output length as recorded: {out_ok} of {len(prof)}")
    print(f"  prompt tokens: mean {prompt // len(prof)}, max {max(r['prompt_tokens'] for r in prof)}; served from device {100 * tiers['device'] / prompt:.1f} %, "
          f"host {100 * tiers['host'] / prompt:.1f} %, storage {100 * tiers['storage'] / prompt:.1f} %, recomputed {100 * (1 - cached / prompt):.1f} %; "
          f"ideal {100 * ideal / prompt:.1f} %, delivered over ideal {cached / ideal if ideal else float('nan'):.3f}")
    short = [r for r in prof if r["ideal_cached"] - sum(int((r.get("cached_details") or {}).get(k) or 0) for k in ("device", "host", "storage")) > 256]
    print(f"  requests served over 256 tokens less than an infinite cache would: {len(short)}")
    ttft = [r["ttft"] for r in prof]
    tpot = [(r["latency"] - r["ttft"]) / (r["completion_tokens"] - 1) for r in prof if (r.get("completion_tokens") or 0) > 1]
    span = max(r["t_send"] + r["latency"] for r in prof) - min(r["t_send"] for r in prof)
    print(f"  TTFT mean {sum(ttft) / len(ttft):.2f} s, p50 {q(ttft, .5):.2f}, p90 {q(ttft, .9):.2f}; time per output token p50 {1000 * q(tpot, .5):.0f} ms; "
          f"{60 * len(prof) / span:.1f} requests per minute")
    print("  TTFT by the wait before the request: " + ", ".join(
        f"{lo:g}-{hi:g} s {sum(v) / len(v):.2f} s ({len(v)})" for lo, hi in BINS
        if (v := [r["ttft"] for r in prof if lo <= r["wait_s"] < hi])))
    hints = [r["hint"] for r in reqs if r.get("hint")]
    if hints:
        matched = collections.Counter(str(h["matched"]) for h in hints)
        named = sum(n for k, n in matched.items() if k not in ("None", "*", "_end"))
        print(f"  hints sent {len(hints)}: named entry picked {named}, default entry {matched['*']}, end {matched['_end']}, nothing to pick {matched['None']}; "
              f"largest table {max(h['list_len'] for h in hints)} entries; most picked {matched.most_common(6)}")
    if any("oracle_gap_s" in r for r in reqs):
        print(f"  exact gaps sent: {sum(1 for r in reqs if 'oracle_gap_s' in r)}")
    return {r["rid"] for r in reqs}


for run_dir in sys.argv[1:]:
    print(f"===== {run_dir}")
    rids = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "programs_*", "client.jsonl"))):
        tag = os.path.basename(os.path.dirname(path)).split("_", 2)[2]
        rids[tag] = client(path)
    events, picks, lines = collections.Counter(), collections.defaultdict(collections.Counter), collections.Counter()
    for line in open(os.path.join(run_dir, "server.log"), errors="replace"):
        if "Traceback" in line:
            lines["tracebacks"] += 1
        if " WARNING " in line or "malformed" in line:
            lines["warnings"] += 1
        if " ERROR " in line:
            lines["errors"] += 1
        m = EVT.search(line)
        if m:
            events[m[1]] += 1
            p = PICK.search(line)
            if p:
                tag = p[1].rsplit("-c", 1)[0]
                kind = "end" if p[3] == "_end" else "none" if p[3] == "none" else "default" if p[3] == "any" else "named"
                picks[tag][kind] += 1
                if kind == "named":
                    picks[tag]["key equals entry" if p[2] == p[3] else "coarser entry"] += 1
    print(f"--- server log: tracebacks {lines['tracebacks']}, warning lines {lines['warnings']}, error lines {lines['errors']}")
    print("  events: " + ", ".join(f"{k} {n}" for k, n in events.most_common(22)))
    for tag, c in picks.items():
        print(f"  engine's picks for {tag}: " + ", ".join(f"{k} {n}" for k, n in c.most_common()))
