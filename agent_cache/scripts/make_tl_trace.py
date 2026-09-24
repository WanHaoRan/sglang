"""Derived trace for the RT-80 campaign: the >=45-turn conversations of the 262K trace, truncated at 50 turns,
with a deterministic TraceLab-cadence gap baked into pre_gap for every (conversation, turn>=1).

Gap(i, t) = tool + think, drawn from random.Random((SEED << 32) ^ (i << 12) ^ t), i = index in the derived list:
  tool  = min(192, 0.1 * exp(3.593 * z1))                     TraceLab (arXiv 2606.30560) Table 7 per-step tool execution:
                                                              p50 0.1 s, p90 10.0 s, cap = p99 3.2 min (avg 16.8 s uncapped)
  think = min(1800, 84 * exp(2.098 * z2)) with prob. 1/8     TraceLab human thinking events: p50 1.4 min, p90 20.6 min;
                                                              "an agent on average takes around 8 steps" per user request
Rationale (agent_cache/results/<campaign>/DECISIONS.md): the source trace's own gaps come from an unattended SWE-bench
harness (p50 0.71 s, no human pauses); TraceLab measures 4,265 real Claude Code / Codex sessions. Synthetic cadence
calibrated to TraceLab, not a replay of recorded timing.
Replay with GAP=1 GAP_CAP=0 (the caps live here). Original gaps kept in pre_gap_orig, the think part in gap_think.

usage: python3 make_tl_trace.py [--seed 1] [--min-turns 45] [--turns 50] [--out PATH] [--dry-run]
"""
import argparse
import json
import math
import random
import statistics as st

SRC = "/home/wanhr/sglang/agent_cache/traces/lmcache_agentic_trace_262k.json"
TOOL_MED, TOOL_P90, TOOL_CAP = 0.1, 10.0, 192.0
THINK_MED, THINK_P90, THINK_CAP = 84.0, 1236.0, 1800.0
P_THINK = 1 / 8.0
Z90 = 1.2816


def gap(seed: int, i: int, t: int) -> tuple:
    r = random.Random((seed << 32) ^ (i << 12) ^ t)
    tool = min(TOOL_CAP, TOOL_MED * math.exp(r.gauss(0, math.log(TOOL_P90 / TOOL_MED) / Z90)))
    think = 0.0
    if r.random() < P_THINK:
        think = min(THINK_CAP, THINK_MED * math.exp(r.gauss(0, math.log(THINK_P90 / THINK_MED) / Z90)))
    return tool + think, think


def pct(a, q):
    a = sorted(a)
    k = (len(a) - 1) * q / 100
    f = int(k)
    c = min(f + 1, len(a) - 1)
    return a[f] + (a[c] - a[f]) * (k - f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--min-turns", type=int, default=45)
    ap.add_argument("--turns", type=int, default=50)
    ap.add_argument("--out", default="lmcache_agentic_trace_262k_ge45_tl_s1.json")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    d = json.load(open(SRC))
    keep = [k for k, c in enumerate(d["conversations"]) if c and len(c) >= a.min_turns]
    convs = []
    for i, k in enumerate(keep):
        c = [dict(t) for t in d["conversations"][k][: a.turns]]
        for t, turn in enumerate(c):
            turn["pre_gap_orig"] = turn.get("pre_gap")
            if t == 0:
                turn["gap_think"] = 0.0
                continue
            g, th = gap(a.seed, i, t)
            turn["pre_gap"] = round(g, 4)
            turn["gap_think"] = round(th, 4)
        convs.append(c)
    flat = [t["pre_gap"] for c in convs for t in c[1:]]
    n = len(flat)
    print(f"{len(convs)} conversations, {sum(len(c) for c in convs)} turns; gap mean {st.mean(flat):.1f} s p50 {pct(flat, 50):.2f} "
          f"p90 {pct(flat, 90):.1f} p99 {pct(flat, 99):.0f}; >1 min {sum(x > 60 for x in flat) / n:.3f} >5 min "
          f"{sum(x > 300 for x in flat) / n:.3f} >10 min {sum(x > 600 for x in flat) / n:.3f}; think pauses "
          f"{sum(1 for c in convs for t in c[1:] if t['gap_think'] > 0) / n:.3f} of gaps")
    if a.dry_run:
        return
    meta = dict(d.get("metadata") or {})
    meta["derived"] = {"from": SRC, "min_turns": a.min_turns, "turns": a.turns, "seed": a.seed,
                       "gap_model": "TraceLab tool LN(p50 0.1,p90 10.0) cap 192 s + think LN(p50 84,p90 1236) cap 1800 s w.p. 1/8",
                       "replay": "GAP=1 GAP_CAP=0"}
    out = {"metadata": meta,
           "session_ids": [d["session_ids"][k] for k in keep] if d.get("session_ids") else [],
           "sessions": [d["sessions"][k] for k in keep] if d.get("sessions") else [],
           "conversations": convs}
    json.dump(out, open(a.out, "w"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
