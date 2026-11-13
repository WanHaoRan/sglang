#!/usr/bin/env python3
"""Item 5, part g: prompts that differ from the trace (replay_prompt_tokens) and what that costs the arm.

The client records the trace's prompt length next to the length the server counted.  Where the two part ways
inside a session, the cached prefix no longer matches and the turn recomputes context in every policy.
"""
import json
import math
import os
import pickle

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "parsed.pkl"), "rb") as f:
    P = pickle.load(f)
RES = {}


def mean(v):
    return math.fsum(v) / len(v) if v else float("nan")


def rc(t):
    return max(t.get("reply_reprefilled") or 0, 0)


M = {a: {(t["conv"], t["turn"]): t for t in P[a]["turns"]} for a in P}

print("=== the three sessions with the largest prompt gap to the trace")
for a, conv, lo, hi in (("hintfree_ctl", 118, 40, 49), ("jit_p50", 17, 26, 34), ("jit_p50_d_cut", 132, 27, 35)):
    other = [b for b in P if b != a and (conv, lo) in M[b]][:2]
    print("-- %s conv %d: turn, prompt, trace prompt, gap, cached, new tokens, recomputed, ttft   | same key in %s: gap, recomputed, ttft" % (a, conv, ", ".join(other)))
    for n in range(lo, hi + 1):
        t = M[a].get((conv, n))
        if not t:
            continue
        s = "   %3d %6d %6d %+6d %6d %6d %6d %8.3f  |" % (n, t["prompt_tokens"], t["replay_prompt_tokens"],
                                                         t["prompt_tokens"] - t["replay_prompt_tokens"], t["cached_tokens"],
                                                         t.get("delta_tokens") or 0, rc(t), t["ttft"])
        for b in other:
            u = M[b].get((conv, n))
            if u:
                s += "  %+5d %6d %7.3f" % (u["prompt_tokens"] - u["replay_prompt_tokens"], rc(u), u["ttft"])
        print(s)

print()
print("=== turns where the gap to the trace changes by more than 8 tokens from the previous turn of the session")
print("    (the step is where the prompt stopped matching what the server had cached)")
print("   %-14s %6s %10s %12s %12s %10s %12s" % ("run", "steps", "max |step|", "recomp_mean", "recomp_other", "ttft_mean", "ttft_other"))
for a in P:
    steps = []
    for (c, n), t in M[a].items():
        if n == 0 or (c, n - 1) not in M[a]:
            continue
        p = M[a][(c, n - 1)]
        g1 = t["prompt_tokens"] - t["replay_prompt_tokens"]
        g0 = p["prompt_tokens"] - p["replay_prompt_tokens"]
        if abs(g1 - g0) > 8:
            steps.append((abs(g1 - g0), c, n, rc(t), t["ttft"]))
    others = [t for k, t in M[a].items() if k[1] > 0]
    r = {"steps_over_8": len(steps), "max_step": max((s[0] for s in steps), default=0),
         "mean_recomputed_on_steps": mean([s[3] for s in steps]) if steps else None,
         "mean_recomputed_all_returning": mean([rc(t) for t in others]),
         "mean_ttft_on_steps": mean([s[4] for s in steps]) if steps else None,
         "mean_ttft_all_returning": mean([t["ttft"] for t in others]),
         "sum_recomputed_on_steps": sum(s[3] for s in steps), "sum_recomputed_all_returning": sum(rc(t) for t in others)}
    RES[a] = r
    print("   %-14s %6d %10d %12.0f %12.0f %10.3f %12.3f   share of the run's recomputed tokens on these turns: %.4f" % (
        a, len(steps), r["max_step"], r["mean_recomputed_on_steps"] or 0, r["mean_recomputed_all_returning"],
        r["mean_ttft_on_steps"] or 0, r["mean_ttft_all_returning"],
        r["sum_recomputed_on_steps"] / max(1, r["sum_recomputed_all_returning"])))

print()
print("=== recomputed tokens: how much is the floor that every arm pays (minutes 0-10, nothing evicted yet)")
print("   %-14s %6s %10s %10s %10s" % ("run", "n", "mean", "median", "max"))
for a in P:
    v = sorted(rc(t) for k, t in M[a].items() if k[1] > 0 and t["t_send"] < 600)
    RES[a].update({"early_recomputed_mean": mean(v), "early_recomputed_median": v[len(v) // 2], "early_recomputed_max": v[-1]})
    print("   %-14s %6d %10.2f %10d %10d" % (a, len(v), mean(v), v[len(v) // 2], v[-1]))

with open(os.path.join(HERE, "item5g_prompt_drift.json"), "w") as f:
    json.dump(RES, f, indent=1, sort_keys=True)
print()
print("wrote item5g_prompt_drift.json")
