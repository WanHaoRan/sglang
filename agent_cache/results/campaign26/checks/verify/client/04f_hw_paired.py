#!/usr/bin/env python3
"""Item 5, part f: same requests, early in the run, before the arms diverge: is the machine equally fast in every run?

For keys sent in the first 25 minutes (all arms still in step there), compare per key against jit_exact:
decode pace (ms per output token after the first), TTFT of returning turns, TTFT of first turns that had to
compute 2000 tokens or more.  Output lengths are identical per key, so the pace ratio is a like-for-like one.
"""
import json
import math
import os
import pickle
import time

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "parsed.pkl"), "rb") as f:
    P = pickle.load(f)
RES = {}


def pct(v, q):
    s = sorted(v)
    return s[min(len(s) - 1, int(q * len(s)))]


def mean(v):
    return math.fsum(v) / len(v) if v else float("nan")


M = {a: {(t["conv"], t["turn"]): t for t in P[a]["turns"]} for a in P}
REF = "jit_exact"


def itl(t):
    return 1000.0 * (t["latency"] - t["ttft"]) / (t["completion_tokens"] - 1)


for lo, hi in ((0, 600), (600, 1500), (0, 1500)):
    print("=== keys sent between minute %d and %d in both runs (send times within 5 s of each other), against %s" % (lo // 60, hi // 60, REF))
    print("   %-14s %11s %6s %10s %10s %10s %10s %7s %10s %7s %10s" % (
        "run", "start_utc", "n", "pace_ms", "pace_ratio", "ratio_p10", "ratio_p90", "n_ret", "ttft_ratio", "n_first", "t0_ratio"))
    for a in P:
        keys = [k for k in M[a] if k in M[REF] and lo <= M[REF][k]["t_send"] < hi and abs(M[a][k]["t_send"] - M[REF][k]["t_send"]) <= 5.0]
        dk = [k for k in keys if M[a][k]["completion_tokens"] > 1 and M[a][k]["completion_tokens"] == M[REF][k]["completion_tokens"]
              and itl(M[REF][k]) > 0]
        r = [itl(M[a][k]) / itl(M[REF][k]) for k in dk]
        ret = [k for k in keys if k[1] > 0 and M[REF][k]["ttft"] > 0]
        rt = [M[a][k]["ttft"] / M[REF][k]["ttft"] for k in ret]
        f0 = [k for k in keys if k[1] == 0 and (M[a][k]["prompt_tokens"] - M[a][k]["cached_tokens"]) >= 2000
              and M[a][k]["cached_tokens"] == M[REF][k]["cached_tokens"] and k != (0, 0)]
        r0 = [M[a][k]["ttft"] / M[REF][k]["ttft"] for k in f0]
        res = {"n": len(dk), "pace_ms_median": pct([itl(M[a][k]) for k in dk], 0.5), "pace_ratio_median": pct(r, 0.5),
               "pace_ratio_p10": pct(r, 0.1), "pace_ratio_p90": pct(r, 0.9), "pace_ratio_of_sums":
               math.fsum(M[a][k]["latency"] - M[a][k]["ttft"] for k in dk) / math.fsum(M[REF][k]["latency"] - M[REF][k]["ttft"] for k in dk),
               "n_returning": len(ret), "ttft_ratio_median": pct(rt, 0.5) if rt else None,
               "n_first_2k": len(f0), "first_ttft_ratio_median": pct(r0, 0.5) if r0 else None,
               "first_ttft_ratio_of_sums": (math.fsum(M[a][k]["ttft"] for k in f0) / math.fsum(M[REF][k]["ttft"] for k in f0)) if f0 else None}
        RES["%d-%d:%s" % (lo // 60, hi // 60, a)] = res
        print("   %-14s %11s %6d %10.3f %10.4f %10.4f %10.4f %7d %10.4f %7d %10s   pace sum-ratio %.4f  first-turn sum-ratio %s" % (
            a, time.strftime("%m-%d %H:%M", time.gmtime(P[a]["run"][0]["t_start"])), len(dk), res["pace_ms_median"],
            res["pace_ratio_median"], res["pace_ratio_p10"], res["pace_ratio_p90"], len(ret), res["ttft_ratio_median"],
            len(f0), ("%.4f" % res["first_ttft_ratio_median"]) if r0 else "-", res["pace_ratio_of_sums"],
            ("%.4f" % res["first_ttft_ratio_of_sums"]) if f0 else "-"))
    print()

# same workload timing in all 13 runs?  per key: planned gap and output length
print("=== per key across all 13 runs: gap_slept, pre_gap, output_length, replay_prompt_tokens, completion_tokens identical?")
for field in ("gap_slept", "pre_gap", "output_length", "replay_prompt_tokens", "completion_tokens", "task_conv"):
    seen = {}
    for a in P:
        for k, t in M[a].items():
            seen.setdefault(k, set()).add(t.get(field))
    multi = sum(1 for k, s in seen.items() if len(s) > 1)
    RES["identity:" + field] = {"keys": len(seen), "keys_with_more_than_one_value": multi}
    print("   %-22s keys %6d  keys with more than one value across runs: %d" % (field, len(seen), multi))

with open(os.path.join(HERE, "item5f_hw_paired.json"), "w") as f:
    json.dump(RES, f, indent=1, sort_keys=True)
print()
print("wrote item5f_hw_paired.json")
