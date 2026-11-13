#!/usr/bin/env python3
"""Item 5, part d: do the runs start from the same server state?  First turns of the first ten minutes."""
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


runs = list(P)
first = {a: {t["conv"]: t for t in P[a]["turns"] if t["turn"] == 0} for a in runs}
print("=== first turns of convs 0..19: ttft seconds (uncached tokens = prompt - cached) per run")
print("%4s %7s " % ("conv", "prompt") + " ".join("%13s" % a[:13] for a in runs))
for c in range(20):
    row = []
    for a in runs:
        t = first[a].get(c)
        row.append("%6.3f(%5d)" % (t["ttft"], t["prompt_tokens"] - t["cached_tokens"]) if t else "-")
    print("%4d %7d " % (c, first[runs[0]][c]["prompt_tokens"]) + " ".join("%13s" % x for x in row))

print()
print("=== first turns sent in minutes 0-10 (51 convs): cached tokens and ttft per run")
print("%-14s %6s %9s %9s %9s %8s %9s %9s %9s" % (
    "run", "n", "cached=0", "sum_unc", "sum_ttft", "tok/s", "ttft_med", "ttft_mean", "ttft_max"))
for a in runs:
    v = [t for t in first[a].values() if t["t_send"] < 600]
    unc = [t["prompt_tokens"] - t["cached_tokens"] for t in v]
    tt = [t["ttft"] for t in v]
    r = {"n": len(v), "cached_zero": sum(1 for t in v if t["cached_tokens"] == 0), "sum_uncached": sum(unc),
         "sum_ttft": math.fsum(tt), "uncached_tok_per_ttft_s": sum(unc) / math.fsum(tt), "ttft_median": pct(tt, 0.5),
         "ttft_mean": mean(tt), "ttft_max": max(tt),
         "start_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(P[a]["run"][0]["t_start"]))}
    RES[a] = r
    print("%-14s %6d %9d %9d %9.2f %8.0f %9.4f %9.4f %9.3f" % (
        a, len(v), r["cached_zero"], r["sum_uncached"], r["sum_ttft"], r["uncached_tok_per_ttft_s"],
        r["ttft_median"], r["ttft_mean"], r["ttft_max"]))

print()
print("=== first turns with cached_tokens == 0 in minutes 0-10: conv, prompt, ttft")
for a in runs:
    v = sorted((t["t_send"], t["conv"], t["prompt_tokens"], round(t["ttft"], 3)) for t in first[a].values()
               if t["t_send"] < 600 and t["cached_tokens"] == 0)
    print("%-14s %s" % (a, [(c, p, x) for _, c, p, x in v]))

print()
print("=== cached tokens of first turns (all first turns of the run): where they come from")
for a in runs:
    v = list(first[a].values())
    dev = sum((t["cached_details"] or {}).get("device", 0) for t in v)
    host = sum((t["cached_details"] or {}).get("host", 0) for t in v)
    sto = sum((t["cached_details"] or {}).get("storage", 0) for t in v)
    tot = sum(t["cached_tokens"] for t in v)
    pr = sum(t["prompt_tokens"] for t in v)
    RES[a].update({"first_turns": len(v), "first_prompt_tokens": pr, "first_cached_tokens": tot,
                   "first_cached_device": dev, "first_cached_host": host, "first_cached_storage": sto})
    print("%-14s first turns %4d  prompt %9d  cached %8d (%.3f)  device %8d  host %7d  storage %7d" % (
        a, len(v), pr, tot, tot / pr, dev, host, sto))

# the very first returning turns: is anything served from host or storage in minute 0-2 (left over from an earlier run)?
print()
print("=== cache tier of the cached tokens on returning turns, minutes 0-5 (a leftover cache would show as host/storage hits)")
for a in runs:
    v = [t for t in P[a]["turns"] if t["turn"] > 0 and t["t_send"] < 300]
    dev = sum((t["cached_details"] or {}).get("device", 0) for t in v)
    host = sum((t["cached_details"] or {}).get("host", 0) for t in v)
    sto = sum((t["cached_details"] or {}).get("storage", 0) for t in v)
    RES[a].update({"early_ret_turns": len(v), "early_device": dev, "early_host": host, "early_storage": sto})
    print("%-14s turns %4d  device %9d  host %7d  storage %7d" % (a, len(v), dev, host, sto))

with open(os.path.join(HERE, "item5d_warm_state.json"), "w") as f:
    json.dump(RES, f, indent=1, sort_keys=True)
print()
print("wrote item5d_warm_state.json")
