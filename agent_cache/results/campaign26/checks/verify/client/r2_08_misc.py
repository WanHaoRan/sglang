#!/usr/bin/env python3
"""r2_08_misc.py -- small client-side checks cited in the answer.

  a. start state: uncached tokens of the first turns sent in minutes 0-10; host/storage hits of
     returning turns in minutes 0-5 (a cache left over from an earlier run would show here)
  b. record consistency: ttft <= 0, ttft > latency, completion_tokens != output_length,
     negative or null reply_reprefilled on returning turns
  c. the floor of recomputed tokens (returning turns of minutes 0-10, before any eviction)
  d. LEAD5 prompt_tokens spread counts (<=1, <=2, <=8)
  e. order of session admission

Run as:  nice -n 19 python3 -I -B r2_08_misc.py
"""
import json
import math
import os
import pickle

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ["baseline", "base_rep", "hintfree", "hintfree_ctl", "jit_p50", "jit_p50_ctl",
        "jit_p50_d_cut", "jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend", "ORDER",
        "jit_p50_d", "ec_r13_rep"]
L5 = ["jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend"]


def pct(v, q):
    s = sorted(v)
    return s[min(len(s) - 1, int(q * len(s)))]


def mean(v):
    return math.fsum(v) / len(v) if v else float("nan")


with open(os.path.join(HERE, "r2_parsed.pkl"), "rb") as f:
    P = pickle.load(f)
with open(os.path.join(HERE, "r2_parsed_extra.pkl"), "rb") as f:
    P.update(pickle.load(f))
T = {a: [r for r in P[a]["records"] if r.get("kind") == "turn"] for a in ARMS}
RES = {}

print("=== a. start state")
RES["start_state"] = {}
for a in ARMS:
    f0 = [r for r in T[a] if r["turn"] == 0 and r["t_send"] < 600]
    unc = sum(r["prompt_tokens"] - (r.get("cached_tokens") or 0) for r in f0)
    n0 = sum(1 for r in f0 if (r.get("cached_tokens") or 0) == 0)
    ret5 = [r for r in T[a] if r["turn"] > 0 and r["t_send"] < 300]
    host = sum(((r.get("cached_details") or {}).get("host") or 0) for r in ret5)
    stor = sum(((r.get("cached_details") or {}).get("storage") or 0) for r in ret5)
    dev = sum(((r.get("cached_details") or {}).get("device") or 0) for r in ret5)
    RES["start_state"][a] = {"first_turns_0_10": len(f0), "uncached_tokens": unc,
                             "first_turns_fully_uncached": n0, "ret_0_5": len(ret5),
                             "host_tokens": host, "storage_tokens": stor, "device_tokens": dev}
    print("   %-14s first turns in min 0-10: %2d, uncached tokens %6d, with nothing cached %d | "
          "returning turns in min 0-5: %3d, cached from device %7d, host %d, storage %d"
          % (a, len(f0), unc, n0, len(ret5), dev, host, stor))

print("\n=== b. record consistency (all turn records)")
RES["records"] = {}
for a in ARMS:
    bad_ttft = sum(1 for r in T[a] if r.get("ttft") is not None and r["ttft"] <= 0)
    gt_lat = sum(1 for r in T[a] if r.get("ttft") is not None and r["ttft"] > r["latency"])
    ct = sum(1 for r in T[a] if r.get("completion_tokens") != r.get("output_length"))
    rr_neg = sum(1 for r in T[a] if (r.get("reply_reprefilled") or 0) < 0)
    rr_null_ret = sum(1 for r in T[a] if r["turn"] > 0 and r.get("reply_reprefilled") is None)
    rr_null_first = sum(1 for r in T[a] if r["turn"] == 0 and r.get("reply_reprefilled") is None)
    cd_null_ret = sum(1 for r in T[a] if r["turn"] > 0 and r.get("cached_details") is None)
    errkey = sum(1 for r in T[a] if "error" in r)
    RES["records"][a] = {"ttft_le_0": bad_ttft, "ttft_gt_latency": gt_lat,
                         "completion_ne_output_length": ct, "recomputed_negative": rr_neg,
                         "recomputed_null_returning": rr_null_ret,
                         "recomputed_null_first": rr_null_first,
                         "cached_details_null_returning": cd_null_ret,
                         "records_with_error_key": errkey}
    print("   %-14s ttft<=0 %d, ttft>latency %d, completion!=output_length %d, reply_reprefilled "
          "negative %d, null on returning turns %d (null on first turns %d), cached_details null on "
          "returning turns %d, records with an 'error' key %d"
          % (a, bad_ttft, gt_lat, ct, rr_neg, rr_null_ret, rr_null_first, cd_null_ret, errkey))

print("\n=== c. floor of recomputed tokens: returning turns sent in minutes 0-10")
RES["recompute_floor"] = {}
for a in ARMS:
    v = [max(r.get("reply_reprefilled") or 0, 0) for r in T[a] if r["turn"] > 0 and r["t_send"] < 600]
    RES["recompute_floor"][a] = {"n": len(v), "mean": mean(v), "median": pct(v, 0.5), "max": max(v)}
    print("   %-14s n %4d  mean %.2f  median %d  max %d" % (a, len(v), mean(v), pct(v, 0.5), max(v)))

print("\n=== d. LEAD5 shared turns: spread of prompt_tokens over the five runs")
KM = {}
for a in L5:
    t = sorted((r for r in T[a] if r.get("ttft") is not None), key=lambda r: r["t_send"])
    thr = t[-1]["t_send"] - 300.0
    KM[a] = {(r["conv"], r["turn"]): r for r in t if r["t_send"] < thr}
keys = None
for a in L5:
    ks = set(k for k in KM[a] if k[1] > 0)
    keys = ks if keys is None else keys & ks
sp = [max(KM[a][k]["prompt_tokens"] for a in L5) - min(KM[a][k]["prompt_tokens"] for a in L5)
      for k in keys]
RES["lead5_prompt_spread"] = {"n": len(sp), "eq": sum(1 for x in sp if x == 0),
                              "le1": sum(1 for x in sp if x <= 1), "le2": sum(1 for x in sp if x <= 2),
                              "le8": sum(1 for x in sp if x <= 8), "max": max(sp),
                              "mean_prompt_tokens": mean([KM["jit_exact"][k]["prompt_tokens"] for k in keys])}
print("   keys %d: spread 0: %d, <=1: %d, <=2: %d, <=8: %d, max %d; mean prompt length %.0f tokens"
      % (len(sp), RES["lead5_prompt_spread"]["eq"], RES["lead5_prompt_spread"]["le1"],
         RES["lead5_prompt_spread"]["le2"], RES["lead5_prompt_spread"]["le8"], max(sp),
         RES["lead5_prompt_spread"]["mean_prompt_tokens"]))

print("\n=== e. sessions are admitted in conv order in every run")
RES["admission"] = {}
for a in ARMS:
    cs = [r for r in P[a]["records"] if r.get("kind") == "conv_start"]
    ids = [r["conv"] for r in cs]
    asc = all(b == a_ + 1 for a_, b in zip(ids, ids[1:]))
    RES["admission"][a] = {"sessions": len(ids), "first": ids[0], "last": ids[-1], "consecutive": asc}
    print("   %-14s sessions %3d, conv ids %d..%d, consecutive ascending: %s"
          % (a, len(ids), ids[0], ids[-1], asc))

with open(os.path.join(HERE, "r2_misc.json"), "w") as f:
    json.dump(RES, f, indent=1, sort_keys=True)
print("\nwrote r2_misc.json")
