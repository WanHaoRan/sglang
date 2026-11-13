#!/usr/bin/env python3
"""Look at the record shapes in the cached client logs (no results computed here)."""
import json
import os
import pickle

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "parsed.pkl"), "rb") as f:
    P = pickle.load(f)

print("=== run records and args differences")
allargs = {}
for label, d in P.items():
    assert len(d["run"]) == 1, (label, len(d["run"]))
    r = d["run"][0]
    allargs[label] = r["args"]
    print(label, "t_start=%.3f" % r["t_start"], "users=%s turns=%s model=%s tag=%s line=%d" % (
        r.get("users"), r.get("turns"), r.get("model"), r.get("tag"), r["_line"]),
        "other_keys=%s" % sorted(set(r) - {"kind", "tag", "t_start", "model", "args", "users", "turns", "_line"}))
keys = sorted({k for a in allargs.values() for k in a})
for k in keys:
    vals = {label: a.get(k, "<absent>") for label, a in allargs.items()}
    distinct = {json.dumps(v, sort_keys=True) for v in vals.values()}
    if k in ("output", "tag"):
        continue
    if len(distinct) == 1:
        print("  same  %-18s %s" % (k, next(iter(distinct))))
    else:
        print("  DIFF  %-18s %s" % (k, json.dumps(vals, sort_keys=True)))

print()
print("=== turn record keys per run (key: count)")
for label, d in P.items():
    kc = {}
    for t in d["turns"]:
        for k in t:
            kc[k] = kc.get(k, 0) + 1
    print(label, len(d["turns"]), json.dumps(kc, sort_keys=True))

print()
print("=== samples of the other kinds")
for label in ("baseline", "hintfree", "jit_p50_d_cut", "ec_r13"):
    d = P[label]
    print("--", label)
    for k, recs in d["other"].items():
        s = json.dumps(recs[0], sort_keys=True)
        print("   ", k, len(recs), s[:700])
        if k in ("summary",):
            print("    FULL summary:", json.dumps(recs[0], sort_keys=True)[:6000])
        if len(recs) > 1:
            print("    last:", json.dumps(recs[-1], sort_keys=True)[:700])
    print("    conv_start[0]:", json.dumps(d["conv_start"][0], sort_keys=True)[:500])
    print("    conv_start[-1]:", json.dumps(d["conv_start"][-1], sort_keys=True)[:500])

print()
print("=== value types of the main turn fields")
for label, d in P.items():
    tc = {}
    for t in d["turns"]:
        for k in ("conv", "turn", "t_send", "ttft", "latency", "prompt_tokens", "completion_tokens",
                  "cached_tokens", "reply_reprefilled", "cached_details", "gap_slept", "error"):
            v = t.get(k, "<absent>")
            ty = "<absent>" if isinstance(v, str) and v == "<absent>" else type(v).__name__
            tc[(k, ty)] = tc.get((k, ty), 0) + 1
    print(label, " ".join("%s:%s=%d" % (k, ty, n) for (k, ty), n in sorted(tc.items())))
