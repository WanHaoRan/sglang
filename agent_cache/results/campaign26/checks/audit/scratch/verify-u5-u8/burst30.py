"""Audit scratch (read-only): recomputed tokens (reply_reprefilled) of returning turns by minute of each run's own clock,
minutes 27-34, and turns recomputing 4K or more, to see whether ec_r13's burst at minute 30 is in the other pricing runs."""
import collections, json, os, sys
RES = "/home/wanhr/sglang/agent_cache/results"
S = "_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70"
RUNS = [("ec_r13", "20261111_070012"), ("ec_r13_rep", "20261111_224624"), ("ec_r1", "20261111_110558"), ("ec_r134", "20261112_153600"),
        ("ec_full", "20261112_214834"), ("nofc_noend", "20261111_130801"), ("nofc_noend_rep", "20261112_174138"), ("jit_exact_d", "20261112_045511")]
print("minute:        " + "".join(f"{m:>12d}" for m in range(27, 35)) + "   (K tokens recomputed / turns of 4K or more)")
for name, stamp in RUNS:
    run = stamp + S
    client = sorted(d for d in os.listdir(f"{RES}/{run}") if d.startswith("client_"))[-1]
    tok, big = collections.Counter(), collections.Counter()
    for line in open(f"{RES}/{run}/{client}/client.jsonl"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("kind") == "turn" and r.get("ttft") is not None and r["turn"] > 0:
            m = int(r["t_send"] // 60)
            rec = max(r.get("reply_reprefilled") or 0, 0)
            tok[m] += rec
            big[m] += rec >= 4000
    print(f"{name:15s}" + "".join(f"{tok[m] / 1000:8.0f}/{big[m]:<3d}" for m in range(27, 35)))
