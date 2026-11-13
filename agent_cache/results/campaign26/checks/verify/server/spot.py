#!/usr/bin/env python3
"""Spot values quoted in the final text.  Run as:  nice -n 19 python3 -I -B spot.py"""
import json
import os

OUT = "/home/wanhr/sglang/agent_cache/results/campaign26/checks/verify/server"
with open(os.path.join(OUT, "compare2.json")) as f:
    rows = json.load(f)
for r in rows:
    if r["file"] == "stall_ec_ctl.out" and r["arm"] == "hintfree_ctl" and r["what"].startswith("stall per reading turn not landed"):
        print("hintfree_ctl not-landed stall per reading turn, float arithmetic:", r["mine_raw"], "file prints", r["theirs"])
with open(os.path.join(OUT, "exact_cmp.json")) as f:
    for r in json.load(f):
        if r["arm"] == "hintfree_ctl" and r["what"].startswith("stall per reading turn"):
            print("  exact:", r)


def load(name):
    with open(os.path.join(OUT, name)) as f:
        return {(v["item"], v["arm"], v["arm_set"], v["metric"]): v["value"] for v in json.load(f)}


FX = load("ext/ext_values.json")
FZ = load("frozen_results_items1to5.json")
for m in ("mean_ttft", "mean_queue_wait"):
    a, b = FX[("2", "ec_r13", "ALL13", m)], FX[("2", "ec_r13_rep", "ALL13", m)]
    print("ALL13 %s ec_r13 %.6f ec_r13_rep %.6f  %+.1f %%" % (m, a, b, 100 * (b / a - 1)))
for m in ("mean_read_stall_per_shared_turn", "stall_read_was_last_wait_per_shared_turn", "stall_gpu_queue_held_as_well_per_shared_turn"):
    a, b = FX[("3", "ec_r13", "ALL13", m)], FX[("3", "ec_r13_rep", "ALL13", m)]
    print("ALL13 %s ec_r13 %.6f ec_r13_rep %.6f  %+.1f %%" % (m, a, b, 100 * (b / a - 1)))
for m in ("mean_ttft", "mean_queue_wait", "mean_ttft_minus_queue_wait"):
    a, b = FX[("2", "ec_r13", "LEAD6", m)], FX[("2", "ec_r13_rep", "LEAD6", m)]
    print("LEAD6 %s ec_r13 %.6f ec_r13_rep %.6f  %+.4f" % (m, a, b, b - a))
arms = ["jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend"]
t = {a: FZ[("2", a, "LEAD5", "mean_ttft")] for a in arms}
print("LEAD5 mean TTFT:", {a: round(v, 6) for a, v in t.items()})
four = arms[1:]
gaps = sorted(abs(t[x] - t[y]) for i, x in enumerate(four) for y in four[i + 1:])
print("gaps among the four:", [round(g, 4) for g in gaps])
print("jit_exact gaps:", [round(t[a] - t["jit_exact"], 4) for a in four])
for a in arms:
    print(a, "LEAD5 evict M", FZ[("1", a, "", "evict_device_tokens_kept_window")] / 1e6, FZ[("1", a, "", "evict_host_tokens_kept_window")] / 1e6)
print("ec_r13_rep evict", FX[("1", "ec_r13_rep", "", "evict_device_tokens_kept_window")], FX[("1", "ec_r13_rep", "", "evict_host_tokens_kept_window")],
      "window", FX[("0", "ec_r13_rep", "", "kept_window_length")], "kept returning", FX[("0", "ec_r13_rep", "", "kept_returning_turns")])
print("windows:", {a: round(FZ[("0", a, "", "kept_window_length")], 1) for a in arms + ["hintfree", "jit_p50_d_cut"]})
