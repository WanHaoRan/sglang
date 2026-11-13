#!/usr/bin/env python3
"""Extra entries for the final value list: item 2 on ALL11's shared turns (the set of
compare_ec.out's 'all arms' table), the two new arms on the same 6715 turns, and a few
figures quoted in the concerns.

Run as:  nice -n 19 python3 -I -B final2_add.py
"""
import json
import os

OUT = "/home/wanhr/sglang/agent_cache/results/campaign26/checks/verify/server"
ALL11 = ["baseline", "base_rep", "hintfree", "jit_p50", "jit_exact", "jit_p50_ctl",
         "hintfree_ctl", "ec_r13", "nofc", "ec_r1", "nofc_noend"]
LEAD5 = ["jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend"]


def load(name):
    with open(os.path.join(OUT, name)) as f:
        return {(v["item"], v["arm"], v["arm_set"], v["metric"]): v["value"] for v in json.load(f)}


FZ = load("frozen_results_items1to5.json")
FX = load("ext/ext_values.json")
out = []


def add(item, arm, arm_set, metric, unit, value):
    out.append({"item": item, "arm": arm, "arm_set": arm_set, "metric": metric, "unit": unit,
                "value": round(value, 6) if isinstance(value, float) else value})


for a in ALL11:
    add("2", a, "ALL11", "mean_queue_wait", "s", FZ[("2", a, "ALL11", "mean_queue_wait")])
    add("2", a, "ALL11", "mean_ttft_minus_queue_wait", "s", FZ[("2", a, "ALL11", "mean_ttft_minus_queue_wait")])
for a in ("jit_p50_d", "ec_r13_rep"):
    add("2", a, "ALL13", "mean_queue_wait", "s", FX[("2", a, "ALL13", "mean_queue_wait")])
    add("2", a, "ALL13", "mean_ttft_minus_queue_wait", "s", FX[("2", a, "ALL13", "mean_ttft_minus_queue_wait")])
for a in LEAD5:
    add("5a", a, "own", "share_of_s2h_io_lines_starting_over_0.05s_before_arrival", "fraction",
        FZ[("5a", a, "", "read_start_minus_arrival_share_below_-0.05s")])
    add("5b", a, "own", "server_span_minus_client_latency_max", "s", FZ[("5b", a, "", "server_minus_client_latency_max")])
    add("5x", a, "own", "entry_time_minus_client_send_time_min", "s", FZ[("5x", a, "", "entry_time_minus_client_send_time_min")])
    add("5x", a, "own", "entry_time_minus_client_send_time_median", "s", FZ[("5x", a, "", "entry_time_minus_client_send_time_median")])
    add("5x", a, "own", "entry_time_minus_client_send_time_p99", "s", FZ[("5x", a, "", "entry_time_minus_client_send_time_p99")])
with open(os.path.join(OUT, "final2_add.txt"), "w") as f:
    for v in out:
        f.write(json.dumps(v, separators=(",", ":")) + "\n")
for v in out:
    print(json.dumps(v, separators=(",", ":")))
