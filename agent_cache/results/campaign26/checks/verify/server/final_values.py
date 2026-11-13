#!/usr/bin/env python3
"""Select the item 1-5 numbers from the frozen results for the final answer.

Reads frozen_results_items1to5.json only; writes final_values.json and prints
it one record per line.  Run as:  nice -n 19 python3 -I -B final_values.py
"""
import json
import os

OUT = "/home/wanhr/sglang/agent_cache/results/campaign26/checks/verify/server"
LEAD5 = ["jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend"]
SEVEN = LEAD5 + ["jit_p50_ctl", "hintfree_ctl"]
RUNS13 = ["baseline", "base_rep", "hintfree", "hintfree_ctl", "jit_p50", "jit_p50_ctl", "jit_p50_d_cut",
          "jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend", "order_run"]

FZ = {}
with open(os.path.join(OUT, "frozen_results_items1to5.json")) as f:
    for v in json.load(f):
        FZ[(v["item"], v["arm"], v["arm_set"], v["metric"])] = v["value"]

out = []


def add(item, arm, arm_set, metric, unit, key):
    v = FZ[key]
    if isinstance(v, float):
        v = round(v, 6)
    out.append({"item": item, "arm": arm, "arm_set": arm_set, "metric": metric, "unit": unit, "value": v})


for s in ("LEAD5", "SEVEN"):
    add("0", "all", s, "shared_returning_turns", "turns", ("0", "all", s, "shared_returning_turns"))
for a in SEVEN:
    add("0", a, "own", "kept_returning_turns", "turns", ("0", a, "", "kept_returning_turns"))

for a in LEAD5:
    s = "LEAD5"
    add("1", a, s, "turns_reading_from_ssd", "turns", ("1", a, s, "turns_that_read_from_ssd"))
    add("1", a, s, "ssd_pages_shared_turns", "pages", ("1", a, s, "ssd_pages_shared_turns"))
    add("1", a, "own", "ssd_pages_kept_returning_turns", "pages", ("1", a, "", "ssd_pages_kept_returning_turns"))
    add("1", a, s, "s2h_io_lines_shared_turns", "lines", ("1", a, s, "s2h_io_lines_shared_turns"))
    add("1", a, s, "reads_per_reading_turn", "reads/turn", ("1", a, s, "reads_per_reading_turn"))
    add("1", a, s, "staged_reads_ntp_stage_lines", "lines", ("1", a, s, "staged_reads_ntp_stage_lines"))
    add("1", a, s, "turns_with_staged_read", "turns", ("1", a, s, "turns_with_staged_read"))
    add("1", a, s, "refused_launches", "lines", ("1", a, s, "refused_launches"))
    add("1", a, s, "turns_with_refused_launch", "turns", ("1", a, s, "turns_with_refused_launch"))
    add("1", a, "own", "evict_device_tokens_kept_window", "tokens", ("1", a, "", "evict_device_tokens_kept_window"))
    add("1", a, "own", "evict_host_tokens_kept_window", "tokens", ("1", a, "", "evict_host_tokens_kept_window"))

for a in RUNS13:
    add("2", a, "own", "keys_with_more_than_one_ReqTimeStats_line", "keys", ("2", a, "", "keys_with_more_than_one_ReqTimeStats_line"))
for s, arms in (("LEAD5", LEAD5), ("SEVEN", SEVEN)):
    for a in arms:
        add("2", a, s, "mean_queue_wait", "s", ("2", a, s, "mean_queue_wait"))
        add("2", a, s, "mean_ttft_minus_queue_wait", "s", ("2", a, s, "mean_ttft_minus_queue_wait"))
for s, arms in (("LEAD5", LEAD5), ("SEVEN", SEVEN)):
    for a in arms:
        add("3", a, s, "mean_read_stall", "s/shared turn", ("3", a, s, "mean_read_stall_per_shared_turn"))
        add("3", a, s, "reading_turns_landed_by_arrival", "turns", ("3", a, s, "reading_turns_landed_by_arrival"))
        add("3", a, s, "stall_read_was_last_wait", "s/shared turn", ("3", a, s, "stall_read_was_last_wait_per_shared_turn"))
        add("3", a, s, "turns_read_was_last_wait", "turns", ("3", a, s, "turns_read_was_last_wait"))
        add("3", a, s, "stall_gpu_queue_held_too", "s/shared turn", ("3", a, s, "stall_gpu_queue_held_as_well_per_shared_turn"))
        add("3", a, s, "turns_gpu_queue_held_too", "turns", ("3", a, s, "turns_gpu_queue_held_as_well"))
for a in LEAD5:
    for tag in ("bound=0.05", "bound=0.2", "bound=0.5", "slack=0", "slack=0.5"):
        add("4", a, "LEAD5", "stall_read_was_last_wait[%s]" % tag, "s/shared turn", ("4", a, "LEAD5", "stall_read_was_last_wait_per_shared_turn[%s]" % tag))
        add("4", a, "LEAD5", "stall_gpu_queue_held_too[%s]" % tag, "s/shared turn", ("4", a, "LEAD5", "stall_gpu_queue_held_as_well_per_shared_turn[%s]" % tag))
for a in ("nofc", "nofc_noend"):
    add("5a", a, "own", "s2h_io_lines", "lines", ("5a", a, "", "s2h_io_lines_with_arrival"))
    add("5a", a, "own", "read_start_minus_arrival_min", "s", ("5a", a, "", "read_start_minus_arrival_min"))
    add("5a", a, "own", "read_start_minus_arrival_p01", "s", ("5a", a, "", "read_start_minus_arrival_p01"))
    add("5a", a, "own", "read_start_minus_arrival_median", "s", ("5a", a, "", "read_start_minus_arrival_median"))
    add("5a", a, "own", "share_below_minus_0.05s", "fraction", ("5a", a, "", "read_start_minus_arrival_share_below_-0.05s"))
for a in LEAD5:
    add("5b", a, "own", "server_span_minus_client_latency_n", "turns", ("5b", a, "", "server_minus_client_latency_n"))
    add("5b", a, "own", "server_span_minus_client_latency_median", "s", ("5b", a, "", "server_minus_client_latency_median"))
    add("5b", a, "own", "server_span_minus_client_latency_p01", "s", ("5b", a, "", "server_minus_client_latency_p01"))
    add("5b", a, "own", "server_span_minus_client_latency_p99", "s", ("5b", a, "", "server_minus_client_latency_p99"))
for a in LEAD5:
    add("5c", a, "LEAD5", "share_queue_wait_over_ttft_plus_0.05s", "fraction", ("5c", a, "LEAD5", "share_shared_turns_queue_wait_exceeds_ttft_by_0.05s"))

with open(os.path.join(OUT, "final_values.json"), "w") as f:
    json.dump(out, f)
print(len(out))
for v in out:
    print(json.dumps(v, separators=(",", ":")))
