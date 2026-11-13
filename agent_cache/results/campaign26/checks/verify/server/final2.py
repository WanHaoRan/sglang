#!/usr/bin/env python3
"""Assemble the value list of the final answer.

Sources:
  frozen_results_items1to5.json   items 1-5 as frozen before any existing output was opened (float arithmetic)
  recheck.json                    second implementation, exact microsecond arithmetic; used for the entries
                                  whose metric ends in "[exact]" (only where it differs from the frozen value)
  ext/ext_values.json             the six-arm set of the regenerated stall_ec.out (computed after opening)

Run as:  nice -n 19 python3 -I -B final2.py
"""
import json
import os

OUT = "/home/wanhr/sglang/agent_cache/results/campaign26/checks/verify/server"
LEAD5 = ["jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend"]
SEVEN = LEAD5 + ["jit_p50_ctl", "hintfree_ctl"]
LEAD6 = ["jit_exact", "ec_r13", "ec_r13_rep", "ec_r1", "nofc", "nofc_noend"]
RUNS13 = ["baseline", "base_rep", "hintfree", "hintfree_ctl", "jit_p50", "jit_p50_ctl", "jit_p50_d_cut",
          "jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend", "order_run"]
SIX = "SIX"


def load(name):
    with open(os.path.join(OUT, name)) as f:
        return {(v["item"], v["arm"], v["arm_set"], v["metric"]): v["value"] for v in json.load(f)}


FZ = load("frozen_results_items1to5.json")
RC = load("recheck.json")
FX = load("ext/ext_values.json")
out = []


def r6(v):
    return round(v, 6) if isinstance(v, float) else v


def add(item, arm, arm_set, metric, unit, value):
    out.append({"item": item, "arm": arm, "arm_set": arm_set, "metric": metric, "unit": unit, "value": r6(value)})


for s in ("LEAD5", "SEVEN", "ALL11", "CUT7"):
    add("0", "all", s, "shared_returning_turns", "turns", FZ[("0", "all", s, "shared_returning_turns")])
add("0", "all", SIX, "shared_returning_turns", "turns", FX[("0", "all", "LEAD6", "shared_returning_turns")])
for a in SEVEN:
    add("0", a, "own", "kept_returning_turns", "turns", FZ[("0", a, "", "kept_returning_turns")])
    add("0", a, "own", "kept_window_length", "s", FZ[("0", a, "", "kept_window_length")])

# item 1
for a in LEAD5:
    s = "LEAD5"
    add("1", a, s, "turns_reading_from_ssd", "turns", FZ[("1", a, s, "turns_that_read_from_ssd")])
    add("1", a, s, "ssd_pages_shared_turns", "pages", FZ[("1", a, s, "ssd_pages_shared_turns")])
    add("1", a, "own", "ssd_pages_kept_returning_turns", "pages", FZ[("1", a, "", "ssd_pages_kept_returning_turns")])
    add("1", a, s, "s2h_io_lines_shared_turns", "lines", FZ[("1", a, s, "s2h_io_lines_shared_turns")])
    add("1", a, s, "reads_per_reading_turn", "reads/turn", FZ[("1", a, s, "reads_per_reading_turn")])
    add("1", a, s, "staged_reads_ntp_stage_lines", "lines", FZ[("1", a, s, "staged_reads_ntp_stage_lines")])
    add("1", a, s, "turns_with_staged_read", "turns", FZ[("1", a, s, "turns_with_staged_read")])
    add("1", a, s, "refused_launches", "lines", FZ[("1", a, s, "refused_launches")])
    add("1", a, s, "turns_with_refused_launch", "turns", FZ[("1", a, s, "turns_with_refused_launch")])
    add("1", a, "own", "evict_device_tokens_kept_window", "tokens", FZ[("1", a, "", "evict_device_tokens_kept_window")])
    add("1", a, "own", "evict_host_tokens_kept_window", "tokens", FZ[("1", a, "", "evict_host_tokens_kept_window")])

# item 2
for a in RUNS13:
    add("2", a, "own", "keys_with_more_than_one_ReqTimeStats_line", "keys", FZ[("2", a, "", "keys_with_more_than_one_ReqTimeStats_line")])
for a in ("jit_p50_d", "ec_r13_rep"):
    add("2", a, "own", "keys_with_more_than_one_ReqTimeStats_line", "keys", FX[("2", a, "", "keys_with_more_than_one_ReqTimeStats_line")])
for s, arms in (("LEAD5", LEAD5), ("SEVEN", SEVEN)):
    for a in arms:
        add("2", a, s, "mean_queue_wait", "s", FZ[("2", a, s, "mean_queue_wait")])
        add("2", a, s, "mean_ttft_minus_queue_wait", "s", FZ[("2", a, s, "mean_ttft_minus_queue_wait")])

# item 3
M3 = (("mean_read_stall", "s per shared turn", "mean_read_stall_per_shared_turn"),
      ("reading_turns_landed_by_arrival", "turns", "reading_turns_landed_by_arrival"),
      ("stall_read_was_last_wait", "s per shared turn", "stall_read_was_last_wait_per_shared_turn"),
      ("turns_read_was_last_wait", "turns", "turns_read_was_last_wait"),
      ("stall_gpu_queue_held_too", "s per shared turn", "stall_gpu_queue_held_as_well_per_shared_turn"),
      ("turns_gpu_queue_held_too", "turns", "turns_gpu_queue_held_as_well"))
n_exact = 0
for s, arms in (("LEAD5", LEAD5), ("SEVEN", SEVEN)):
    for a in arms:
        for name, unit, key in M3:
            fv = FZ[("3", a, s, key)]
            add("3", a, s, name, unit, fv)
            ev = RC[("3", a, s, key)]
            differs = (fv != ev) if isinstance(fv, int) else abs(fv - ev) > 1e-6
            if differs:
                add("3", a, s, name + "[exact]", unit, ev)
                n_exact += 1

# item 4
for a in LEAD5:
    for tag in ("bound=0.05", "bound=0.2", "bound=0.5", "slack=0", "slack=0.5"):
        add("4", a, "LEAD5", "stall_read_was_last_wait[%s]" % tag, "s per shared turn",
            FZ[("4", a, "LEAD5", "stall_read_was_last_wait_per_shared_turn[%s]" % tag)])
        add("4", a, "LEAD5", "stall_gpu_queue_held_too[%s]" % tag, "s per shared turn",
            FZ[("4", a, "LEAD5", "stall_gpu_queue_held_as_well_per_shared_turn[%s]" % tag)])
        if tag.startswith("bound"):
            add("4", a, "LEAD5", "turns_read_was_last_wait[%s]" % tag, "turns", FZ[("4", a, "LEAD5", "turns_read_was_last_wait[%s]" % tag)])
            add("4", a, "LEAD5", "turns_gpu_queue_held_too[%s]" % tag, "turns", FZ[("4", a, "LEAD5", "turns_gpu_queue_held_as_well[%s]" % tag)])

# item 5
for a in ("nofc", "nofc_noend"):
    add("5a", a, "own", "s2h_io_lines", "lines", FZ[("5a", a, "", "s2h_io_lines_with_arrival")])
    add("5a", a, "own", "read_start_minus_arrival_min", "s", FZ[("5a", a, "", "read_start_minus_arrival_min")])
    add("5a", a, "own", "read_start_minus_arrival_p01", "s", FZ[("5a", a, "", "read_start_minus_arrival_p01")])
    add("5a", a, "own", "read_start_minus_arrival_median", "s", FZ[("5a", a, "", "read_start_minus_arrival_median")])
    add("5a", a, "own", "share_below_minus_0.05s", "fraction", FZ[("5a", a, "", "read_start_minus_arrival_share_below_-0.05s")])
for a in LEAD5:
    add("5b", a, "own", "server_span_minus_client_latency_n", "turns", FZ[("5b", a, "", "server_minus_client_latency_n")])
    add("5b", a, "own", "server_span_minus_client_latency_median", "s", FZ[("5b", a, "", "server_minus_client_latency_median")])
    add("5b", a, "own", "server_span_minus_client_latency_p01", "s", FZ[("5b", a, "", "server_minus_client_latency_p01")])
    add("5b", a, "own", "server_span_minus_client_latency_p99", "s", FZ[("5b", a, "", "server_minus_client_latency_p99")])
for a in LEAD5:
    add("5c", a, "LEAD5", "share_queue_wait_over_ttft_plus_0.05s", "fraction", FZ[("5c", a, "LEAD5", "share_shared_turns_queue_wait_exceeds_ttft_by_0.05s")])

# six-arm set of the regenerated stall_ec.out
for a in LEAD6:
    s = "LEAD6"
    add("1", a, SIX, "turns_reading_from_ssd", "turns", FX[("1", a, s, "turns_that_read_from_ssd")])
    add("1", a, SIX, "ssd_pages_shared_turns", "pages", FX[("1", a, s, "ssd_pages_shared_turns")])
    add("2", a, SIX, "mean_ttft", "s", FX[("2", a, s, "mean_ttft")])
    add("2", a, SIX, "mean_queue_wait", "s", FX[("2", a, s, "mean_queue_wait")])
    add("2", a, SIX, "mean_ttft_minus_queue_wait", "s", FX[("2", a, s, "mean_ttft_minus_queue_wait")])
    for name, unit, key in M3:
        add("3", a, SIX, name, unit, FX[("3", a, s, key)])

with open(os.path.join(OUT, "final2_values.json"), "w") as f:
    json.dump(out, f)
with open(os.path.join(OUT, "final2_values.txt"), "w") as f:
    for v in out:
        f.write(json.dumps(v, separators=(",", ":")) + "\n")
print(len(out), "values;", n_exact, "exact-arithmetic variants")
