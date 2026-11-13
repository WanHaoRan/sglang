#!/usr/bin/env python3
"""r2_10_final_values.py -- assemble the numbers of the final answer, one compact JSON per line.

Sources: r2_items_1_4.json (frozen), r2_item5.json, r2_misc.json, r2_item6_compare.json.
Run as:  nice -n 19 python3 -I -B r2_10_final_values.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load(n):
    with open(os.path.join(HERE, n)) as f:
        return json.load(f)


M = load("r2_items_1_4.json")
C = load("r2_item5.json")
X = load("r2_misc.json")
OUT = []


def put(item, arm_set, arm, metric, value, unit):
    e = {"item": item}
    if arm_set:
        e["arm_set"] = arm_set
    e["arm"] = arm
    e["metric"] = metric
    e["value"] = value
    e["unit"] = unit
    OUT.append(e)


def r4(x):
    return round(x, 4)


SETORDER = ["ALL11", "LEAD5", "SEVEN", "CUT7"]
for s in SETORDER:
    d = M["sets"][s]
    put("1", s, "(set)", "shared_turns", d["n_shared"], "turns")
    put("1", s, "(set)", "episode_turns", d["n_shared_in_episodes"], "turns")
    put("1", s, "(set)", "elsewhere_turns", d["n_shared_elsewhere"], "turns")
    for a in d["arms"]:
        st = d["per_arm"][a]
        put("1", s, a, "mean_ttft", r4(st["mean"]), "s")
        put("1", s, a, "median_ttft", r4(st["median"]), "s")
        put("1", s, a, "p90_ttft", r4(st["p90"]), "s")
        put("1", s, a, "p99_ttft", r4(st["p99"]), "s")
        put("1", s, a, "turns_over_10s", st["n_over_10s"], "turns")
        put("1", s, a, "mean_ttft_episodes", r4(st["mean_episodes"]), "s")
        put("1", s, a, "mean_ttft_elsewhere", r4(st["mean_elsewhere"]), "s")
        put("1", s, a, "mean_recomputed_tokens", round(st["mean_recomputed"], 2), "tokens")
for s in ("LEAD5", "CUT7"):
    for name, p in M["pairs"][s].items():
        put("2", s, name, "paired_diff", r4(p["mean_diff"]), "s")
        put("2", s, name, "paired_diff_episodes", r4(p["mean_diff_episodes"]), "s")
        put("2", s, name, "paired_diff_elsewhere", r4(p["mean_diff_elsewhere"]), "s")
for a in M["sets"]["ALL11"]["arms"]:
    r = M["ratios"]["ALL11"][a]
    put("3", "ALL11", a, "gain_share_vs_baseline", r4(r["vs_baseline"]), "ratio")
    put("3", "ALL11", a, "gain_share_vs_base_rep", r4(r["vs_base_rep"]), "ratio")
n123 = len(OUT)

RUNS = ["baseline", "base_rep", "hintfree", "hintfree_ctl", "jit_p50", "jit_p50_ctl",
        "jit_p50_d_cut", "jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend", "ORDER"]
tot = dup = err = 0
for a in RUNS:
    i = M["runs"][a]
    tot += i["n_turn_records"]
    dup += i["duplicate_keys_all_turn_records"]
    err += i["n_error_or_null_ttft"]
    put("4", "", a, "turn_records", i["n_turn_records"], "turns")
    put("4", "", a, "kept_turns", i["n_kept"], "turns")
    put("4", "", a, "kept_returning_turns", i["n_kept_returning"], "turns")
    put("4", "", a, "last_kept_send_minute", round(i["last_kept_send_minute"], 2), "min")
    if i["n_kept_returning_keys_missing_from_order_run"]:
        put("4", "", a, "kept_returning_keys_missing_from_order_run",
            i["n_kept_returning_keys_missing_from_order_run"], "turns")
put("4", "", "13 runs", "turn_records_total", tot, "turns")
put("4", "", "13 runs", "duplicate_conv_turn_keys", dup, "keys")
put("4", "", "13 runs", "turn_records_with_error_or_null_ttft", err, "turns")
w = M["workload"]
x = w["LEAD5_shared_prompt_tokens"]
ps = X["lead5_prompt_spread"]
put("4", "LEAD5", "(set)", "prompt_tokens_keys_compared", x["n_compared"], "keys")
put("4", "LEAD5", "(set)", "prompt_tokens_identical_in_all_five", x["n_identical"], "keys")
put("4", "LEAD5", "(set)", "prompt_tokens_share_identical", r4(x["share_identical"]), "ratio")
put("4", "LEAD5", "(set)", "prompt_tokens_spread_le_1", ps["le1"], "keys")
put("4", "LEAD5", "(set)", "prompt_tokens_spread_le_8", ps["le8"], "keys")
put("4", "LEAD5", "(set)", "prompt_tokens_spread_max", x["spread_max"], "tokens")
f = w["first_turn_prompt_tokens_all_runs"]
put("4", "", "13 runs", "first_turn_convs_in_2plus_runs", f["n_convs_in_at_least_two_runs"], "convs")
put("4", "", "13 runs", "first_turn_prompt_tokens_identical", f["n_identical"], "convs")
put("4", "", "13 runs", "first_turn_convs_in_all_13_runs", f["n_convs_in_all_13_runs"], "convs")
put("4", "", "13 runs", "keys_in_2plus_runs", w["all_runs_same_key_prompt_tokens"]["n_keys_in_2plus_runs"], "keys")
put("4", "", "13 runs", "keys_in_2plus_runs_prompt_tokens_identical",
    w["all_runs_same_key_prompt_tokens"]["n_identical"], "keys")
put("4", "", "13 runs", "keys_in_2plus_runs_trace_prompt_length_identical",
    w["all_runs_same_key_replay_prompt_tokens"]["n_identical"], "keys")
for s in SETORDER:
    put("4", s, "(set)", "shared_keys_missing_from_order_run",
        M["sets"][s]["n_shared_missing_from_order_run"], "turns")
n4 = len(OUT) - n123

# ---- item 5 (key numbers; the rest is in r2_item5.json)
R = C["repeats"]
for key in ("ALL11:base_rep-baseline", "ALL11:hintfree_ctl-hintfree", "ALL11:jit_p50_ctl-jit_p50",
            "ALL11:jit_p50_d-jit_p50_ctl", "ALL11:jit_p50_d-jit_p50", "ALL11:ec_r13_rep-ec_r13"):
    r = R[key]
    pair = key.split(":")[1]
    put("5", "ALL11", pair, "repeat_paired_diff", r4(r["diff"]), "s")
    put("5", "ALL11", pair, "repeat_diff_relative_to_earlier_run", r4(r["rel_diff_to_first"]), "ratio")
r = R["LEAD5+rep:ec_r13_rep-ec_r13"]
S6 = "LEAD5+ec_r13_rep"
put("5", S6, "(set)", "shared_turns", r["n"], "turns")
put("5", S6, "ec_r13_rep-ec_r13", "repeat_paired_diff", r4(r["diff"]), "s")
put("5", S6, "ec_r13_rep-ec_r13", "repeat_diff_relative_to_earlier_run", r4(r["rel_diff_to_first"]), "ratio")
put("5", S6, "ec_r13_rep-ec_r13", "repeat_paired_diff_episodes", r4(r["diff_episodes"]), "s")
put("5", S6, "ec_r13_rep-ec_r13", "repeat_paired_diff_elsewhere", r4(r["diff_elsewhere"]), "s")
put("5", S6, "ec_r13", "turns_over_10s", r["over10_first"], "turns")
put("5", S6, "ec_r13_rep", "turns_over_10s", r["over10_second"], "turns")
put("5", S6, "ec_r13", "p99_ttft", r4(r["p99_first"]), "s")
put("5", S6, "ec_r13_rep", "p99_ttft", r4(r["p99_second"]), "s")
for a in ("jit_exact", "ec_r13", "ec_r13_rep", "ec_r1", "nofc", "nofc_noend"):
    put("5", S6, a, "mean_ttft", r4(C["six_arm_means"][a]["mean"]), "s")
put("5", "LEAD5", "(set)", "pairs_with_abs_diff_not_above_ec_r13_repeat_diff",
    C["lead5_pairs_not_larger_than_ec_r13_repeat"], "pairs of 10")
r = R["own:jit_p50_d-jit_p50_d_cut"]
put("5", "CUT7 keys", "jit_p50_d-jit_p50_d_cut", "repeat_shared_turns", r["n"], "turns")
put("5", "CUT7 keys", "jit_p50_d-jit_p50_d_cut", "repeat_paired_diff", r4(r["diff"]), "s")
put("5", "CUT7 keys", "jit_p50_d-jit_p50_d_cut", "repeat_paired_diff_episodes", r4(r["diff_episodes"]), "s")
put("5", "CUT7 keys", "jit_p50_d-jit_p50_d_cut", "repeat_paired_diff_elsewhere", r4(r["diff_elsewhere"]), "s")
for a in ("jit_p50_d", "ec_r13_rep"):
    g = C["gain_share_13"][a]
    put("5", "ALL11 keys", a, "mean_ttft", r4(g["mean"]), "s")
    put("5", "ALL11 keys", a, "gain_share_vs_baseline", r4(g["vs_baseline"]), "ratio")
    put("5", "ALL11 keys", a, "gain_share_vs_base_rep", r4(g["vs_base_rep"]), "ratio")
for a in ("jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend"):
    c = C["coverage"]["ALL11"]["arms"][a]
    put("5", "ALL11", a, "share_of_arm_kept_returning_turns_in_set", r4(c["share_inside"]), "ratio")
    put("5", "ALL11", a, "mean_ttft_of_arm_turns_outside_set", r4(c["mean_outside"]), "s")
for s in SETORDER:
    put("5", s, "(set)", "episode2_turns", C["coverage"][s]["episode2"], "turns")
put("5", "ALL11", "(set)", "largest_order_position", C["coverage"]["ALL11"]["max_position"], "position")
put("5", "CUT7", "(set)", "episode1_turns", C["coverage"]["CUT7"]["episode1"], "turns")
for key in ("SEVEN:jit_p50_ctl-jit_exact", "ALL11:base_rep-jit_exact"):
    s, pair = key.split(":")
    r = C["send_time_shift"][key]
    put("5", s, pair, "same_key_send_time_shift_mean", round(r["mean"], 1), "s")
    put("5", s, pair, "same_key_send_time_shift_share_over_600s", r4(r["share_over_600s"]), "ratio")
la = C["lead5_absent"]
put("5", "LEAD5", "(set)", "elsewhere_turns_without_order_position", la["absent"], "turns")
for a in ("jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend"):
    v = la["arms"][a]
    put("5", "LEAD5", a, "mean_ttft_turns_without_order_position", r4(v["mean_absent"]), "s")
    put("5", "LEAD5", a, "mean_ttft_elsewhere_turns_with_position", r4(v["mean_elsewhere_with_position"]), "s")
    t = C["tails"]["LEAD5:" + a]
    put("5", "LEAD5", a, "share_of_summed_ttft_in_turns_over_10s", r4(t["share_sum_over10"]), "ratio")
    put("5", "LEAD5", a, "largest_burst_turns_over_10s", t["largest_burst_turns"], "turns")
    put("5", "LEAD5", a, "mean_ttft_of_turns_at_or_under_10s", r4(t["mean_le10"]), "s")
    put("5", "LEAD5", a, "mean_first_turn_ttft", r4(C["first_turns"]["LEAD5:" + a]["mean"]), "s")
put("5", "LEAD5", "ec_r13-jit_exact", "paired_diff_elsewhere_turns_with_position",
    r4(la["pairs"]["ec_r13-jit_exact"]["elsewhere_with_position"]), "s")
put("5", "LEAD5", "ec_r13-jit_exact", "paired_diff_turns_without_order_position",
    r4(la["pairs"]["ec_r13-jit_exact"]["absent"]), "s")
put("5", "", "14 arms", "median_ms_per_output_token_minutes_0_10_min", round(C["pace_spread_0_10"]["min"], 3), "ms")
put("5", "", "14 arms", "median_ms_per_output_token_minutes_0_10_max", round(C["pace_spread_0_10"]["max"], 3), "ms")
put("5", "", "ec_r13", "first_request_of_run_ttft", round(C["pace"]["ec_r13"]["conv0_turn0_ttft"], 3), "s")
put("5", "", "nofc_noend", "first_request_of_run_ttft", round(C["pace"]["nofc_noend"]["conv0_turn0_ttft"], 3), "s")
fl = [X["recompute_floor"][a]["mean"] for a in RUNS]
put("5", "", "13 runs", "recomputed_tokens_per_returning_turn_minutes_0_10_min", round(min(fl), 2), "tokens")
put("5", "", "13 runs", "recomputed_tokens_per_returning_turn_minutes_0_10_max", round(max(fl), 2), "tokens")
n5 = len(OUT) - n123 - n4

# ---- item 6
T = load("r2_item6_compare.json")["totals"]
for cat, lab in (("A", "items 1-3 numbers, five named files"), ("A4", "item 4 numbers, two compare files"),
                 ("B", "items 1-3 numbers, three other files"), ("X", "other recomputed numbers, eight files")):
    put("6", "", lab, "printed_numbers_compared", T[cat][0], "numbers")
    put("6", "", lab, "printed_numbers_equal", T[cat][1], "numbers")
n6 = len(OUT) - n123 - n4 - n5

with open(os.path.join(HERE, "r2_final_values.jsonl"), "w") as fo:
    for e in OUT:
        fo.write(json.dumps(e, separators=(",", ":")) + "\n")
print("entries:", len(OUT), {"1-3": n123, "4": n4, "5": n5, "6": n6})
print("bytes:", os.path.getsize(os.path.join(HERE, "r2_final_values.jsonl")))
