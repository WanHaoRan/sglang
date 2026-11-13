#!/usr/bin/env python3
"""Assemble the numbers of the final answer from the saved JSON files, one compact JSON object per line."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    with open(os.path.join(HERE, name)) as f:
        return json.load(f)


M = load("items_1_4.json")
A = load("item5a_spread.json")
B = load("item5b_shared_bias.json")
E = load("item5e_more.json")
I = load("item5i_pre_entry.json")

OUT = []


def put(item, arm_set, arm, metric, value, unit):
    OUT.append({"item": item, "arm_set": arm_set, "arm": arm, "metric": metric, "value": value, "unit": unit})


def r5(x):
    return round(x, 5)


SHORT = {"mean_ttft": ("mean_ttft", "s", 5), "median_ttft": ("median_ttft", "s", 5), "p90_ttft": ("p90_ttft", "s", 5),
         "p99_ttft": ("p99_ttft", "s", 5), "turns_over_10s": ("turns_over_10s", "turns", 0),
         "mean_ttft_load_episodes": ("mean_ttft_load_episodes", "s", 5), "mean_ttft_elsewhere": ("mean_ttft_elsewhere", "s", 5),
         "mean_recomputed_tokens": ("mean_recomputed_tokens", "tokens", 2),
         "shared_turns": ("shared_turns", "turns", 0), "shared_turns_in_load_episodes": ("turns_in_load_episodes", "turns", 0),
         "shared_turns_elsewhere": ("turns_elsewhere", "turns", 0),
         "paired_mean_ttft_diff": ("paired_diff_left_minus_right", "s", 5),
         "paired_mean_ttft_diff_load_episodes": ("paired_diff_left_minus_right_load_episodes", "s", 5),
         "paired_mean_ttft_diff_elsewhere": ("paired_diff_left_minus_right_elsewhere", "s", 5),
         "share_of_gain_vs_baseline": ("share_of_gain_vs_baseline", "fraction", 5),
         "share_of_gain_vs_base_rep": ("share_of_gain_vs_base_rep", "fraction", 5)}
for e in M["values"]:
    if e["item"] not in ("1", "2", "3"):
        continue
    name, unit, nd = SHORT[e["metric"]]
    v = int(e["value"]) if nd == 0 else round(e["value"], nd)
    put(e["item"], e["arm_set"], e["arm"], name, v, unit)

# item 4
runs = M["sanity"]["per_run"]
order = ["baseline", "base_rep", "hintfree", "hintfree_ctl", "jit_p50", "jit_p50_ctl", "jit_p50_d_cut", "jit_exact", "ec_r13",
         "ec_r1", "nofc", "nofc_noend", "ORDER"]
tot = 0
for a in order:
    s = runs[a]
    tot += s["turn_records"]
    put("4", "", a, "turn_records", s["turn_records"], "turns")
    put("4", "", a, "kept_turns", s["kept_turns"], "turns")
    put("4", "", a, "kept_returning_turns", s["kept_returning"], "turns")
    put("4", "", a, "last_kept_send_minute", round(s["last_kept_minute"], 2), "min")
    put("4", "", a, "kept_keys_missing_from_order_run", s["kept_keys_missing_from_order_run"], "turns")
    put("4", "", a, "kept_returning_keys_missing_from_order_run", s["kept_returning_keys_missing_from_order_run"], "turns")
put("4", "", "all 13 runs", "turn_records_total", tot, "turns")
put("4", "", "all 13 runs", "duplicate_conv_turn_keys", sum(runs[a]["duplicate_keys"] for a in order), "keys")
put("4", "", "all 13 runs", "turn_records_with_error", sum(runs[a]["error_truthy"] + runs[a]["error_key_present"] for a in order), "turns")
put("4", "", "all 13 runs", "turn_records_with_null_ttft", sum(runs[a]["ttft_null"] for a in order), "turns")
w = M["sanity"]["workload_identity"]
x = w["lead5_shared_returning_kept.prompt_tokens"]
put("4", "LEAD5", "(set)", "prompt_tokens_keys_compared", x["keys"], "keys")
put("4", "LEAD5", "(set)", "prompt_tokens_identical_in_all_five", x["identical"], "keys")
put("4", "LEAD5", "(set)", "prompt_tokens_share_identical", r5(x["share_identical"]), "fraction")
put("4", "LEAD5", "(set)", "prompt_tokens_spread_le_1_token", x["spread_le_1"], "keys")
put("4", "LEAD5", "(set)", "prompt_tokens_spread_le_8_tokens", x["spread_le_8"], "keys")
put("4", "LEAD5", "(set)", "prompt_tokens_max_spread", x["max_spread"], "tokens")
put("4", "LEAD5", "(set)", "prompt_tokens_p99_spread", x["p99_spread_all"], "tokens")
x = w["lead5_all_common_keys.replay_prompt_tokens"]
put("4", "LEAD5", "(set)", "trace_prompt_length_keys_compared_first_turns_included", x["keys"], "keys")
put("4", "LEAD5", "(set)", "trace_prompt_length_identical", x["identical"], "keys")
x = w["first_turn_prompt_tokens"]
put("4", "", "all 13 runs", "first_turn_convs_in_more_than_one_run", x["convs_multi"], "convs")
put("4", "", "all 13 runs", "first_turn_prompt_tokens_identical_everywhere", x["identical_multi"], "convs")
put("4", "", "all 13 runs", "first_turn_convs_in_all_13_runs", x["convs_in_all_13"], "convs")
put("4", "", "all 13 runs", "first_turn_prompt_tokens_identical_in_all_13", x["identical_in_all_13"], "convs")
for s in ("ALL11", "LEAD5", "SEVEN", "CUT7"):
    put("4", s, "(set)", "shared_keys_missing_from_order_run", M["sets"][s]["missing_from_order"], "turns")

# item 5 highlights
for name, d in A["repeat_pairs"].items():
    pair, scope = name.split("@")
    sset = "ALL11" if scope == "ALL11" else "pair's own shared turns"
    if pair == "base_rep-baseline" and scope == "pairwise":
        continue
    put("5", sset, pair, "repeat_shared_turns", d["n"], "turns")
    put("5", sset, pair, "repeat_diff_left_minus_right", r5(d["diff"]), "s")
    put("5", sset, pair, "repeat_diff_relative_to_right", r5(d["rel_diff"]), "fraction")
    put("5", sset, pair, "repeat_diff_load_episodes", r5(d["diff_episodes"]), "s")
    put("5", sset, pair, "repeat_diff_elsewhere", r5(d["diff_elsewhere"]), "s")
for a in ("jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend"):
    c = B["coverage"]["ALL11:" + a]
    put("5", "ALL11", a, "share_of_arm_kept_returning_turns_inside_set", r5(c["covered_share"]), "fraction")
    put("5", "ALL11", a, "mean_ttft_of_arm_turns_outside_set", r5(c["mean_unshared"]), "s")
    put("5", "ALL11", a, "turns_over_10s_outside_set", c["over10_unshared"], "turns")
for s in ("ALL11", "LEAD5", "SEVEN", "CUT7"):
    c = E["episode_coverage"][s]
    put("5", s, "(set)", "episode1_turns", c["episode1"], "turns")
    put("5", s, "(set)", "episode2_turns", c["episode2"], "turns")
    put("5", s, "(set)", "largest_order_position", c["max_position"], "position")
la = B["lead5_absent"]
put("5", "LEAD5", "(set)", "elsewhere_turns_with_no_order_position", la["absent"], "turns")
for a in ("jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend"):
    put("5", "LEAD5", a, "mean_ttft_turns_with_no_order_position", r5(la[a]["mean_absent"]), "s")
    put("5", "LEAD5", a, "mean_ttft_elsewhere_turns_with_a_position", r5(la[a]["mean_elsewhere_with_position"]), "s")
    put("5", "LEAD5", a, "mean_ttft_episode1", r5(la[a]["mean_episode1"]), "s")
    put("5", "LEAD5", a, "mean_ttft_episode2", r5(la[a]["mean_episode2"]), "s")
    c = A["concentration"]["LEAD5:" + a]
    put("5", "LEAD5", a, "share_of_summed_ttft_in_turns_over_10s", r5(c["over10_share_of_sum"]), "fraction")
    put("5", "LEAD5", a, "bursts_of_turns_over_10s", c["bursts_60s"], "bursts")
    put("5", "LEAD5", a, "turns_over_10s_in_largest_burst", c["biggest_burst_turns"], "turns")
    put("5", "LEAD5", a, "mean_first_turn_ttft", r5(B["first_turns"]["LEAD5:" + a]["mean"]), "s")
    put("5", "LEAD5", a, "mean_send_to_queue_entry", r5(I["LEAD5:" + a]["mean_pre_entry"]), "s")
for a in ("baseline", "base_rep", "hintfree", "hintfree_ctl", "jit_p50", "jit_p50_ctl", "jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend"):
    put("5", "", a, "turn_records_per_minute", round(B["closed_loop"][a]["turns_per_min"], 2), "turns/min")

with open(os.path.join(HERE, "final_values.jsonl"), "w") as f:
    for e in OUT:
        f.write(json.dumps(e, separators=(",", ":")) + "\n")
print(len(OUT), "entries written to final_values.jsonl")
by = {}
for e in OUT:
    by[e["item"]] = by.get(e["item"], 0) + 1
print(by)
# figures needed for the text
for a in ("base_rep", "hintfree"):
    s = runs[a]
    print(a, "max t_send %.3f, cut-off minute %.4f, last kept send %.3f = minute %.4f, gap to the cut-off %.2f s" % (
        s["max_t_send_s"], (s["max_t_send_s"] - 300) / 60, s["last_kept_t_send_s"], s["last_kept_minute"],
        s["max_t_send_s"] - 300 - s["last_kept_t_send_s"]))
