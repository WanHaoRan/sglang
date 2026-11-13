#!/usr/bin/env python3
"""r2_03_cross_check.py -- compare the fresh recomputation (r2_items_1_4.json) with the
values frozen by the interrupted attempt (items_1_4.json, sha256 in FROZEN_items_1_4.sha256).

Both were produced from the task definitions without opening the existing analysis.
Run as:  nice -n 19 python3 -I -B r2_03_cross_check.py
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main():
    old_p = os.path.join(HERE, "items_1_4.json")
    new_p = os.path.join(HERE, "r2_items_1_4.json")
    frozen = open(os.path.join(HERE, "FROZEN_items_1_4.sha256")).read().split()
    print("frozen sha  :", frozen[0])
    print("items_1_4   :", sha(old_p), "match" if sha(old_p) == frozen[0] else "DIFFERENT")
    old = json.load(open(old_p))
    new = json.load(open(new_p))
    metrics = {}
    for v in old["values"]:
        metrics.setdefault((v["item"], v["metric"]), []).append(v)
    print("\nmetric names in the earlier frozen list:")
    for k in sorted(metrics):
        ex = metrics[k][0]
        print("  item %-3s %-45s n=%3d  e.g. arm=%s set=%s value=%r"
              % (k[0], k[1], len(metrics[k]), ex.get("arm"), ex.get("arm_set"), ex.get("value")))

    # mapping: earlier metric name -> function(new, arm_set, arm) -> my value
    def per_arm(field):
        return lambda n, s, a: n["sets"][s]["per_arm"][a][field]

    def set_level(field):
        return lambda n, s, a: n["sets"][s][field]

    def pair(field):
        return lambda n, s, a: n["pairs"][s][a][field]

    def ratio(field):
        return lambda n, s, a: n["ratios"][s][a][field]

    MAP = {
        ("1", "shared_turns"): set_level("n_shared"),
        ("1", "shared_turns_in_load_episodes"): set_level("n_shared_in_episodes"),
        ("1", "shared_turns_elsewhere"): set_level("n_shared_elsewhere"),
        ("1", "mean_ttft"): per_arm("mean"),
        ("1", "median_ttft"): per_arm("median"),
        ("1", "p90_ttft"): per_arm("p90"),
        ("1", "p99_ttft"): per_arm("p99"),
        ("1", "turns_over_10s"): per_arm("n_over_10s"),
        ("1", "mean_ttft_load_episodes"): per_arm("mean_episodes"),
        ("1", "mean_ttft_elsewhere"): per_arm("mean_elsewhere"),
        ("1", "turns_load_episodes"): per_arm("n_episodes"),
        ("1", "turns_elsewhere"): per_arm("n_elsewhere"),
        ("1", "mean_recomputed_tokens"): per_arm("mean_recomputed"),
        ("2", "paired_mean_ttft_diff"): pair("mean_diff"),
        ("2", "paired_mean_ttft_diff_load_episodes"): pair("mean_diff_episodes"),
        ("2", "paired_mean_ttft_diff_elsewhere"): pair("mean_diff_elsewhere"),
        ("3", "share_of_gain_vs_baseline"): ratio("vs_baseline"),
        ("3", "share_of_gain_vs_base_rep"): ratio("vs_base_rep"),
    }
    n_ok = n_bad = n_unmapped = 0
    unmapped = {}
    worst = 0.0
    for v in old["values"]:
        key = (v["item"], v["metric"])
        f = MAP.get(key)
        if f is None:
            n_unmapped += 1
            unmapped[key] = unmapped.get(key, 0) + 1
            continue
        try:
            mine = f(new, v.get("arm_set"), v.get("arm"))
        except KeyError as e:
            print("  cannot map", v, "KeyError", e)
            n_unmapped += 1
            continue
        theirs = v["value"]
        if isinstance(theirs, (int, float)) and isinstance(mine, (int, float)):
            err = abs(mine - theirs)
            rel = err / max(abs(theirs), 1e-12)
            worst = max(worst, min(err, rel))
            if err <= 1e-9 or rel <= 1e-9:
                n_ok += 1
            else:
                n_bad += 1
                print("  DIFF", v, "mine", mine)
        else:
            if mine == theirs:
                n_ok += 1
            else:
                n_bad += 1
                print("  DIFF", v, "mine", mine)
    print("\nvalues in the earlier frozen list: %d; mapped and equal: %d; different: %d; "
          "not mapped: %d; worst min(abs, rel) error among equal ones: %.3g"
          % (len(old["values"]), n_ok, n_bad, n_unmapped, worst))
    if unmapped:
        print("unmapped metric names:", unmapped)

    # sanity block: compare the per-run counts that both files hold
    S = {
        "turn_records": "n_turn_records", "ttft_non_null": "n_with_ttft",
        "ttft_null": "n_null_ttft", "error_truthy": "n_error_truthy",
        "duplicate_keys": "duplicate_keys_all_turn_records", "kept_turns": "n_kept",
        "kept_returning": "n_kept_returning", "kept_first_turns": "n_kept_first_turns",
        "last_kept_t_send_s": "last_kept_t_send_s", "last_kept_minute": "last_kept_send_minute",
        "max_t_send_s": "max_t_send_with_ttft",
        "kept_keys_missing_from_order_run": "n_kept_keys_missing_from_order_run",
        "kept_returning_keys_missing_from_order_run":
            "n_kept_returning_keys_missing_from_order_run",
        "summary_present": "has_summary",
    }
    s_ok = s_bad = 0
    for label, o in old["sanity"]["per_run"].items():
        for ko, kn in S.items():
            a, b = o[ko], new["runs"][label][kn]
            same = (a == b) or (isinstance(a, float) and abs(a - b) < 1e-3)
            if same:
                s_ok += 1
            else:
                s_bad += 1
                print("  SANITY DIFF", label, ko, a, kn, b)
    print("per-run sanity counts compared: %d equal, %d different" % (s_ok, s_bad))
    print("\nearlier workload_identity block keys:", list(old["sanity"]["workload_identity"].keys()))


if __name__ == "__main__":
    main()
