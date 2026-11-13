#!/usr/bin/env python3
"""r2_02_compute.py -- items 1 to 4 from the task definitions (fresh implementation).

Input : r2_parsed.pkl (written by r2_01_parse.py)
Output: r2_items_1_4.json  (machine readable)  and stdout (readable)

Run as:  nice -n 19 python3 -I -B r2_02_compute.py
"""
import json
import math
import os
import pickle

HERE = os.path.dirname(os.path.abspath(__file__))

ARMS = ["baseline", "base_rep", "hintfree", "hintfree_ctl", "jit_p50", "jit_p50_ctl",
        "jit_p50_d_cut", "jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend"]
SETS = {
    "ALL11": ["baseline", "base_rep", "hintfree", "jit_p50", "jit_exact", "jit_p50_ctl",
              "hintfree_ctl", "ec_r13", "nofc", "ec_r1", "nofc_noend"],
    "LEAD5": ["jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend"],
    "SEVEN": ["jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend", "jit_p50_ctl", "hintfree_ctl"],
    "CUT7": ["jit_p50", "jit_p50_ctl", "jit_p50_d_cut", "hintfree_ctl", "nofc_noend", "ec_r13",
             "jit_exact"],
}
EPISODES = [(5500, 6999), (8750, 10999)]  # inclusive position ranges
TAIL_S = 300.0


def pct(v, q):
    s = sorted(v)
    return s[min(len(s) - 1, int(q * len(s)))]


def mean(v):
    return math.fsum(v) / len(v) if v else float("nan")


def recomputed(r):
    x = r.get("reply_reprefilled")
    return max(x or 0, 0)


def in_episode(p):
    if p is None:
        return False
    for lo, hi in EPISODES:
        if lo <= p <= hi:
            return True
    return False


def load():
    with open(os.path.join(HERE, "r2_parsed.pkl"), "rb") as f:
        return pickle.load(f)


def turn_views(parsed):
    """label -> dict with all turn records, the with-ttft list sorted by t_send, the kept list."""
    views = {}
    for label, d in parsed.items():
        turns = [r for r in d["records"] if r.get("kind") == "turn"]
        ok = [r for r in turns if r.get("ttft") is not None]
        ok_sorted = sorted(ok, key=lambda r: r["t_send"])  # stable: ties keep file order
        tmax_ok = max(r["t_send"] for r in ok_sorted)
        tmax_all = max(r["t_send"] for r in turns if r.get("t_send") is not None)
        thr = tmax_ok - TAIL_S
        kept = [r for r in ok_sorted if r["t_send"] < thr]
        views[label] = {
            "turns": turns, "ok": ok_sorted, "kept": kept,
            "tmax_ok": tmax_ok, "tmax_all": tmax_all, "thr": thr,
        }
    return views


def arm_stats(recs, pos):
    t = [r["ttft"] for r in recs]
    ep = [r["ttft"] for r in recs if in_episode(pos.get((r["conv"], r["turn"])))]
    el = [r["ttft"] for r in recs if not in_episode(pos.get((r["conv"], r["turn"])))]
    return {
        "n": len(t),
        "mean": mean(t),
        "median": pct(t, 0.5),
        "p90": pct(t, 0.9),
        "p99": pct(t, 0.99),
        "n_over_10s": sum(1 for x in t if x > 10.0),
        "n_ge_10s": sum(1 for x in t if x >= 10.0),
        "max": max(t),
        "mean_episodes": mean(ep),
        "n_episodes": len(ep),
        "mean_elsewhere": mean(el),
        "n_elsewhere": len(el),
        "mean_recomputed": mean([recomputed(r) for r in recs]),
        "sum_ttft": math.fsum(t),
    }


def main():
    parsed = load()
    views = turn_views(parsed)
    res = {"definitions": {
        "kept": "turn records with ttft != null, sorted by t_send, t_send < max(t_send over those) - 300",
        "shared": "returning (turn>0) keys kept in every arm of the set",
        "percentile": "sorted(v)[min(len(v)-1, int(q*len(v)))]",
        "position": "0-based index in ORDER run's with-ttft turns sorted by t_send (stable)",
        "episodes": "positions 5500..6999 and 8750..10999 inclusive",
        "over_10s": "ttft > 10.0",
    }}

    # ---- positions from the ORDER run ------------------------------------------------
    order_ok = views["ORDER"]["ok"]
    pos = {}
    dup_order = 0
    for i, r in enumerate(order_ok):
        k = (r["conv"], r["turn"])
        if k in pos:
            dup_order += 1
        pos[k] = i
    ts = [r["t_send"] for r in order_ok]
    ties = sum(1 for a, b in zip(ts, ts[1:]) if a == b)
    # ties that straddle an episode boundary would make the split order dependent
    bnd = [5500, 7000, 8750, 11000]
    tie_at_boundary = [b for b in bnd if 0 < b < len(ts) and ts[b - 1] == ts[b]]
    res["order_run"] = {
        "n_turn_records": len(views["ORDER"]["turns"]),
        "n_with_ttft": len(order_ok),
        "duplicate_keys": dup_order,
        "t_send_ties_adjacent": ties,
        "ties_at_episode_boundaries": tie_at_boundary,
        "n_positions_in_episode1": sum(1 for p in pos.values() if 5500 <= p <= 6999),
        "n_positions_in_episode2": sum(1 for p in pos.values() if 8750 <= p <= 10999),
        "t_send_at_pos_5500": ts[5500], "t_send_at_pos_6999": ts[6999],
        "t_send_at_pos_8750": ts[8750], "t_send_last": ts[-1],
    }
    print("ORDER run:", json.dumps(res["order_run"]))

    # ---- item 4 a-c: per run sanity -----------------------------------------------------
    res["runs"] = {}
    kept_by = {}
    print("\n== item 4: per-run sanity ==")
    for label in ARMS + ["ORDER"]:
        v = views[label]
        turns = v["turns"]
        keys_all = [(r["conv"], r["turn"]) for r in turns]
        seen, dup_all = set(), 0
        for k in keys_all:
            if k in seen:
                dup_all += 1
            seen.add(k)
        seen2, dup_ok = set(), 0
        for r in v["ok"]:
            k = (r["conv"], r["turn"])
            if k in seen2:
                dup_ok += 1
            seen2.add(k)
        n_err = sum(1 for r in turns if r.get("error"))
        n_err_key = sum(1 for r in turns if "error" in r)
        n_null = sum(1 for r in turns if r.get("ttft") is None)
        n_err_or_null = sum(1 for r in turns if r.get("error") or r.get("ttft") is None)
        n_err_with_ttft = sum(1 for r in turns if r.get("error") and r.get("ttft") is not None)
        n_bad_ttft = sum(1 for r in v["ok"]
                         if not isinstance(r["ttft"], (int, float)) or r["ttft"] != r["ttft"]
                         or r["ttft"] <= 0)
        kept = v["kept"]
        kd = {}
        for r in kept:
            kd[(r["conv"], r["turn"])] = r
        kept_by[label] = kd
        kept_ret = [r for r in kept if r["turn"] > 0]
        miss_all = sum(1 for k in kd if k not in pos)
        miss_ret = sum(1 for k in kd if k[1] > 0 and k not in pos)
        run_rec = [r for r in parsed[label]["records"] if r.get("kind") == "run"]
        summ = [r for r in parsed[label]["records"] if r.get("kind") == "summary"]
        info = {
            "folder": parsed[label]["folder"],
            "client_dir": parsed[label]["client_dir"],
            "n_client_dirs": parsed[label]["n_client_dirs"],
            "n_run_records": len(run_rec),
            "t_start": run_rec[0].get("t_start") if run_rec else None,
            "has_summary": bool(summ),
            "n_turn_records": len(turns),
            "duplicate_keys_all_turn_records": dup_all,
            "duplicate_keys_with_ttft": dup_ok,
            "n_error_truthy": n_err,
            "n_records_with_error_field": n_err_key,
            "n_null_ttft": n_null,
            "n_error_or_null_ttft": n_err_or_null,
            "n_error_but_ttft_present": n_err_with_ttft,
            "n_ttft_nonpositive_or_nan": n_bad_ttft,
            "n_with_ttft": len(v["ok"]),
            "max_t_send_with_ttft": v["tmax_ok"],
            "max_t_send_all_turn_records": v["tmax_all"],
            "cut_threshold_s": v["thr"],
            "n_kept": len(kept),
            "n_kept_distinct_keys": len(kd),
            "n_kept_returning": len(kept_ret),
            "n_kept_first_turns": len(kept) - len(kept_ret),
            "n_dropped_tail_300s": len(v["ok"]) - len(kept),
            "last_kept_t_send_s": kept[-1]["t_send"],
            "last_kept_send_minute": kept[-1]["t_send"] / 60.0,
            "n_kept_keys_missing_from_order_run": miss_all,
            "n_kept_returning_keys_missing_from_order_run": miss_ret,
            "max_ttft_with_ttft": max(r["ttft"] for r in v["ok"]),
            "n_ttft_over_300s": sum(1 for r in v["ok"] if r["ttft"] > 300.0),
            "mean_ttft_all_kept_returning": mean([r["ttft"] for r in kept_ret]),
        }
        res["runs"][label] = info
        print("%-14s turns=%5d dupkeys=%d err=%d nullttft=%d err|null=%d withttft=%5d "
              "max_t_send=%.1f thr=%.1f kept=%5d (ret %5d, first %3d) last_kept_min=%.2f "
              "missing_from_order=%d (ret %d) summary=%s"
              % (label, len(turns), dup_all, n_err, n_null, n_err_or_null, len(v["ok"]),
                 v["tmax_ok"], v["thr"], len(kept), len(kept_ret), len(kept) - len(kept_ret),
                 kept[-1]["t_send"] / 60.0, miss_all, miss_ret, bool(summ)))

    # ---- items 1-3 ------------------------------------------------------------------------
    res["sets"] = {}
    res["pairs"] = {}
    shared_by_set = {}
    for sname, arms in SETS.items():
        shared = None
        for a in arms:
            ks = set(k for k in kept_by[a] if k[1] > 0)
            shared = ks if shared is None else (shared & ks)
        shared = sorted(shared)
        shared_by_set[sname] = shared
        n_ep = sum(1 for k in shared if in_episode(pos.get(k)))
        n_missing = sum(1 for k in shared if k not in pos)
        sres = {"arms": arms, "n_shared": len(shared), "n_shared_in_episodes": n_ep,
                "n_shared_elsewhere": len(shared) - n_ep,
                "n_shared_missing_from_order_run": n_missing,
                "n_shared_convs": len(set(k[0] for k in shared)), "per_arm": {}}
        print("\n== item 1: %s  shared turns = %d (episodes %d, elsewhere %d, missing from "
              "order run %d, convs %d) ==" % (sname, len(shared), n_ep, len(shared) - n_ep,
                                              n_missing, sres["n_shared_convs"]))
        print("%-14s %9s %8s %8s %8s %6s %10s %6s %10s %6s %10s"
              % ("arm", "mean", "median", "p90", "p99", ">10s", "mean_ep", "n_ep", "mean_else",
                 "n_else", "recomputed"))
        for a in arms:
            recs = [kept_by[a][k] for k in shared]
            st = arm_stats(recs, pos)
            sres["per_arm"][a] = st
            print("%-14s %9.4f %8.4f %8.4f %8.3f %6d %10.4f %6d %10.4f %6d %10.2f"
                  % (a, st["mean"], st["median"], st["p90"], st["p99"], st["n_over_10s"],
                     st["mean_episodes"], st["n_episodes"], st["mean_elsewhere"],
                     st["n_elsewhere"], st["mean_recomputed"]))
        res["sets"][sname] = sres

    def paired(sname, second, first):
        shared = shared_by_set[sname]
        d = [kept_by[second][k]["ttft"] - kept_by[first][k]["ttft"] for k in shared]
        dep = [x for x, k in zip(d, shared) if in_episode(pos.get(k))]
        del_ = [x for x, k in zip(d, shared) if not in_episode(pos.get(k))]
        return {
            "second": second, "first": first, "n": len(d),
            "mean_diff": mean(d), "median_diff": pct(d, 0.5),
            "mean_diff_episodes": mean(dep), "n_episodes": len(dep),
            "mean_diff_elsewhere": mean(del_), "n_elsewhere": len(del_),
            "n_second_slower": sum(1 for x in d if x > 0),
            "n_second_faster": sum(1 for x in d if x < 0),
        }

    print("\n== item 2: paired mean TTFT differences (second minus first), s per shared turn ==")
    res["pairs"]["LEAD5"] = {}
    lead = SETS["LEAD5"]
    for i in range(len(lead)):
        for j in range(i + 1, len(lead)):
            p = paired("LEAD5", lead[j], lead[i])
            res["pairs"]["LEAD5"]["%s-%s" % (lead[j], lead[i])] = p
            print("LEAD5 %-12s minus %-12s n=%d  all %+8.4f   episodes %+8.4f (n=%d)   "
                  "elsewhere %+8.4f (n=%d)"
                  % (lead[j], lead[i], p["n"], p["mean_diff"], p["mean_diff_episodes"],
                     p["n_episodes"], p["mean_diff_elsewhere"], p["n_elsewhere"]))
    res["pairs"]["CUT7"] = {}
    for second, first in [("jit_p50_d_cut", "jit_p50_ctl"), ("jit_p50_d_cut", "jit_p50"),
                          ("jit_p50_ctl", "jit_p50")]:
        p = paired("CUT7", second, first)
        res["pairs"]["CUT7"]["%s-%s" % (second, first)] = p
        print("CUT7  %-13s minus %-12s n=%d  all %+8.4f   episodes %+8.4f (n=%d)   "
              "elsewhere %+8.4f (n=%d)"
              % (second, first, p["n"], p["mean_diff"], p["mean_diff_episodes"],
                 p["n_episodes"], p["mean_diff_elsewhere"], p["n_elsewhere"]))

    print("\n== item 3: share of the gain to jit_exact on ALL11 (mean TTFT on shared turns) ==")
    res["ratios"] = {"ALL11": {}}
    m = {a: res["sets"]["ALL11"]["per_arm"][a]["mean"] for a in SETS["ALL11"]}
    for a in SETS["ALL11"]:
        r1 = (m["baseline"] - m[a]) / (m["baseline"] - m["jit_exact"])
        r2 = (m["base_rep"] - m[a]) / (m["base_rep"] - m["jit_exact"])
        res["ratios"]["ALL11"][a] = {"vs_baseline": r1, "vs_base_rep": r2, "mean": m[a]}
        print("%-14s mean %8.4f   (baseline-arm)/(baseline-jit_exact) = %8.4f   "
              "(base_rep-arm)/(base_rep-jit_exact) = %8.4f" % (a, m[a], r1, r2))

    # ---- item 4 d-e: is (conv, turn) the same workload item in every run? ---------------
    print("\n== item 4: workload identity ==")
    wl = {}

    def spread_hist(spreads):
        edges = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 10**9]
        names = ["0", "1", "2", "3-5", "6-10", "11-20", "21-50", "51-100", "101-200",
                 "201-500", "501-1000", ">1000"]
        h = dict((n, 0) for n in names)
        for s in spreads:
            for e, n in zip(edges, names):
                if s <= e:
                    h[n] += 1
                    break
        return h

    def compare_field(labels, keys, field, src):
        """src[label][key] -> record.  Returns stats on the per-key spread of `field`."""
        spreads, n_ident, n_none = [], 0, 0
        rel = []
        for k in keys:
            vals = [src[a][k].get(field) for a in labels]
            if any(x is None for x in vals):
                n_none += 1
                continue
            sp = max(vals) - min(vals)
            spreads.append(sp)
            if sp == 0:
                n_ident += 1
            if min(vals) > 0:
                rel.append(sp / float(min(vals)))
        n = len(spreads)
        out = {"n_keys": len(keys), "n_compared": n, "n_with_null": n_none,
               "n_identical": n_ident,
               "share_identical": (n_ident / float(n)) if n else None}
        if n:
            out.update({
                "spread_mean": mean(spreads), "spread_median": pct(spreads, 0.5),
                "spread_p90": pct(spreads, 0.9), "spread_p99": pct(spreads, 0.99),
                "spread_max": max(spreads), "spread_hist": spread_hist(spreads),
                "rel_spread_p50": pct(rel, 0.5) if rel else None,
                "rel_spread_p99": pct(rel, 0.99) if rel else None,
                "rel_spread_max": max(rel) if rel else None,
            })
        return out

    lead_shared = shared_by_set["LEAD5"]
    for field in ["prompt_tokens", "replay_prompt_tokens", "output_length", "completion_tokens",
                  "task_conv", "gap_slept"]:
        c = compare_field(lead, lead_shared, field, kept_by)
        wl["LEAD5_shared_%s" % field] = c
        print("LEAD5 shared returning turns, %-22s: compared %d, identical in all five %d "
              "(%.4f); spread mean %.3f median %s p90 %s p99 %s max %s"
              % (field, c["n_compared"], c["n_identical"], c["share_identical"] or 0.0,
                 c.get("spread_mean", float("nan")), c.get("spread_median"), c.get("spread_p90"),
                 c.get("spread_p99"), c.get("spread_max")))
        if field == "prompt_tokens":
            print("   spread histogram (max-min over the five runs, tokens):", c["spread_hist"])
            print("   relative spread: p50 %.6f p99 %.6f max %.6f"
                  % (c["rel_spread_p50"], c["rel_spread_p99"], c["rel_spread_max"]))

    # pairwise prompt_tokens differences versus jit_exact on the LEAD5 shared turns
    wl["LEAD5_prompt_tokens_pairwise_vs_jit_exact"] = {}
    for a in lead[1:]:
        d = [kept_by[a][k]["prompt_tokens"] - kept_by["jit_exact"][k]["prompt_tokens"]
             for k in lead_shared]
        ad = [abs(x) for x in d]
        o = {"n": len(d), "n_equal": sum(1 for x in d if x == 0),
             "share_equal": sum(1 for x in d if x == 0) / float(len(d)),
             "mean_diff": mean(d), "mean_abs_diff": mean(ad), "median_abs": pct(ad, 0.5),
             "p90_abs": pct(ad, 0.9), "p99_abs": pct(ad, 0.99), "max_abs": max(ad),
             "min_diff": min(d), "max_diff": max(d)}
        wl["LEAD5_prompt_tokens_pairwise_vs_jit_exact"][a] = o
        print("   prompt_tokens %-11s minus jit_exact: equal %d of %d (%.4f), mean %+.3f, "
              "mean|d| %.3f, p50|d| %d, p90|d| %d, p99|d| %d, max|d| %d, range [%d, %d]"
              % (a, o["n_equal"], o["n"], o["share_equal"], o["mean_diff"], o["mean_abs_diff"],
                 o["median_abs"], o["p90_abs"], o["p99_abs"], o["max_abs"], o["min_diff"],
                 o["max_diff"]))

    # first-turn prompt_tokens per conv across all runs (12 arms + ORDER)
    all_labels = ARMS + ["ORDER"]
    first = {}
    for a in all_labels:
        for r in views[a]["turns"]:
            if r["turn"] == 0 and r.get("prompt_tokens") is not None:
                first.setdefault(r["conv"], {})[a] = r
    n_conv = len(first)
    n_multi = sum(1 for c in first if len(first[c]) >= 2)
    n_all13 = sum(1 for c in first if len(first[c]) == len(all_labels))
    ident = 0
    spreads = []
    worst = []
    for c, d in first.items():
        if len(d) < 2:
            continue
        vals = [d[a]["prompt_tokens"] for a in d]
        sp = max(vals) - min(vals)
        spreads.append(sp)
        if sp == 0:
            ident += 1
        else:
            worst.append((sp, c, dict((a, d[a]["prompt_tokens"]) for a in d)))
    worst.sort(reverse=True)
    task_mismatch = 0
    replay_mismatch = 0
    for c, d in first.items():
        if len(set(d[a].get("task_conv") for a in d)) > 1:
            task_mismatch += 1
        if len(set(d[a].get("replay_prompt_tokens") for a in d)) > 1:
            replay_mismatch += 1
    wl["first_turn_prompt_tokens_all_runs"] = {
        "n_convs_with_a_first_turn_in_any_run": n_conv,
        "n_convs_in_at_least_two_runs": n_multi,
        "n_convs_in_all_13_runs": n_all13,
        "n_identical": ident,
        "share_identical": ident / float(n_multi) if n_multi else None,
        "spread_hist": spread_hist(spreads),
        "spread_max": max(spreads) if spreads else None,
        "worst_examples": worst[:5],
        "n_convs_task_conv_differs": task_mismatch,
        "n_convs_replay_prompt_tokens_differs": replay_mismatch,
    }
    print("first-turn prompt_tokens per conv over all 13 runs: convs %d, in >=2 runs %d, in all "
          "13 runs %d, identical %d (%.4f), spread hist %s, max spread %s; task_conv differs for "
          "%d convs, replay_prompt_tokens differs for %d convs"
          % (n_conv, n_multi, n_all13, ident, ident / float(n_multi), spread_hist(spreads),
             max(spreads) if spreads else None, task_mismatch, replay_mismatch))
    for sp, c, d in worst[:5]:
        print("   conv %d spread %d: %s" % (c, sp, d))

    # conv -> (task_conv, session_id) mapping from conv_start records, across all runs
    cs = {}
    for a in all_labels:
        for r in parsed[a]["records"]:
            if r.get("kind") == "conv_start":
                cs.setdefault(r["conv"], {})[a] = (r.get("task_conv"), r.get("session_id"),
                                                   r.get("turns"), r.get("arrival_s"))
    n_map_diff = sum(1 for c in cs if len(set(v[:3] for v in cs[c].values())) > 1)
    n_arr_diff = sum(1 for c in cs if len(set(v[3] for v in cs[c].values())) > 1)
    wl["conv_start_mapping"] = {
        "n_convs": len(cs),
        "n_convs_task_or_session_or_turns_differ": n_map_diff,
        "n_convs_arrival_s_differs": n_arr_diff,
    }
    print("conv_start mapping over all runs: convs %d, (task_conv, session_id, turns) differs "
          "for %d, arrival_s differs for %d" % (len(cs), n_map_diff, n_arr_diff))

    # same key, all runs: replay_prompt_tokens / output_length / task_conv identical?
    allk = {}
    for a in all_labels:
        for r in views[a]["turns"]:
            allk.setdefault((r["conv"], r["turn"]), {})[a] = r
    for field in ["replay_prompt_tokens", "output_length", "task_conv", "prompt_tokens",
                  "gap_slept"]:
        n_cmp = n_same = 0
        for k, d in allk.items():
            vals = [d[a].get(field) for a in d if d[a].get(field) is not None]
            if len(vals) < 2:
                continue
            n_cmp += 1
            if max(vals) == min(vals):
                n_same += 1
        wl["all_runs_same_key_%s" % field] = {"n_keys_in_2plus_runs": n_cmp, "n_identical": n_same,
                                              "share_identical": n_same / float(n_cmp)}
        print("all 13 runs, same key, %-22s: keys in >=2 runs %d, identical %d (%.4f)"
              % (field, n_cmp, n_same, n_same / float(n_cmp)))
    res["workload"] = wl

    # ---- missing from order run, per arm ---------------------------------------------------
    print("\n== item 4: kept keys missing from the ORDER run ==")
    for a in ARMS:
        i = res["runs"][a]
        print("%-14s kept keys %5d, missing from order run %4d (returning %4d)"
              % (a, i["n_kept_distinct_keys"], i["n_kept_keys_missing_from_order_run"],
                 i["n_kept_returning_keys_missing_from_order_run"]))
    for sname in SETS:
        print("%-6s shared keys missing from order run: %d of %d"
              % (sname, res["sets"][sname]["n_shared_missing_from_order_run"],
                 res["sets"][sname]["n_shared"]))

    with open(os.path.join(HERE, "r2_items_1_4.json"), "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    print("\nwrote r2_items_1_4.json")


if __name__ == "__main__":
    main()
