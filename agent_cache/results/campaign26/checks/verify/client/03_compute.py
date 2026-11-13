#!/usr/bin/env python3
"""Items 1 to 4: independent recomputation from the task's definitions.

Input: parsed.pkl (written by 01_parse.py).  Output: items_1_4.json and the
text printed to stdout.  Nothing here reads the existing analysis.
"""
import json
import math
import os
import pickle
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "parsed.pkl"), "rb") as f:
    P = pickle.load(f)

LEAD5 = ["jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend"]
ARMSETS = {
    "ALL11": ["baseline", "base_rep", "hintfree", "jit_p50", "jit_exact", "jit_p50_ctl",
              "hintfree_ctl", "ec_r13", "nofc", "ec_r1", "nofc_noend"],
    "LEAD5": LEAD5,
    "SEVEN": LEAD5 + ["jit_p50_ctl", "hintfree_ctl"],
    "CUT7": ["jit_p50", "jit_p50_ctl", "jit_p50_d_cut", "hintfree_ctl", "nofc_noend", "ec_r13", "jit_exact"],
}
ARMS = [a for a in P if a != "ORDER"]
EPISODES = [(5500, 6999), (8750, 10999)]
TAIL_S = 300.0


def pct(v, q):
    s = sorted(v)
    return s[min(len(s) - 1, int(q * len(s)))]


def mean(v):
    return math.fsum(v) / len(v) if v else float("nan")


def recomputed(t):
    return max(t.get("reply_reprefilled") or 0, 0)


def with_ttft_sorted(turns):
    v = [t for t in turns if t.get("ttft") is not None]
    v.sort(key=lambda t: t["t_send"])  # stable: ties keep file order
    return v


def kept_turns(turns):
    v = with_ttft_sorted(turns)
    if not v:
        return [], None
    cut = v[-1]["t_send"] - TAIL_S
    return [t for t in v if t["t_send"] < cut], v[-1]["t_send"]


def in_episode(p):
    return p is not None and any(lo <= p <= hi for lo, hi in EPISODES)


OUT = {"values": [], "sanity": {}, "sets": {}}


def put(item, arm_set, arm, metric, value, unit=""):
    OUT["values"].append({"item": item, "arm_set": arm_set, "arm": arm, "metric": metric,
                          "value": value, "unit": unit})


# ---------------------------------------------------------------- kept turns
KEPT = {}      # label -> list of kept turn records
KEPTMAP = {}   # label -> {(conv, turn): record}
MAXSEND = {}
for label, d in P.items():
    k, mx = kept_turns(d["turns"])
    KEPT[label] = k
    MAXSEND[label] = mx
    m = {}
    for t in k:
        m[(t["conv"], t["turn"])] = t
    KEPTMAP[label] = m

order_sorted = with_ttft_sorted(P["ORDER"]["turns"])
POS = {}
for i, t in enumerate(order_sorted):
    POS[(t["conv"], t["turn"])] = i
ties = 0
for a, b in zip(order_sorted, order_sorted[1:]):
    if a["t_send"] == b["t_send"]:
        ties += 1
OUT["sanity"]["order_run"] = {"turns_with_ttft": len(order_sorted), "distinct_keys": len(POS),
                              "adjacent_equal_t_send": ties}
print("ORDER run: %d turns with a ttft, %d distinct keys, %d adjacent equal t_send"
      % (len(order_sorted), len(POS), ties))
# ties that sit on an episode boundary would make membership depend on tie order
for b in (5500, 7000, 8750, 11000):
    if 0 < b < len(order_sorted):
        print("  boundary %d: t_send[%d]=%.3f t_send[%d]=%.3f" % (
            b, b - 1, order_sorted[b - 1]["t_send"], b, order_sorted[b]["t_send"]))

# ---------------------------------------------------------------- items 1-3
SHARED = {}
for sname, arms in ARMSETS.items():
    keysets = [{k for k in KEPTMAP[a] if k[1] > 0} for a in arms]
    shared = set.intersection(*keysets)
    SHARED[sname] = shared
    n = len(shared)
    skeys = sorted(shared)
    epi = [k for k in skeys if in_episode(POS.get(k))]
    els = [k for k in skeys if not in_episode(POS.get(k))]
    missing = [k for k in skeys if k not in POS]
    epi1 = [k for k in skeys if POS.get(k) is not None and in_episode(POS[k] + 1)]  # 1-based variant
    OUT["sets"][sname] = {"arms": arms, "shared": n, "episode": len(epi), "elsewhere": len(els),
                          "missing_from_order": len(missing), "episode_if_1_based": len(epi1),
                          "convs": len({k[0] for k in skeys}),
                          "max_conv": max(k[0] for k in skeys), "max_turn": max(k[1] for k in skeys)}
    put("1", sname, "(set)", "shared_turns", n, "turns")
    put("1", sname, "(set)", "shared_turns_in_load_episodes", len(epi), "turns")
    put("1", sname, "(set)", "shared_turns_elsewhere", len(els), "turns")
    print()
    print("=== %s: %d shared returning turns (%d in load episodes, %d elsewhere; %d keys absent from the"
          " order run; %d convs; episode count with 1-based positions %d)"
          % (sname, n, len(epi), len(els), len(missing), len({k[0] for k in skeys}), len(epi1)))
    print("%-14s %8s %8s %8s %8s %6s %10s %10s %10s %9s %9s" % (
        "arm", "mean", "median", "p90", "p99", ">10s", "epi_mean", "else_mean", "recomp", "med_stat", "sum"))
    means = {}
    for a in arms:
        m = KEPTMAP[a]
        v = [m[k]["ttft"] for k in skeys]
        ve = [m[k]["ttft"] for k in epi]
        vo = [m[k]["ttft"] for k in els]
        rc = [recomputed(m[k]) for k in skeys]
        mu = mean(v)
        means[a] = mu
        over = sum(1 for x in v if x > 10.0)
        put("1", sname, a, "mean_ttft", mu, "s")
        put("1", sname, a, "median_ttft", pct(v, 0.5), "s")
        put("1", sname, a, "p90_ttft", pct(v, 0.9), "s")
        put("1", sname, a, "p99_ttft", pct(v, 0.99), "s")
        put("1", sname, a, "turns_over_10s", over, "turns")
        put("1", sname, a, "mean_ttft_load_episodes", mean(ve), "s")
        put("1", sname, a, "mean_ttft_elsewhere", mean(vo), "s")
        put("1", sname, a, "mean_recomputed_tokens", mean(rc), "tokens")
        put("1x", sname, a, "median_ttft_statistics_median", statistics.median(v), "s")
        print("%-14s %8.4f %8.4f %8.4f %8.4f %6d %10.4f %10.4f %10.2f %9.4f %9.1f" % (
            a, mu, pct(v, 0.5), pct(v, 0.9), pct(v, 0.99), over, mean(ve), mean(vo), mean(rc),
            statistics.median(v), math.fsum(v)))

    def paired(first, second, keys):
        ma, mb = KEPTMAP[first], KEPTMAP[second]
        return mean([mb[k]["ttft"] - ma[k]["ttft"] for k in keys])

    if sname == "LEAD5":
        pairs = [(arms[i], arms[j]) for i in range(len(arms)) for j in range(i + 1, len(arms))]
    elif sname == "CUT7":
        pairs = [("jit_p50_ctl", "jit_p50_d_cut"), ("jit_p50", "jit_p50_d_cut"), ("jit_p50", "jit_p50_ctl")]
    else:
        pairs = []
    if pairs:
        print("paired mean TTFT differences, second minus first (s per shared turn): all / load episodes / elsewhere")
    for first, second in pairs:
        d_all = paired(first, second, skeys)
        d_epi = paired(first, second, epi)
        d_els = paired(first, second, els)
        name = "%s-%s" % (second, first)
        put("2", sname, name, "paired_mean_ttft_diff", d_all, "s")
        put("2", sname, name, "paired_mean_ttft_diff_load_episodes", d_epi, "s")
        put("2", sname, name, "paired_mean_ttft_diff_elsewhere", d_els, "s")
        print("  %-28s %+9.4f %+9.4f %+9.4f" % (name, d_all, d_epi, d_els))

    if sname == "ALL11":
        print("share of the gain to jit_exact: (ref - arm)/(ref - jit_exact) for ref = baseline, base_rep")
        for a in arms:
            f1 = (means["baseline"] - means[a]) / (means["baseline"] - means["jit_exact"])
            f2 = (means["base_rep"] - means[a]) / (means["base_rep"] - means["jit_exact"])
            put("3", sname, a, "share_of_gain_vs_baseline", f1, "fraction")
            put("3", sname, a, "share_of_gain_vs_base_rep", f2, "fraction")
            print("  %-14s %8.4f %8.4f" % (a, f1, f2))

# ---------------------------------------------------------------- item 4
print()
print("=== sanity per run")
print("%-14s %7s %7s %6s %6s %8s %9s %9s %9s %9s %8s %8s" % (
    "run", "turns", "ttftOK", "error", "dupkey", "kept", "kept_ret", "max_send", "last_kept", "last_min",
    "dup_kept", "miss_ord"))
san = {}
for label, d in P.items():
    turns = d["turns"]
    seen = {}
    for t in turns:
        k = (t["conv"], t["turn"])
        seen[k] = seen.get(k, 0) + 1
    dup = sum(1 for k, c in seen.items() if c > 1)
    dup_extra = sum(c - 1 for c in seen.values() if c > 1)
    nerr = sum(1 for t in turns if t.get("error"))
    nerr_key = sum(1 for t in turns if "error" in t)
    nnull = sum(1 for t in turns if t.get("ttft") is None)
    nbadnum = sum(1 for t in turns if t.get("ttft") is not None and not (t["ttft"] >= 0 and t["ttft"] == t["ttft"]))
    k = KEPT[label]
    kseen = {}
    for t in k:
        kk = (t["conv"], t["turn"])
        kseen[kk] = kseen.get(kk, 0) + 1
    dupk = sum(1 for c in kseen.values() if c > 1)
    kept_ret = sum(1 for t in k if t["turn"] > 0)
    miss = [kk for kk in kseen if kk not in POS]
    miss_ret = [kk for kk in miss if kk[1] > 0]
    last_kept = k[-1]["t_send"] if k else float("nan")
    san[label] = {
        "turn_records": len(turns), "ttft_non_null": len(turns) - nnull, "ttft_null": nnull,
        "error_truthy": nerr, "error_key_present": nerr_key, "ttft_negative_or_nan": nbadnum,
        "duplicate_keys": dup, "duplicate_extra_records": dup_extra,
        "kept_turns": len(k), "kept_returning": kept_ret, "kept_first_turns": len(k) - kept_ret,
        "max_t_send_s": MAXSEND[label], "last_kept_t_send_s": last_kept,
        "last_kept_minute": last_kept / 60.0, "duplicate_keys_in_kept": dupk,
        "kept_keys_missing_from_order_run": len(miss), "kept_returning_keys_missing_from_order_run": len(miss_ret),
        "convs_started": len(d["conv_start"]), "convs_with_a_turn": len({t["conv"] for t in turns}),
        "max_conv_kept": max(t["conv"] for t in k), "summary_present": "summary" in d["other"],
    }
    print("%-14s %7d %7d %6d %6d %8d %9d %9.1f %9.1f %9.2f %8d %8d" % (
        label, len(turns), len(turns) - nnull, nerr, dup, len(k), kept_ret, MAXSEND[label], last_kept,
        last_kept / 60.0, dupk, len(miss)))
OUT["sanity"]["per_run"] = san

print()
print("=== per set: shared keys missing from the order run")
for sname in ARMSETS:
    print("  %-6s %d of %d" % (sname, OUT["sets"][sname]["missing_from_order"], OUT["sets"][sname]["shared"]))

# same workload item?  prompt_tokens of the same key across the LEAD5 runs
print()
print("=== prompt_tokens of the same key across the LEAD5 runs")


def spread_report(name, keys, maps, field):
    ident = 0
    spreads = []
    for k in keys:
        vals = [maps[a][k].get(field) for a in maps]
        if all(v == vals[0] for v in vals):
            ident += 1
            spreads.append(0)
        else:
            nums = [v for v in vals if isinstance(v, (int, float))]
            spreads.append((max(nums) - min(nums)) if len(nums) == len(vals) else float("inf"))
    n = len(keys)
    res = {"keys": n, "identical": ident, "share_identical": (ident / n) if n else float("nan")}
    if n:
        nz = sorted(s for s in spreads if s != 0)
        res.update({"differing": len(nz), "max_spread": max(spreads),
                    "p50_spread_all": pct(spreads, 0.5), "p99_spread_all": pct(spreads, 0.99),
                    "spread_le_1": sum(1 for s in spreads if s <= 1), "spread_le_8": sum(1 for s in spreads if s <= 8),
                    "spread_gt_100": sum(1 for s in spreads if s > 100)})
        if nz:
            res.update({"p50_spread_nonzero": pct(nz, 0.5), "p90_spread_nonzero": pct(nz, 0.9)})
    print("  %-48s %s" % (name, json.dumps(res)))
    return res


allmaps = {}
for a in P:
    m = {}
    for t in P[a]["turns"]:
        m[(t["conv"], t["turn"])] = t
    allmaps[a] = m
lead_maps_kept = {a: KEPTMAP[a] for a in LEAD5}
lead_maps_all = {a: allmaps[a] for a in LEAD5}
wl = {}
wl["lead5_shared_returning_kept.prompt_tokens"] = spread_report(
    "LEAD5 shared returning turns: prompt_tokens", sorted(SHARED["LEAD5"]), lead_maps_kept, "prompt_tokens")
common_all = set.intersection(*[set(allmaps[a]) for a in LEAD5])
wl["lead5_all_common_keys.prompt_tokens"] = spread_report(
    "LEAD5 all common keys (turn 0 included): prompt_tokens", sorted(common_all), lead_maps_all, "prompt_tokens")
for field in ("replay_prompt_tokens", "output_length", "completion_tokens", "gap_slept", "pre_gap", "task_conv", "source"):
    wl["lead5_all_common_keys." + field] = spread_report(
        "LEAD5 all common keys: " + field, sorted(common_all), lead_maps_all, field)
# pairwise signed differences against jit_exact
print("  signed prompt_tokens difference (arm - jit_exact) on the LEAD5 shared returning turns:")
for a in LEAD5[1:]:
    dif = [KEPTMAP[a][k]["prompt_tokens"] - KEPTMAP["jit_exact"][k]["prompt_tokens"] for k in sorted(SHARED["LEAD5"])]
    nzero = sum(1 for x in dif if x == 0)
    r = {"n": len(dif), "zero": nzero, "min": min(dif), "p01": pct(dif, 0.01), "p50": pct(dif, 0.5),
         "p99": pct(dif, 0.99), "max": max(dif), "mean": mean(dif)}
    wl["lead5_prompt_tokens_diff_%s_minus_jit_exact" % a] = r
    print("    %-12s %s" % (a, json.dumps(r)))
# prompt_tokens against the trace value recorded by the client in the same record
print("  prompt_tokens == replay_prompt_tokens within each run (all turn records):")
for a in P:
    n = len(P[a]["turns"])
    eq = sum(1 for t in P[a]["turns"] if t.get("prompt_tokens") == t.get("replay_prompt_tokens"))
    dif = [t["prompt_tokens"] - t["replay_prompt_tokens"] for t in P[a]["turns"]
           if isinstance(t.get("replay_prompt_tokens"), int)]
    wl["prompt_vs_replay." + a] = {"n": n, "equal": eq, "min_diff": min(dif), "max_diff": max(dif)}
    print("    %-14s n=%d equal=%d min_diff=%d max_diff=%d" % (a, n, eq, min(dif), max(dif)))

# first-turn prompt_tokens per conv across all runs
print()
print("=== first-turn (turn 0) prompt_tokens per conv across all 13 runs")
first = {}
for a in P:
    for t in P[a]["turns"]:
        if t["turn"] == 0:
            first.setdefault(t["conv"], {})[a] = t["prompt_tokens"]
nconv = len(first)
multi = {c: v for c, v in first.items() if len(v) > 1}
agree = sum(1 for v in multi.values() if len(set(v.values())) == 1)
in_all = {c: v for c, v in first.items() if len(v) == len(P)}
agree_all = sum(1 for v in in_all.values() if len(set(v.values())) == 1)
maxspread = max((max(v.values()) - min(v.values())) for v in multi.values())
print("  convs with a first turn in any run: %d; in more than one run: %d, of which identical everywhere: %d;"
      " in all 13 runs: %d, identical: %d; largest spread: %d tokens"
      % (nconv, len(multi), agree, len(in_all), agree_all, maxspread))
wl["first_turn_prompt_tokens"] = {"convs_any": nconv, "convs_multi": len(multi), "identical_multi": agree,
                                  "convs_in_all_13": len(in_all), "identical_in_all_13": agree_all,
                                  "max_spread": maxspread}
bad = [(c, v) for c, v in sorted(multi.items()) if len(set(v.values())) > 1]
for c, v in bad[:10]:
    print("    differing conv", c, v)
# conv -> session identity from conv_start
ident = {}
for a in P:
    for r in P[a]["conv_start"]:
        ident.setdefault(r["conv"], {})[a] = (r.get("task_conv"), r.get("session_id"), r.get("turns"), r.get("arrival_s"))
multi_i = {c: v for c, v in ident.items() if len(v) > 1}
agree_i = sum(1 for v in multi_i.values() if len(set(v.values())) == 1)
print("  conv_start (task_conv, session_id, turns, arrival_s) per conv: %d convs in more than one run, %d identical everywhere"
      % (len(multi_i), agree_i))
wl["conv_start_identity"] = {"convs_multi": len(multi_i), "identical": agree_i}
# returning turns across ALL runs: prompt_tokens identical for every key seen in more than one run?
allkeys = {}
for a in P:
    for k, t in allmaps[a].items():
        allkeys.setdefault(k, {})[a] = t["prompt_tokens"]
mk = {k: v for k, v in allkeys.items() if len(v) > 1}
agree_k = sum(1 for v in mk.values() if len(set(v.values())) == 1)
spreadk = [max(v.values()) - min(v.values()) for v in mk.values()]
print("  every key seen in more than one of the 13 runs: %d keys, prompt_tokens identical in %d; largest spread %d"
      % (len(mk), agree_k, max(spreadk)))
wl["all_runs_prompt_tokens"] = {"keys_multi": len(mk), "identical": agree_k, "max_spread": max(spreadk),
                                "spread_gt_0": sum(1 for s in spreadk if s > 0)}
OUT["sanity"]["workload_identity"] = wl

with open(os.path.join(HERE, "items_1_4.json"), "w") as f:
    json.dump(OUT, f, indent=1, sort_keys=True)
print()
print("wrote", os.path.join(HERE, "items_1_4.json"), "with", len(OUT["values"]), "values")
