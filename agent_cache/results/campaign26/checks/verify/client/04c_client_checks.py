#!/usr/bin/env python3
"""Item 5, part c: client-side signs of unequal conditions between runs, and record-level checks."""
import json
import math
import os
import pickle

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "parsed.pkl"), "rb") as f:
    P = pickle.load(f)

RES = {}


def pct(v, q):
    s = sorted(v)
    return s[min(len(s) - 1, int(q * len(s)))]


def mean(v):
    return math.fsum(v) / len(v) if v else float("nan")


def with_ttft_sorted(turns):
    v = [t for t in turns if t.get("ttft") is not None]
    v.sort(key=lambda t: t["t_send"])
    return v


print("=== 1. the first minutes of every run, when the same few sessions are live in every arm")
print("    (returning turns sent in minutes 0-10; first turns sent in minutes 0-10)")
print("   %-14s %5s %9s %9s %9s %9s %6s %9s %11s %11s %10s" % (
    "run", "n_ret", "ttft_med", "ttft_mean", "ttft_p90", "itl_ms_med", "n_t0", "t0_med", "t0_tok/s_med", "c0t0_ttft", "date"))
import time
RES["early"] = {}
for a, d in P.items():
    v = with_ttft_sorted(d["turns"])
    ret = [t for t in v if t["turn"] > 0 and t["t_send"] < 600]
    t0 = [t for t in v if t["turn"] == 0 and t["t_send"] < 600]
    itl = [1000.0 * (t["latency"] - t["ttft"]) / (t["completion_tokens"] - 1) for t in ret if t["completion_tokens"] > 1]
    rate = [t["prompt_tokens"] / t["ttft"] for t in t0 if t["cached_tokens"] == 0 and t["ttft"] > 0]
    c0 = next((t for t in v if t["conv"] == 0 and t["turn"] == 0), None)
    day = time.strftime("%m-%d %H:%M", time.gmtime(d["run"][0]["t_start"]))
    r = {"n_returning": len(ret), "ttft_median": pct([t["ttft"] for t in ret], 0.5), "ttft_mean": mean([t["ttft"] for t in ret]),
         "ttft_p90": pct([t["ttft"] for t in ret], 0.9), "itl_ms_median": pct(itl, 0.5), "itl_ms_mean": mean(itl),
         "n_first": len(t0), "first_ttft_median": pct([t["ttft"] for t in t0], 0.5),
         "first_uncached_prefill_tok_per_s_median": pct(rate, 0.5) if rate else None,
         "conv0_turn0_ttft": c0["ttft"] if c0 else None, "start_utc": day}
    RES["early"][a] = r
    print("   %-14s %5d %9.4f %9.4f %9.4f %9.3f %6d %9.4f %11.0f %11.3f %10s" % (
        a, len(ret), r["ttft_median"], r["ttft_mean"], r["ttft_p90"], r["itl_ms_median"], len(t0),
        r["first_ttft_median"], r["first_uncached_prefill_tok_per_s_median"] or float("nan"),
        r["conv0_turn0_ttft"], day))

print()
print("=== 2. decode pace over the whole run: ms per output token after the first (kept-style, all turns with ttft)")
print("   %-14s %9s %9s %9s %9s" % ("run", "median", "mean", "p90", "p99"))
RES["itl"] = {}
for a, d in P.items():
    v = with_ttft_sorted(d["turns"])
    itl = [1000.0 * (t["latency"] - t["ttft"]) / (t["completion_tokens"] - 1) for t in v if t["completion_tokens"] > 1]
    RES["itl"][a] = {"median": pct(itl, 0.5), "mean": mean(itl), "p90": pct(itl, 0.9), "p99": pct(itl, 0.99)}
    print("   %-14s %9.3f %9.3f %9.3f %9.3f" % (a, pct(itl, 0.5), mean(itl), pct(itl, 0.9), pct(itl, 0.99)))

print()
print("=== 3. does the client send on time?  t_send - (previous t_send + previous latency + gap_slept), seconds")
print("   %-14s %7s %9s %9s %9s %9s %8s" % ("run", "n", "median", "p99", "max", "min", ">1s"))
RES["lateness"] = {}
for a, d in P.items():
    byc = {}
    for t in d["turns"]:
        byc.setdefault(t["conv"], {})[t["turn"]] = t
    late = []
    gaps_missing = 0
    noncontig = 0
    for c, m in byc.items():
        ks = sorted(m)
        if ks != list(range(ks[0], ks[0] + len(ks))) or ks[0] != 0:
            noncontig += 1
        for n in ks:
            if n == 0 or (n - 1) not in m:
                continue
            p, t = m[n - 1], m[n]
            late.append(t["t_send"] - (p["t_send"] + p["latency"] + t["gap_slept"]))
    r = {"n": len(late), "median": pct(late, 0.5), "p99": pct(late, 0.99), "max": max(late), "min": min(late),
         "over_1s": sum(1 for x in late if x > 1.0), "convs_with_gaps_in_turn_numbers": noncontig}
    RES["lateness"][a] = r
    print("   %-14s %7d %9.4f %9.4f %9.3f %9.3f %8d   convs with non-contiguous turns: %d" % (
        a, len(late), r["median"], r["p99"], r["max"], r["min"], r["over_1s"], noncontig))

print()
print("=== 4. record-level consistency")
print("   %-14s %8s %8s %9s %9s %10s %10s %9s %9s" % (
    "run", "ttft>lat", "rr<0", "rr_null_ret", "ident_bad", "max_ttft", "max_lat", "cd_null_ret", "ct!=outlen"))
RES["records"] = {}
for a, d in P.items():
    tt = d["turns"]
    gt = sum(1 for t in tt if t["ttft"] > t["latency"] + 0.01)
    neg = sum(1 for t in tt if isinstance(t.get("reply_reprefilled"), int) and t["reply_reprefilled"] < 0)
    nullret = sum(1 for t in tt if t["turn"] > 0 and t.get("reply_reprefilled") is None)
    bad = 0
    for t in tt:
        if t.get("reply_reprefilled") is not None and t.get("delta_tokens") is not None:
            if t["cached_tokens"] + t["delta_tokens"] + t["reply_reprefilled"] != t["prompt_tokens"]:
                bad += 1
    cdnull = sum(1 for t in tt if t["turn"] > 0 and t.get("cached_details") is None)
    ctdiff = sum(1 for t in tt if t["completion_tokens"] != t["output_length"])
    r = {"ttft_gt_latency": gt, "reprefilled_negative": neg, "reprefilled_null_on_returning": nullret,
         "cached_plus_delta_plus_reprefilled_ne_prompt": bad, "max_ttft": max(t["ttft"] for t in tt),
         "max_latency": max(t["latency"] for t in tt), "cached_details_null_on_returning": cdnull,
         "completion_ne_output_length": ctdiff}
    RES["records"][a] = r
    print("   %-14s %8d %8d %9d %9d %10.2f %10.2f %9d %9d" % (
        a, gt, neg, nullret, bad, r["max_ttft"], r["max_latency"], cdnull, ctdiff))

print()
print("=== 5. prompt_tokens against the trace's value (replay_prompt_tokens): the largest gaps per run")
RES["prompt_outliers"] = {}
for a, d in P.items():
    dif = sorted(((t["prompt_tokens"] - t["replay_prompt_tokens"], t["conv"], t["turn"], t["prompt_tokens"],
                   t["replay_prompt_tokens"], t["completion_tokens"], t["output_length"]) for t in d["turns"]))
    big = [x for x in dif if abs(x[0]) > 64]
    RES["prompt_outliers"][a] = {"abs_gt_64": len(big), "worst": dif[0][:5], "hist": {
        "0": sum(1 for x in dif if x[0] == 0), "1..8": sum(1 for x in dif if 0 < abs(x[0]) <= 8),
        "9..64": sum(1 for x in dif if 8 < abs(x[0]) <= 64), ">64": len(big)}}
    print("   %-14s |diff|=0: %5d  1..8: %5d  9..64: %4d  >64: %3d   worst (diff, conv, turn, prompt, replay): %s" % (
        a, sum(1 for x in dif if x[0] == 0), sum(1 for x in dif if 0 < abs(x[0]) <= 8),
        sum(1 for x in dif if 8 < abs(x[0]) <= 64), len(big), dif[0][:5]))
    for x in big[:6]:
        print("        ", x)

print()
print("=== 6. turns lost at the end of a run (a session still open whose next turn was due but has no record)")
print("    the next turn's planned gap is taken from any other run that has that key")
gap_of = {}
for a, d in P.items():
    for t in d["turns"]:
        gap_of.setdefault((t["conv"], t["turn"]), t["gap_slept"])
RES["censoring"] = {}
for a, d in P.items():
    v = with_ttft_sorted(d["turns"])
    mx = v[-1]["t_send"]
    cut = mx - 300.0
    ended = {r["conv"] for r in d["other"].get("conv_end", [])}
    last = {}
    for t in d["turns"]:
        if t["conv"] not in last or t["turn"] > last[t["conv"]]["turn"]:
            last[t["conv"]] = t
    started = {r["conv"]: r for r in d["conv_start"]}
    open_convs = [c for c in started if c not in ended]
    due_before_cut = []
    unknown = 0
    no_turn = 0
    for c in open_convs:
        if c not in last:
            no_turn += 1
            # first turn sent at start_s; lost if the session started before the cut
            if started[c]["start_s"] < cut:
                due_before_cut.append((c, 0, started[c]["start_s"]))
            continue
        t = last[c]
        nk = (c, t["turn"] + 1)
        if nk not in gap_of:
            unknown += 1
            continue
        due = t["t_send"] + t["latency"] + gap_of[nk]
        if due < cut:
            due_before_cut.append((c, t["turn"] + 1, round(due, 1)))
    r = {"sessions_started": len(started), "sessions_ended": len(ended), "sessions_open_at_end": len(open_convs),
         "open_without_any_turn": no_turn, "next_turn_gap_unknown": unknown,
         "next_turn_due_before_the_300s_cut_but_no_record": len(due_before_cut), "examples": due_before_cut[:8],
         "max_t_send": mx}
    RES["censoring"][a] = r
    print("   %-14s open at end %3d  (no turn at all %d, next gap unknown %d)  due before the 300 s cut with no record: %d %s" % (
        a, len(open_convs), no_turn, unknown, len(due_before_cut), due_before_cut[:5]))

with open(os.path.join(HERE, "item5c_client_checks.json"), "w") as f:
    json.dump(RES, f, indent=1, sort_keys=True)
print()
print("wrote item5c_client_checks.json")
