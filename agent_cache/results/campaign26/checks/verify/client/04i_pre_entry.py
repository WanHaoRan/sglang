#!/usr/bin/env python3
"""Item 5, part i: the time between the client's send and the server's entry_time.

entry_time is the scheduler's wait-queue entry (req_time_stats.py prints wait_queue_entry_time), so
"entry - send" is HTTP + chat template + tokenizer + hand-over to the scheduler loop.  It is inside the
client's TTFT and outside the server's queue_duration.  Uses parsed.pkl and server_reqstats.pkl.
"""
import json
import math
import os
import pickle

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "parsed.pkl"), "rb") as f:
    P = pickle.load(f)
with open(os.path.join(HERE, "server_reqstats.pkl"), "rb") as f:
    S = pickle.load(f)

LEAD5 = ["jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend"]
ALL11 = ["baseline", "base_rep", "hintfree", "jit_p50", "jit_exact", "jit_p50_ctl", "hintfree_ctl", "ec_r13", "nofc", "ec_r1", "nofc_noend"]
EPISODES = [(5500, 6999), (8750, 10999)]
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


def kept_map(turns):
    v = with_ttft_sorted(turns)
    cut = v[-1]["t_send"] - 300.0
    return {(t["conv"], t["turn"]): t for t in v if t["t_send"] < cut}


KM = {a: kept_map(P[a]["turns"]) for a in P}
POS = {(t["conv"], t["turn"]): i for i, t in enumerate(with_ttft_sorted(P["ORDER"]["turns"]))}
T0 = {a: P[a]["run"][0]["t_start"] for a in P}


def in_episode(k):
    p = POS.get(k)
    return p is not None and any(lo <= p <= hi for lo, hi in EPISODES)


def pre(a, k):
    return S[a]["rows"][k][6] - (T0[a] + KM[a][k]["t_send"])


def queue(a, k):
    return S[a]["rows"][k][4] / 1000.0


def shared_of(arms):
    return sorted(set.intersection(*[{k for k in KM[a] if k[1] > 0} for a in arms]))


for sname, arms in (("ALL11", ALL11), ("LEAD5", LEAD5)):
    keys = shared_of(arms)
    epi = [k for k in keys if in_episode(k)]
    els = [k for k in keys if not in_episode(k)]
    print("=== %s shared turns (%d): TTFT = (entry - send) + server queue wait + the rest" % (sname, len(keys)))
    print("   %-14s %8s %8s %8s %8s %9s %9s %8s %8s %7s %9s %9s" % (
        "arm", "ttft", "pre", "queue", "rest", "pre/ttft", "pre_med", "pre_p90", "pre_p99", ">1s", "pre_epi", "pre_else"))
    for a in arms:
        pv = [pre(a, k) for k in keys]
        qv = [queue(a, k) for k in keys]
        tv = [KM[a][k]["ttft"] for k in keys]
        r = {"mean_ttft": mean(tv), "mean_pre_entry": mean(pv), "mean_queue": mean(qv), "mean_rest": mean(tv) - mean(pv) - mean(qv),
             "pre_entry_share_of_ttft": mean(pv) / mean(tv), "pre_median": pct(pv, 0.5), "pre_p90": pct(pv, 0.9), "pre_p99": pct(pv, 0.99),
             "pre_over_1s": sum(1 for x in pv if x > 1.0), "pre_mean_episodes": mean([pre(a, k) for k in epi]),
             "pre_mean_elsewhere": mean([pre(a, k) for k in els]),
             "pre_over_1s_in_episodes": sum(1 for k in epi if pre(a, k) > 1.0),
             "rest_negative": sum(1 for k in keys if KM[a][k]["ttft"] - pre(a, k) - queue(a, k) < -0.005)}
        RES["%s:%s" % (sname, a)] = r
        print("   %-14s %8.4f %8.4f %8.4f %8.4f %9.3f %9.4f %8.4f %8.4f %7d %9.4f %9.4f   (>1s inside episodes: %d; rest<0: %d)" % (
            a, r["mean_ttft"], r["mean_pre_entry"], r["mean_queue"], r["mean_rest"], r["pre_entry_share_of_ttft"], r["pre_median"],
            r["pre_p90"], r["pre_p99"], r["pre_over_1s"], r["pre_mean_episodes"], r["pre_mean_elsewhere"],
            r["pre_over_1s_in_episodes"], r["rest_negative"]))
    if sname == "LEAD5":
        print("   paired differences (second minus first), seconds per shared turn: TTFT = pre-entry + queue + rest")
        for i in range(len(arms)):
            for j in range(i + 1, len(arms)):
                a, b = arms[i], arms[j]
                dt = mean([KM[b][k]["ttft"] - KM[a][k]["ttft"] for k in keys])
                dp = mean([pre(b, k) - pre(a, k) for k in keys])
                dq = mean([queue(b, k) - queue(a, k) for k in keys])
                RES["LEAD5:%s-%s" % (b, a)] = {"ttft": dt, "pre_entry": dp, "queue": dq, "rest": dt - dp - dq}
                print("     %-24s %+8.4f = %+8.4f %+8.4f %+8.4f" % ("%s-%s" % (b, a), dt, dp, dq, dt - dp - dq))
    print()

print("=== entry - send against prompt length (jit_exact and nofc_noend, all kept returning turns): median, p90 by prompt tokens")
for a in ("jit_exact", "nofc_noend", "baseline"):
    b = {}
    for k, t in KM[a].items():
        if k[1] == 0:
            continue
        p = t["prompt_tokens"]
        key = "<8K" if p < 8000 else "8-16K" if p < 16000 else "16-32K" if p < 32000 else "32-64K" if p < 64000 else ">=64K"
        b.setdefault(key, []).append(pre(a, k))
    print("   %-12s " % a + "  ".join("%s: n=%d med=%.3f p90=%.3f" % (kk, len(b[kk]), pct(b[kk], 0.5), pct(b[kk], 0.9))
                                    for kk in ("<8K", "8-16K", "16-32K", "32-64K", ">=64K") if kk in b))

print()
print("=== when do the long ones (entry - send over 1 s) happen?  per arm: count, in how many 60-s groups, largest group")
for a in LEAD5 + ["baseline", "jit_p50_ctl"]:
    v = sorted((t["t_send"], pre(a, k)) for k, t in KM[a].items() if k[1] > 0 and pre(a, k) > 1.0)
    groups = []
    for ts, x in v:
        if groups and ts - groups[-1][-1] <= 60.0:
            groups[-1].append(ts)
        else:
            groups.append([ts])
    big = max((len(g) for g in groups), default=0)
    RES["long_pre:%s" % a] = {"count": len(v), "groups_60s": len(groups), "largest_group": big}
    print("   %-14s %4d turns in %3d groups, largest %3d; first minutes of the five largest groups: %s" % (
        a, len(v), len(groups), big, [round(g[0] / 60, 1) for g in sorted(groups, key=len, reverse=True)[:5]]))

with open(os.path.join(HERE, "item5i_pre_entry.json"), "w") as f:
    json.dump(RES, f, indent=1, sort_keys=True)
print()
print("wrote item5i_pre_entry.json")
