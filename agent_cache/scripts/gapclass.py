#!/usr/bin/env python3
"""gapclass.py --manifest compare_<t>/manifest.txt [--window 1800,7200] [--out DIR] [--boot 2000]

Per arm and per gap class (the idle time before a returning turn: <1 min, 1-5 min, 5-10 min, >=10 min), inside the
steady-state window of client time: number of turns, where the prefix was found (device / host / storage / recompute),
TTFT mean (95% bootstrap CI) / p50 / p90, and the server-side queue wait (ReqTimeStats queue_duration, when logged).
Then the same (conv, turn) pairs compared across arms, against the first arm in the manifest.
Writes gapclass.csv and gapclass_paired.csv next to the manifest (or to --out).
"""
import argparse
import csv
import json
import os
import random
import re
import statistics as st

SH = 7808  # shared system prompt, tokens (cached for every returning turn)
CLASSES = [("<1 min", 0, 60), ("1-5 min", 60, 300), ("5-10 min", 300, 600), (">=10 min", 600, 1e12)]
TIERS = ("device", "host", "storage", "recompute")
TSL = re.compile(r"ReqTimeStats\(rid=([^,]+),.*?queue_duration=([0-9.]+)(ms|s),")


def tier_of(r):
    cd = r.get("cached_details") or {}
    p = r.get("prompt_tokens") or 1
    if p - (r.get("cached_tokens") or 0) > 0.5 * max(p - SH, 1):
        return "recompute"
    return "storage" if (cd.get("storage") or 0) > 0 else "host" if (cd.get("host") or 0) > 0 else "device"


def q(xs, f):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(f * len(xs)))] if xs else float("nan")


def boot_ci(xs, n, rng):
    if len(xs) < 2:
        return (float("nan"), float("nan"))
    ms = sorted(st.mean(rng.choice(xs) for _ in xs) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n) - 1]


def load(results, arm, run, client):
    rows = [json.loads(l) for l in open(os.path.join(results, run, client, "client.jsonl")) if l.strip()]
    queue = {}
    for line in open(os.path.join(results, run, "server.log"), errors="replace"):
        if "queue_duration=" in line:
            m = TSL.search(line)
            if m:
                queue[m[1]] = float(m[2]) / (1000.0 if m[3] == "ms" else 1.0)
    out = {}
    for r in rows:
        if r.get("kind") != "turn" or "t_send" not in r or r["turn"] == 0 or not r.get("ttft"):
            continue
        rid = f"cmp_{arm}-c{r['conv']}-t{r['turn']}"
        out[(r["conv"], r["turn"])] = dict(t=r["t_send"], gap=r.get("gap_slept") or 0.0, ttft=r["ttft"], tier=tier_of(r),
                                           queue=queue.get(rid))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--window", default="1800,7200", help="client-time window in seconds (start,end)")
    ap.add_argument("--out", default="")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    w0, w1 = (float(x) for x in a.window.split(","))
    results = os.path.dirname(os.path.dirname(os.path.abspath(a.manifest)))
    out = a.out or os.path.dirname(os.path.abspath(a.manifest))
    arms = [l.split() for l in open(a.manifest) if l.strip()]
    data = {arm: load(results, arm, run, client) for arm, run, client in arms}
    rng = random.Random(0)
    rows = []
    print(f"window {w0:.0f}-{w1:.0f} s of client time")
    for arm, _, _ in arms:
        T = [v for v in data[arm].values() if w0 <= v["t"] < w1]
        for name, lo, hi in [("all", 0, 1e12)] + CLASSES:
            c = [v for v in T if lo <= v["gap"] < hi]
            if not c:
                continue
            tt = [v["ttft"] for v in c]
            qq = [v["queue"] for v in c if v["queue"] is not None]
            lo_ci, hi_ci = boot_ci(tt, a.boot, rng)
            row = dict(arm=arm, gap_class=name, n=len(c), **{f"share_{k}": round(sum(v["tier"] == k for v in c) / len(c), 4) for k in TIERS},
                       ttft_mean=round(st.mean(tt), 3), ttft_mean_ci_lo=round(lo_ci, 3), ttft_mean_ci_hi=round(hi_ci, 3),
                       ttft_p50=round(q(tt, .5), 3), ttft_p90=round(q(tt, .9), 3),
                       queue_mean=round(st.mean(qq), 3) if qq else "", queue_measured=len(qq))
            rows.append(row)
            shares = " ".join(f"{k[:3]} {100 * row['share_' + k]:5.1f}%" for k in TIERS)
            print(f"{arm:14s} {name:9s} n {len(c):5d}  {shares}  TTFT mean {row['ttft_mean']:7.2f} [{lo_ci:6.2f},{hi_ci:6.2f}]"
                  f" p50 {row['ttft_p50']:6.2f} p90 {row['ttft_p90']:7.2f}" + (f"  queue {row['queue_mean']:6.2f}" if qq else ""))
    with open(os.path.join(out, "gapclass.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    # paired: the same (conv, turn) in the reference arm and each other arm, both inside the window
    ref = arms[0][0]
    prow = []
    print(f"\npaired against {ref} (same conversation and turn, both in the window)")
    for arm, _, _ in arms[1:]:
        keys = [k for k, v in data[arm].items() if k in data[ref] and w0 <= v["t"] < w1 and w0 <= data[ref][k]["t"] < w1]
        for name, lo, hi in [("all", 0, 1e12)] + CLASSES:
            ks = [k for k in keys if lo <= data[ref][k]["gap"] < hi]
            if len(ks) < 2:
                continue
            d = [data[arm][k]["ttft"] - data[ref][k]["ttft"] for k in ks]
            lo_ci, hi_ci = boot_ci(d, a.boot, rng)
            prow.append(dict(arm=arm, ref=ref, gap_class=name, pairs=len(ks), diff_mean=round(st.mean(d), 3),
                             diff_ci_lo=round(lo_ci, 3), diff_ci_hi=round(hi_ci, 3), diff_p50=round(q(d, .5), 3)))
            print(f"{arm:14s} {name:9s} pairs {len(ks):5d}  TTFT {arm} - {ref}: mean {st.mean(d):+8.2f} s [{lo_ci:+7.2f},{hi_ci:+7.2f}]  p50 {q(d, .5):+7.2f}")
    if prow:
        with open(os.path.join(out, "gapclass_paired.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(prow[0].keys())); wr.writeheader(); wr.writerows(prow)
    print("wrote", os.path.join(out, "gapclass.csv"))


if __name__ == "__main__":
    main()
