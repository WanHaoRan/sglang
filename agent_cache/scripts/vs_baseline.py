#!/usr/bin/env python3
"""vs_baseline.py --c12 M --base M --base-label c7 --iostat-c12 LOG --iostat-base LOG --out DIR

Whole-run statistics of one campaign-12 scale against its burst baseline (same arms): turns, errors, capped
conversations, wall, time by which 90 % of conversations ended, returning-turn TTFT, server queue wait
(ReqTimeStats queue_duration), where returning turns found their prefix, SSD traffic (iostat over the client window);
then paired TTFT differences for the same (conversation, turn): campaign 12 minus baseline per arm, and within each
campaign three_tier_to - hbm_host, three_tier_wc - three_tier_to, hbm_lru - hbm_host. Writes stats.csv and paired.csv.
"""
import argparse, csv, datetime, json, os, random, re, statistics as st

SH = 7808  # shared system prompt, tokens (cached for every returning turn)
TIERS = ("device", "host", "storage", "recompute")
TSL = re.compile(r"ReqTimeStats\(rid=([^,]+),.*?queue_duration=([0-9.]+)(ms|s),")
STAMP = re.compile(r"(\d\d/\d\d/\d{2,4}) (\d\d:\d\d:\d\d)(?: ([AP]M))?\s*$")


def tier_of(r):
    cd = r.get("cached_details") or {}
    p = r.get("prompt_tokens") or 1
    if p - (r.get("cached_tokens") or 0) > 0.5 * max(p - SH, 1):
        return "recompute"
    return "storage" if (cd.get("storage") or 0) > 0 else "host" if (cd.get("host") or 0) > 0 else "device"


def q(xs, f):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(f * len(xs)))] if xs else float("nan")


def boot_ci(d, n=2000, seed=0):
    rng, k = random.Random(seed), len(d)
    ms = sorted(sum(rng.choices(d, k=k)) / k for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n) - 1]


def load_iostat(path):
    """(epoch, rkB/s, wkB/s, %util) per 10-s record; skips each sampler's first record (averages since boot)."""
    recs, t, cols, skip = [], None, None, False
    for line in open(path, errors="replace"):
        if line.startswith("Linux "):
            skip = True
            continue
        m = STAMP.match(line)
        if m:
            day = "%m/%d/%Y" if len(m[1]) == 10 else "%m/%d/%y"
            fmt, s = (day + " %I:%M:%S %p", f"{m[1]} {m[2]} {m[3]}") if m[3] else (day + " %H:%M:%S", f"{m[1]} {m[2]}")
            t = datetime.datetime.strptime(s, fmt).replace(tzinfo=datetime.timezone.utc).timestamp()
        elif line.startswith("Device"):
            cols = line.split()
        elif line.startswith("vdc") and t is not None and cols:
            if skip:
                skip = False
                continue
            v = line.split()
            recs.append((t, float(v[cols.index("rkB/s")]), float(v[cols.index("wkB/s")]), float(v[cols.index("%util")])))
    return recs


def load_arm(results, arm, run, client, iostat):
    rows = [json.loads(l) for l in open(os.path.join(results, run, client, "client.jsonl")) if l.strip()]
    run_rec = next(r for r in rows if r.get("kind") == "run")
    summ = next((r for r in rows if r.get("kind") == "summary"), {})
    tag = run_rec.get("tag") or f"cmp_{arm}"
    turns = [r for r in rows if r.get("kind") == "turn" and "t_send" in r]
    ends = {r["conv"]: r["end_s"] for r in rows if r.get("kind") == "conv_end" and "end_s" in r}
    last = {}
    for r in turns:
        if r.get("latency") is not None:
            last[r["conv"]] = max(last.get(r["conv"], 0.0), r["t_send"] + r["latency"])
    for c, t in last.items():
        ends.setdefault(c, t)   # runs that predate conv_end.end_s: a conversation ends with its last reply
    queue = {}
    for line in open(os.path.join(results, run, "server.log"), errors="replace"):
        if "ReqTimeStats(rid=" + tag in line:
            m = TSL.search(line)
            if m:
                queue[m[1]] = float(m[2]) / (1000.0 if m[3] == "ms" else 1.0)
    ret = {(r["conv"], r["turn"]): r for r in turns if r["turn"] > 0 and r.get("ttft")}
    t0, wall = run_rec["t_start"], summ.get("elapsed_s") or max(ends.values())
    io = [x for x in iostat if t0 < x[0] <= t0 + wall + 10] if iostat else []
    tt = [r["ttft"] for r in ret.values()]
    qq = [queue[f"{tag}-c{c}-t{k}"] for (c, k) in ret if f"{tag}-c{c}-t{k}" in queue]
    row = dict(arm=arm, turns=summ.get("turns", len(turns)), errors=summ.get("errors", ""),
               capped_convs=summ.get("capped_convs", ""), wall_s=round(wall, 1),
               conv90_min=round(q(list(ends.values()), .9) / 60, 1), returning=len(tt),
               ttft_mean=round(st.mean(tt), 3), ttft_p50=round(q(tt, .5), 3), ttft_p90=round(q(tt, .9), 3),
               ttft_p99=round(q(tt, .99), 3), queue_mean=round(st.mean(qq), 3) if qq else "", queue_n=len(qq),
               uncached_m=round(sum((r.get("prompt_tokens") or 0) - (r.get("cached_tokens") or 0) for r in turns) / 1e6, 2),
               **{f"n_{k}": sum(tier_of(r) == k for r in ret.values()) for k in TIERS},
               ssd_read_tb=round(sum(x[1] for x in io) * 10 / 1e9, 2) if io else "",
               ssd_written_tb=round(sum(x[2] for x in io) * 10 / 1e9, 2) if io else "",
               util_mean=round(st.mean(x[3] for x in io), 1) if io else "",
               util_p90=round(q([x[3] for x in io], .9), 1) if io else "", util_max=round(max(x[3] for x in io), 1) if io else "",
               iostat_cover=round(len(io) * 10 / wall, 2) if io else 0.0)
    return row, {k: r["ttft"] for k, r in ret.items()}


def load(manifest, iostat_path):
    results = os.path.dirname(os.path.dirname(os.path.abspath(manifest)))
    iostat = load_iostat(iostat_path) if iostat_path else []
    out = {}
    for arm, run, client in (l.split() for l in open(manifest) if l.strip()):
        out[arm] = load_arm(results, arm, run, client, iostat)
    return out


def paired(label, a, b, ta, tb):
    keys = sorted(set(ta) & set(tb))
    d = [ta[k] - tb[k] for k in keys]
    if len(d) < 2:
        return None
    lo, hi = boot_ci(d)
    return dict(comparison=label, a=a, b=b, pairs=len(d), diff_mean=round(st.mean(d), 2), ci_lo=round(lo, 2),
                ci_hi=round(hi, 2), diff_p50=round(q(d, .5), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c12", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--base-label", required=True)
    ap.add_argument("--iostat-c12", default="")
    ap.add_argument("--iostat-base", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    c12, base = load(a.c12, a.iostat_c12), load(a.base, a.iostat_base)
    rows = [dict(campaign="c12", **r) for r, _ in c12.values()] + [dict(campaign=a.base_label, **r) for r, _ in base.values()]
    with open(os.path.join(a.out, "stats.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(" ".join(f"{k}={v}" for k, v in r.items()))
    prs = []
    for arm in c12:
        if arm in base:
            prs.append(paired(f"c12 - {a.base_label}", f"c12 {arm}", f"{a.base_label} {arm}", c12[arm][1], base[arm][1]))
    for lab, camp in (("c12", c12), (a.base_label, base)):
        for x, y in (("three_tier_to", "hbm_host"), ("three_tier_wc", "three_tier_to"), ("hbm_lru", "hbm_host")):
            if x in camp and y in camp:
                prs.append(paired(f"{lab}: {x} - {y}", f"{lab} {x}", f"{lab} {y}", camp[x][1], camp[y][1]))
    prs = [p for p in prs if p]
    with open(os.path.join(a.out, "paired.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prs[0].keys()))
        w.writeheader()
        w.writerows(prs)
    for p in prs:
        print(f"{p['a']:22s} - {p['b']:22s} pairs {p['pairs']:5d}  mean {p['diff_mean']:+8.2f} [{p['ci_lo']:+8.2f}, {p['ci_hi']:+8.2f}]  p50 {p['diff_p50']:+7.2f}")


if __name__ == "__main__":
    main()
