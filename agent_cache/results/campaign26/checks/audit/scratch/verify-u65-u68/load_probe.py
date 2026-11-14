#!/usr/bin/env python3
"""Read-only feasibility probe for audit item U67 (NOT a result for the record).
Applies the method of frozen/oracle_scripts_staging/c23_engine/load_curve_c44.py (per-minute load = context of live
sessions not parked in a gap of 60 s or more; band means from minute 20; log interpolation between band means with
3 minutes or more) to campaign 26's 44-user runs, from client.jsonl only. Standard library; reads, writes nothing
outside stdout. Run from anywhere:  nice -n 10 python3 -I -B load_probe.py
"""
import bisect, collections, json, math, os, statistics as st

RES = "/home/wanhr/sglang/agent_cache/results"
LONG, WARM = 60, 20
BANDS = ((0, 800), (800, 900), (900, 1000), (1000, 1050), (1050, 1100), (1100, 1200), (1200, 1400))
RUNS = [("baseline", "20261101_082229_three_tier_wc_custom_L1nat_H70", None),
        ("base_rep", "20261105_075438_three_tier_wc_custom_L1nat_H70", None),
        ("hintfree", "20261101_122705_three_tier_wc_dl_poll0_custom_L1nat_H70", None),
        ("oracle_v2", "20261101_102434_three_tier_wc_oracle_v2_nu_custom_L1nat_H70", None),
        ("jit_p50", "20261101_222451_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70", None),
        ("jit_exact", "20261101_202142_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70", None)]
for line in open(os.path.join(RES, "campaign26", "manifest.txt")):
    f = line.split()
    if f and not f[0].endswith("_c128"):
        RUNS.append((f[0], f[1], f[2]))
RUNS.append(("nofc_noend_h100_cut", "20261113_060157_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H100", None))
GROUPS = collections.OrderedDict([
    ("original HiCache (2 runs)", ("baseline", "base_rep")),
    ("hint-free, read pins (2 runs)", ("hintfree", "hintfree_ctl")),
    ("old oracle v2 (1 run)", ("oracle_v2",)),
    ("JIT p50 (3 runs)", ("jit_p50", "jit_p50_ctl", "jit_p50_d")),
    ("rule 3 alone (1 run)", ("ec_r3",)),
    ("fq90 (1 run)", ("fq90",)),
    ("rule 1 alone (1 run)", ("ec_r1",)),
    ("rules 1+3 (2 runs)", ("ec_r13", "ec_r13_rep")),
    ("rules 1+3+4 (1 run)", ("ec_r134",)),
    ("all four rules (1 run)", ("ec_full",)),
    ("no gap forecast (2 runs)", ("nofc", "nofc_rep")),
    ("no hints (2 runs)", ("nofc_noend", "nofc_noend_rep")),
    ("JIT exact (2 runs)", ("jit_exact", "jit_exact_d")),
    ("no hints, 100 GB, cut at 84 min", ("nofc_noend_h100_cut",)),
])


def load(run, client):
    if client is None:
        client = sorted(d for d in os.listdir(os.path.join(RES, run)) if d.startswith("client_"))[-1]
    recs = [json.loads(line) for line in open(os.path.join(RES, run, client, "client.jsonl")) if line.strip()]
    return sorted((r for r in recs if r.get("kind") == "turn" and r.get("ttft") is not None), key=lambda r: r["t_send"])


def sessions(turns):
    conv = collections.defaultdict(list)
    for r in turns:
        conv[r["conv"]].append(r)
    return [(v, [r["t_send"] for r in v]) for v in conv.values()]


def state_at(convs, t):
    active = 0.0
    for v, sends in convs:
        if not sends[0] <= t < sends[-1] + v[-1]["latency"]:
            continue
        i = bisect.bisect_right(sends, t) - 1
        size = ((v[i].get("prompt_tokens") or 0) + (v[i].get("completion_tokens") or 0)) / 1000
        now = t < sends[i] + v[i]["latency"]
        if not now and i + 1 < len(v) and (v[i + 1].get("gap_slept") or 0) >= LONG:
            continue
        active += size
    return active


def minutes(turns, convs):
    by = collections.defaultdict(list)
    for r in turns:
        by[int(r["t_send"] // 60)].append(r["ttft"])
    rows = []
    for m in range(int(turns[-1]["t_send"] // 60)):
        if by[m]:
            rows.append(dict(m=m, ttft=st.mean(by[m]), turns=len(by[m]), load=state_at(convs, m * 60 + 30)))
    return rows


M = {}
for name, run, client in RUNS:
    try:
        t = load(run, client)
    except Exception as e:  # a missing run is reported, not fatal
        print("skip", name, repr(e)[:120])
        continue
    M[name] = minutes(t, sessions(t))
    print(f"loaded {name:20s} turns {len(t):6d} minutes {len(M[name]):4d} max load {max(r['load'] for r in M[name]):6.0f} K")


def band_table(names):
    table = []
    for lo, hi in BANDS:
        x = [r for n in names if n in M for r in M[n] if r["m"] >= WARM and lo <= r["load"] < hi]
        if x:
            table.append(dict(lo=lo, hi=hi, mid=st.mean(r["load"] for r in x), minutes=len(x), ttft=st.mean(r["ttft"] for r in x)))
    return table


def at_target(table):
    rows = [r for r in table if r["minutes"] >= 3]
    out = []
    for target in (1, 2, 3, 5, 10):
        hit = next((a["mid"] + (b["mid"] - a["mid"]) * (math.log(target) - math.log(a["ttft"])) / (math.log(b["ttft"]) - math.log(a["ttft"]))
                    for a, b in zip(rows, rows[1:]) if a["ttft"] < target <= b["ttft"]), None)
        out.append(f"{target} s: {hit:5.0f}" if hit else f"{target} s:  none")
    return "  ".join(out)


print("\nload band K: minutes, mean TTFT per band, by group, from minute", WARM)
print(f"  {'':34s}" + "".join(f"{lo:>6d}-{hi:<6d}" for lo, hi in BANDS))
for g, names in GROUPS.items():
    tb = {(r["lo"], r["hi"]): r for r in band_table(names)}
    print(f"  {g:34s}" + "".join((f"{tb[b]['minutes']:4d} {tb[b]['ttft']:6.2f}  " if b in tb else " " * 13) for b in BANDS))

print("\nload K at which mean TTFT passes a target, by group (bands with 3 minutes or more; rough, see load_curve_c44.py)")
for g, names in GROUPS.items():
    print(f"  {g:34s} {at_target(band_table(names))}")

print("\nthe same, run by run")
for name, _, _ in RUNS:
    if name in M:
        print(f"  {name:34s} {at_target(band_table((name,)))}")
