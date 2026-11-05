"""Read-only: TTFT of turns whose p50 plan expired, against the same turns with exact hints (128 users, 160 GB).

Usage: python3 -I -B expired_c128.py <results_dir>
"""
import datetime
import json
import os
import re
import sys

R = sys.argv[1]
ARMS = {
    "baseline": "20261018_103544_three_tier_wc_NAT160",
    "oracle": "20261018_154254_three_tier_wc_oracle_v2_nu_NAT160",
    "old_p50": "20261031_204526_three_tier_wc_oracle_v2_nu_NAT160",
    "new_exact": "20261102_002657_three_tier_wc_oracle_v2_nu_owned_NAT160",
    "new_p50": "20261102_023031_three_tier_wc_oracle_v2_nu_owned_NAT160",
}
RID = re.compile(r"-c(\d+)-t(\d+)$")
TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\]")


def client(arm):
    d = os.path.join(R, ARMS[arm])
    sub = [x for x in os.listdir(d) if x.startswith("client_")][0]
    t0 = None
    turns = {}
    with open(os.path.join(d, sub, "client.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r.get("kind") == "run":
                t0 = r["t_start"]
            elif r.get("kind") == "turn" and r.get("ttft") is not None:
                turns[(r["conv"], r["turn"])] = r
    return t0, turns


def expiries(arm):
    out = {}
    with open(os.path.join(R, ARMS[arm], "server.log"), errors="replace") as f:
        for line in f:
            if "ntp_expire" not in line:
                continue
            m = TS.match(line)
            rid = re.search(r"rid=(\S+)", line).group(1)
            k = RID.search(rid)
            ts = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f").replace(
                tzinfo=datetime.timezone.utc).timestamp()
            staged = "staged=1" in line
            out[(int(k.group(1)), int(k.group(2)))] = (ts, staged)
    return out


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


data = {a: client(a) for a in ARMS}
for p50_arm, exact_arm in (("old_p50", "oracle"), ("new_p50", "new_exact")):
    t0, turns = data[p50_arm]
    exp = expiries(p50_arm)
    for window in (7200.0, 1e9):
        rows = []
        for key, (ts, staged) in exp.items():
            r = turns.get(key)
            if r is None:
                continue
            send = t0 + r["t_send"]
            if send < ts or r["t_send"] > window:
                continue
            others = {a: data[a][1].get(key) for a in ARMS}
            if any(v is None for v in others.values()) and window < 1e8:
                pass
            rows.append((key, staged, r, others))
        both = [x for x in rows if x[3][exact_arm] is not None]
        allfive = [x for x in rows if all(v is not None for v in x[3].values())]
        w = "first 2 h" if window < 1e8 else "whole run"
        print(f"{p50_arm} ({w}): expiries {len(exp)}, turns sent after expiry {len(rows)}, "
              f"of them also in {exact_arm} {len(both)}")
        if both:
            print(f"   mean TTFT {p50_arm} {mean([x[2]['ttft'] for x in both]):.2f} s, "
                  f"{exact_arm} {mean([x[3][exact_arm]['ttft'] for x in both]):.2f} s; "
                  f"true gap median {sorted(x[2]['gap_slept'] for x in both)[len(both)//2]:.0f} s; "
                  f"SSD tokens {mean([(x[2].get('cached_details') or {}).get('storage', 0) for x in both]):.0f} vs "
                  f"{mean([(x[3][exact_arm].get('cached_details') or {}).get('storage', 0) for x in both]):.0f}")
        if allfive:
            print(f"   on {len(allfive)} also in all five arms: " + ", ".join(
                f"{a} {mean([x[3][a]['ttft'] for x in allfive]):.2f}" for a in ARMS))
