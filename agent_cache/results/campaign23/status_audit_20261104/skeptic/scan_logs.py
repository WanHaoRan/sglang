"""Read-only scan of server logs: arrival offset d from ntp_plan, expiry timing, long admission waits."""
import re, sys, statistics as st
from datetime import datetime

R = "/home/wanhr/sglang/agent_cache/results/"
RUNS = {
    "jit_exact_c44": "20261101_202142_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70",
    "jit_p50_c44": "20261101_222451_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70",
    "jit_exact_c128": "20261102_002657_three_tier_wc_oracle_v2_nu_owned_NAT160",
    "jit_p50_c128": "20261102_023031_three_tier_wc_oracle_v2_nu_owned_NAT160",
    "table_smoke": "20261101_194908_three_tier_wc_oracle_v2_nu_table_custom_L1nat_H70",
    "c21_oracle": "20261101_102434_three_tier_wc_oracle_v2_nu_custom_L1nat_H70",
    "c21_p50": "20261101_062014_three_tier_wc_oracle_v2_nu_custom_L1nat_H70",
    "jit_70": "20261101_050233_three_tier_wc_oracle_v2_nu_jit_custom_L1nat_H70",
}
TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\]")
def ts(line):
    m = TS.match(line)
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f").timestamp() if m else None

for name, d in RUNS.items():
    t0 = None
    ds, gaps = [], []
    exp_t = []
    admit_waits = []
    first_plan_t = None
    with open(R + d + "/server.log", errors="replace") as f:
        for line in f:
            if "HICACHE_EVT" not in line:
                continue
            if " ntp_plan " in line:
                t = ts(line)
                if first_plan_t is None:
                    first_plan_t = t
                g = int(re.search(r"gap_ms=(\d+)", line).group(1))
                p = int(re.search(r"pred_arr_ms=(-?\d+)", line).group(1))
                ds.append((p - g) / 1000.0)
            elif " ntp_expire " in line:
                exp_t.append((ts(line), "reason=timeout" in line))
            elif " ntp_admit " in line:
                m = re.search(r"wait_ms=(\d+)", line)
                if m:
                    admit_waits.append(int(m.group(1)) / 1000.0)
    # client start/end from client.log
    import glob
    cl = glob.glob(R + d + "/client*/client.log")
    t_end = None
    if first_plan_t is not None and exp_t:
        rel = [(t - first_plan_t) / 60.0 for t, _ in exp_t]
    else:
        rel = []
    q = lambda xs, p: sorted(xs)[int(p * (len(xs) - 1))] if xs else float("nan")
    print(f"== {name}: plans {len(ds)}  d median {st.median(ds) if ds else float('nan'):.3f}  p90 {q(ds,0.9):.3f}  max {max(ds) if ds else float('nan'):.3f}")
    n_to = sum(1 for _, to in exp_t if to)
    before118 = sum(1 for (t, to), r in zip(exp_t, rel) if to and r < 118)
    print(f"   expire {len(exp_t)} (timeout {n_to}); timeout before 118 min after first plan: {before118}; last expire at {max(rel) if rel else float('nan'):.1f} min")
    print(f"   admits {len(admit_waits)}; wait>=60 s: {sum(1 for w in admit_waits if w >= 60)}; max {max(admit_waits) if admit_waits else float('nan'):.1f} s")
