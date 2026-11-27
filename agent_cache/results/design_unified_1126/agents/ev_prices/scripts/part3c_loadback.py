"""Part 3 check: time after admission beyond the prefill of the new tokens (host-to-GPU load), at light load.
usage: part3c_loadback.py"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402
import qstate  # noqa: E402
import readers  # noqa: E402

FIT = json.load(open(f"{lib.OUT}/prefill_fit.json"))


def t_fit(f, n, p):
    return f["a_s"] + f["b_s_per_tok"] * n + f["c_s_per_tok_Mtok"] * n * (p + n / 2) / 1e6


print("Time after admission beyond the prefill of the new tokens (TTFT - queue wait - idle-fit prefill), light load "
      "(pool use < 0.90, no ready turn waiting at arrival)")
for camp, arm, hi in (("c34", "ec_full", 28), ("c34", "hicache", 28), ("c33", "hicache", 110), ("c33", "jit_p50", 110), ("c33", "ec_full", 110)):
    f = FIT[f"{camp}:{arm}"]["idle"]
    st = qstate.build(camp, arm)
    rows = readers.annotate(st)
    grp = {"device only": [], "host hit, no SSD": [], "SSD read": []}
    for r in rows:
        if r["k"] == 0 or r["arr"] >= hi * 60 or r["ttft"] is None or r["q_rdy_arr"] > 0 or r["use_arr"] >= 0.90:
            continue
        cd = r["cd"] or {}
        new = r["in_len"] - r["cached_srv"]
        res = r["ttft"] - r["queue"] - t_fit(f, new, r["cached_srv"])
        h, s = cd.get("host") or 0, cd.get("storage") or 0
        if r["n_reads"] > 0:
            grp["SSD read"].append((res, h + s, s))
        elif h > 0:
            grp["host hit, no SSD"].append((res, h, 0))
        elif not r["reads"]:
            grp["device only"].append((res, 0, 0))
    print(f"{camp}:{arm} to minute {hi}")
    for k, v in grp.items():
        if v:
            print(f"   {k:18s} n {len(v):5d}: residual mean {lib.mean(x[0] for x in v)*1e3:7.1f} ms p50 {lib.quantile([x[0] for x in v], .5)*1e3:6.1f} "
                  f"p90 {lib.quantile([x[0] for x in v], .9)*1e3:6.1f}; tokens loaded from host memory mean {lib.mean(x[1] for x in v)/1e3:5.1f} K")
