"""Referee: the anchors, measured (referee/measured.json, from ref_measured.py) against simulated (the referee's own
runs under referee/runs/anchors: seed 0 = the campaigns' users, seeds 5-12 = fresh users).

  python -I -B ref_anchors.py [runs folder] > anchors_table.md
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import reflib  # noqa: E402

RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "runs", "anchors")
SIMFILE = sys.argv[2] if len(sys.argv) > 2 else "kvsim.py"
FRESH = (5, 6, 7, 8, 9, 10, 11, 12)
ARMS = ("hicache", "jit_p50", "jit_exact", "ec_full", "ec_exact")
W = ("0-30", "30-60", "60-90", "90-110")


def fmt(x, nd=2):
    if x is None:
        return "none"
    if isinstance(x, str):
        return x
    return f"{x:.{nd}f}"


def rel(m, s):
    if m is None or s is None or isinstance(m, str) or isinstance(s, str):
        return ""
    if abs(m) < 1e-12:
        return "" if abs(s) < 1e-12 else "(measured 0)"
    return f"{100 * (s - m) / abs(m):+.0f} %"


def verdict(m, s, tol_ok=0.10, tol_rough=0.30):
    if m is None or s is None or isinstance(m, str) or isinstance(s, str):
        return "reproduced" if m == s else "not_reproduced"
    if abs(m) < 1e-12:
        return "reproduced" if abs(s) < 1e-12 else "not_reproduced"
    e = abs(s - m) / abs(m)
    return "reproduced" if e <= tol_ok else ("roughly" if e <= tol_rough else "not_reproduced")


def spread(vals, nd=2):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "none"
    a = np.array(vals, float)
    return f"{a.mean():.{nd}f} ± {a.std(ddof=1) if len(a) > 1 else 0:.{nd}f} [{a.min():.{nd}f}, {a.max():.{nd}f}]"


def main() -> int:
    with open(os.path.join(HERE, "measured.json")) as f:
        meas = json.load(f)
    sims = {}
    for setting in ("default", "fixes=off"):
        for arm in ARMS:
            for seed in (0,) + FRESH:
                folder = os.path.join(RUNS, setting, arm, f"seed{seed}")
                if os.path.exists(os.path.join(folder, "summary.json")):
                    sims[(setting, arm, seed)] = reflib.sim_stats(folder)
    rows_json = []

    def row(table, anchor, m, s0, fresh, nd=2, tol=(0.10, 0.30), v=None):
        vd = v or verdict(m, s0, *tol)
        rows_json.append({"table": table, "anchor": anchor, "measured": m, "sim_seed0": s0, "verdict": vd,
                          "fresh": [x for x in fresh if x is not None]})
        print(f"| {anchor} | {fmt(m, nd)} | {fmt(s0, nd)} | {rel(m, s0)} | {spread(fresh, nd)} | {vd} |")

    def head(title):
        print(f"\n### {title}\n")
        print("| anchor | measured | simulated, seed 0 | error | simulated, fresh seeds 5-12: mean ± sd [min, max] | verdict |")
        print("|---|---|---|---|---|---|")

    for camp, setting, label in (("campaign34", "default", "Campaign 34 (fixes on)"),
                                 ("campaign32", "fixes=off", "Campaign 32 (fixes off)")):
        head(f"{label}: client side, requests sent before 6,600 s")
        for arm in ARMS:
            m = meas[f"{camp}/{arm}"]
            s0 = sims[(setting, arm, 0)]
            fr = [sims[(setting, arm, k)] for k in FRESH if (setting, arm, k) in sims]
            row(label, f"{arm}: mean TTFT, s", m["ttft_mean"], s0["ttft_mean"], [x["ttft_mean"] for x in fr])
            for i, w in enumerate(W):
                row(label, f"{arm}: mean TTFT, minutes {w}", m["windows"][i]["ttft_mean"], s0["windows"][i]["ttft_mean"],
                    [x["windows"][i]["ttft_mean"] for x in fr if x["windows"][i]])
            row(label, f"{arm}: p50 TTFT, s", m["ttft_p50"], s0["ttft_p50"], [x["ttft_p50"] for x in fr])
            row(label, f"{arm}: p90 TTFT, s", m["ttft_p90"], s0["ttft_p90"], [x["ttft_p90"] for x in fr])
            row(label, f"{arm}: requests per minute to 6,600 s", m["req_per_min"], s0["req_per_min"],
                [x["req_per_min"] for x in fr], 1, (0.03, 0.10))
            row(label, f"{arm}: requests per minute, minutes 90-109", m["req_per_min_90_109"], s0["req_per_min_90_109"],
                [x["req_per_min_90_109"] for x in fr], 1, (0.05, 0.15))
            row(label, f"{arm}: users with a request", m["users"], s0["users"], [x["users"] for x in fr], 0, (0.03, 0.1))
            row(label, f"{arm}: prompt tokens from GPU, share", m["share_gpu"], s0["share_gpu"], [x["share_gpu"] for x in fr], 3)
            row(label, f"{arm}: from host, share", m["share_host"], s0["share_host"], [x["share_host"] for x in fr], 3)
            row(label, f"{arm}: from SSD, share", m["share_ssd"], s0["share_ssd"], [x["share_ssd"] for x in fr], 3)
            for i, w in enumerate(W[1:], 1):
                row(label, f"{arm}: from SSD, minutes {w}", m["windows"][i]["share_ssd"], s0["windows"][i]["share_ssd"],
                    [x["windows"][i]["share_ssd"] for x in fr if x["windows"][i]], 3)
            row(label, f"{arm}: history recomputed, share", m["history_recomputed"], s0["history_recomputed"],
                [x["history_recomputed"] for x in fr], 4, (0.15, 0.40))
            for i, w in enumerate(W[1:], 1):
                row(label, f"{arm}: history recomputed, M tokens, minutes {w}", m["windows"][i]["recomputed_M"],
                    s0["windows"][i]["recomputed_M"], [x["windows"][i]["recomputed_M"] for x in fr if x["windows"][i]],
                    2, (0.15, 0.40))
            row(label, f"{arm}: returning requests that read from the SSD, minutes 90-110", m["windows"][3]["returning_reading"],
                s0["windows"][3]["returning_reading"], [x["windows"][3]["returning_reading"] for x in fr if x["windows"][3]], 3)
            row(label, f"{arm}: their mean TTFT, s", m["windows"][3]["ttft_readers"], s0["windows"][3]["ttft_readers"],
                [x["windows"][3]["ttft_readers"] for x in fr if x["windows"][3]], 1)
            row(label, f"{arm}: the others' mean TTFT, s", m["windows"][3]["ttft_others"], s0["windows"][3]["ttft_others"],
                [x["windows"][3]["ttft_others"] for x in fr if x["windows"][3]], 1)
        head(f"{label}: server side")
        for arm in ARMS:
            m = meas[f"{camp}/{arm}"]["server"]
            s0 = sims[(setting, arm, 0)]["server"]
            fr = [sims[(setting, arm, k)]["server"] for k in FRESH if (setting, arm, k) in sims]
            for w in ("30-60", "60-90", "90-110"):
                row(label, f"{arm}: running requests, median, minutes {w}", m["decode"][w]["running_median"],
                    s0["decode"][w]["running_median"], [x["decode"][w]["running_median"] for x in fr], 1, (0.05, 0.15))
            row(label, f"{arm}: running requests, mean, minutes 90-110", m["decode"]["90-110"]["running_mean"],
                s0["decode"]["90-110"]["running_mean"], [x["decode"]["90-110"]["running_mean"] for x in fr], 1, (0.05, 0.15))
            row(label, f"{arm}: queued requests, median, minutes 90-110", m["decode"]["90-110"]["queue_median"],
                s0["decode"]["90-110"]["queue_median"], [x["decode"]["90-110"]["queue_median"] for x in fr], 0)
            row(label, f"{arm}: queued requests, median, minutes 60-90", m["decode"]["60-90"]["queue_median"],
                s0["decode"]["60-90"]["queue_median"], [x["decode"]["60-90"]["queue_median"] for x in fr], 0)
            row(label, f"{arm}: GPU pool used, median, minutes 90-110", m["decode"]["90-110"]["usage_median"],
                s0["decode"]["90-110"]["usage_median"], [x["decode"]["90-110"]["usage_median"] for x in fr], 3, (0.03, 0.1))
            for w in ("30-60", "60-90", "90-110"):
                row(label, f"{arm}: read thread busy share, minutes {w}", m["io"][w]["read_busy"], s0["io"][w]["read_busy"],
                    [x["io"][w]["read_busy"] for x in fr], 3, (0.10, 0.30))
                row(label, f"{arm}: write thread busy share, minutes {w}", m["io"][w]["write_busy"], s0["io"][w]["write_busy"],
                    [x["io"][w]["write_busy"] for x in fr], 3, (0.10, 0.30))
            mc = [c[0] for c in m["cleaner_passes"]]
            sc = [c[0] for c in s0["cleaner_passes"]]
            row(label, f"{arm}: first cleaner pass, s", mc[0] if mc else None, sc[0] if sc else None,
                [x["cleaner_passes"][0][0] if x["cleaner_passes"] else None for x in fr], 0, (0.03, 0.10))
            row(label, f"{arm}: cleaner passes before 6,600 s", sum(1 for t in mc if t < 6600), sum(1 for t in sc if t < 6600),
                [sum(1 for c in x["cleaner_passes"] if c[0] < 6600) for x in fr], 0, (0.0, 0.5))
            if arm != "hicache":
                row(label, f"{arm}: reads refused host room before 6,600 s", m["dispatch"]["refused_before_cut"],
                    s0["refused_before_cut"], [x["refused_before_cut"] for x in fr], 0, (0.15, 0.40))
                row(label, f"{arm}: loop onset minute (3 minutes in a row with 5 or more refusals)", m["loop_onset_min_3"],
                    s0["loop_onset_min_3"], [x["loop_onset_min_3"] for x in fr], 0, (0.10, 0.30))
                row(label, f"{arm}: loop onset minute (5 in a row)", m["loop_onset_min_5"],
                    s0["loop_onset_min_5"], [x["loop_onset_min_5"] for x in fr], 0, (0.10, 0.30))
                row(label, f"{arm}: write hold switched on, times", m["write_holds_on"], s0["counters"]["write_hold_on"],
                    [x["counters"]["write_hold_on"] for x in fr], 0, (0.15, 0.40))
                row(label, f"{arm}: reads staged before arrival", m["counts"]["ntp_stage"], s0["counters"]["staged"],
                    [x["counters"]["staged"] for x in fr], 0, (0.10, 0.30))
            row(label, f"{arm}: SSD reads in the run", m["reads"]["n"], s0["counters"]["reads"],
                [x["counters"]["reads"] for x in fr], 0)
            row(label, f"{arm}: tokens read from the SSD, M (pages x 64)", m["reads"]["tokens_M"],
                s0["counters"]["read_tokens"] / 1e6, [x["counters"]["read_tokens"] / 1e6 for x in fr], 1)
            row(label, f"{arm}: tokens written to the SSD, M (submitted, incl. skipped)", m["writes"]["tokens_M"],
                (s0["counters"]["write_tokens"]) / 1e6, [x["counters"]["write_tokens"] / 1e6 for x in fr], 1)
            if "census" in m:
                c, sc2 = m["census"]["60-110"], s0["census"]["60-110"]
                frc = [x["census"]["60-110"] for x in fr]
                row(label, f"{arm}: host mirror, median of minutes 60-110, K", c["mirror"] / 1e3, sc2["host_mirror"] / 1e3,
                    [x["host_mirror"] / 1e3 for x in frc], 0, (0.03, 0.10))
                row(label, f"{arm}: host free, median, K", c["free"] / 1e3, sc2["host_free"] / 1e3,
                    [x["host_free"] / 1e3 for x in frc], 1, (0.15, 0.40))
                row(label, f"{arm}: pinned tokens (census 'pinned': pins of arrived and staged turns), median, K",
                    c["pinned"] / 1e3, sc2["host_pinned"] / 1e3, [x["host_pinned"] / 1e3 for x in frc], 0, (0.15, 0.40))
                row(label, f"{arm}: pinned tokens, p90, K", c["pinned_p90"] / 1e3, sc2["host_pinned_p90"] / 1e3,
                    [x["host_pinned_p90"] / 1e3 for x in frc], 0, (0.15, 0.40))
                row(label, f"{arm}: pending writes (census 'pending_write'), p90, K", c["pending_write_p90"] / 1e3,
                    sc2["host_dirty_p90"] / 1e3, [x["host_dirty_p90"] / 1e3 for x in frc], 1, (0.15, 0.40))
                row(label, f"{arm}: host-only room held by pending writes (census 'h_write', with ancestors; sim: dirty "
                    f"host-only tokens), p90, K", c["h_write_p90"] / 1e3, sc2["host_dirty_only_p90"] / 1e3,
                    [x["host_dirty_only_p90"] / 1e3 for x in frc], 1, (0.15, 0.40))
    # paired differences
    print("\n### Paired differences between arms (B - A on the requests both sent before 6,600 s; interval over users)\n")
    print("| campaign | pair | measured (referee's recomputation) | simulated, seed 0 | simulated, fresh seeds: mean [min, max] | sign over the 9 seeds | verdict |")
    print("|---|---|---|---|---|---|---|")
    pairs = (("jit_p50", "ec_full"), ("jit_p50", "jit_exact"), ("ec_full", "jit_exact"), ("jit_p50", "ec_exact"),
             ("ec_full", "ec_exact"), ("jit_exact", "ec_exact"), ("hicache", "ec_exact"), ("hicache", "ec_full"))
    for camp, setting in (("campaign34", "default"), ("campaign32", "fixes=off")):
        mreq = {arm: reflib.measured_requests(camp, arm) for arm in ARMS}
        for a, b in pairs:
            pm = reflib.paired(mreq[a], mreq[b])
            ps = reflib.paired(reflib.sim_requests(os.path.join(RUNS, setting, a, "seed0")),
                               reflib.sim_requests(os.path.join(RUNS, setting, b, "seed0")))
            fr = []
            for k in FRESH:
                fa, fb = os.path.join(RUNS, setting, a, f"seed{k}"), os.path.join(RUNS, setting, b, f"seed{k}")
                if os.path.exists(os.path.join(fa, "requests.csv")) and os.path.exists(os.path.join(fb, "requests.csv")):
                    fr.append(reflib.paired(reflib.sim_requests(fa), reflib.sim_requests(fb), draws=200)["diff"])
            allv = [ps["diff"]] + fr
            sign = "same" if all(v > 0 for v in allv) or all(v < 0 for v in allv) else "changes"
            inside = pm["lo"] <= ps["diff"] <= pm["hi"]
            same_sign = (pm["diff"] > 0) == (ps["diff"] > 0)
            sig = not (pm["lo"] <= 0 <= pm["hi"])
            if inside:
                vd = "reproduced"
            elif sig and same_sign and abs(ps["diff"] - pm["diff"]) <= 0.5 * abs(pm["diff"]):
                vd = "roughly"
            else:
                vd = "not_reproduced"
            rows_json.append({"table": "paired " + camp, "anchor": f"{a} -> {b}", "measured": pm, "sim_seed0": ps,
                              "fresh": fr, "sign": sign, "verdict": vd})
            print(f"| {camp} | {a} -> {b} | {pm['diff']:+.2f} [{pm['lo']:+.2f}, {pm['hi']:+.2f}] on {pm['n']} | "
                  f"{ps['diff']:+.2f} [{ps['lo']:+.2f}, {ps['hi']:+.2f}] on {ps['n']} | "
                  f"{np.mean(fr):+.2f} [{min(fr):+.2f}, {max(fr):+.2f}] | {sign} | {vd} |")
    # the fixes: campaign 34 minus 32 on each run's own requests
    print("\n### The fixes: fixes on minus fixes off, mean TTFT of each run's own requests, s\n")
    print("| arm | measured (34 - 32) | simulated, seed 0 | simulated, fresh seeds: mean [min, max] |")
    print("|---|---|---|---|")
    for arm in ARMS:
        m = meas[f"campaign34/{arm}"]["ttft_mean"] - meas[f"campaign32/{arm}"]["ttft_mean"]
        s = sims[("default", arm, 0)]["ttft_mean"] - sims[("fixes=off", arm, 0)]["ttft_mean"]
        fr = [sims[("default", arm, k)]["ttft_mean"] - sims[("fixes=off", arm, k)]["ttft_mean"] for k in FRESH
              if ("default", arm, k) in sims and ("fixes=off", arm, k) in sims]
        rows_json.append({"table": "fixes", "anchor": arm, "measured": m, "sim_seed0": s, "fresh": fr,
                          "verdict": verdict(m, s, 0.15, 0.40)})
        print(f"| {arm} | {m:+.2f} | {s:+.2f} | {np.mean(fr):+.2f} [{min(fr):+.2f}, {max(fr):+.2f}] |")
    with open(os.path.join(os.path.dirname(RUNS.rstrip('/')), "..", f"anchors_rows_{os.path.basename(RUNS.rstrip('/'))}.json"), "w") as f:
        json.dump(rows_json, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
