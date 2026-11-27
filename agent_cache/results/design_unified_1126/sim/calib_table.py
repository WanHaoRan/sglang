"""The calibration table: measured anchors (campaigns 34 and 32) against kvsim, seed 0 and five seeds.

Reads measured_anchors.json (calib_measured.py: the client records) and the runs of
  rerun_calibration.sh  (calibration_runs/calib, calibration_runs/triton, calibration_runs/noise)
and prints markdown tables (calibration_table.md) and CALIBRATION.md (calibration_notes.md plus the tables). Anchors that come from server logs are quoted from the campaign
records, with the record named in the row.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ("hicache", "jit_p50", "jit_exact", "ec_full", "ec_exact")
HINTED = ("jit_p50", "jit_exact", "ec_full", "ec_exact")
SEEDS = (0, 1, 2, 3, 4)
OUT = os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else "calibration_runs")
RUNS = os.path.join(OUT, "calib")


def load(setting: str, arm: str, seed: int) -> dict:
    with open(os.path.join(RUNS, setting, arm, f"seed{seed}", "summary.json")) as f:
        return json.load(f)


def series(setting: str, arm: str, seed: int) -> list:
    with open(os.path.join(RUNS, setting, arm, f"seed{seed}", "series.csv")) as f:
        return [{k: (float(v) if v not in ("", "None") else None) for k, v in row.items()} for row in csv.DictReader(f)]


def onset(ser: list, need: int = 5, in_a_row: int = 3):
    """First of `in_a_row` minutes in a row with `need` or more reads refused host room (campaign 34's definition)."""
    per_min = {}
    for x in ser:
        m = int((x["t"] - 1e-6) // 60)
        per_min[m] = per_min.get(m, 0) + (x["refusals"] or 0)
    run = 0
    for m in range(0, 121):
        run = run + 1 if per_min.get(m, 0) >= need else 0
        if run >= in_a_row:
            return m - in_a_row + 1
    return None


def rel(sim, meas):
    if meas in (None, 0) or sim is None:
        return ""
    return f"{100.0 * (sim - meas) / meas:+.0f} %"


def fmt(x, nd=2):
    if x is None:
        return "none"
    return f"{x:.{nd}f}"


def row(name, meas, sim0, seeds_vals, nd=2, source=""):
    vals = [v for v in seeds_vals if v is not None]
    if vals:
        spread = f"{np.mean(vals):.{nd}f} ± {np.std(vals, ddof=1) if len(vals) > 1 else 0:.{nd}f} [{min(vals):.{nd}f}, {max(vals):.{nd}f}]"
    else:
        spread = "none"
    return f"| {name} | {fmt(meas, nd)} | {fmt(sim0, nd)} | {rel(sim0, meas)} | {spread} | {source} |"


def main() -> int:
    with open(os.path.join(HERE, "measured_anchors.json")) as f:
        meas = json.load(f)
    out = []
    S = {(st, arm, k): load(st, arm, k) for st in ("default", "fixes=off") for arm in ARMS for k in SEEDS}
    head = "| anchor | measured | simulated, seed 0 | relative error | simulated, seeds 0-4: mean ± sd [min, max] | measured from |\n|---|---|---|---|---|---|"

    def sims(st, arm, fn):
        return fn(S[(st, arm, 0)]), [fn(S[(st, arm, k)]) for k in SEEDS]

    # ----- campaign 34 -----
    out.append("### Campaign 34 (fixes on): requests sent before 6,600 s\n")
    out.append(head)
    for arm in ARMS:
        m = meas[f"campaign34/{arm}"]
        s0, sv = sims("default", arm, lambda s: s["ttft_mean"])
        out.append(row(f"{arm}: mean TTFT, s", m["ttft_mean"], s0, sv, source="client.jsonl"))
    mh = np.mean([meas[f"campaign34/{a}"]["ttft_mean"] for a in HINTED])
    ratio_m = meas["campaign34/hicache"]["ttft_mean"] / mh
    ratios = [S[("default", "hicache", k)]["ttft_mean"] / np.mean([S[("default", a, k)]["ttft_mean"] for a in HINTED]) for k in SEEDS]
    out.append(row("hicache over the mean of the four hinted arms", ratio_m, ratios[0], ratios, source="client.jsonl"))
    spread_m = max(meas[f"campaign34/{a}"]["ttft_mean"] for a in HINTED) - min(meas[f"campaign34/{a}"]["ttft_mean"] for a in HINTED)
    spreads = [max(S[("default", a, k)]["ttft_mean"] for a in HINTED) - min(S[("default", a, k)]["ttft_mean"] for a in HINTED) for k in SEEDS]
    out.append(row("hinted arms: slowest minus fastest, s", spread_m, spreads[0], spreads, source="client.jsonl"))
    for a, b in (("jit_p50", "ec_full"), ("ec_full", "jit_exact"), ("ec_full", "ec_exact")):
        dm = meas[f"campaign34/{b}"]["ttft_mean"] - meas[f"campaign34/{a}"]["ttft_mean"]
        ds = [S[("default", b, k)]["ttft_mean"] - S[("default", a, k)]["ttft_mean"] for k in SEEDS]
        out.append(row(f"{a} -> {b}: difference of means, s", dm, ds[0], ds, source="client.jsonl"))
    for arm in ARMS:
        m = meas[f"campaign34/{arm}"]
        for key, label, nd in (("ttft_p50", "p50 TTFT, s", 2), ("ttft_p90", "p90 TTFT, s", 1), ("ttft_p99", "p99 TTFT, s", 1)):
            s0, sv = sims("default", arm, lambda s, key=key: s[key])
            out.append(row(f"{arm}: {label}", m[key], s0, sv, nd=nd, source="client.jsonl"))
    out.append("\n### Campaign 34: by half hour of send time (mean TTFT, s)\n")
    out.append(head)
    for arm in ARMS:
        m = meas[f"campaign34/{arm}"]
        for i, w in enumerate(m["windows"]):
            s0, sv = sims("default", arm, lambda s, i=i: s["windows"][i]["ttft_mean"] if i < len(s["windows"]) else None)
            out.append(row(f"{arm}: minutes {w['window'].replace(' min', '')}", w["ttft_mean"], s0, sv, source="client.jsonl"))
    out.append("\n### Campaign 34: throughput, tiers, memory, device\n")
    out.append(head)
    for arm in ARMS:
        m = meas[f"campaign34/{arm}"]
        s0, sv = sims("default", arm, lambda s: s["req_per_min"])
        out.append(row(f"{arm}: requests per minute (to 6,600 s)", m["req_per_min"], s0, sv, nd=1, source="client.jsonl"))
        s0, sv = sims("default", arm, lambda s: s.get("req_per_min_90_109"))
        out.append(row(f"{arm}: requests per minute, minutes 90-109", m["req_per_min_90_109"], s0, sv, nd=1, source="client.jsonl"))
    for arm in ARMS:
        m = meas[f"campaign34/{arm}"]
        for key, label in (("share_gpu", "prompt tokens from GPU"), ("share_host", "from host"), ("share_ssd", "from SSD"),
                           ("history_recomputed", "history recomputed")):
            s0, sv = sims("default", arm, lambda s, key=key: s[key])
            out.append(row(f"{arm}: {label}, share", m[key], s0, sv, nd=4 if key == "history_recomputed" else 3, source="client.jsonl"))
    for arm in ARMS:
        m = meas[f"campaign34/{arm}"]
        for i, w in enumerate(m["windows"][1:], start=1):
            s0, sv = sims("default", arm, lambda s, i=i: s["windows"][i]["share_ssd"])
            out.append(row(f"{arm}: from SSD, minutes {w['window'].replace(' min', '')}", w["share_ssd"], s0, sv, nd=3, source="client.jsonl"))
    quoted = {
        "first cleaner pass, s": ({"jit_p50": 5301, "ec_full": 5275, "jit_exact": 5272, "ec_exact": 5252, "hicache": None},
                                  lambda s: s["first_cleaner_pass_s"], 0, "campaign34/DECISIONS.md (Run)"),
        "reads refused host room before the cut": ({"jit_p50": 0, "ec_full": 0, "jit_exact": 0, "ec_exact": 0},
                                                   lambda s: s["refusals_before_cut"], 0, "campaign34/DECISIONS.md"),
        "pinned host tokens, median of minutes 60-110, K": ({"jit_p50": 136.0, "ec_full": 0.0, "jit_exact": 0.0},
                                                            lambda s: s["census_60_110"]["host_pinned"] / 1e3, 0,
                                                            "campaign34/DECISIONS.md (census: pins 0 to 25 K; jit_p50's early reads 136 K)"),
        "pinned host tokens, p90, K": ({"jit_p50": 280.0}, lambda s: s["pinned_p90_60_110"] / 1e3, 0,
                                       "campaign34/DECISIONS.md (census)"),
        "host mirror, median of minutes 60-110, K": ({"ec_full": 650.0}, lambda s: s["census_60_110"]["host_mirror"] / 1e3, 0,
                                                     "campaign34/DECISIONS.md (census)"),
        "host free, median, K": ({"ec_full": 17.0}, lambda s: s["census_60_110"]["host_free"] / 1e3, 1,
                                 "campaign34/DECISIONS.md (census: 16 to 18 K)"),
        "pending writes, p90 of minutes 60-110, K": ({"ec_full": 80.0}, lambda s: s["pending_write_p90_60_110"] / 1e3, 0,
                                                     "campaign34/DECISIONS.md (census p90 65 to 94 K)"),
        "running requests, minutes 90-110 (median)": ({"ec_full": 19.9, "jit_exact": 19.9, "jit_p50": 19.9},
                                                      lambda s: s["census_90_110"]["running"], 1,
                                                      "critique.md (19.7 to 20.1)"),
        "queued requests, minutes 90-110 (median)": ({"ec_full": 60.0}, lambda s: s["census_90_110"]["queue"], 0,
                                                     "campaign34/checks/by_minute_ec_full.out"),
        "GPU pool used while queued (median)": ({"ec_full": 0.974}, lambda s: s["census_90_110"]["pool_used"], 3,
                                                "campaign34/checks/by_minute_ec_full.out"),
        "SSD reads in the run": ({"jit_p50": 3268, "ec_full": 2678, "jit_exact": 2221}, lambda s: s["counters"]["reads"], 0,
                                 "campaign34/checks/read_thread.out"),
        "tokens read from the SSD, M": ({"jit_p50": 82.5, "ec_full": 64.1, "jit_exact": 54.4},
                                        lambda s: s["counters"]["read_tokens"] / 1e6, 1, "campaign34/checks/read_thread.out"),
        "reads staged before arrival": ({"jit_p50": 1208, "ec_full": 1, "jit_exact": 99}, lambda s: s["counters"]["staged"], 0,
                                        "facts_fetch.md (ntp_stage)"),
        "write hold switched on, times": ({"jit_p50": 30, "ec_full": 30, "jit_exact": 28}, lambda s: s["counters"]["write_hold_on"], 0,
                                          "facts_evict.md (srs_write_hold)"),
        "users with a request before the cut": ({"jit_p50": 617, "ec_full": 620, "jit_exact": 622, "ec_exact": 621, "hicache": 448},
                                                lambda s: s["users_with_request"], 0, "campaign34/DECISIONS.md (Against campaign 32)"),
    }
    quoted["updates of the learned price in the run"] = ({"ec_full": 1451}, lambda s: s["policy"]["price_updates"], 0,
                                                         "facts_fetch.md (ec_price lines)")
    quoted["of them, updates that moved the price"] = ({"ec_full": 0}, lambda s: s["policy"]["price_moved"], 0,
                                                       "facts_fetch.md (lambda stayed at 5e-5)")
    for label, (vals, fn, nd, src) in quoted.items():
        for arm, mv in vals.items():
            s0, sv = sims("default", arm, fn)
            out.append(row(f"{arm}: {label}", mv, s0, sv, nd=nd, source=src))
    # ----- campaign 32 -----
    # The loop's start is a tipping point: with the same users, 5 ms of send jitter moves it by minutes. The
    # simulated column is therefore the median of six replicas of seed 0 (no jitter, and five with jitter).
    def reps(arm):
        out_ = [(S[("fixes=off", arm, 0)], series("fixes=off", arm, 0))]
        for j in range(1, 6):
            d = os.path.join(OUT, "noise", f"fixes=off_noise=0.005_noise_seed={j}", arm, "seed0")
            if os.path.exists(os.path.join(d, "summary.json")):
                with open(os.path.join(d, "summary.json")) as f:
                    sm = json.load(f)
                with open(os.path.join(d, "series.csv")) as f:
                    sr = [{k: (float(v) if v not in ("", "None") else None) for k, v in r_.items()} for r_ in csv.DictReader(f)]
                out_.append((sm, sr))
        return out_

    def row32(name, mv, vals, seeds_vals, nd=2, source="client.jsonl"):
        v = [x for x in vals if x is not None]
        none = len(vals) - len(v)
        med = float(np.median(v)) if v else None
        rep_txt = "none" if not v else f"{med:.{nd}f} [{min(v):.{nd}f}, {max(v):.{nd}f}]" + (f"; none in {none} of {len(vals)}" if none else "")
        sv = [x for x in seeds_vals if x is not None]
        seed_txt = "none" if not sv else f"{np.mean(sv):.{nd}f} ± {np.std(sv, ddof=1) if len(sv) > 1 else 0:.{nd}f} [{min(sv):.{nd}f}, {max(sv):.{nd}f}]"
        if len(sv) < len(seeds_vals) and sv:
            seed_txt += f"; none in {len(seeds_vals) - len(sv)} of {len(seeds_vals)}"
        return f"| {name} | {fmt(mv, nd)} | {rep_txt} | {rel(med, mv)} | {seed_txt} | {source} |"

    out.append("\n### Campaign 32 (fixes off): requests sent before 6,600 s\n")
    out.append("| anchor | measured | simulated, seed 0: median of 6 replicas [min, max] | relative error of the median | simulated, seeds 0-4 (no jitter): mean ± sd [min, max] | measured from |\n|---|---|---|---|---|---|")
    R = {arm: reps(arm) for arm in ARMS}
    for arm in ARMS:
        m = meas[f"campaign32/{arm}"]
        for key, label, nd in (("ttft_mean", "mean TTFT, s", 2), ("req_per_min", "requests per minute", 1),
                               ("history_recomputed", "history recomputed, share", 3)):
            out.append(row32(f"{arm}: {label}", m[key], [x[0][key] for x in R[arm]],
                             [S[("fixes=off", arm, k)][key] for k in SEEDS], nd=nd))
    for arm in ARMS:
        m = meas[f"campaign32/{arm}"]
        for i, w in enumerate(m["windows"]):
            out.append(row32(f"{arm}: mean TTFT, minutes {w['window'].replace(' min', '')}", w["ttft_mean"],
                             [x[0]["windows"][i]["ttft_mean"] for x in R[arm]],
                             [S[("fixes=off", arm, k)]["windows"][i]["ttft_mean"] for k in SEEDS]))
    onset_m = {"jit_p50": 42, "ec_exact": 79, "jit_exact": 80, "ec_full": 84}
    for arm, mv in onset_m.items():
        out.append(row32(f"{arm}: loop onset, minute (first of 3 in a row with 5 or more refusals)", mv,
                         [onset(x[1]) for x in R[arm]], [onset(series("fixes=off", arm, k)) for k in SEEDS], nd=0,
                         source="campaign34/DECISIONS.md (Arms)"))
    out.append(row32("jit_p50: loop onset, minute (5 in a row)", 67, [onset(x[1], in_a_row=5) for x in R["jit_p50"]],
                     [onset(series("fixes=off", "jit_p50", k), in_a_row=5) for k in SEEDS], nd=0,
                     source="campaign34/DECISIONS.md (Arms)"))
    for arm in HINTED:
        on = [onset(series("default", arm, k)) for k in SEEDS]
        out.append(row(f"{arm}, fixes on: loop onset", None, on[0], on, nd=0, source="campaign34/DECISIONS.md: none"))
    for arm in ("jit_p50", "ec_full", "jit_exact", "ec_exact"):
        pk = lambda sr: max((x["host_dirty"] or 0) for x in sr if x["t"] >= 3600) / 1e3
        out.append(row32(f"{arm}: pending SSD writes at their peak after minute 60, K tokens",
                         {"jit_p50": 675, "ec_full": 610, "jit_exact": 523, "ec_exact": 576}[arm],
                         [pk(x[1]) for x in R[arm]], [pk(series("fixes=off", arm, k)) for k in SEEDS], nd=0,
                         source="campaign34/DECISIONS.md (campaign 32 column)"))
    for arm in ARMS:
        m = meas[f"campaign32/{arm}"]
        out.append(row32(f"{arm}: requests per minute, minutes 90-109", m["req_per_min_90_109"],
                         [x[0].get("req_per_min_90_109") for x in R[arm]],
                         [S[("fixes=off", arm, k)].get("req_per_min_90_109") for k in SEEDS], nd=1))
    out.append("\n### Campaign 34 minus campaign 32 (the fixes), difference of mean TTFT, s\n")
    out.append(head)
    for arm in ARMS:
        dm = meas[f"campaign34/{arm}"]["ttft_mean"] - meas[f"campaign32/{arm}"]["ttft_mean"]
        ds = [S[("default", arm, k)]["ttft_mean"] - S[("fixes=off", arm, k)]["ttft_mean"] for k in SEEDS]
        d0 = S[("default", arm, 0)]["ttft_mean"] - float(np.median([x[0]["ttft_mean"] for x in R[arm]]))
        out.append(row(f"{arm}: fixes on minus fixes off (each run's own requests; seed 0 against the median of its 6 replicas)",
                       dm, d0, ds, source="client.jsonl"))
    # ----- paired differences between arms (campaign 34) -----
    sys.path.insert(0, HERE)
    import kvsim_compare as kc
    out.append("\n### Campaign 34: paired differences between arms (B - A on the requests both sent before 6,600 s), s\n")
    out.append("| pair | measured [95 % interval over users] | simulated, seed 0 [interval] | simulated, seeds 0-4: mean [min, max] | sign |\n|---|---|---|---|---|")
    pairs = (("jit_p50", "ec_full", "-0.62 [-1.02, -0.22]"), ("jit_p50", "jit_exact", "-0.67 [-1.07, -0.28]"),
             ("ec_full", "jit_exact", "-0.06 [-0.19, +0.06]"), ("jit_p50", "ec_exact", "-1.01 [-1.42, -0.62]"),
             ("ec_full", "ec_exact", "-0.40 [-0.51, -0.28]"), ("jit_exact", "ec_exact", "-0.32 [-0.45, -0.19]"),
             ("hicache", "ec_exact", "-26.96 [-30.53, -23.61]"))
    rng = np.random.default_rng(0)
    reqs = {}

    def get(arm, k):
        if (arm, k) not in reqs:
            reqs[(arm, k)] = kc.load_requests(os.path.join(RUNS, "default", arm, f"seed{k}"), 6600.0)
        return reqs[(arm, k)]

    for a, b, mtxt in pairs:
        ps = [kc.paired(get(a, k), get(b, k), rng) for k in SEEDS]
        d = [p["diff"] for p in ps]
        sign = "same in every seed" if all(x > 0 for x in d) or all(x < 0 for x in d) else "changes between seeds"
        out.append(f"| {a} -> {b} | {mtxt} | {ps[0]['diff']:+.2f} [{ps[0]['lo']:+.2f}, {ps[0]['hi']:+.2f}] | "
                   f"{np.mean(d):+.2f} [{min(d):+.2f}, {max(d):+.2f}] | {sign} |")
    # ----- who waits (calib_structure.py) -----
    spath = os.path.join(HERE, "measured_structure.json")
    if os.path.exists(spath):
        with open(spath) as f:
            st = json.load(f)
        out.append("\n### Campaign 34: who waits (returning requests)\n")
        out.append(head)
        for arm in ARMS:
            if arm not in st:
                continue
            m, sv = st[arm]["measured"], st[arm]["sim"]
            for key, label, nd in (("reader_share", "share of returning requests that read from the SSD, minutes 90-110", 3),
                                   ("ttft_readers", "their mean TTFT, s", 1), ("ttft_others", "mean TTFT of the others, s", 1)):
                out.append(row(f"{arm}: {label}", m[key], sv[0][key], [x[key] for x in sv], nd=nd, source="client.jsonl"))
            for lo, hi in ((0, 1), (1, 10), (10, 60), (60, 300), (300, 1800)):
                key = f"wait_{lo}_{hi}"
                out.append(row(f"{arm}: mean TTFT of requests sent after a wait of {lo}-{hi} s", m[key], sv[0][key],
                               [x[key] for x in sv], nd=1, source="client.jsonl"))
    # ----- campaign 28: the Triton cost set, not fitted here -----
    tri = os.path.join(OUT, "triton", "backend=triton_fixes=off")
    if os.path.isdir(tri):
        out.append("\n### Campaign 28 (Triton backend, same users, no fixes): requests sent before 6,600 s; the Triton costs come from the records and were not fitted here\n")
        out.append(head)
        m28 = {"hicache": (115.69, 41.3, 0.16, 0.401, 128.41, 245.68), "jit_p50": (49.38, 77.8, 0.42, 0.012, 52.69, 86.23),
               "jit_exact": (49.79, 76.9, 0.41, 0.014, 53.93, 85.91), "ec_full": (48.27, 78.9, 0.43, 0.004, 51.72, 80.24),
               "ec_exact": (48.56, 78.6, 0.44, 0.005, 51.93, 81.78)}
        for arm, (ttft, rpm, ssd, rec, p50, p90) in m28.items():
            ss = []
            for k in SEEDS:
                with open(os.path.join(tri, arm, f"seed{k}", "summary.json")) as f:
                    ss.append(json.load(f))
            src = "campaign28/compare_arms.out"
            out.append(row(f"{arm}: mean TTFT, s", ttft, ss[0]["ttft_mean"], [x["ttft_mean"] for x in ss], source=src))
            out.append(row(f"{arm}: p50 TTFT, s", p50, ss[0]["ttft_p50"], [x["ttft_p50"] for x in ss], nd=1, source=src))
            out.append(row(f"{arm}: p90 TTFT, s", p90, ss[0]["ttft_p90"], [x["ttft_p90"] for x in ss], nd=1, source=src))
            out.append(row(f"{arm}: requests per minute", rpm, ss[0]["req_per_min"], [x["req_per_min"] for x in ss], nd=1, source=src))
            out.append(row(f"{arm}: prompt tokens from SSD, share", ssd, ss[0]["share_ssd"], [x["share_ssd"] for x in ss], nd=3, source=src))
            out.append(row(f"{arm}: history recomputed, share", rec, ss[0]["history_recomputed"],
                           [x["history_recomputed"] for x in ss], nd=3, source=src))
    # ----- the same users with 5 ms of send jitter -----
    noise = os.path.join(OUT, "noise")
    if os.path.isdir(noise):
        out.append("\n### Same users (seed 0), five replicas with up to 5 ms of send jitter: mean TTFT, s\n")
        out.append("| arm | fixes | replicas | min | max | max - min |\n|---|---|---|---|---|---|")
        for fixes, prefix in (("on", "noise=0.005_noise_seed="), ("off", "fixes=off_noise=0.005_noise_seed=")):
            for arm in ARMS:
                vals = []
                for j in range(1, 6):
                    path = os.path.join(noise, f"{prefix}{j}", arm, "seed0", "summary.json")
                    if os.path.exists(path):
                        with open(path) as f:
                            vals.append(json.load(f)["ttft_mean"])
                if vals:
                    out.append(f"| {arm} | {fixes} | {', '.join(f'{v:.2f}' for v in vals)} | {min(vals):.2f} | {max(vals):.2f} | {max(vals) - min(vals):.2f} |")
    text = "\n".join(out) + "\n"
    with open(os.path.join(HERE, "calibration_table.md"), "w") as f:
        f.write(text)
    notes = os.path.join(HERE, "calibration_notes.md")
    if os.path.exists(notes):
        with open(notes) as f:
            head_text = f.read()
        with open(os.path.join(HERE, "CALIBRATION.md"), "w") as f:
            f.write(head_text.rstrip() + "\n\n## 7. The table\n\n" + text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
