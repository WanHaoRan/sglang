"""Referee helpers: the statistics of a simulated run in the same form as ref_measured.py gives for a GPU run, and
the paired comparison per user request (resampling users)."""
from __future__ import annotations

import csv
import json
import os

import numpy as np

RESULTS = "/home/wanhr/sglang/agent_cache/results"
CUT = 6600.0
WINDOWS = ((0, 1800), (1800, 3600), (3600, 5400), (5400, 6600))
SRV_WINDOWS = ((1800, 3600), (3600, 5400), (5400, 6600), (3600, 6600))
WKEY = {(0, 1800): "0-30", (1800, 3600): "30-60", (3600, 5400): "60-90", (5400, 6600): "90-110", (3600, 6600): "60-110"}


def read_csv(path: str) -> tuple:
    with open(path) as f:
        rd = csv.reader(f)
        cols = next(rd)
        rows = [r for r in rd]
    return cols, rows


def sim_stats(folder: str, cut: float = CUT) -> dict:
    cols, rows = read_csv(os.path.join(folder, "requests.csv"))
    ix = {c: i for i, c in enumerate(cols)}

    def col(name):
        return np.array([float(r[ix[name]]) for r in rows])
    t, ttft = col("t_send"), col("ttft")
    prompt, out, ideal = col("prompt"), col("out"), col("ideal")
    gpu, host, ssd = col("tok_gpu"), col("tok_host"), col("tok_ssd")
    first = col("first") > 0
    play = col("play")
    cached = gpu + host + ssd
    m = t < cut
    res = {
        "requests_all": int(len(t)), "requests": int(m.sum()), "req_per_min": float(m.sum() / (cut / 60)),
        "users": int(len(set(play[m].astype(int)))),
        "ttft_mean": float(ttft[m].mean()), "ttft_p50": float(np.percentile(ttft[m], 50)),
        "ttft_p90": float(np.percentile(ttft[m], 90)), "ttft_p99": float(np.percentile(ttft[m], 99)),
        "share_gpu": float(gpu[m].sum() / prompt[m].sum()), "share_host": float(host[m].sum() / prompt[m].sum()),
        "share_ssd": float(ssd[m].sum() / prompt[m].sum()),
        "history_recomputed": float(1 - cached[m].sum() / max(1.0, ideal[m].sum())),
        "prompt_mean": float(prompt[m].mean()), "out_mean": float(out[m].mean()),
        "new_mean": float((prompt[m] - ideal[m]).mean()), "latency_mean": float(col("latency")[m].mean()),
        "queue_wait_mean": float(col("queue_wait")[m].mean()),
        "windows": [],
    }
    for lo, hi in WINDOWS:
        w = m & (t >= lo) & (t < hi)
        ret = w & ~first
        if not w.any():
            res["windows"].append(None)
            continue
        res["windows"].append({
            "window": WKEY[(lo, hi)], "requests": int(w.sum()), "req_per_min": float(w.sum() / ((hi - lo) / 60)),
            "ttft_mean": float(ttft[w].mean()), "ttft_p50": float(np.percentile(ttft[w], 50)),
            "ttft_p90": float(np.percentile(ttft[w], 90)),
            "share_gpu": float(gpu[w].sum() / prompt[w].sum()), "share_host": float(host[w].sum() / prompt[w].sum()),
            "share_ssd": float(ssd[w].sum() / prompt[w].sum()),
            "history_recomputed": float(1 - cached[w].sum() / max(1.0, ideal[w].sum())),
            "recomputed_M": float((ideal[w] - cached[w]).clip(min=0).sum() / 1e6),
            "returning_reading": float((ssd[ret] > 0).mean()) if ret.any() else None,
            "ttft_readers": float(ttft[ret & (ssd > 0)].mean()) if (ret & (ssd > 0)).any() else None,
            "ttft_others": float(ttft[ret & (ssd == 0)].mean()) if (ret & (ssd == 0)).any() else None,
        })
    sat = (t >= 5400) & (t < 6540)
    res["req_per_min_90_109"] = float(sat.sum() / 19.0)
    # the 10 s series
    scols, srows = read_csv(os.path.join(folder, "series.csv"))
    six = {c: i for i, c in enumerate(scols)}

    def scol(name):
        return np.array([float(r[six[name]]) if r[six[name]] != "" else np.nan for r in srows])
    st = scol("t")
    sv = {"decode": {}, "io": {}, "census": {}}
    for lo, hi in SRV_WINDOWS:
        w = (st > lo) & (st <= hi)          # a sample at t covers (t - 10, t]
        if not w.any():
            continue
        key = WKEY[(lo, hi)]
        sv["decode"][key] = {"running_median": float(np.median(scol("running")[w])),
                             "running_mean": float(scol("running")[w].mean()),
                             "queue_median": float(np.median(scol("queue")[w])),
                             "usage_median": float(np.median(scol("pool_used")[w])),
                             "ctx_K_median": float(np.median(scol("ctx_run")[w]) / 1e3)}
        sv["io"][key] = {"read_busy": float(scol("read_busy")[w].mean()),
                         "write_busy": float(scol("write_busy")[w].mean())}
    for lo, hi in ((3600, 6600), (5400, 6600)):
        w = (st >= lo) & (st < hi)
        if not w.any():
            continue
        c = {}
        for name in ("host_mirror", "host_idle", "host_dirty", "host_dirty_only", "host_pinned", "host_inflight",
                     "host_free", "parked"):
            c[name] = float(np.median(scol(name)[w]))
        for name in ("host_dirty", "host_dirty_only", "host_pinned"):
            c[name + "_p90"] = float(np.percentile(scol(name)[w], 90))
            c[name + "_max"] = float(scol(name)[w].max())
        sv["census"][WKEY[(lo, hi)]] = c
    refus = scol("refusals")
    per_min = np.zeros(125, int)
    for tt, rf in zip(st, refus):
        if 0 < tt <= 7500 and rf > 0:
            per_min[int((tt - 5.0) // 60)] += int(rf)

    def onset(run: int):
        for i in range(0, 125 - run):
            if (per_min[i:i + run] >= 5).all():
                return i
        return None
    sv["loop_onset_min_3"], sv["loop_onset_min_5"] = onset(3), onset(5)
    sv["refused_before_cut"] = int(refus[st <= cut].sum())
    sv["refused_by_window"] = {WKEY[(lo, hi)]: int(per_min[lo // 60:hi // 60].sum()) for lo, hi in SRV_WINDOWS}
    dirty = scol("host_dirty")
    sv["pending_write_peak_K"] = {WKEY[(lo, hi)]: float(dirty[(st > lo) & (st <= hi)].max() / 1e3)
                                  for lo, hi in SRV_WINDOWS if ((st > lo) & (st <= hi)).any()}
    with open(os.path.join(folder, "summary.json")) as f:
        summ = json.load(f)
    sv["counters"] = summ["counters"]
    sv["cleaner_passes"] = summ["cleaner_passes"]
    sv["policy"] = summ.get("policy")
    res["server"] = sv
    return res


def sim_requests(folder: str, cut: float = CUT) -> dict:
    cols, rows = read_csv(os.path.join(folder, "requests.csv"))
    ix = {c: i for i, c in enumerate(cols)}
    out = {}
    for r in rows:
        if float(r[ix["t_send"]]) < cut:
            out[(int(r[ix["play"]]), int(r[ix["stream"]]), int(r[ix["k"]]))] = float(r[ix["ttft"]])
    return out


def manifest(campaign: str) -> dict:
    out = {}
    with open(os.path.join(RESULTS, campaign, "manifest.txt")) as f:
        for line in f:
            p = line.split()
            if len(p) >= 3 and not line.startswith("#"):
                out[p[0]] = os.path.join(RESULTS, p[1], p[2], "client.jsonl")
    return out


def measured_requests(campaign: str, arm: str, cut: float = CUT) -> dict:
    out = {}
    with open(manifest(campaign)[arm]) as f:
        for line in f:
            if line.startswith('{"kind":"req"'):
                d = json.loads(line)
                if d.get("ttft") is None or d.get("error") or d["t_send"] >= cut:
                    continue
                out[(d["play"], d["stream"], d["k"])] = d["ttft"]
    return out


def paired(a: dict, b: dict, *, draws: int = 2000, seed: int = 0) -> dict:
    """Mean of b - a over the requests both sent, with a 95 % interval over users (plays)."""
    keys = [k for k in a if k in b]
    if not keys:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    plays = np.array([k[0] for k in keys])
    diff = np.array([b[k] - a[k] for k in keys])
    users, inv = np.unique(plays, return_inverse=True)
    sums = np.bincount(inv, weights=diff)
    counts = np.bincount(inv).astype(float)
    idx = rng.integers(0, len(users), size=(draws, len(users)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"n": len(keys), "users": int(len(users)), "diff": float(diff.mean()), "lo": float(lo), "hi": float(hi)}
