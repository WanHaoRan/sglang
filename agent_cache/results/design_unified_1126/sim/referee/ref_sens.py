"""Referee: sensitivity of the arm ordering to each fitted constant, +-20 % one at a time.

  python -I -B ref_sens.py run   [--sim kvsim] [--seeds 0,5,7] [--off-seeds 0] [--procs 4] [--out runs/sens]
  python -I -B ref_sens.py report [--out runs/sens]

Every fitted constant of CALIBRATION.md section 3 is scaled by 0.8 and by 1.2 with the others at their values; the
five anchored arms run on the same users (seed) under each setting, with the fixes on (and off for --off-seeds).
Each run keeps its summary and, per request sent before 6,600 s, (play, stream, k, t_send, ttft) for the pairing.
At most 4 processes, each at nice 19.
"""
from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.dirname(HERE)
sys.path.insert(0, SIM_DIR)
sys.path.insert(0, HERE)

ARMS = ("ec_full", "hicache", "jit_p50", "jit_exact", "ec_exact")
# name -> where it lives: 'gpu' = kvsim.BACKENDS['fa3'][name], 'cfg' = Config field
CONSTANTS = (
    ("a", "gpu", "prefill s per token at position 0 (1.45e-5)"),
    ("b", "gpu", "prefill s per token per position (1.18e-9)"),
    ("batch_s", "gpu", "per prefill batch (15 ms)"),
    ("h2d", "gpu", "per token loaded from host memory (0.85 us)"),
    ("step0", "gpu", "decode step constant (3.3 ms)"),
    ("step_n", "gpu", "decode step per running request up to 8 (0.30 ms)"),
    ("step_k", "gpu", "decode step per token of running context (0.0277 us)"),
    ("first_token_s", "cfg", "ingestion + first-token delivery (0.03 s)"),
    ("client_lag", "cfg", "reply -> next send (0.03 s)"),
    ("read_alone", "cfg", "read thread alone (24.0 K tokens/s)"),
    ("write_alone", "cfg", "write thread alone (18.8 K tokens/s)"),
    ("beta", "cfg", "both threads busy, total (21.4 K tokens/s)"),
    ("phi", "cfg", "read share while both run (0.73)"),
    ("lookup_lag", "cfg", "lookup lag (0.02 s)"),
    ("land_lag", "cfg", "landing lag (0.03 s)"),
    ("ssd_tokens", "cfg", "SSD capacity (24.69 M tokens)"),
)
CUT = 6600.0


def jobs_for(seeds, off_seeds, factors=(0.8, 1.2)):
    out = []
    for seed in seeds:
        for fixes in (True, False):
            if not fixes and seed not in off_seeds:
                continue
            for arm in ARMS:
                out.append(("base", None, 1.0, arm, seed, fixes))
            for name, _, _ in CONSTANTS:
                for f in factors:
                    for arm in ARMS:
                        out.append((f"{name}x{f}", name, f, arm, seed, fixes))
    return out


def folder_of(out, label, arm, seed, fixes):
    return os.path.join(out, label, "fixes_on" if fixes else "fixes_off", arm, f"seed{seed}")


def work(job):
    simname, out, label, name, factor, arm, seed, fixes = job
    try:
        os.nice(19)
    except OSError:
        pass
    folder = folder_of(out, label, arm, seed, fixes)
    spath = os.path.join(folder, "summary.json")
    if os.path.exists(spath) and os.path.exists(os.path.join(folder, "req.npz")):
        return job, 0.0
    kvsim = importlib.import_module(simname)
    kw = dict(arm=arm, seed=seed, fixes=fixes)
    where = {n: w for n, w, _ in CONSTANTS}
    if name is not None:
        if where[name] == "gpu":
            kvsim.BACKENDS["fa3"][name] *= factor
        else:
            base = getattr(kvsim.Config(), name)
            v = base * factor
            kw[name] = int(round(v)) if isinstance(base, int) else v
    t0 = time.process_time()
    sim = kvsim.Sim(kvsim.Config(**kw)).run()
    s = kvsim.summary(sim, cut=CUT)
    rows = [x for x in sim.request_rows() if x["t_send"] < CUT]
    os.makedirs(folder, exist_ok=True)
    np.savez_compressed(os.path.join(folder, "req.npz"),
                        play=np.array([x["play"] for x in rows], np.int32), stream=np.array([x["stream"] for x in rows], np.int16),
                        k=np.array([x["k"] for x in rows], np.int32), t_send=np.array([x["t_send"] for x in rows], np.float32),
                        ttft=np.array([x["ttft"] for x in rows], np.float32),
                        ssd=np.array([x["tok_ssd"] for x in rows], np.int32),
                        recomputed=np.array([x["tok_recomputed"] for x in rows], np.int32))
    ser = sim.series
    per_min = np.zeros(125, int)
    for x in ser:
        if x["refusals"] and 0 < x["t"] <= 7500:
            per_min[int((x["t"] - 5.0) // 60)] += int(x["refusals"])
    onset = next((i for i in range(0, 122) if (per_min[i:i + 3] >= 5).all()), None)
    keep = {k: s[k] for k in ("requests", "req_per_min", "ttft_mean", "ttft_p50", "ttft_p90", "share_gpu", "share_host",
                              "share_ssd", "history_recomputed", "req_per_min_90_109", "first_cleaner_pass_s",
                              "refusals_before_cut", "users_with_request", "counters") if k in s}
    keep["windows"] = [{"window": w["window"], "ttft_mean": w["ttft_mean"], "requests": w["requests"]} for w in s["windows"]]
    keep["loop_onset_min_3"] = onset
    keep["label"], keep["factor"], keep["constant"], keep["arm"], keep["seed"], keep["fixes"] = label, factor, name, arm, seed, fixes
    with open(spath + ".tmp", "w") as f:
        json.dump(keep, f)
    os.replace(spath + ".tmp", spath)
    return job, time.process_time() - t0


def load(out, label, arm, seed, fixes):
    folder = folder_of(out, label, arm, seed, fixes)
    with open(os.path.join(folder, "summary.json")) as f:
        s = json.load(f)
    z = np.load(os.path.join(folder, "req.npz"))
    req = {(int(p), int(st), int(k)): float(t) for p, st, k, t in zip(z["play"], z["stream"], z["k"], z["ttft"])}
    return s, req


def run(a):
    seeds = [int(x) for x in a.seeds.split(",")]
    off = [int(x) for x in a.off_seeds.split(",")] if a.off_seeds else []
    jobs = [(a.sim, a.out) + j for j in jobs_for(seeds, off)]
    t0 = time.time()
    done = 0
    with mp.Pool(max(1, min(4, a.procs)), maxtasksperchild=1) as pool:
        for job, cpu in pool.imap_unordered(work, jobs):
            done += 1
            if done % 20 == 0 or done == len(jobs):
                print(f"[{time.time() - t0:6.0f} s] {done}/{len(jobs)} runs", flush=True)


def report(a):
    import reflib
    seeds = [int(x) for x in a.seeds.split(",")]
    off = [int(x) for x in a.off_seeds.split(",")] if a.off_seeds else []
    labels = ["base"] + [f"{n}x{f}" for n, _, _ in CONSTANTS for f in (0.8, 1.2)]
    desc = {n: d for n, _, d in CONSTANTS}
    result = []
    for fixes in (True, False):
        for seed in seeds:
            if not fixes and seed not in off:
                continue
            print(f"\n### seed {seed}, fixes {'on' if fixes else 'off'}: mean TTFT per arm (requests sent before 6,600 s) and paired differences B - ec_full, s\n")
            print("| setting | ec_full | hicache | jit_p50 | jit_exact | ec_exact | hicache - ec_full | jit_p50 - ec_full [95 %] | jit_exact - ec_full [95 %] | ec_exact - ec_full [95 %] | refused before cut (ec_full / jit_p50) | loop onset minute (ec_full / jit_p50 / jit_exact / ec_exact) |")
            print("|---|---|---|---|---|---|---|---|---|---|---|---|")
            for label in labels:
                try:
                    data = {arm: load(a.out, label, arm, seed, fixes) for arm in ARMS}
                except FileNotFoundError:
                    continue
                base_req = data["ec_full"][1]
                pd = {arm: reflib.paired(base_req, data[arm][1], draws=1000) for arm in ARMS[1:]}
                rec = {"seed": seed, "fixes": fixes, "label": label,
                       "ttft": {arm: data[arm][0]["ttft_mean"] for arm in ARMS},
                       "rpm": {arm: data[arm][0]["req_per_min"] for arm in ARMS},
                       "paired": pd,
                       "refused": {arm: data[arm][0]["refusals_before_cut"] for arm in ARMS},
                       "onset": {arm: data[arm][0]["loop_onset_min_3"] for arm in ARMS},
                       "recomputed": {arm: data[arm][0]["history_recomputed"] for arm in ARMS}}
                result.append(rec)
                t = rec["ttft"]
                print(f"| {label} | {t['ec_full']:.2f} | {t['hicache']:.2f} | {t['jit_p50']:.2f} | {t['jit_exact']:.2f} | {t['ec_exact']:.2f} | "
                      f"{pd['hicache']['diff']:+.2f} | {pd['jit_p50']['diff']:+.2f} [{pd['jit_p50']['lo']:+.2f}, {pd['jit_p50']['hi']:+.2f}] | "
                      f"{pd['jit_exact']['diff']:+.2f} [{pd['jit_exact']['lo']:+.2f}, {pd['jit_exact']['hi']:+.2f}] | "
                      f"{pd['ec_exact']['diff']:+.2f} [{pd['ec_exact']['lo']:+.2f}, {pd['ec_exact']['hi']:+.2f}] | "
                      f"{rec['refused']['ec_full']} / {rec['refused']['jit_p50']} | "
                      f"{rec['onset']['ec_full']} / {rec['onset']['jit_p50']} / {rec['onset']['jit_exact']} / {rec['onset']['ec_exact']} |")
    with open(os.path.join(a.out, "sens_report.json"), "w") as f:
        json.dump(result, f, indent=1)
    # which conclusions survive
    print("\n### Which ordering conclusions survive (count of settings where the sign holds, over the 32 perturbed settings per seed)\n")
    print("| seed | fixes | hicache behind ec_full | jit_p50 behind ec_full (point) | ... with the interval excluding 0 | |jit_exact - ec_full| < 0.2 s | |ec_exact - ec_full| < 0.2 s | every hinted arm loops (fixes off) | no hinted arm loops (fixes on) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for fixes in (True, False):
        for seed in seeds:
            rs = [r for r in result if r["seed"] == seed and r["fixes"] == fixes and r["label"] != "base"]
            if not rs:
                continue
            n = len(rs)
            c1 = sum(1 for r in rs if r["paired"]["hicache"]["diff"] > 0)
            c2 = sum(1 for r in rs if r["paired"]["jit_p50"]["diff"] > 0)
            c2s = sum(1 for r in rs if r["paired"]["jit_p50"]["lo"] > 0)
            c3 = sum(1 for r in rs if abs(r["paired"]["jit_exact"]["diff"]) < 0.2)
            c4 = sum(1 for r in rs if abs(r["paired"]["ec_exact"]["diff"]) < 0.2)
            hinted = ("ec_full", "jit_p50", "jit_exact", "ec_exact")
            loops = sum(1 for r in rs if all(r["onset"][h] is not None for h in hinted))
            noloop = sum(1 for r in rs if all(r["onset"][h] is None for h in hinted))
            print(f"| {seed} | {'on' if fixes else 'off'} | {c1}/{n} | {c2}/{n} | {c2s}/{n} | {c3}/{n} | {c4}/{n} | "
                  f"{loops if not fixes else '-'}/{n} | {noloop if fixes else '-'}/{n} |")
    print("\nSettings where a conclusion fails:")
    for r in result:
        if r["label"] == "base":
            continue
        fails = []
        if r["paired"]["hicache"]["diff"] <= 0:
            fails.append(f"hicache ahead of ec_full ({r['paired']['hicache']['diff']:+.2f})")
        if r["paired"]["jit_p50"]["diff"] <= 0:
            fails.append(f"jit_p50 ahead of ec_full ({r['paired']['jit_p50']['diff']:+.2f})")
        if r["fixes"] and abs(r["paired"]["jit_exact"]["diff"]) >= 0.2:
            fails.append(f"jit_exact - ec_full {r['paired']['jit_exact']['diff']:+.2f}")
        if r["fixes"] and abs(r["paired"]["ec_exact"]["diff"]) >= 0.2:
            fails.append(f"ec_exact - ec_full {r['paired']['ec_exact']['diff']:+.2f}")
        hinted = ("ec_full", "jit_p50", "jit_exact", "ec_exact")
        if r["fixes"] and any(r["onset"][h] is not None for h in hinted):
            fails.append("a loop with the fixes on: " + ", ".join(f"{h} minute {r['onset'][h]}" for h in hinted if r["onset"][h] is not None))
        if not r["fixes"] and any(r["onset"][h] is None for h in hinted):
            fails.append("no loop with the fixes off: " + ", ".join(h for h in hinted if r["onset"][h] is None))
        if fails:
            name = r["label"].rsplit("x", 1)[0]
            print(f"- seed {r['seed']}, fixes {'on' if r['fixes'] else 'off'}, {r['label']} ({desc.get(name, '')}): " + "; ".join(fails))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "report"))
    ap.add_argument("--sim", default="kvsim")
    ap.add_argument("--seeds", default="0,5,7")
    ap.add_argument("--off-seeds", default="0")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "sens"))
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a)
    else:
        report(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
