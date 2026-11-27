"""Policy comparison on kvsim: arms x settings x seeds, at most 4 processes, with the paired comparison per user
request that frozen/oracle_scripts_staging/c28_users/compare_arms.py makes on GPU runs.

  kvsim_compare.py --arms hicache ec_full "unified@F1+R4" "ec_full@mirror=off" \
      --set cap=128,host_gb=160 --set cap=44,host_gb=70 --seeds 0-4 --out <folder> [--procs 4] [--cut 6600]

An arm is NAME or NAME@OPTION+OPTION...: an option is a DESIGN.md section 9 switch (F1 ... PR1, or all; arm unified)
or a Config field as key=value (mirror=off, fixes=off, backend=triton, hints=exact, d2h_s_per_token=0, ...). A setting
is a comma-joined list of Config fields (cap, rate, host_gb, duration, gap_scale, gpu_tokens, ...). Every run is one
process at nice 19; its folder <out>/<setting>/<arm>/seed<k>/ holds requests.csv, series.csv and summary.json, and a
finished run is not repeated unless --force is given (it then overwrites the run's files).

Printed per setting: each arm's statistics over the seeds (mean and the spread between seeds), then every arm
against the first one, paired: requests pair by user (play), stream and index, among those both arms sent before the
cut; the interval of a paired difference resamples users (2,000 draws, seed 0) within a seed, and the last columns
give the mean and the range of the paired difference across seeds.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kvsim  # noqa: E402
from kvsim_policy import SWITCHES  # noqa: E402

DRAWS = 2000


def parse_value(cfg: kvsim.Config, key: str, text: str):
    cur = getattr(cfg, key)
    if isinstance(cur, bool):
        return text.lower() in ("1", "on", "true", "yes")
    if isinstance(cur, int):
        return int(float(text))
    if isinstance(cur, float):
        return float(text)
    return text


def make_config(arm: str, setting: str, seed: int) -> kvsim.Config:
    name, _, opts = arm.partition("@")
    cfg = kvsim.Config(arm=name, seed=seed)
    for item in filter(None, setting.split(",")):
        key, _, val = item.partition("=")
        setattr(cfg, key, parse_value(cfg, key, val))
    switches = []
    for opt in filter(None, opts.split("+")):
        if "=" in opt:
            key, _, val = opt.partition("=")
            setattr(cfg, key, parse_value(cfg, key, val))
        elif opt.upper() in SWITCHES or opt.lower() == "all":
            switches.append(opt.upper() if opt.lower() != "all" else "all")
        else:
            raise ValueError(f"arm option {opt!r}: a switch {SWITCHES}, 'all', or key=value")
    if switches:
        cfg.switches = "all" if "all" in switches else ",".join(switches)
    return cfg


def slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "._-=" else "_" for c in text) or "default"


def run_one(job: tuple) -> dict:
    arm, setting, seed, out, cut, force = job
    try:
        os.nice(19)
    except OSError:
        pass
    folder = os.path.join(out, slug(setting), slug(arm), f"seed{seed}")
    spath = os.path.join(folder, "summary.json")
    t0 = time.process_time()
    if force or not os.path.exists(spath):
        cfg = make_config(arm, setting, seed)
        cfg.out = folder
        if cut:
            cfg.cut = cut
        kvsim.run(cfg)
    with open(spath) as f:
        s = json.load(f)
    return {"arm": arm, "setting": setting, "seed": seed, "folder": folder, "summary": s,
            "cpu_s": round(time.process_time() - t0, 1)}


def load_requests(folder: str, cut: float) -> dict:
    """(play, stream, k) -> ttft, of requests sent before the cut."""
    out = {}
    with open(os.path.join(folder, "requests.csv")) as f:
        cols = f.readline().strip().split(",")
        ip, istr, ik, it, ittft = (cols.index(c) for c in ("play", "stream", "k", "t_send", "ttft"))
        for line in f:
            v = line.split(",")
            if float(v[it]) < cut:
                out[(int(v[ip]), int(v[istr]), int(v[ik]))] = float(v[ittft])
    return out


def paired(a: dict, b: dict, rng: np.random.Generator) -> dict:
    """Mean of b - a over the requests both sent, and its 95 % interval over users."""
    keys = [k for k in a if k in b]
    if not keys:
        return {"n": 0}
    plays = np.array([k[0] for k in keys])
    diff = np.array([b[k] - a[k] for k in keys])
    users, inv = np.unique(plays, return_inverse=True)
    sums = np.bincount(inv, weights=diff)
    counts = np.bincount(inv).astype(float)
    idx = rng.integers(0, len(users), size=(DRAWS, len(users)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"n": len(keys), "users": len(users), "diff": float(diff.mean()), "lo": float(lo), "hi": float(hi),
            "a_mean": float(np.mean([a[k] for k in keys])), "b_mean": float(np.mean([b[k] for k in keys]))}


def expand_seeds(text: str) -> list:
    out = []
    for part in text.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def run_matrix(arms: list, settings: list, seeds: list, out: str, *, procs: int = 4, cut: float | None = None,
               quiet: bool = False, force: bool = False) -> list:
    """Run every (arm, setting, seed) with at most `procs` (<= 4) processes; returns the result records."""
    procs = max(1, min(4, procs))
    jobs = [(arm, setting, seed, out, cut, force) for setting in settings for seed in seeds for arm in arms]
    results = []
    t0 = time.time()
    with mp.Pool(procs, maxtasksperchild=1) as pool:
        for res in pool.imap_unordered(run_one, jobs):
            results.append(res)
            if not quiet:
                s = res["summary"]
                print(f"  [{time.time() - t0:5.0f} s] {res['setting'] or 'default'} | {res['arm']} seed {res['seed']}: "
                      f"TTFT {s['ttft_mean']:.2f} s, {s['req_per_min']:.1f} req/min, CPU {res['cpu_s']} s",
                      file=sys.stderr, flush=True)
    return results


def report(results: list, arms: list, settings: list, seeds: list) -> None:
    by = {(r["arm"], r["setting"], r["seed"]): r for r in results}
    rng = np.random.default_rng(0)
    for setting in settings:
        print(f"== setting: {setting or 'default (campaign 34: 128 users, 160 GB, 2 h)'}; seeds {seeds}")
        print(f"{'arm':28s} | TTFT mean (sd over seeds) [min, max] |   p50 |   p90 |   p99 | req/min | SSD share | "
              f"history recomputed | refused reads | first cleaner pass s")
        for arm in arms:
            ss = [by[(arm, setting, k)]["summary"] for k in seeds if (arm, setting, k) in by]
            if not ss:
                continue
            m = np.array([s["ttft_mean"] for s in ss])
            fc = [s["first_cleaner_pass_s"] for s in ss if s["first_cleaner_pass_s"] is not None]
            print(f"{arm:28s} | {m.mean():7.2f} ({m.std(ddof=1) if len(m) > 1 else 0:5.2f}) [{m.min():6.2f}, {m.max():6.2f}] | "
                  f"{np.mean([s['ttft_p50'] for s in ss]):5.2f} | {np.mean([s['ttft_p90'] for s in ss]):5.1f} | "
                  f"{np.mean([s['ttft_p99'] for s in ss]):5.1f} | {np.mean([s['req_per_min'] for s in ss]):7.1f} | "
                  f"{np.mean([s['share_ssd'] for s in ss]):9.3f} | {np.mean([s['history_recomputed'] for s in ss]):18.4f} | "
                  f"{np.mean([s['refusals_before_cut'] for s in ss]):13.1f} | "
                  f"{(np.mean(fc) if fc else float('nan')):8.0f}")
        base = arms[0]
        print(f"paired against {base} (B - A on the requests both sent before the cut; interval over users, 95 %):")
        for arm in arms[1:]:
            diffs, lines = [], []
            for k in seeds:
                if (base, setting, k) not in by or (arm, setting, k) not in by:
                    continue
                cut = by[(base, setting, k)]["summary"]["cut"]
                a = load_requests(by[(base, setting, k)]["folder"], cut)
                b = load_requests(by[(arm, setting, k)]["folder"], cut)
                p = paired(a, b, rng)
                if p["n"]:
                    diffs.append(p["diff"])
                    lines.append(f"seed {k}: {p['diff']:+.2f} [{p['lo']:+.2f}, {p['hi']:+.2f}] on {p['n']}")
            if diffs:
                d = np.array(diffs)
                print(f"  {base} -> {arm}: mean over seeds {d.mean():+.2f} s (sd {d.std(ddof=1) if len(d) > 1 else 0:.2f}, "
                      f"range {d.min():+.2f} to {d.max():+.2f}; sign {'same in every seed' if (d > 0).all() or (d < 0).all() else 'changes between seeds'})")
                for line in lines:
                    print(f"      {line}")
        print()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--set", dest="settings", action="append", default=None, help="a setting: key=value,key=value")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=4, help="processes at once (at most 4)")
    ap.add_argument("--cut", type=float, default=None, help="count requests sent before this (default duration - 600)")
    ap.add_argument("--force", action="store_true", help="run again even where a finished run exists (overwrites it)")
    args = ap.parse_args(argv)
    settings = args.settings or [""]
    seeds = expand_seeds(args.seeds)
    results = run_matrix(args.arms, settings, seeds, args.out, procs=args.procs, cut=args.cut, force=args.force)
    report(results, args.arms, settings, seeds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
