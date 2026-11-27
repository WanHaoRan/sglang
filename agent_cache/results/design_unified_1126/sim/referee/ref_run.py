"""Referee: run a matrix of arms x settings x seeds on a chosen simulator module (kvsim or kvsim_patched), in the
folder layout of kvsim_compare.py (<out>/<setting>/<arm>/seed<k>/{requests.csv,series.csv,summary.json}).

  python -I -B ref_run.py --sim kvsim_patched --arms hicache ec_full --seeds 0,5 --set "" --set fixes=off --out runs/x

A setting is key=value,key=value over Config fields ("" = defaults). At most 4 processes (default 3), each at nice 19.
"""
from __future__ import annotations

import argparse
import importlib
import multiprocessing as mp
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.dirname(HERE)
sys.path.insert(0, SIM_DIR)
sys.path.insert(0, HERE)


def slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "._-=" else "_" for c in text) or "default"


def parse_value(cur, text: str):
    if isinstance(cur, bool):
        return text.lower() in ("1", "on", "true", "yes")
    if isinstance(cur, int):
        return int(float(text))
    if isinstance(cur, float):
        return float(text)
    return text


def work(job):
    simname, arm, setting, seed, out, force = job
    try:
        os.nice(19)
    except OSError:
        pass
    folder = os.path.join(out, slug(setting), slug(arm), f"seed{seed}")
    if not force and os.path.exists(os.path.join(folder, "summary.json")):
        return job, 0.0, None
    kvsim = importlib.import_module(simname)
    name, _, opts = arm.partition("@")
    cfg = kvsim.Config(arm=name, seed=seed)
    for item in filter(None, setting.split(",")):
        key, _, val = item.partition("=")
        setattr(cfg, key, parse_value(getattr(cfg, key), val))
    switches = []
    for opt in filter(None, opts.split("+")):
        if "=" in opt:
            key, _, val = opt.partition("=")
            setattr(cfg, key, parse_value(getattr(cfg, key), val))
        else:
            switches.append(opt.upper() if opt.lower() != "all" else "all")
    if switches:
        cfg.switches = "all" if "all" in switches else ",".join(switches)
    cfg.out = folder
    t0 = time.process_time()
    sim = kvsim.run(cfg)
    s = kvsim.summary(sim)
    return job, time.process_time() - t0, s["ttft_mean"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="kvsim")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--set", dest="settings", action="append", default=None)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    seeds = []
    for part in a.seeds.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    jobs = [(a.sim, arm, setting, seed, a.out, a.force) for setting in (a.settings or [""]) for seed in seeds for arm in a.arms]
    t0 = time.time()
    with mp.Pool(max(1, min(4, a.procs)), maxtasksperchild=1) as pool:
        for job, cpu, ttft in pool.imap_unordered(work, jobs):
            if ttft is not None:
                print(f"[{time.time() - t0:5.0f} s] {job[0]} {job[2] or 'default'} | {job[1]} seed {job[3]}: TTFT {ttft:.2f} s, CPU {cpu:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
