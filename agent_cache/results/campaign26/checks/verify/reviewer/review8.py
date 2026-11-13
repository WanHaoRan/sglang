"""Reviewer check 8: bytes the device actually read and wrote (iostat on vdc, 10 s samples) against the pages the engine
read from 'SSD' in the same window. A large shortfall would mean the kernel page cache serves part of the reads."""
import os, sys, json, datetime as dt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
PAGE_B = 64 * 98304
path = os.path.join(HERE, "..", "..", "..", "iostat_vdc.log")
samples = []  # (epoch, rkB/s, wkB/s, util)
t = None
for line in open(path, errors="replace"):
    line = line.strip()
    if len(line) >= 22 and line[2] == "/" and line[5] == "/":
        try:
            t = dt.datetime.strptime(line, "%m/%d/%Y %I:%M:%S %p").replace(tzinfo=dt.timezone.utc).timestamp()
        except ValueError:
            t = None
    elif line.startswith("vdc") and t is not None:
        f = line.split()
        samples.append((t, float(f[2]), float(f[8]), float(f[-1])))
print(f"iostat samples: {len(samples)}, from {dt.datetime.fromtimestamp(samples[0][0], dt.timezone.utc)} to {dt.datetime.fromtimestamp(samples[-1][0], dt.timezone.utc)}")
for a in ("jit_p50_ctl", "hintfree_ctl", "ec_r13", "nofc", "ec_r1", "nofc_noend", "jit_p50_d_cut"):
    R = Run(a)
    lo, hi = R.t0, R.t0 + (6900 if a != "jit_p50_d_cut" else 74.7 * 60)
    s = [x for x in samples if lo <= x[0] < hi]
    if not s:
        print(a, "no iostat samples"); continue
    rd = sum(x[1] for x in s) * 10 * 1024  # bytes
    wr = sum(x[2] for x in s) * 10 * 1024
    pages = sum(p for (c, tt, tend, p, ms) in R.raw["io"] if lo <= tend < hi)
    ms = sum(m for (c, tt, tend, p, m) in R.raw["io"] if lo <= tend < hi)
    new = sum((r[F["prompt"]] or 0) - (r[F["cached"]] or 0) + (r[F["completion"]] or 0) for r in R.every if r[F["t_send"]] < hi - lo)
    util = sorted(x[3] for x in s)
    print(f"{a:13s} window {len(s) * 10 / 60:5.1f} min | device read {rd / 1e12:5.2f} TB vs engine s2h {pages * PAGE_B / 1e12:5.2f} TB ({pages} pages; ratio {rd / (pages * PAGE_B):4.2f}) | read thread busy {ms / 1000:6.0f} s -> {pages * PAGE_B / 1e9 / (ms / 1000):4.2f} GB/s while reading | device write {wr / 1e12:5.2f} TB vs tokens computed x 98,304 B {new * 98304 / 1e12:5.2f} TB | %util median {util[len(util) // 2]:4.1f} p90 {util[int(.9 * len(util))]:4.1f}")
