"""Reviewer check 4: (a) reads by the gap before the turn, in replay seconds and in the trace's own seconds (gap / 10);
(b) ec_r13's private slow stretch at positions 3000-3500; (c) running requests in the episodes; (d) rule 3 against its rivals in read share.
  nice -n 19 python3 -I -B review4.py > review4.out
"""
import os, sys
import statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

rank = order_rank()
runs = {a: Run(a) for a in LABELS}
keys5 = shared(runs, LEAD5, rank)

print("== (a) returning turns by the gap slept before them (replay seconds = trace seconds x 10): turns, share that read from SSD, mean TTFT")
BINS = ((0, 2), (2, 5), (5, 10), (10, 15), (15, 30), (30, 60), (60, 120), (120, 1e9))
for a in ("nofc_noend", "ec_r13", "jit_exact", "hintfree_ctl", "baseline"):
    R = runs[a]
    ks = [k for k in R.by if k[1] > 0]
    tot_idle = sum((R.by[k][F["gap_slept"]] or 0) for k in ks)
    tot_reads = sum(R.pages.get(k, 0) > 0 for k in ks)
    print(f"\n{a}: {len(ks)} returning turns, {tot_reads} read from SSD ({100 * tot_reads / len(ks):.1f} %); gap median {st.median([(R.by[k][F['gap_slept']] or 0) for k in ks]):.2f} s")
    for lo, hi in BINS:
        sel = [k for k in ks if lo <= (R.by[k][F["gap_slept"]] or 0) < hi]
        if not sel:
            continue
        rd = sum(R.pages.get(k, 0) > 0 for k in sel)
        print(f"   gap {lo:>4.0f}-{hi if hi < 1e8 else float('inf'):<6.0f} s: {len(sel):6d} turns ({100 * len(sel) / len(ks):5.1f} %), {100 * sum((R.by[k][F['gap_slept']] or 0) for k in sel) / tot_idle:5.1f} % of idle time | read {rd:5d} = {100 * rd / len(sel):5.1f} % of the bin, {100 * rd / max(1, tot_reads):5.1f} % of all reading turns | mean TTFT {mean([R.ttft(k) for k in sel]):6.2f} s")

print("\n\n== (b) ec_r13's slow stretch at positions 3000-3500 (the block where it trails the three other same-day arms by 0.22-0.24 s per turn)")
blk = [k for k in keys5 if 3000 <= rank.get(k, -1) < 3500]
for a in LEAD5:
    R = runs[a]
    v = [R.ttft(k) for k in blk]
    print(f"   {a:11s} mean {mean(v):.3f} p90 {pct(v, .9):.2f} max {max(v):.1f} | turns > 2 s {sum(x > 2 for x in v):3d} | read turns {sum(R.pages.get(k, 0) > 0 for k in blk):3d} staged {sum(k in R.staged for k in blk):3d} refused {sum(k in R.refused for k in blk):3d} | recomputed tokens {sum(R.by[k][F['rec']] for k in blk):7d}")
R = runs["ec_r13"]
worst = sorted(blk, key=lambda k: -R.ttft(k))[:25]
print("   ec_r13's 25 slowest turns there: key, minute, TTFT, queue wait, pages read, staged?, refused, recomputed, gap before | same turn's TTFT in nofc, ec_r1, nofc_noend")
for k in sorted(worst, key=lambda k: R.by[k][F["t_send"]]):
    r = R.by[k]
    q = R.queue.get(k, [float("nan")])[0]
    print(f"     c{k[0]}-t{k[1]:<3d} min {r[F['t_send']] / 60:5.1f} ttft {r[F['ttft']]:6.2f} queue {q:6.2f} pages {R.pages.get(k, 0):4d} staged {int(k in R.staged)} refused {R.refused.get(k, 0)} rec {r[F['rec']]:6d} gap {r[F['gap_slept']] or 0:7.1f} | {runs['nofc'].ttft(k):6.2f} {runs['ec_r1'].ttft(k):6.2f} {runs['nofc_noend'].ttft(k):6.2f}")
tmin = min(R.by[k][F["t_send"]] for k in worst)
tmax = max(R.by[k][F["t_send"]] for k in worst)
print(f"   those 25 turns were sent within minutes {tmin / 60:.1f}-{tmax / 60:.1f}; sum of their TTFT {sum(R.ttft(k) for k in worst):.0f} s of the block's {sum(R.ttft(k) for k in blk):.0f} s; the block's excess over nofc {sum(R.ttft(k) - runs['nofc'].ttft(k) for k in blk):+.0f} s")

print("\n\n== (c) decode batches in the two episodes on each arm's own clock: mean running requests, mean queue, ms per decode step")
for a in LEAD5 + ["jit_p50_ctl", "hintfree_ctl"]:
    R = runs[a]
    # the arm's own time window covering the episode positions among its kept turns
    out = []
    for lo, hi in EPISODES:
        ts = [r[F["t_send"]] for r in R.kept if lo <= rank.get((r[0], r[1]), -1) < hi]
        if len(ts) < 50:
            out.append("      (not reached)      ")
            continue
        t_lo, t_hi = pct(ts, .02), pct(ts, .98)
        d = [x for x in R.raw["decode"] if t_lo <= x[0] <= t_hi]
        run = mean([x[1] for x in d])
        que = mean([x[4] for x in d])
        step = mean([1000 * x[1] / x[3] for x in d if x[3] > 0])
        out.append(f"min {t_lo / 60:5.1f}-{t_hi / 60:5.1f}: running {run:5.2f} queue {que:5.2f} step {step:5.1f} ms")
    print(f"   {a:13s} " + " | ".join(out))

print("\n\n== (d) rule 3 against its rivals: turns reading from SSD on the LEAD5 shared turns, by episode")
for a in LEAD5:
    R = runs[a]
    ep1 = [k for k in keys5 if 5500 <= rank.get(k, -1) < 7000]
    ep2 = [k for k in keys5 if 8750 <= rank.get(k, -1) < 11000]
    calm = [k for k in keys5 if not in_episode(rank, k)]
    f = lambda ks: f"{sum(R.pages.get(k, 0) > 0 for k in ks):5d} ({100 * sum(R.pages.get(k, 0) > 0 for k in ks) / len(ks):4.1f} %)"
    print(f"   {a:11s} all {f(keys5)} | episode 1 {f(ep1)} | episode 2 {f(ep2)} | elsewhere {f(calm)} | pages {sum(R.pages.get(k, 0) for k in keys5):7d} | recomputed tokens/turn {mean([R.by[k][F['rec']] for k in keys5]):5.0f}")
