"""Reviewer check 1: replicate the headline means, then metrics the write-up does not lead with.
  nice -n 19 python3 -I -B review1.py > review1.out
"""
import os, sys
import statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I drops the script folder from sys.path
from common import *

rank = order_rank()
runs = {a: Run(a) for a in LABELS}

print("== replication: mean TTFT on shared returning turns")
for name, labels in (("ALL11", ALL11), ("LEAD5", LEAD5), ("SEVEN", SEVEN), ("CUT7", CUT7)):
    keys = shared(runs, labels, rank)
    ep = [k for k in keys if in_episode(rank, k)]
    calm = [k for k in keys if not in_episode(rank, k)]
    print(f"\n{name}: {len(keys)} shared returning turns ({len(ep)} in episodes, {len(calm)} elsewhere)")
    print(f"  {'arm':14s} {'mean':>6s} {'med':>6s} {'p90':>6s} {'p95':>6s} {'p99':>6s} {'>2s':>6s} {'>5s':>6s} {'>10s':>6s} | {'episode':>7s} {'calm':>6s} | {'lat mean':>8s} {'lat med':>7s} {'compl':>6s} | {'trim1%':>6s} | read turns, pages, rec/turn")
    # trimmed: drop the 1 % of keys with the largest max-over-arms TTFT
    mx = sorted(keys, key=lambda k: max(runs[a].ttft(k) for a in labels))
    trimmed = mx[: int(len(mx) * 0.99)]
    for a in labels:
        R = runs[a]
        v = [R.ttft(k) for k in keys]
        lat = [R.by[k][F["latency"]] for k in keys]
        comp = [R.by[k][F["completion"]] or 0 for k in keys]
        print(f"  {a:14s} {mean(v):6.3f} {st.median(v):6.3f} {pct(v, .9):6.2f} {pct(v, .95):6.2f} {pct(v, .99):6.2f} "
              f"{sum(x > 2 for x in v):6d} {sum(x > 5 for x in v):6d} {sum(x > 10 for x in v):6d} | "
              f"{mean([R.ttft(k) for k in ep]):7.3f} {mean([R.ttft(k) for k in calm]):6.3f} | {mean(lat):8.2f} {st.median(lat):7.2f} {mean(comp):6.0f} | "
              f"{mean([R.ttft(k) for k in trimmed]):6.3f} | {sum(R.pages.get(k, 0) > 0 for k in keys):5d} {sum(R.pages.get(k, 0) for k in keys):8d} {mean([R.by[k][F['rec']] for k in keys]):6.0f}")

print("\n== throughput on each arm's own clock (all turns with a ttft, first turns included)")
print(f"  {'arm':14s} {'turns<115min':>12s} {'compl tok<115min (M)':>20s} {'convs started':>13s} {'turns<74.7min':>13s} {'first-turn mean TTFT':>20s} {'n first':>7s}")
for a in LABELS:
    R = runs[a]
    w = [r for r in R.every if r[F["t_send"]] < 6900]
    w2 = [r for r in R.every if r[F["t_send"]] < 74.7 * 60]
    first = [r[F["ttft"]] for r in R.kept if r[F["turn"]] == 0]
    print(f"  {a:14s} {len(w):12d} {sum((r[F['completion']] or 0) for r in w) / 1e6:20.3f} {len(set(r[0] for r in w)):13d} {len(w2):13d} {mean(first):20.3f} {len(first):7d}")

print("\n== first turns (turn 0) shared by LEAD5: mean TTFT")
fk = sorted(set.intersection(*(set(k for k in runs[a].by if k[1] == 0) for a in LEAD5)))
print("  n", len(fk), " ".join(f"{a} {mean([runs[a].ttft(k) for k in fk]):.3f}" for a in LEAD5))
keys = shared(runs, LEAD5, rank)
allk = keys + fk
print("  all turns (returning + first) shared by LEAD5:", len(allk), " ".join(f"{a} {mean([runs[a].ttft(k) for k in allk]):.3f}" for a in LEAD5))

print("\n== where the mean comes from: share of an arm's summed TTFT (LEAD5 shared turns) by position stretch and by slowest turns")
for a in LEAD5:
    R = runs[a]
    v = [(R.ttft(k), k) for k in keys]
    tot = sum(x for x, _ in v)
    epi = sum(x for x, k in v if in_episode(rank, k))
    srt = sorted((x for x, _ in v), reverse=True)
    print(f"  {a:12s} total {tot:8.0f} s | in episodes {100 * epi / tot:5.1f} % ({sum(in_episode(rank, k) for k in keys)} turns = {100 * sum(in_episode(rank, k) for k in keys) / len(keys):.1f} % of turns) | slowest 1 % {100 * sum(srt[: len(srt) // 100]) / tot:5.1f} % | slowest 5 % {100 * sum(srt[: len(srt) // 20]) / tot:5.1f} % | turns over 2 s {100 * sum(x for x in srt if x > 2) / tot:5.1f} %")
