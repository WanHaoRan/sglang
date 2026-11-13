"""Reviewer check 6: the LEAD5 shared turns that have no position in the order run; the rule-1 arms' reading turns elsewhere."""
import os, sys
import statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
rank = order_rank()
runs = {a: Run(a) for a in LEAD5 + ["jit_p50_ctl", "hintfree_ctl", "baseline", "base_rep"]}
keys5 = shared(runs, LEAD5, rank)
nopos = [k for k in keys5 if k not in rank]
haspos = [k for k in keys5 if k in rank]
print(f"LEAD5 shared returning turns: {len(keys5)}; without a position in the order run: {len(nopos)} ({100 * len(nopos) / len(keys5):.1f} %); order run has {len(rank)} positions")
for a in LEAD5:
    R = runs[a]
    ts = [R.by[k][F['t_send']] / 60 for k in nopos]
    print(f"  {a:11s} no-position turns: mean TTFT {mean([R.ttft(k) for k in nopos]):.3f} s, sent minutes {min(ts):.1f}-{max(ts):.1f} (median {st.median(ts):.1f}); read from SSD {sum(R.pages.get(k, 0) > 0 for k in nopos)} | 'elsewhere' with them {mean([R.ttft(k) for k in keys5 if not in_episode(rank, k)]):.3f}, without them {mean([R.ttft(k) for k in haspos if not in_episode(rank, k)]):.3f}")
calm_pos = [k for k in haspos if not in_episode(rank, k)]
print(f"  elsewhere with a position: {len(calm_pos)} turns")
# when in each arm's own clock do the no-position turns fall relative to the second episode window
print("\nsessions (conv ids) of the no-position turns: min", min(k[0] for k in nopos), "max", max(k[0] for k in nopos), "| distinct", len(set(k[0] for k in nopos)))
# share of all TTFT differences nofc_noend - ec_r13 by group
E, N = runs["ec_r13"], runs["nofc_noend"]
tot = sum(N.ttft(k) - E.ttft(k) for k in keys5)
for name, ks in (("episode 1", [k for k in haspos if 5500 <= rank[k] < 7000]), ("episode 2", [k for k in haspos if 8750 <= rank[k] < 11000]), ("elsewhere with a position", calm_pos), ("no position", nopos)):
    s = sum(N.ttft(k) - E.ttft(k) for k in ks)
    print(f"  nofc_noend - ec_r13 in {name:26s}: {len(ks):5d} turns, {s:+8.0f} s = {100 * s / tot:5.1f} % of the net {tot:+.0f} s; per turn {s / len(ks):+.3f}")
