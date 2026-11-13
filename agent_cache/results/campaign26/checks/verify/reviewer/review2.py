"""Reviewer check 2: how fragile are the differences between the leading arms, and what does an A/A pair look like
under the same interval. Then the false-positive rate and power of the write-up's rule for the repeats.
  nice -n 19 python3 -I -B review2.py > review2.out
"""
import os, sys, random
import statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

rank = order_rank()
runs = {a: Run(a) for a in LABELS}


def pair(a, b, keys, title, verbose=True):
    """a minus b on keys."""
    d = [(k, runs[a].ttft(k) - runs[b].ttft(k)) for k in keys]
    m, lo, hi, blocks = block_boot(d, rank, 500)
    _, lo1, hi1, _ = block_boot(d, rank, 1000)
    _, lo2, hi2, _ = block_boot(d, rank, 250)
    clo, chi, ncl = conv_boot(d)
    tot = sum(x for _, x in d)
    bs = sorted(blocks.items())
    sums = [s for _, (s, n) in bs]
    pos = sum(s > 0 for s in sums)
    absn = sorted((abs(s) for s in sums), reverse=True)
    kish = sum(absn) ** 2 / sum(x * x for x in absn)
    loo = [(tot - s) / (len(d) - n) for _, (s, n) in bs]
    srt = sorted((x for _, x in d), key=abs, reverse=True)
    top1 = sum(srt[: len(srt) // 100])
    sd = st.pstdev([x for _, x in d])
    print(f"\n{title}: {a} - {b} on {len(d)} turns: mean {m:+.3f} s  block500 [{lo:+.2f},{hi:+.2f}]  block1000 [{lo1:+.2f},{hi1:+.2f}]  block250 [{lo2:+.2f},{hi2:+.2f}]  by-session ({ncl}) [{clo:+.2f},{chi:+.2f}]")
    print(f"   per-turn sd {sd:.2f} s; median paired diff {st.median([x for _, x in d]):+.4f}; turns a>b by >0.5 s: {sum(x > .5 for _, x in d)}, a<b by >0.5 s: {sum(x < -.5 for _, x in d)}")
    print(f"   blocks of 500 positions: {len(sums)}; with a>b: {pos}, a<b: {len(sums) - pos}; effective number of blocks (Kish on |sum|) {kish:.1f}; top-3 blocks hold {100 * sum(absn[:3]) / sum(absn):.0f} % of the absolute block sums")
    print(f"   leave-one-block-out mean: min {min(loo):+.3f} max {max(loo):+.3f}; the 1 % of turns with the largest |diff| sum to {top1:+.0f} s of the net {tot:+.0f} s")
    if verbose:
        print("   block: n, mean diff | " + "  ".join(f"{b * 500}:{n},{s / n:+.2f}" for b, (s, n) in bs))
    return m


keys5 = shared(runs, LEAD5, rank)
print("== between-arm differences among the leading arms (LEAD5 shared returning turns)")
for a, b in (("nofc_noend", "ec_r13"), ("nofc", "ec_r13"), ("ec_r1", "ec_r13"), ("ec_r13", "jit_exact"), ("nofc_noend", "jit_exact"), ("nofc_noend", "nofc"), ("nofc", "ec_r1")):
    pair(a, b, keys5, "LEAD5")
keys11 = shared(runs, ALL11, rank)
print("\n== the same on the ALL11 set (stops after the first episode)")
for a, b in (("nofc_noend", "ec_r13"), ("ec_r13", "jit_exact"), ("nofc_noend", "jit_exact")):
    pair(a, b, keys11, "ALL11", verbose=False)

print("\n\n== A/A: the same arm run twice, on the returning turns the two runs share; does the within-pair interval cover zero?")
for a, b in (("base_rep", "baseline"), ("hintfree_ctl", "hintfree"), ("jit_p50_ctl", "jit_p50"), ("jit_p50_d_cut", "jit_p50_ctl"), ("jit_p50_d_cut", "jit_p50")):
    ks = shared(runs, [a, b], rank)
    m = pair(a, b, ks, "A/A all shared", verbose=False)
    ma, mb = mean([runs[a].ttft(k) for k in ks]), mean([runs[b].ttft(k) for k in ks])
    print(f"   means {ma:.3f} vs {mb:.3f}: |diff| is {100 * abs(m) / ((ma + mb) / 2):.0f} % of the pair's mean")
    ep = [k for k in ks if in_episode(rank, k)]
    d = [(k, runs[a].ttft(k) - runs[b].ttft(k)) for k in ep]
    mm, lo, hi, _ = block_boot(d, rank, 500)
    ea, eb = mean([runs[a].ttft(k) for k in ep]), mean([runs[b].ttft(k) for k in ep])
    print(f"   in the episodes ({len(ep)} turns): {mm:+.3f} [{lo:+.2f},{hi:+.2f}]; means {ea:.2f} vs {eb:.2f}: {100 * abs(mm) / ((ea + eb) / 2):.0f} % of the pair's episode mean")
    # restricted to the ALL11 set for comparability with the write-up
    ks2 = [k for k in keys11 if k in runs[a].by and k in runs[b].by]
    if ks2:
        ma, mb = mean([runs[a].ttft(k) for k in ks2]), mean([runs[b].ttft(k) for k in ks2])
        print(f"   on ALL11's {len(ks2)} turns: {ma:.3f} vs {mb:.3f}, diff {ma - mb:+.3f} = {100 * abs(ma - mb) / ((ma + mb) / 2):.0f} % of the mean")

print("\n\n== the write-up's rule for repeats: 'every pair of runs has the same sign and the smallest between-arm difference exceeds the largest move between two runs of either arm'")
print("   Monte Carlo, normal run-to-run noise with sd sigma on each run's mean, true difference delta (in units of sigma); n runs per arm")
rng = random.Random(1)
N = 200000
for n in (2, 3, 4):
    row = []
    for delta in (0, 0.5, 1, 2, 3, 4, 6):
        hit = same = 0
        for _ in range(N if delta == 0 else N // 4):
            A = [rng.gauss(0, 1) for _ in range(n)]
            B = [rng.gauss(delta, 1) for _ in range(n)]
            if max(A) < min(B):
                same += 1
                if min(B) - max(A) > max(max(A) - min(A), max(B) - min(B)):
                    hit += 1
            elif delta == 0 and max(B) < min(A):
                same += 1
                if min(A) - max(B) > max(max(A) - min(A), max(B) - min(B)):
                    hit += 1
        tot = N if delta == 0 else N // 4
        row.append(f"delta={delta}: same-sign {same / tot:.3f}, rule fires {hit / tot:.3f}")
    print(f"   n={n}: " + " | ".join(row))
print("   (delta = 0 row counts either direction; other rows count the true direction only)")
