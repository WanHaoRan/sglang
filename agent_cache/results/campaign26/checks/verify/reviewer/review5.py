"""Reviewer check 5: (a) where ec_r13's lead over the never-stage arms sits (turns it staged, turns whose reads had landed, the rest);
(b) total context of the sessions in progress against the pools, to size the host memory that would hold it;
(c) large recomputations and refused launches per arm; (d) power of a plain two-sample test on per-run means.
  nice -n 19 python3 -I -B review5.py > review5.out
"""
import os, sys, random, bisect
import statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

rank = order_rank()
runs = {a: Run(a) for a in LABELS}
keys5 = shared(runs, LEAD5, rank)
E = runs["ec_r13"]

print("== (a) ec_r13's lead by what ec_r13 did with the turn (LEAD5 shared returning turns); seconds per shared turn")
landed = set()
for k in keys5:
    if E.pages.get(k, 0) > 0 and k in E.arrive and max(x[0] for x in E.io[k]) <= E.arrive[k][0]:
        landed.add(k)
parts = {
    "staged before arrival, all reads landed by arrival": [k for k in keys5 if k in E.staged and k in landed],
    "staged before arrival, not landed in time": [k for k in keys5 if k in E.staged and k not in landed],
    "read from SSD, never staged": [k for k in keys5 if k not in E.staged and E.pages.get(k, 0) > 0],
    "no SSD read in ec_r13": [k for k in keys5 if k not in E.staged and E.pages.get(k, 0) == 0],
}
for other in ("nofc", "nofc_noend", "ec_r1", "jit_exact"):
    O = runs[other]
    tot = sum(O.ttft(k) - E.ttft(k) for k in keys5)
    print(f"\n  {other} - ec_r13: {tot / len(keys5):+.3f} s per shared turn")
    for name, ks in parts.items():
        s = sum(O.ttft(k) - E.ttft(k) for k in ks)
        rd = sum(O.pages.get(k, 0) > 0 for k in ks)
        print(f"     {name:52s} {len(ks):6d} turns: {s / len(keys5):+.3f} s per shared turn ({100 * s / tot:5.1f} % of the lead) | mean TTFT here {mean([E.ttft(k) for k in ks]):5.2f} vs {mean([O.ttft(k) for k in ks]):5.2f} | of them reading in {other}: {rd}")

print("\n\n== (b) context held by the sessions in progress (prompt + completion of each session's latest finished turn), sampled every 30 s")
TOK_B = 98304
for a in ("nofc_noend", "ec_r13", "baseline"):
    R = runs[a]
    by_conv = {}
    for r in R.every:
        by_conv.setdefault(r[0], []).append(r)
    spans = []
    for c, v in by_conv.items():
        v.sort(key=lambda r: r[F["t_send"]])
        ends = [(r[F["t_send"]] + (r[F["latency"]] or 0), (r[F["prompt"]] or 0) + (r[F["completion"]] or 0)) for r in v]
        spans.append((v[0][F["t_send"]], ends[-1][0], [e[0] for e in ends], [e[1] for e in ends], [r[F["t_send"]] for r in v], [(r[F["prompt"]] or 0) for r in v]))
    samples = []
    t = 600.0
    while t < R.end:
        n = 0
        tok = 0
        for t_first, t_last, ends, sizes, sends, prompts in spans:
            if not t_first <= t <= t_last:
                continue
            n += 1
            i = bisect.bisect_right(ends, t) - 1
            if i >= 0:
                tok += sizes[i]
            else:
                tok += prompts[0]
        samples.append((t, n, tok))
        t += 30
    toks = [s[2] for s in samples]
    print(f"  {a:11s} sessions in progress: median {st.median([s[1] for s in samples]):.0f}; context tokens: median {st.median(toks) / 1e6:.2f} M, p90 {pct(toks, .9) / 1e6:.2f} M, max {max(toks) / 1e6:.2f} M = {max(toks) * TOK_B / 1e9:.0f} GB of KV (p90 {pct(toks, .9) * TOK_B / 1e9:.0f} GB); GPU pool 0.668 M, host pool 0.712 M tokens (70 GB)")
    over = sum(x > 712128 for x in toks)
    print(f"              samples with more context than the 70 GB host pool holds: {100 * over / len(toks):.0f} %; than a 1.628 M-token pool (160 GB): {100 * sum(x > 1627648 for x in toks) / len(toks):.0f} %")

print("\n\n== (c) large recomputations and refused read launches on the LEAD5 shared returning turns")
print(f"  {'arm':11s} {'rec tokens (M)':>14s} {'turns rec>=8K':>13s} {'their TTFT sum s':>16s} {'share of rec in them':>20s} {'turns refused room':>18s} {'refused & rec>=8K':>17s}")
for a in LEAD5:
    R = runs[a]
    rec = [(R.by[k][F["rec"]], k) for k in keys5]
    tot = sum(x for x, _ in rec)
    big = [(x, k) for x, k in rec if x >= 8000]
    print(f"  {a:11s} {tot / 1e6:14.2f} {len(big):13d} {sum(R.ttft(k) for _, k in big):16.0f} {100 * sum(x for x, _ in big) / tot:19.0f}% {sum(k in R.refused for k in keys5):18d} {sum(k in R.refused for _, k in big):17d}")

print("\n\n== (d) a pooled two-sample t test on per-run means, alpha 0.05 two-sided: power by runs per arm and true difference (in run-to-run sd)")
TCRIT = {2: 4.303, 3: 2.776, 4: 2.447, 5: 2.306, 6: 2.228, 8: 2.145}
rng = random.Random(2)
for n, tc in TCRIT.items():
    row = []
    for delta in (0, 1, 1.5, 2, 3, 4):
        hit = 0
        N = 40000
        for _ in range(N):
            A = [rng.gauss(0, 1) for _ in range(n)]
            B = [rng.gauss(delta, 1) for _ in range(n)]
            sp = ((st.variance(A) + st.variance(B)) / 2) ** 0.5
            t = (st.fmean(B) - st.fmean(A)) / (sp * (2 / n) ** 0.5)
            hit += abs(t) > tc
        row.append(f"{delta}: {hit / N:.2f}")
    print(f"   n={n} per arm: " + "  ".join(row))
