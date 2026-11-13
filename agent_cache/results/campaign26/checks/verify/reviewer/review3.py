"""Reviewer check 3: end-to-end turn latency instead of TTFT; what the hint says on the turns that need an SSD read;
what rule 1 decided for the turns that ended up reading; first turns.
  nice -n 19 python3 -I -B review3.py > review3.out
"""
import os, sys
import statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

rank = order_rank()
runs = {a: Run(a) for a in LABELS}
keys5 = shared(runs, LEAD5, rank)
keys11 = shared(runs, ALL11, rank)
lat = lambda a, k: runs[a].by[k][F["latency"]]

print("== end-to-end turn latency (client 'latency', s) and its part after the first token, on shared returning turns")
for name, labels, keys in (("ALL11", ALL11, keys11), ("LEAD5", LEAD5, keys5)):
    print(f"\n{name}: {len(keys)} turns")
    print(f"  {'arm':14s} {'TTFT':>6s} {'latency':>8s} {'after 1st token':>15s} {'lat p50':>8s} {'lat p90':>8s} {'lat p99':>8s} | summed latency h | ms per completion token after the 1st token")
    for a in labels:
        v = [lat(a, k) for k in keys]
        t = [runs[a].ttft(k) for k in keys]
        post = [l - x for l, x in zip(v, t)]
        comp = sum((runs[a].by[k][F["completion"]] or 0) for k in keys)
        print(f"  {a:14s} {mean(t):6.3f} {mean(v):8.3f} {mean(post):15.3f} {st.median(v):8.2f} {pct(v, .9):8.2f} {pct(v, .99):8.2f} | {sum(v) / 3600:8.2f} | {1000 * sum(post) / comp:6.1f}")

print("\n== paired differences in latency and in TTFT, block bootstrap over 500 positions [5 %, 95 %]")
for name, keys in (("ALL11", keys11), ("LEAD5", keys5)):
    for a, b in (("ec_r13", "nofc_noend"), ("ec_r13", "nofc"), ("jit_exact", "nofc_noend"), ("jit_exact", "nofc"), ("ec_r13", "jit_exact"), ("ec_r1", "nofc")):
        dl = [(k, lat(a, k) - lat(b, k)) for k in keys]
        dt_ = [(k, runs[a].ttft(k) - runs[b].ttft(k)) for k in keys]
        m, lo, hi, _ = block_boot(dl, rank)
        mt, lot, hit, _ = block_boot(dt_, rank)
        print(f"  {name} {a:10s} - {b:10s}: latency {m:+.3f} [{lo:+.2f},{hi:+.2f}]   TTFT {mt:+.3f} [{lot:+.2f},{hit:+.2f}]   after first token {m - mt:+.3f}")

print("\n== A/A pairs on latency (same arm twice), all returning turns the pair shares")
for a, b in (("base_rep", "baseline"), ("hintfree_ctl", "hintfree"), ("jit_p50_ctl", "jit_p50"), ("jit_p50_d_cut", "jit_p50_ctl")):
    ks = shared(runs, [a, b], rank)
    dl = [(k, lat(a, k) - lat(b, k)) for k in ks]
    m, lo, hi, _ = block_boot(dl, rank)
    print(f"  {a:14s} - {b:12s} ({len(ks)} turns): latency {m:+.3f} [{lo:+.2f},{hi:+.2f}]  means {mean([lat(a, k) for k in ks]):.2f} vs {mean([lat(b, k) for k in ks]):.2f}")


def hint_for(R, k):
    """The hint sent with the session's previous turn: what the engine was told about the gap before turn k."""
    prev = R.by.get((k[0], k[1] - 1))
    if prev is None:
        prev = next((r for r in R.every if r[0] == k[0] and r[1] == k[1] - 1), None)
    return prev


print("\n\n== what the three-quantile hint said about the gap before a turn, by whether the turn read from SSD")
for a in ("nofc_noend", "ec_r13", "jit_p50_ctl"):
    R = runs[a]
    prev_of = {}
    for r in R.every:
        prev_of[(r[0], r[1] + 1)] = r
    rows = {"read": [], "no read": []}
    for k in R.by:
        if k[1] == 0:
            continue
        p = prev_of.get(k)
        if p is None:
            continue
        rows["read" if R.pages.get(k, 0) > 0 else "no read"].append((k, p))
    print(f"\n{a}: returning turns {sum(len(v) for v in rows.values())}")
    for cls, v in rows.items():
        gaps = [R.by[k][F["gap_slept"]] or 0.0 for k, _ in v]
        have = [(k, p) for k, p in v if p[F["p50"]] is not None and p[F["true_gap"]] is not None]
        three = [(k, p) for k, p in have if p[F["p10"]] is not None and p[F["p90"]] is not None]
        tg = [p[F["true_gap"]] for _, p in three]
        above = sum(p[F["true_gap"]] > p[F["p90"]] for _, p in three)
        below = sum(p[F["true_gap"]] < p[F["p10"]] for _, p in three)
        pooled = sum(p[F["matched"]] == "*" for _, p in have)
        nohint = len(v) - len(have)
        err = [p[F["true_gap"]] - p[F["p50"]] for _, p in have]
        within2 = sum(abs(e) <= 2 for e in err)
        within5 = sum(abs(e) <= 5 for e in err)
        width = [p[F["p90"]] - p[F["p10"]] for _, p in three]
        print(f"  {cls:8s} n {len(v):5d} | gap before the turn: median {st.median(gaps):7.1f} s, p90 {pct(gaps, .9):7.1f}, share >= 30 s {100 * sum(g >= 30 for g in gaps) / len(gaps):5.1f} %, >= 60 s {100 * sum(g >= 60 for g in gaps) / len(gaps):5.1f} %, >= 120 s {100 * sum(g >= 120 for g in gaps) / len(gaps):5.1f} %")
        print(f"           hint: none or ended-entry {100 * nohint / len(v):5.1f} %, pooled '*' entry {100 * pooled / max(1, len(have)):5.1f} % of hinted | true gap above p90 {100 * above / max(1, len(three)):5.1f} %, below p10 {100 * below / max(1, len(three)):5.1f} % | |true - p50| <= 2 s {100 * within2 / max(1, len(have)):5.1f} %, <= 5 s {100 * within5 / max(1, len(have)):5.1f} % | p90-p10 width median {st.median(width) if width else float('nan'):6.1f} s | true gap median {st.median(tg) if tg else float('nan'):7.1f}, p50 hint median {st.median([p[F['p50']] for _, p in have]) if have else float('nan'):6.1f}")

print("\n\n== rule 1 on the turns that read from SSD (ec arms): last start decision logged before the turn arrived")
for a in ("ec_r13", "ec_r1"):
    R = runs[a]
    dec = {}
    for c, t, ts, decision, tokens, start_ms, idle_ms, read_ms, wait_ms in R.raw["ec_start"]:
        dec.setdefault((c, t), []).append((ts, decision, tokens, start_ms, idle_ms))
    readers = [k for k in keys5 if R.pages.get(k, 0) > 0]
    out = {}
    stall_by = {}
    for k in readers:
        arr = R.arrive.get(k)
        tarr = arr[0] if arr else None
        last = None
        for ts, decision, tokens, start_ms, idle_ms in dec.get(k, ()):
            if tarr is None or ts <= tarr:
                last = decision
        staged = k in R.staged
        first_io_end = min(x[0] for x in R.io[k])
        landed = tarr is not None and max(x[0] for x in R.io[k]) <= tarr
        cls = ("staged" if staged else "not staged") + ", last decision " + str(last)
        out[cls] = out.get(cls, 0) + 1
        s = stall_by.setdefault(cls, [0, 0.0])
        s[0] += landed
        s[1] += R.ttft(k)
    print(f"\n{a}: {len(readers)} reading turns among the LEAD5 shared turns")
    for cls, n in sorted(out.items(), key=lambda x: -x[1]):
        print(f"   {cls:42s} {n:5d} ({100 * n / len(readers):4.1f} %)  all reads landed by arrival: {stall_by[cls][0]:4d}  mean TTFT {stall_by[cls][1] / n:5.2f} s")
    # the gap before these turns against the hint
    prev_of = {(r[0], r[1] + 1): r for r in R.every}
    for cls_name, sel in (("reading turns never staged", [k for k in readers if k not in R.staged]), ("reading turns staged", [k for k in readers if k in R.staged])):
        three = [prev_of[k] for k in sel if k in prev_of and prev_of[k][F["p10"]] is not None and prev_of[k][F["p90"]] is not None and prev_of[k][F["true_gap"]] is not None]
        if three:
            print(f"   {cls_name}: {len(sel)}; with a three-quantile hint {len(three)}: true gap above p90 {100 * sum(p[F['true_gap']] > p[F['p90']] for p in three) / len(three):.0f} %, between p10 and p90 {100 * sum(p[F['p10']] <= p[F['true_gap']] <= p[F['p90']] for p in three) / len(three):.0f} %, below p10 {100 * sum(p[F['true_gap']] < p[F['p10']] for p in three) / len(three):.0f} %; median true gap {st.median([p[F['true_gap']] for p in three]):.1f} s, median p50 {st.median([p[F['p50']] for p in three]):.1f} s, median p90 {st.median([p[F['p90']] for p in three]):.1f} s")

print("\n\n== first turns (turn 0): paired TTFT differences on the first turns shared by LEAD5, block bootstrap by position")
fk = sorted(set.intersection(*(set(k for k in runs[a].by if k[1] == 0) for a in LEAD5)), key=lambda k: rank.get(k, 1 << 30))
for a, b in (("ec_r13", "nofc_noend"), ("ec_r13", "nofc"), ("ec_r1", "nofc"), ("ec_r13", "jit_exact")):
    d = [(k, runs[a].ttft(k) - runs[b].ttft(k)) for k in fk]
    m, lo, hi, _ = block_boot(d, rank)
    print(f"  {a} - {b} on {len(fk)} first turns: {m:+.3f} [{lo:+.2f},{hi:+.2f}]  medians {st.median([runs[a].ttft(k) for k in fk]):.3f} vs {st.median([runs[b].ttft(k) for k in fk]):.3f}")
