"""Test (f13): does anything about the leading arms hold in both runs of rules 1 and 3?
A. The six runs under statistics that one congestion event cannot carry: capped means, quantiles, turns over a threshold.
B. Each run of rules 1 and 3 against each other arm: sessions and 250-position bins in which it is lower / higher, and
   the paired difference with TTFT capped at 2 and 5 s.
C. Leave one 250-position bin out: the largest change of each pairwise difference, and the bin that makes it.
D. The bin where the two runs of rules 1 and 3 differ most: what the logs show there.
E. Test (a) again with the load matched on live context alone (requests in flight are an outcome of the policy), and
   the time each run spends with live context over 1,300K tokens.
  nice -n 19 python3 -I -B tg_robust.py"""
import itertools, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import lib2 as L

rank = C.order_rank()
NPOS = len(rank)
SIX = C.LEAD5 + ["ec_r13_rep"]
D = {a: C.load(a) for a in SIX}
B = {a: C.by_key(C.kept(D[a])) for a in SIX}
READ = {a: L.read_pages(D[a]) for a in SIX}
keys = C.shared_keys(B, SIX, rank)
send = lambda k: C.mean(B[a][k]["t_send"] for a in SIX)
BINS = L.pos_bins(keys, rank, NPOS, 250, send)

print(f"A. the six runs on their {len(keys)} shared returning turns")
print("  arm          mean | capped at 10 s, 5 s, 2 s, 1 s | p50   p75   p90   p95    p99 | turns over 1, 2, 5, 10, 20 s | share of the summed TTFT in turns over 10 s")
for a in SIX:
    v = [B[a][k]["ttft"] for k in keys]
    print(f"  {a:11s} {C.mean(v):.3f} | {C.mean(min(x, 10) for x in v):.3f} {C.mean(min(x, 5) for x in v):.3f} {C.mean(min(x, 2) for x in v):.3f} {C.mean(min(x, 1) for x in v):.3f} | "
          f"{C.pct(v, .5):.3f} {C.pct(v, .75):.3f} {C.pct(v, .9):.3f} {C.pct(v, .95):.3f} {C.pct(v, .99):6.2f} | " + " ".join(f"{sum(1 for x in v if x > c):5d}" for c in (1, 2, 5, 10, 20))
          + f" | {100 * sum(x for x in v if x > 10) / sum(v):.1f} %")
for cap in (None, 10, 5, 2, 1):
    m = {a: C.mean((min(B[a][k]["ttft"], cap) if cap else B[a][k]["ttft"]) for k in keys) for a in SIX}
    print(f"  order with TTFT {'uncapped' if cap is None else 'capped at ' + str(cap) + ' s':15s}: " + " < ".join(f"{a} {m[a]:.3f}" for a in sorted(m, key=m.get)))

print("\nB. each run of rules 1 and 3 against each other arm (other - rules 1+3 run): mean | capped at 5 s | capped at 2 s | sessions where the rules-1+3 run is lower / higher (p) | 250-bins lower / higher (p) | outside the fixed windows")
for r13 in ("ec_r13", "ec_r13_rep"):
    for o in ("jit_exact", "ec_r1", "nofc", "nofc_noend"):
        d = {k: B[o][k]["ttft"] - B[r13][k]["ttft"] for k in keys}
        sess = {}
        for k in keys:
            sess[k[0]] = sess.get(k[0], 0.0) + d[k]
        sl, sh = sum(1 for v in sess.values() if v > 0), sum(1 for v in sess.values() if v < 0)
        bl = sum(1 for _, _, ks in BINS if sum(d[k] for k in ks) > 0)
        bh = sum(1 for _, _, ks in BINS if sum(d[k] for k in ks) < 0)
        calm = [k for k in keys if not C.in_episode(rank, k)]
        print(f"  {o:10s} - {r13:10s} {C.mean(d.values()):+6.3f} | {C.mean(min(B[o][k]['ttft'], 5) - min(B[r13][k]['ttft'], 5) for k in keys):+6.3f} | {C.mean(min(B[o][k]['ttft'], 2) - min(B[r13][k]['ttft'], 2) for k in keys):+6.3f} | "
              f"{sl:3d} / {sh:3d} (p {L.p2(sl, sl + sh):.3g}) | {bl:2d} / {bh:2d} (p {L.p2(bl, bl + bh):.3f}) | {C.mean(d[k] for k in calm):+6.3f}")
d = {k: B["ec_r13_rep"][k]["ttft"] - B["ec_r13"][k]["ttft"] for k in keys}
print(f"  the same arm twice, ec_r13_rep - ec_r13: {C.mean(d.values()):+6.3f} | capped at 5 s {C.mean(min(B['ec_r13_rep'][k]['ttft'], 5) - min(B['ec_r13'][k]['ttft'], 5) for k in keys):+6.3f} | "
      f"capped at 2 s {C.mean(min(B['ec_r13_rep'][k]['ttft'], 2) - min(B['ec_r13'][k]['ttft'], 2) for k in keys):+6.3f}")

print("\nC. leave one 250-position bin out: pairwise mean difference (second minus first), its range over the 47 leave-one-out sets, and the bin whose removal changes it most")
for a, b in itertools.combinations(SIX, 2):
    d = {k: B[b][k]["ttft"] - B[a][k]["ttft"] for k in keys}
    tot, n = sum(d.values()), len(keys)
    loo = [((tot - sum(d[k] for k in ks)) / (n - len(ks)), label) for label, _, ks in BINS]
    worst = max(loo, key=lambda x: abs(x[0] - tot / n))
    print(f"  {b:10s} - {a:10s} {tot / n:+6.3f}  range {min(x[0] for x in loo):+6.3f}..{max(x[0] for x in loo):+6.3f}  without bin {worst[1]:>5s}: {worst[0]:+6.3f}" + ("  <-- sign changes" if min(x[0] for x in loo) * max(x[0] for x in loo) < 0 else ""))

print("\nD. the bin where the two runs of rules 1 and 3 differ most")
d = {k: B["ec_r13_rep"][k]["ttft"] - B["ec_r13"][k]["ttft"] for k in keys}  # section C reused the name for every pair
worst = max(BINS, key=lambda b: abs(sum(d[k] for k in b[2])))
label, lo, ks = worst
tot = sum(d.values())
print(f"  bin {label} ({len(ks)} turns, {100 * len(ks) / len(keys):.1f} % of the shared turns) carries {100 * sum(d[k] for k in ks) / tot:+.1f} % of the summed difference ec_r13_rep - ec_r13; without it the two runs are "
      f"{(tot - sum(d[k] for k in ks)) / (len(keys) - len(ks)):+.3f} s apart")
print("  per run in that bin: mean TTFT, median, turns over 10 s | mean queue wait s | recomputed tokens per turn, turns recomputing over 4K | turns reading from SSD, pages | minutes sent (first to last) | live context K (mean)")
LS = {a: L.load_state(D[a], True) for a in SIX}
for a in SIX:
    v = [B[a][k]["ttft"] for k in ks]
    q = [D[a]["req"].get(f"{k[0]}-{k[1]}", [0])[0] / 1000 for k in ks]
    ts = [B[a][k]["t_send"] for k in ks]
    print(f"    {a:11s} {C.mean(v):6.3f} {C.pct(v, .5):6.3f} {sum(1 for x in v if x > 10):3d} | {C.mean(q):6.3f} | {C.mean(C.rec(B[a][k]) for k in ks):6.0f} {sum(1 for k in ks if C.rec(B[a][k]) > 4000):3d} | "
          f"{sum(1 for k in ks if READ[a].get(k, 0) > 0):3d} {sum(READ[a].get(k, 0) for k in ks):6d} | {min(ts) / 60:5.1f}-{max(ts) / 60:5.1f} | {C.mean(LS[a][k][1] for k in ks) / 1000:7.1f}")
print("  in the wall-clock window of that bin in each run (first to last send of its turns): Decode lines with a queue, mean queued requests, evicted tokens GPU / host (K), reads refused room, prefill tokens computed (K)")
for a in SIX:
    t0 = D[a]["t0"]
    ts = [B[a][k]["t_send"] for k in ks]
    w = lambda t: min(ts) <= t - t0 < max(ts)
    dec = [x for x in D[a]["dec"] if w(x[0])]
    print(f"    {a:11s} {(max(ts) - min(ts)) / 60:4.1f} min | {100 * C.mean(1 if x[5] > 0 else 0 for x in dec):5.1f} % of {len(dec)} lines, {C.mean(x[5] for x in dec):5.2f} queued | "
          f"{sum(x[1] for x in D[a]['evict_device'] if w(x[0])) / 1000:7.0f} / {sum(x[1] for x in D[a]['evict_host'] if w(x[0])) / 1000:7.0f} | {sum(1 for x in D[a]['disp'] if w(x[0]) and x[5] == 0):3d} | {sum(x[2] for x in D[a]['pre'] if w(x[0])) / 1000:7.0f}")

print("\nE. test (a) with the load matched on live context alone (sessions cut by the run's end kept in progress)")
print("  pair: raw paired difference on all shared turns | on the turns with live context within 1 % in the two runs (n) | within 0.5 % (n) | the same three inside the fixed windows")
ep = [k for k in keys if C.in_episode(rank, k)]
for a, b in itertools.combinations(SIX, 2):
    cells = []
    for kk in (keys, ep):
        dd = {k: B[b][k]["ttft"] - B[a][k]["ttft"] for k in kk}
        m1 = [k for k in kk if abs(LS[b][k][1] - LS[a][k][1]) <= 0.01 * 0.5 * (LS[b][k][1] + LS[a][k][1])]
        m05 = [k for k in kk if abs(LS[b][k][1] - LS[a][k][1]) <= 0.005 * 0.5 * (LS[b][k][1] + LS[a][k][1])]
        cells.append(f"{C.mean(dd.values()):+6.3f} | {C.mean(dd[k] for k in m1):+6.3f} ({len(m1):5d}) | {C.mean(dd[k] for k in m05):+6.3f} ({len(m05):5d})")
    print(f"  {b:10s} - {a:10s} " + " || ".join(cells))
print("  time with live context over a threshold, per run (minutes of the kept window, from the load state at every send), and the shared turns sent then with their mean TTFT")
for a in SIX:
    turns = sorted(C.kept(D[a]), key=lambda t: t["t_send"])
    cells = []
    for thr in (1200e3, 1300e3, 1350e3):
        mins = sum((t2["t_send"] - t1["t_send"]) for t1, t2 in zip(turns, turns[1:]) if LS[a][(t1["conv"], t1["turn"])][1] >= thr) / 60
        hot = [k for k in keys if LS[a][k][1] >= thr]
        cells.append(f"over {thr / 1e3:.0f}K: {mins:5.1f} min, {len(hot):4d} turns at {C.mean(B[a][k]['ttft'] for k in hot):6.3f} s")
    peak = max(LS[a][(t["conv"], t["turn"])][1] for t in turns)
    print(f"    {a:11s} " + " | ".join(cells) + f" | peak {peak / 1e3:.0f}K")

print("\nE2. by the sign of the live-context difference at the same turn (second minus first run): turns and mean paired TTFT difference where the second run's live context is over 1 % lower | within 1 % | over 1 % higher; "
      "sum of the two outer means (zero when the two runs answer load alike)")
for a, b in itertools.combinations(SIX, 2):
    lo_, mid, hi_ = [], [], []
    for k in keys:
        r = (LS[b][k][1] - LS[a][k][1]) / (0.5 * (LS[b][k][1] + LS[a][k][1]))
        (lo_ if r < -0.01 else hi_ if r > 0.01 else mid).append(B[b][k]["ttft"] - B[a][k]["ttft"])
    print(f"  {b:10s} - {a:10s} lower {len(lo_):5d} {C.mean(lo_):+6.3f} | within {len(mid):5d} {C.mean(mid):+6.3f} | higher {len(hi_):5d} {C.mean(hi_):+6.3f} | outer sum {C.mean(lo_) + C.mean(hi_):+6.3f}")

print("\nE3. mean TTFT standardized to one distribution of live context (the pooled six-run share of turns per bin), with the top of the range in 25K bins; raw mean beside it")
edges = [0, 700, 800, 900, 1000, 1050, 1100, 1150, 1200, 1225, 1250, 1275, 1300, 1325, 1350, 1375, 10 ** 6]
cell = {}
for lo, hi in zip(edges, edges[1:]):
    for a in SIX:
        kk = [k for k in keys if lo * 1000 <= LS[a][k][1] < hi * 1000]
        cell[(a, lo)] = (len(kk), C.mean(B[a][k]["ttft"] for k in kk))
tot = {lo: sum(cell[(a, lo)][0] for a in SIX) for lo in edges[:-1]}
print("  bin K (turns, mean TTFT per run): ")
for lo, hi in zip(edges, edges[1:]):
    if lo >= 1200:
        print(f"    {lo:5d}-{hi if hi < 10 ** 6 else 'up':>5} " + "  ".join(f"{a} {cell[(a, lo)][0]:4d} {cell[(a, lo)][1]:6.3f}" for a in SIX))
for a in SIX:
    num = den = 0.0
    for lo in edges[:-1]:
        n, m = cell[(a, lo)]
        if n:
            num += tot[lo] * m
            den += tot[lo]
    print(f"    {a:11s} raw {C.mean(B[a][k]['ttft'] for k in keys):.3f}  standardized {num / den:.3f}  (bins with no turn of this run hold {100 * (1 - den / sum(tot.values())):.1f} % of the pooled turns)")

print("\nF. calibration: the same statistics between two runs of the SAME arm, each pair on its own shared returning turns (second minus first)")
print("  pair: turns | mean | capped at 10 s | capped at 5 s | capped at 2 s | median | sessions where the first is lower / higher (p) | 250-bins lower / higher (p) | turns over 1 s, first / second | turns over 10 s")
PAIRS = (("ec_r13", "ec_r13_rep"), ("jit_p50_ctl", "jit_p50_d"), ("jit_p50_d_cut", "jit_p50_d"), ("jit_p50_ctl", "jit_p50_d_cut"), ("jit_p50", "jit_p50_ctl"), ("jit_p50", "jit_p50_d"), ("hintfree", "hintfree_ctl"), ("baseline", "base_rep"))
for a, b in PAIRS:
    Da, Db = C.by_key(C.kept(C.load(a))), C.by_key(C.kept(C.load(b)))
    kk = sorted(set(k for k in Da if k[1] > 0) & set(k for k in Db if k[1] > 0), key=lambda k: (rank.get(k, 1 << 30), k))
    dd = {k: Db[k]["ttft"] - Da[k]["ttft"] for k in kk}
    sess = {}
    for k in kk:
        sess[k[0]] = sess.get(k[0], 0.0) + dd[k]
    sl, sh = sum(1 for v in sess.values() if v > 0), sum(1 for v in sess.values() if v < 0)
    bb = L.pos_bins(kk, rank, NPOS, 250, lambda k: Da[k]["t_send"])
    bl = sum(1 for _, _, ks_ in bb if sum(dd[k] for k in ks_) > 0)
    bh = sum(1 for _, _, ks_ in bb if sum(dd[k] for k in ks_) < 0)
    cap = lambda c: C.mean(min(Db[k]["ttft"], c) - min(Da[k]["ttft"], c) for k in kk)
    print(f"  {b:13s} - {a:13s} {len(kk):5d} | {C.mean(dd.values()):+6.3f} | {cap(10):+6.3f} | {cap(5):+6.3f} | {cap(2):+6.3f} | {C.pct([Db[k]['ttft'] for k in kk], .5) - C.pct([Da[k]['ttft'] for k in kk], .5):+7.4f} | "
          f"{sl:3d} / {sh:3d} (p {L.p2(sl, sl + sh):.3g}) | {bl:2d} / {bh:2d} (p {L.p2(bl, bl + bh):.3f}) | {sum(1 for k in kk if Da[k]['ttft'] > 1)} / {sum(1 for k in kk if Db[k]['ttft'] > 1)} | "
          f"{sum(1 for k in kk if Da[k]['ttft'] > 10)} / {sum(1 for k in kk if Db[k]['ttft'] > 10)}")
print("  the same between arms, LEAD5 on the six-run shared turns (second minus first): mean | capped at 10 s | 5 s | 2 s | turns over 1 s | over 10 s")
for a, b in itertools.combinations(C.LEAD5, 2):
    cap = lambda c: C.mean(min(B[b][k]["ttft"], c) - min(B[a][k]["ttft"], c) for k in keys)
    print(f"  {b:13s} - {a:13s} {C.mean(B[b][k]['ttft'] - B[a][k]['ttft'] for k in keys):+6.3f} | {cap(10):+6.3f} | {cap(5):+6.3f} | {cap(2):+6.3f} | {sum(1 for k in keys if B[a][k]['ttft'] > 1)} / {sum(1 for k in keys if B[b][k]['ttft'] > 1)} | "
          f"{sum(1 for k in keys if B[a][k]['ttft'] > 10)} / {sum(1 for k in keys if B[b][k]['ttft'] > 10)}")
