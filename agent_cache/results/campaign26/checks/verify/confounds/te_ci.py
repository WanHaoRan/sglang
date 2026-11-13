"""Test (e), intervals: is a 3 % gap in the decode step between two runs more than the sampling noise of the measure?
A. Decode ms per step (1000 * running / gen throughput, Decode lines with #queue-req 0, first 60 minutes), median by
   bucket of running requests, with a bootstrap over 5-minute blocks of the run's clock (Decode lines next to each other
   share a batch): 5 % and 95 % of each run's deviation from the median of the task's seven runs.
B. One model for all lines: ms per step = b1 * running + b2 * tokens in the batch (K) + a constant per run (least
   squares, lines with 8-31 running requests). The per-run constant, as a deviation from the mean of the seven in % of
   the mean step, is the run's speed with the batch mix held equal; block bootstrap as in A. A Decode line's throughput
   covers the 40 steps since the previous Decode line and any prefill the scheduler ran between them, so the measure
   carries the arm's prefill work. B2 adds the prefill batches and the new tokens they computed in that interval
   (Prefill lines) to the model; B3 keeps only the lines with no Prefill line in the interval.
C. Trend with the time of day: slope of the per-run constants on the start hour over the runs of 2026-11-11 on
   GPU-605dfc41.
The two runs that finished after the task's list was written (GPU-a7198925) are shown but are not in the seven.
  nice -n 19 python3 -I -B te_ci.py"""
import os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import lib2 as L

SEVEN = ["jit_exact", "jit_p50_ctl", "hintfree_ctl", "ec_r13", "nofc", "ec_r1", "nofc_noend"]
MORE = ["jit_p50_d_cut", "jit_p50_d", "ec_r13_rep"]
RUNS = SEVEN + MORE
D = {a: C.load(a) for a in RUNS}
DRAWS = 1000
NB = 12  # 5-minute blocks in the first hour
hour = {a: (D[a]["t0"] % 86400) / 3600 for a in RUNS}

print("runs by start (UTC): " + "; ".join(f"{a} {time.strftime('%m-%d %H:%M', time.gmtime(D[a]['t0']))}" for a in sorted(RUNS, key=lambda a: D[a]["t0"])))

print("\nA. bucket medians, first 60 minutes: median ms per step (n) | deviation from the median of the seven [5 %, 95 % over 5-minute blocks]")
blocks = {}
for a in RUNS:
    for i, (lo, hi) in enumerate(L.BUCKETS):
        bl = [[] for _ in range(NB)]
        for t, ms, tok in L.decode_steps(D[a], 0, 3600, lo, hi):
            bl[min(NB - 1, int(t // 300))].append(ms)
        blocks[(a, i)] = bl
rng = random.Random(0)
draws = {(a, i): [] for a in RUNS for i in range(3)}
for _ in range(DRAWS):
    for a in RUNS:
        pick = [rng.randrange(NB) for _ in range(NB)]
        for i in range(3):
            v = [x for b in pick for x in blocks[(a, i)][b]]
            draws[(a, i)].append(L.med(v) if v else float("nan"))
point = {(a, i): L.med([x for b in blocks[(a, i)] for x in b]) for a in RUNS for i in range(3)}
ref = [L.med([point[(a, i)] for a in SEVEN]) for i in range(3)]
for a in RUNS:
    cells = []
    for i, (lo, hi) in enumerate(L.BUCKETS):
        dev = []
        for j in range(DRAWS):
            r = L.med([draws[(b, i)][j] for b in SEVEN])
            x = draws[(a, i)][j]
            if x == x and r == r:
                dev.append(100 * (x / r - 1))
        n = sum(len(b) for b in blocks[(a, i)])
        cells.append(f"{lo}-{hi}: {point[(a, i)]:6.2f} ({n:4d}) {100 * (point[(a, i)] / ref[i] - 1):+5.1f} % [{C.pct(dev, .05):+5.1f}, {C.pct(dev, .95):+5.1f}]")
    print(f"  {a:14s} {hour[a]:5.2f}h " + " | ".join(cells) + ("" if a in SEVEN else "   (not among the seven)"))
print("  largest gap between two of the seven per bucket, and its 5 % to 95 % range over the draws (the gap between the fastest and slowest of seven noisy medians is above zero by construction):")
for i, (lo, hi) in enumerate(L.BUCKETS):
    gap = 100 * (max(point[(a, i)] for a in SEVEN) / min(point[(a, i)] for a in SEVEN) - 1)
    g = [100 * (max(draws[(a, i)][j] for a in SEVEN) / min(draws[(a, i)][j] for a in SEVEN) - 1) for j in range(DRAWS)]
    print(f"    {lo}-{hi}: {gap:.1f} % [{C.pct(g, .05):.1f}, {C.pct(g, .95):.1f}]")
print("  noise of one run's bucket median (half the 5 % to 95 % width of its own draws, in % of the median): " + "; ".join(
    f"{lo}-{hi}: " + " ".join(f"{100 * (C.pct([x for x in draws[(a, i)] if x == x], .95) - C.pct([x for x in draws[(a, i)] if x == x], .05)) / 2 / point[(a, i)]:.1f}" for a in SEVEN) for i, (lo, hi) in enumerate(L.BUCKETS)))

day = ["jit_p50_ctl", "hintfree_ctl", "ec_r13", "nofc", "ec_r1", "nofc_noend", "jit_p50_d_cut"]
ROWS = {}
for a in RUNS:
    d = D[a]
    t0 = d["t0"]
    pre = sorted(d["pre"])
    j = 0
    prev = None
    rows = []
    for t, run, token, usage, thr, q in d["dec"]:
        nb = ntok = 0
        while j < len(pre) and pre[j][0] <= t:
            if prev is not None and pre[j][0] > prev:
                nb += 1
                ntok += pre[j][2]
            j += 1
        if prev is not None and 0 <= t - t0 < 3600 and 8 <= run <= 31 and q == 0 and thr > 0:
            # (block, running, tokens K, prefill batches in the interval, new prefill tokens K in the interval, ms per step, seconds since the previous Decode line)
            rows.append((min(NB - 1, int((t - t0) // 300)), run, token / 1000.0, nb, ntok / 1000.0, 1000.0 * run / thr, t - prev))
        prev = t
    ROWS[a] = rows
print("\nB. one model for all Decode lines with no queue, 8-31 running requests, first 60 minutes; per run: lines, share with no Prefill line since the previous Decode line, prefill batches and new tokens (K) per interval, median seconds between Decode lines")
for a in RUNS:
    r = ROWS[a]
    print(f"    {a:14s} {len(r):5d} | pure {100 * sum(1 for x in r if x[3] == 0) / len(r):5.1f} % | {C.mean(x[3] for x in r):5.2f} batches {C.mean(x[4] for x in r):6.2f}K | {L.med([x[6] for x in r]):5.2f} s")


def solve(xx, xy):
    n = len(xy)
    a = [row[:] + [xy[i]] for i, row in enumerate(xx)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(a[r][i]))
        a[i], a[p] = a[p], a[i]
        piv = a[i][i]
        for j in range(i, n + 1):
            a[i][j] /= piv
        for r in range(n):
            if r != i and a[r][i]:
                f = a[r][i]
                for j in range(i, n + 1):
                    a[r][j] -= f * a[i][j]
    return [a[i][n] for i in range(n)]


def fit(title, feats, names, keep=lambda x: True, seed=1):
    """ms per step = sum_k b_k * feats(x)[k] + constant per run; block bootstrap over 5-minute blocks within each run."""
    F = len(names)
    P = F + len(RUNS)

    def suff(rows, c):
        xx = [[0.0] * P for _ in range(P)]
        xy = [0.0] * P
        for x in rows:
            v = feats(x)
            y = x[5]
            for i in range(F):
                for j in range(F):
                    xx[i][j] += v[i] * v[j]
                xx[i][c] += v[i]
                xx[c][i] += v[i]
                xy[i] += v[i] * y
            xx[c][c] += 1.0
            xy[c] += y
        return xx, xy

    def add(stats):
        xx = [[0.0] * P for _ in range(P)]
        xy = [0.0] * P
        for sxx, sxy in stats:
            for i in range(P):
                xy[i] += sxy[i]
                row, srow = xx[i], sxx[i]
                for j in range(P):
                    row[j] += srow[j]
        return xx, xy

    S = {a: [suff([x for x in ROWS[a] if x[0] == b and keep(x)], F + r) for b in range(NB)] for r, a in enumerate(RUNS)}
    used = [(r, x) for r, a in enumerate(RUNS) for x in ROWS[a] if keep(x)]
    beta = solve(*add([s for a in RUNS for s in S[a]]))
    mean_ms = C.mean(x[5] for _, x in used)
    sse = sum((x[5] - sum(b * v for b, v in zip(beta, feats(x))) - beta[F + r]) ** 2 for r, x in used)
    sst = sum((x[5] - mean_ms) ** 2 for _, x in used)

    def effects(b):
        base = C.mean(b[F + RUNS.index(a)] for a in SEVEN)
        return {a: 100 * (b[F + RUNS.index(a)] - base) / mean_ms for a in RUNS}

    def slope(e):
        xs = [hour[a] for a in day]
        ys = [e[a] for a in day]
        mx, my = C.mean(xs), C.mean(ys)
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)

    eff = effects(beta)
    rng = random.Random(seed)
    bs = {a: [] for a in RUNS}
    slopes = []
    for _ in range(DRAWS):
        e = effects(solve(*add([S[a][rng.randrange(NB)] for a in RUNS for _ in range(NB)])))
        for a in RUNS:
            bs[a].append(e[a])
        slopes.append(slope(e))
    print(f"\n{title}")
    print(f"  {len(used)} lines; " + ", ".join(f"{n} {b:.4f}" for n, b in zip(names, beta)) + f"; mean step {mean_ms:.2f} ms; R2 {1 - sse / sst:.3f}; residual sd {(sse / len(used)) ** .5:.2f} ms")
    print("  per run: deviation of its constant from the mean of the seven, % of the mean step [5 %, 95 %] | lines")
    for r, a in enumerate(RUNS):
        lo, hi = C.pct(bs[a], .05), C.pct(bs[a], .95)
        flag = "  <-- interval wholly outside +-3 %" if lo > 3 or hi < -3 else ("  <-- point over 3 %" if abs(eff[a]) > 3 else "")
        print(f"    {a:14s} {hour[a]:5.2f}h {eff[a]:+5.2f} % [{lo:+5.2f}, {hi:+5.2f}] | {sum(1 for rr, _ in used if rr == r):5d}" + ("" if a in SEVEN else "   (not among the seven)") + flag)
    print(f"  largest gap between two of the seven: {max(eff[a] for a in SEVEN) - min(eff[a] for a in SEVEN):.2f} points ({max(SEVEN, key=eff.get)} against {min(SEVEN, key=eff.get)}); among LEAD5: "
          f"{max(eff[a] for a in C.LEAD5) - min(eff[a] for a in C.LEAD5):.2f} points ({max(C.LEAD5, key=eff.get)} against {min(C.LEAD5, key=eff.get)})")
    print(f"  GPU units: runs of campaign 26 on GPU-605dfc41 {C.mean(eff[a] for a in day):+.2f} %, on GPU-a7198925 {C.mean(eff[a] for a in ('jit_p50_d', 'ec_r13_rep')):+.2f} %, campaign 23's unit (jit_exact) {eff['jit_exact']:+.2f} %")
    s = slope(eff)
    print(f"  C. trend over the seven runs of 2026-11-11 on GPU-605dfc41 (02:53 to 15:12): {s:+.3f} % per hour [{C.pct(slopes, .05):+.3f}, {C.pct(slopes, .95):+.3f}] = {s * (hour[day[-1]] - hour[day[0]]):+.2f} % over "
          f"{hour[day[-1]] - hour[day[0]]:.1f} h; draws with a positive slope {100 * sum(1 for x in slopes if x > 0) / len(slopes):.0f} %; per run " + " ".join(f"{eff[a]:+.2f}" for a in day))
    return eff


fit("B1. ms per step = b1 * running + b2 * tokens(K) + constant per run", lambda x: (x[1], x[2]), ("per running request", "per K tokens in the batch"))
fit("B2. the same plus the prefill work in the interval: + b3 * prefill batches / 40 + b4 * new prefill tokens(K) / 40", lambda x: (x[1], x[2], x[3] / 40.0, x[4] / 40.0),
    ("per running request", "per K tokens in the batch", "ms per prefill batch", "ms per K new prefill tokens"))
fit("B3. only the lines with no Prefill line since the previous Decode line: ms per step = b1 * running + b2 * tokens(K) + constant per run", lambda x: (x[1], x[2]),
    ("per running request", "per K tokens in the batch"), keep=lambda x: x[3] == 0)

print("\nB3 medians: lines with no Prefill line in the interval, median ms per step (n) by bucket; deviation from the median of the seven")
pm = {(a, i): [x[5] for x in ROWS[a] if x[3] == 0 and lo <= x[1] <= hi] for a in RUNS for i, (lo, hi) in enumerate(L.BUCKETS)}
pref = [L.med([L.med(pm[(a, i)]) for a in SEVEN]) for i in range(3)]
for a in RUNS:
    print(f"    {a:14s} " + " | ".join(f"{lo}-{hi}: {L.med(pm[(a, i)]):6.2f} ({len(pm[(a, i)]):4d}) {100 * (L.med(pm[(a, i)]) / pref[i] - 1):+5.1f} %" for i, (lo, hi) in enumerate(L.BUCKETS)) + ("" if a in SEVEN else "   (not among the seven)"))
print("  largest gap between two of the seven per bucket: " + ", ".join(f"{lo}-{hi}: {100 * (max(L.med(pm[(a, i)]) for a in SEVEN) / min(L.med(pm[(a, i)]) for a in SEVEN) - 1):.1f} %" for i, (lo, hi) in enumerate(L.BUCKETS)))
