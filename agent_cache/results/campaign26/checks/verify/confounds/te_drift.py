"""Test (e): hardware drift across the day.
A. Decode ms per step = 1000 * running / gen throughput on Decode batch lines with #queue-req: 0, median by bucket of running
   requests (8-15, 16-23, 24-31), over the first 60 minutes of each run (from the client's t_start).
B. The same by 20-minute third. C. The same matched on the batch's #token (100K bins). D. The first 8 minutes, where every
   arm serves the same requests in the same cache state: per-token decode time and first-turn prefill, paired by turn.
E. Trend with the time of day over the runs of 2026-11-11.
  nice -n 19 python3 -I -B te_drift.py"""
import math, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

ARMS = ["jit_exact", "jit_p50_ctl", "hintfree_ctl", "ec_r13", "nofc", "ec_r1", "nofc_noend"]  # by start time
EXTRA = ["jit_p50_d_cut", "jit_p50", "hintfree", "baseline", "base_rep"]
BUCKETS = ((8, 15), (16, 23), (24, 31))
D = {a: C.load(a) for a in ARMS + EXTRA}
hour = {a: (D[a]["t0"] % 86400) / 3600 for a in D}
print("runs by start (UTC): " + "; ".join(f"{a} {time.strftime('%m-%d %H:%M', time.gmtime(D[a]['t0']))}" for a in sorted(D, key=lambda a: D[a]['t0'])))


def med(v):
    v = sorted(v)
    n = len(v)
    return float("nan") if not n else (v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2]))


def steps(a, t_lo, t_hi, lo, hi, tok=None):
    t0 = D[a]["t0"]
    return [1000.0 * run / thr for t, run, token, usage, thr, q in D[a]["dec"]
            if t_lo <= t - t0 < t_hi and lo <= run <= hi and q == 0 and thr > 0 and (tok is None or tok[0] <= token < tok[1])]


print("\nA. first 60 minutes, Decode lines with #queue-req 0: median ms per step (n) by running requests; deviation from the median of the seven runs")
tab = {}
for a in ARMS + EXTRA:
    tab[a] = [steps(a, 0, 3600, lo, hi) for lo, hi in BUCKETS]
ref = [med([med(tab[a][i]) for a in ARMS]) for i in range(3)]
for a in ARMS + EXTRA:
    cells = [f"{lo}-{hi}: {med(tab[a][i]):6.2f} ({len(tab[a][i]):4d}) {100 * (med(tab[a][i]) / ref[i] - 1):+5.1f} %" for i, (lo, hi) in enumerate(BUCKETS)]
    flag = " <-- over 3 %" if any(abs(med(tab[a][i]) / ref[i] - 1) > 0.03 for i in range(3) if len(tab[a][i]) >= 20) else ""
    sep = "  (not among the seven) " if a in EXTRA else "  "
    print(f"  {a:14s} start {hour[a]:5.2f}h{sep}" + " | ".join(cells) + flag)
print("  largest gap between two of the seven runs per bucket: " + ", ".join(
    f"{lo}-{hi}: {100 * (max(med(tab[a][i]) for a in ARMS) / min(med(tab[a][i]) for a in ARMS) - 1):.1f} %" for i, (lo, hi) in enumerate(BUCKETS)))
print("  share of Decode lines in the first 60 minutes that have #queue-req 0, and mean #token (K) on the lines used, per bucket:")
for a in ARMS:
    t0 = D[a]["t0"]
    lines = [x for x in D[a]["dec"] if 0 <= x[0] - t0 < 3600]
    cells = []
    for lo, hi in BUCKETS:
        x = [r for r in lines if lo <= r[1] <= hi and r[5] == 0 and r[4] > 0]
        cells.append(f"{C.mean(r[2] for r in x) / 1000:6.0f}K")
    print(f"    {a:14s} {100 * sum(1 for r in lines if r[5] == 0) / len(lines):5.1f} % of {len(lines)} | " + " ".join(cells))

print("\nB. by 20-minute third of the first hour: median ms per step (n)")
for a in ARMS:
    cells = []
    for m in (0, 20, 40):
        cells.append(" ".join(f"{med(steps(a, m * 60, (m + 20) * 60, lo, hi)):6.2f} ({len(steps(a, m * 60, (m + 20) * 60, lo, hi)):3d})" for lo, hi in BUCKETS))
    print(f"  {a:14s} " + " | ".join(cells))

print("\nC. matched on the batch's #token (100K bins), first 60 minutes: cells (running bucket x token bin) with 8 or more lines in every one of the seven runs;")
print("   per run the geometric mean over cells of (run's cell median / median of the seven cell medians)")
cells = []
for lo, hi in BUCKETS:
    for tb in range(0, 2000, 100):
        v = {a: steps(a, 0, 3600, lo, hi, (tb * 1000, (tb + 100) * 1000)) for a in ARMS}
        if all(len(v[a]) >= 8 for a in ARMS):
            cells.append(((lo, hi, tb), {a: med(v[a]) for a in ARMS}, min(len(v[a]) for a in ARMS)))
print(f"   {len(cells)} cells: " + " ".join(f"{c[0][0]}-{c[0][1]}@{c[0][2]}K" for c in cells))
for a in ARMS:
    logs = [math.log(c[1][a] / med(list(c[1].values()))) for c in cells]
    print(f"  {a:14s} geometric mean {100 * (math.exp(C.mean(logs)) - 1):+5.2f} %   median cell {100 * (math.exp(med(logs)) - 1):+5.2f} %   (cell range {100 * (math.exp(min(logs)) - 1):+5.1f} .. {100 * (math.exp(max(logs)) - 1):+5.1f} %)")
print("   cell medians (ms per step) and the smallest line count in the cell:")
for c in cells:
    print(f"     {c[0][0]:2d}-{c[0][1]:2d} running, {c[0][2]:4d}K tokens (n>={c[2]:3d}): " + " ".join(f"{a} {c[1][a]:6.2f}" for a in ARMS))

print("\nD. the first 8 minutes (before any SSD read in any arm): the same turns, paired")
B8 = {a: {(t["conv"], t["turn"]): t for t in D[a]["turns"] if t["ttft"] is not None and t["t_send"] + (t["latency"] or 0) < 480} for a in ARMS}
k8 = sorted(set.intersection(*(set(B8[a]) for a in ARMS)))
first_read = {a: min((x[0] - D[a]["t0"] for x in D[a]["s2h"]), default=float("nan")) for a in ARMS}
print("  first s2h read (s after start): " + " ".join(f"{a} {first_read[a]:.0f}" for a in ARMS))
print(f"  {len(k8)} turns finished within 480 s in all seven runs; prompt and completion tokens identical across runs for "
      f"{sum(1 for k in k8 if len(set((B8[a][k]['pt'], B8[a][k]['ct']) for a in ARMS)) == 1)} of them")
same = [k for k in k8 if len(set((B8[a][k]["pt"], B8[a][k]["ct"]) for a in ARMS)) == 1 and B8[ARMS[0]][k]["ct"] >= 20]
print(f"  on {len(same)} turns with identical token counts and 20 or more completion tokens: per run, median decode ms per token ((latency - ttft) / completion tokens), mean latency s, mean ttft s, "
      "median paired ratio of decode time to the seven-run median of the same turn")
for a in ARMS:
    per = [1000 * (B8[a][k]["latency"] - B8[a][k]["ttft"]) / B8[a][k]["ct"] for k in same]
    ratio = [(B8[a][k]["latency"] - B8[a][k]["ttft"]) / med([B8[b][k]["latency"] - B8[b][k]["ttft"] for b in ARMS]) for k in same]
    print(f"  {a:14s} {med(per):6.2f} ms/token | latency {C.mean(B8[a][k]['latency'] for k in same):6.3f} | ttft {C.mean(B8[a][k]['ttft'] for k in same):6.3f} | ratio {med(ratio):6.4f} ({100 * (med(ratio) - 1):+5.2f} %) "
          f"p10 {C.pct(ratio, .1):.3f} p90 {C.pct(ratio, .9):.3f}")
print("  first turns (turn 0) among them with over 2K prompt tokens and no cached tokens: ms per prompt token of (ttft - server queue wait), median; paired ratio to the seven-run median")
t0k = [k for k in k8 if k[1] == 0 and all((B8[a][k]["pt"] or 0) > 2000 and not (B8[a][k]["cached"] or 0) and f"{k[0]}-0" in D[a]["req"] for a in ARMS)]
for a in ARMS:
    pre = {k: (B8[a][k]["ttft"] - D[a]["req"][f"{k[0]}-0"][0] / 1000) / B8[a][k]["pt"] * 1000 for k in t0k}
    D[a]["_pre"] = pre
for a in ARMS:
    ratio = [D[a]["_pre"][k] / med([D[b]["_pre"][k] for b in ARMS]) for k in t0k]
    print(f"  {a:14s} {med(list(D[a]['_pre'].values())):7.4f} ms/token ({len(t0k)} turns) | ratio {med(ratio):6.4f} ({100 * (med(ratio) - 1):+5.2f} %)")

print("\nD2. whole run: first turns with no queue wait (under 50 ms) and over 2K uncomputed tokens, ms per token of (ttft - queue), median (n); SSD ms per page in reads of 500 pages or more, median (n)")
for a in ARMS + EXTRA:
    fresh = []
    for t in D[a]["turns"]:
        if t["turn"] == 0 and t["ttft"] is not None:
            r = D[a]["req"].get(f"{t['conv']}-0")
            unc = (t["pt"] or 0) - (t["cached"] or 0)
            if r and r[0] < 50 and unc > 2000:
                fresh.append((t["ttft"] - r[0] / 1000) / unc * 1000)
    big = [x[5] / x[3] for x in D[a]["s2h"] if x[3] >= 500]
    print(f"  {a:14s} prefill {med(fresh):7.4f} ({len(fresh):3d}) | SSD {med(big):6.3f} ms/page ({len(big):4d})")

print("\nE. trend with the time of day, runs of 2026-11-11 in start order: least-squares slope of the bucket median on the start hour, in % of the mean per hour; Spearman rank correlation")
day = ["jit_p50_ctl", "hintfree_ctl", "ec_r13", "nofc", "ec_r1", "nofc_noend", "jit_p50_d_cut"]


def slope(xs, ys):
    mx, my = C.mean(xs), C.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx


def spearman(xs, ys):
    rx = {v: i for i, v in enumerate(sorted(xs))}
    ry = {v: i for i, v in enumerate(sorted(ys))}
    n = len(xs)
    return 1 - 6 * sum((rx[x] - ry[y]) ** 2 for x, y in zip(xs, ys)) / (n * (n * n - 1))


for i, (lo, hi) in enumerate(BUCKETS):
    xs = [hour[a] for a in day]
    ys = [med(tab[a][i]) for a in day]
    print(f"  {lo}-{hi} running: " + " ".join(f"{a} {y:.2f}" for a, y in zip(day, ys)) + f" | slope {100 * slope(xs, ys) / C.mean(ys):+.2f} % per hour, {100 * slope(xs, ys) / C.mean(ys) * (xs[-1] - xs[0]):+.1f} % over {xs[-1] - xs[0]:.1f} h | Spearman {spearman(xs, ys):+.2f}")
xs = [hour[a] for a in day[:6]]
for name, f in (("first-8-minute decode ms/token", lambda a: med([1000 * (B8[a][k]["latency"] - B8[a][k]["ttft"]) / B8[a][k]["ct"] for k in same])),
                ("first-8-minute prefill ms/token", lambda a: med(list(D[a]["_pre"].values())))):
    ys = [f(a) for a in day[:6]]
    print(f"  {name}: " + " ".join(f"{a} {y:.4f}" for a, y in zip(day[:6], ys)) + f" | slope {100 * slope(xs, ys) / C.mean(ys):+.2f} % per hour | Spearman {spearman(xs, ys):+.2f}")
