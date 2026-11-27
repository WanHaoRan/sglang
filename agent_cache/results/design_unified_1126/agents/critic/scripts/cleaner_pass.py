"""What a cleaner pass does to the rest of the engine while it runs: decode step time (40-step log intervals
with no prefill batch), read and write ms per page, lookup latency; the pass window is [end - secs, end]."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xlib

for spec in sys.argv[1:]:
    camp, arm = spec.split(":")
    d = xlib.load(camp, arm)
    print("==", spec)
    D = [(i, x) for i, x in enumerate(d["batch"]) if x[1] == "D"]
    steps = []
    for (i0, x), (i1, y) in zip(D, D[1:]):
        if any(p[1] == "P" for p in d["batch"][i0 + 1:i1]):
            continue
        if y[0] - x[0] < 5:
            steps.append((x[0], y[0], (y[0] - x[0]) / 40 * 1e3, x[2]))
    allsteps = []
    for (i0, x), (i1, y) in zip(D, D[1:]):
        if y[0] - x[0] < 10:
            allsteps.append((x[0], y[0], (y[0] - x[0])))
    rd = [(t - f["ms"] / 1e3, t, f["ms"] / max(f["pages"], 1), f["pages"]) for t, f in d["ev"].get("s2h_io", []) if f.get("pages", 0) >= 32]
    wr = [(t - f["ms"] / 1e3, t, f["ms"] / max(f["pages"], 1), f["pages"]) for t, f in d["ev"].get("h2s_group", []) if f.get("pages", 0) >= 16]
    for t_end, groups, files, gib, secs in d["cleanup"]:
        a, b = t_end - secs, t_end
        def sel(rows, lo, hi, idx=2):
            return [r[idx] for r in rows if r[0] < hi and r[1] > lo]
        def fmt(v):
            return f"n {len(v):3d} p50 {xlib.pct(v,.5):6.2f}" if v else "n   0"
        pre = (a - 120, a - 5)
        post = (b + 5, b + 120)
        print(f"pass ending {t_end:7.0f} s ({secs:.1f} s, {groups} groups):")
        print(f"   pure decode step ms: before {fmt(sel(steps, *pre))} | during {fmt(sel(steps, a, b))} | after {fmt(sel(steps, *post))}")
        print(f"   40-step interval s (any): before {fmt(sel(allsteps, *pre))} | during {fmt(sel(allsteps, a, b))} | after {fmt(sel(allsteps, *post))}")
        print(f"   read ms/page: before {fmt(sel(rd, *pre))} | during {fmt(sel(rd, a, b))} | after {fmt(sel(rd, *post))}")
        print(f"   merged write ms/page: before {fmt(sel(wr, *pre))} | during {fmt(sel(wr, a, b))} | after {fmt(sel(wr, *post))}")
