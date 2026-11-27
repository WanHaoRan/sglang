"""Lookup latency (prefetch_start -> s2h_query of the same rid) by window, result and size; lookups in flight."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xlib

for spec in sys.argv[1:]:
    camp, arm = spec.split(":")
    d = xlib.load(camp, arm)
    starts = {}
    for t, f in d["ev"].get("prefetch_start", []):
        starts.setdefault(str(f.get("rid")), []).append(t)
    rows = []
    for t, f in d["ev"].get("s2h_query", []):
        rid = str(f.get("rid"))
        cand = [s for s in starts.get(rid, []) if s <= t]
        if not cand:
            continue
        s = max(cand)
        rows.append((s, t - s, f.get("tokens_req", 0), f.get("storage_hit", 0)))
    print("==", spec, "lookups", len(rows))
    print("window  n  miss%  latency ms p50 p90 p99 max | misses p50 p90 | hits p50 p90 | pages asked p50 p90 | ms per page asked (p50) for lookups of >= 64 pages")
    for lo, hi in [(0, 28), (28, 70), (70, 90), (90, 110)]:
        a, b = lo * 60, hi * 60
        w = [r for r in rows if a <= r[0] < b]
        if not w:
            continue
        lat = [r[1] * 1e3 for r in w]
        miss = [r[1] * 1e3 for r in w if r[3] <= 0]
        hit = [r[1] * 1e3 for r in w if r[3] > 0]
        pg = [r[2] / 64 for r in w]
        per = [r[1] * 1e3 / (r[2] / 64) for r in w if r[2] >= 64 * 64]
        print(f"{lo:3d}-{hi:<3d} {len(w):5d} {100*len(miss)/len(w):4.0f}%  {xlib.pct(lat,.5):6.1f} {xlib.pct(lat,.9):6.1f} {xlib.pct(lat,.99):7.1f} {max(lat):7.0f} | "
              f"{xlib.pct(miss,.5):5.1f} {xlib.pct(miss,.9):6.1f} | {xlib.pct(hit,.5):5.1f} {xlib.pct(hit,.9):6.1f} | {xlib.pct(pg,.5):5.0f} {xlib.pct(pg,.9):5.0f} | {xlib.pct(per,.5):.3f}")
    # latency against pages asked (all windows)
    print("latency by pages asked: bin n p50 p90 ms")
    for lo_, hi_ in [(0, 8), (8, 32), (32, 128), (128, 400), (400, 2000)]:
        w = [r[1] * 1e3 for r in rows if lo_ <= r[2] / 64 < hi_]
        if w:
            print(f"   {lo_:4d}-{hi_:<4d} {len(w):5d} {xlib.pct(w,.5):6.1f} {xlib.pct(w,.9):6.1f}")
    # lookups under way at once: sample at each lookup start
    ends = sorted(r[0] + r[1] for r in rows)
    import bisect
    st = sorted(r[0] for r in rows)
    conc = []
    for s in st:
        conc.append(bisect.bisect_right(st, s) - bisect.bisect_right(ends, s))
    print(f"lookups under way when one starts (itself included): mean {xlib.mean(conc):.2f} p90 {xlib.pct(conc,.9):.0f} max {max(conc)}")
    tot = sum(r[1] for r in rows)
    print(f"sum of lookup latencies {tot:.1f} s; tokens asked {sum(r[2] for r in rows)/1e6:.1f} M = {sum(r[2] for r in rows)/64*2/1e3:.0f} K file stats (2 files a page)")
