"""GPU KV pool use on the logged batch lines, by window, when a queue exists; and the h2d copy-rate shape."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xlib

for spec in sys.argv[1:]:
    camp, arm = spec.split(":")
    d = xlib.load(camp, arm)
    print("==", spec)
    print("window  lines(queue>=5)  usage mean p10 p50 p90  unused tokens mean p50 | running mean | queue mean | share of lines with usage<0.95, <0.97, <0.98")
    for lo, hi in [(0, 28), (28, 70), (70, 90), (90, 110)]:
        a, b = lo * 60, hi * 60
        us, run, q = [], [], []
        for x in d["batch"]:
            if not (a <= x[0] < b):
                continue
            if x[1] == "D":
                usage, running, queue = x[4], x[2], x[5]
            else:
                usage, running, queue = x[5], x[6], x[7]
            if queue >= 5:
                us.append(usage); run.append(running); q.append(queue)
        if not us:
            continue
        n = len(us)
        print(f"{lo:3d}-{hi:<3d} {n:8d}   {xlib.mean(us):.3f} {xlib.pct(us,.1):.2f} {xlib.pct(us,.5):.2f} {xlib.pct(us,.9):.2f}   "
              f"{(1-xlib.mean(us))*xlib.POOL_GPU:8.0f} {(1-xlib.pct(us,.5))*xlib.POOL_GPU:8.0f} | {xlib.mean(run):5.1f} | {xlib.mean(q):5.1f} | "
              f"{sum(u<0.95 for u in us)/n:.2f} {sum(u<0.97 for u in us)/n:.2f} {sum(u<0.98 for u in us)/n:.2f}")
    # h2d rate histogram
    done = d["ev"].get("h2d_done", [])
    rates = [(t, f["tokens"] / f["ms"]) for t, f in done if f["ms"] > 0 and f["tokens"] >= 8192]
    edges = [0, 150, 200, 250, 300, 350, 400, 450, 500, 600, 10**9]
    hist = [0] * (len(edges) - 1)
    for _, r in rates:
        for i in range(len(edges) - 1):
            if edges[i] <= r < edges[i + 1]:
                hist[i] += 1
                break
    print("h2d rate (tokens/ms) histogram, copies >= 8192 tokens:", ", ".join(f"{edges[i]}-{edges[i+1] if edges[i+1] < 10**9 else 'inf'}: {hist[i]}" for i in range(len(hist))))
    # is a slow copy one that overlapped an SSD read's I/O or another copy?
    s2h = [(t - f["ms"] / 1000.0, t) for t, f in d["ev"].get("s2h_io", [])]
    import bisect
    starts = sorted(s2h)
    def overlaps_read(t0, t1):
        i = bisect.bisect_left(starts, (t1, 0))
        for s, e in starts[max(0, i - 6):i]:
            if e > t0 and s < t1:
                return True
        return False
    slow = fast = slow_r = fast_r = 0
    for t, f in done:
        if f["ms"] <= 0 or f["tokens"] < 8192:
            continue
        r = f["tokens"] / f["ms"]
        # the copy ended some time before the ack was logged; use submit-aligned window [t - ms, t] as an approximation
        ov = overlaps_read(t - f["ms"] / 1000.0 - 0.15, t)
        if r < 350:
            slow += 1; slow_r += ov
        else:
            fast += 1; fast_r += ov
    print(f"copies under 350 tokens/ms: {slow} ({slow_r} near an SSD read's I/O window); faster: {fast} ({fast_r} near a read)")
    # rate against copy size
    for lo_, hi_ in [(8192, 16384), (16384, 32768), (32768, 65536), (65536, 10**9)]:
        rs = [f["tokens"] / f["ms"] for t, f in done if f["ms"] > 0 and lo_ <= f["tokens"] < hi_]
        if rs:
            print(f"  size {lo_}-{hi_}: n {len(rs)} rate p10 {xlib.pct(rs,.1):.0f} p50 {xlib.pct(rs,.5):.0f} p90 {xlib.pct(rs,.9):.0f}")
    nodes = {}
    for t, f in done:
        if f["ms"] > 0 and f["tokens"] >= 8192:
            nodes.setdefault(min(f["nodes"], 4), []).append(f["tokens"] / f["ms"])
    print("  rate p50 by merged nodes:", {k: (len(v), round(xlib.pct(v, .5))) for k, v in sorted(nodes.items())})
