"""Head-of-line blocking on GPU pool room. At each logged batch line with a queue: the pool's unused tokens,
and whether a queued turn that is ready (no lookup or read of its own under way) would fit in them.
Open loop: it sizes the idle room first-fit admission could use, not the closed-loop gain."""
import os, sys, bisect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xlib

RESERVE = 512  # tokens kept for the new request's output


def reading_windows(d):
    """rid -> list of (start, end) during which the turn has a lookup or read under way."""
    per = {}
    for name in ("prefetch_start", "s2h_query", "srs_park", "srs_dispatch", "s2h_io", "srs_pin"):
        for t, f in d["ev"].get(name, []):
            rid = f.get("rid")
            if rid is not None:
                per.setdefault(str(rid), []).append((t, name, f))
    out = {}
    for rid, evs in per.items():
        evs.sort(key=lambda e: e[0])
        wins, start, hit = [], None, False
        for t, name, f in evs:
            if name == "prefetch_start":
                if start is None:
                    start, hit = t, False
            elif name == "s2h_query":
                if f.get("storage_hit", 0) <= 0:
                    if start is not None:
                        wins.append((start, t)); start = None
                else:
                    hit = True
            elif name == "srs_dispatch" and f.get("launched", 1) == 0:
                if start is not None:
                    wins.append((start, t)); start = None
            elif name in ("srs_pin", "s2h_io"):
                if start is not None and name == "srs_pin":
                    wins.append((start, t)); start = None
        if start is not None and hit:
            wins.append((start, start + 3600))
        out[rid] = wins
    return out


def main(spec):
    camp, arm = spec.split(":")
    d = xlib.load(camp, arm)
    rw = reading_windows(d)
    reqs = sorted((r[6], r[6] + r[4], r[1], rid) for rid, r in d["rts"].items())  # entry, admit, input_len
    entries = [r[0] for r in reqs]
    print("==", spec)
    print("window  lines  unused tok mean | lines where a ready queued turn fits: share, mean tokens greedy first-fit would place, as % of pool |"
          " share where the head ready turn itself fits (should be ~0) | ready queued mean, their input_len p10/p50")
    for lo, hi in [(28, 70), (70, 90), (90, 110)]:
        a, b = lo * 60, hi * 60
        n = fit = headfit = 0
        placed = unused_sum = ready_sum = 0
        sizes = []
        for x in d["batch"]:
            t = x[0]
            if not (a <= t < b):
                continue
            usage, queue = (x[4], x[5]) if x[1] == "D" else (x[5], x[7])
            if queue < 5 or x[1] != "D":
                continue
            free = (1 - usage) * xlib.POOL_GPU - 0.005 * xlib.POOL_GPU  # logged to 2 decimals: take the low end
            j = bisect.bisect_right(entries, t)
            ready = []
            for e, adm, L, rid in reqs[max(0, j - 400):j]:
                if adm <= t:
                    continue
                if any(s <= t < en for s, en in rw.get(rid, ())):
                    continue
                ready.append((e, L))
            ready.sort()
            n += 1
            unused_sum += max(free, 0)
            ready_sum += len(ready)
            sizes.extend(L for _, L in ready[:1])
            rem, got = free, 0
            for k, (e, L) in enumerate(ready):
                if L + RESERVE <= rem:
                    rem -= L + RESERVE; got += L
                    if k == 0:
                        headfit += 1
            if got:
                fit += 1; placed += got
        if n:
            print(f"{lo:3d}-{hi:<3d} {n:5d} {unused_sum/n:9.0f} | {fit/n:.2f}  {placed/n:8.0f} tok = {100*placed/n/xlib.POOL_GPU:.2f}% of pool | {headfit/n:.3f} | "
                  f"{ready_sum/n:5.1f}, head ready input_len p10 {xlib.pct(sizes,.1):.0f} p50 {xlib.pct(sizes,.5):.0f} p90 {xlib.pct(sizes,.9):.0f}")

for spec in sys.argv[1:]:
    main(spec)
