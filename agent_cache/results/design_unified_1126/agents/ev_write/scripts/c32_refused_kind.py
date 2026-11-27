"""Campaign 32: the refused reads of the first bursts: timed or 'now', sizes, and what the eviction before them freed."""
import sys, os, bisect, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import PAGE, load_trees
from model import load
trees = load_trees()
for label in sys.argv[1:]:
    m = load("campaign32", label, trees)
    ref = sorted((d for d in m.dispatches if not d["launched"]), key=lambda d: d["t"])
    bursts = []
    for d in ref:
        if not bursts or d["t"] - bursts[-1][-1]["t"] > 120.0: bursts.append([d])
        else: bursts[-1].append(d)
    eh = m.evict_host; eht = [e[0] for e in eh]
    print(f"campaign32 {label}: {len(ref)} refused or cut dispatches in {len(bursts)} bursts")
    for b in bursts[:4]:
        kinds = collections.Counter("now" if (d["park"] and d["park"]["deadline"] == "now") else "timed" for d in b)
        rids = {d["rid"] for d in b}
        pages = [d["pages"] for d in b]
        # the short evictions in the burst
        k0, k1 = bisect.bisect_left(eht, b[0]["t"] - 1), bisect.bisect_right(eht, b[-1]["t"] + 1)
        short = [(e[1], e[2]) for e in eh[k0:k1] if e[1] < e[2]]
        print(f"   burst {b[0]['t']:.0f}-{b[-1]['t']:.0f} s: {len(b)} refused dispatches of {len(rids)} requests; {dict(kinds)}; pages median {sorted(pages)[len(pages)//2]} "
              f"({sorted(pages)[len(pages)//2] * PAGE / 1e3:.0f} K tokens); short evictions {len(short)}: asked median {sorted(x[1] for x in short)[len(short)//2] / 1e3 if short else 0:.0f} K, "
              f"freed median {sorted(x[0] for x in short)[len(short)//2] / 1e3 if short else 0:.0f} K")
