"""The dirty set against room, from the census lines (campaign 34), minutes 30 to 120."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import load_trees
from model import load
def q(xs, p):
    ys = sorted(xs); return ys[min(len(ys) - 1, int(p * len(ys)))]
trees = load_trees()
print("arm: lines | pending_write p50/p90/max K | held by writes only (h_write) p50/p90/max K | room = free+leaf+cascade p10/p50 K | pending/room p90, max | h_write/room p90, max | h_write/host_only p90, max | lines with h_write over 100 K")
for label in sys.argv[2:]:
    m = load(sys.argv[1], label, trees)
    cz = [c for c in m.census if 1800 <= c["t"] <= 7200]
    pend = [c["pending_write"] for c in cz]; hw = [c["h_write"] for c in cz]
    room = [c["free"] + c["evict_leaf"] + c["evict_casc"] for c in cz]
    r1 = [c["pending_write"] / max(c["free"] + c["evict_leaf"] + c["evict_casc"], 1) for c in cz]
    r2 = [c["h_write"] / max(c["free"] + c["evict_leaf"] + c["evict_casc"], 1) for c in cz]
    r3 = [c["h_write"] / max(c["host_only"], 1) for c in cz]
    print(f"{label}: {len(cz)} | {q(pend,.5)/1e3:.0f}/{q(pend,.9)/1e3:.0f}/{max(pend)/1e3:.0f} | {q(hw,.5)/1e3:.0f}/{q(hw,.9)/1e3:.0f}/{max(hw)/1e3:.0f} | "
          f"{q(room,.1)/1e3:.0f}/{q(room,.5)/1e3:.0f} | {q(r1,.9):.3f}, {max(r1):.3f} | {q(r2,.9):.3f}, {max(r2):.3f} | {q(r3,.9):.3f}, {max(r3):.3f} | {sum(1 for x in hw if x > 100000)}")
