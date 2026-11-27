"""ev_fetch 3b, a cruder independent replay of the hold margin: for each read that was held, shift its landing by
the change of margin (a larger margin releases earlier, at the earliest at the park time; a smaller one later),
the line as it was.  late: landing + 0.4 s after the GPU turn D (runtab.py); stall = (landing + 0.4 - D)+;
pin = (D - landing)+ x tokens."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib, runtab
from vlib import mean
for arm in ("ec_full", "jit_exact"):
    tab = runtab.build(34, arm); reqs, reads, stall = tab["reqs"], tab["reads"], tab["stall"]
    held = [r for r in reads if r["hold"] and r["pin"] is not None and not r["staged"] and r["rid"] in stall and not r.get("dead")]
    print(f"c34 {arm}: held reads {len(held)}")
    for m in (-2, 0, 1, 2, 3, 4, 6, 99):
        late = 0; st = 0.0; pin = 0.0
        for r in held:
            D = stall[r["rid"]]["D"]
            room = max(r["disp"] - r["park"], 0.0)          # how much earlier it could have been released
            shift = min(m - 2.0, room) if m > 2 else (m - 2.0)
            l = r["pin"] - shift
            if l + 0.4 > D: late += 1
            st += max(l + 0.4 - D, 0.0); pin += max(D - l, 0.0) * r["pages"] * 64
        n = len(held)
        print(f"   margin {'park' if m == 99 else '%+d s' % m}: late {100*late/n:5.1f} %  stall {st/n:.2f} s  pin {pin/n/1e3:6.1f} K token-s per read")
