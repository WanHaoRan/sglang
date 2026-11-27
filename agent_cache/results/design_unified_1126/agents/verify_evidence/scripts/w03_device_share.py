"""Device shares by window, reads counted as pages x 64 (the s2h_io 'tokens' field is not the read's total)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib
from vlib import kv
for camp, arm in [(34, "ec_full"), (34, "jit_exact"), (34, "jit_p50"), (34, "ec_exact"), (34, "hicache"), (32, "ec_full")]:
    ev = vlib.load(camp, arm)
    rd = [(t, int(kv(r)["pages"]) * 64, float(kv(r)["ms"]) / 1e3) for t, r in ev["s2h_io"]]
    wr = []
    for t, r in ev["h2s_io"]:
        d = kv(r)
        wr.append((t, (int(d["pages"]) - int(d.get("skipped", 0))) * 64))
    # skipped pages inside groups are not written: subtract them at the group's time
    for t, r in ev.get("h2s_group", []):
        d = kv(r)
        wr.append((t, -int(d["skipped"]) * 64))
    tot_r = sum(x[1] for x in rd)
    print(f"c{camp} {arm}: reads {len(rd)} ops, {tot_r/1e6:.1f} M tokens in the run; written (not skipped) {sum(x[1] for x in wr)/1e6:.1f} M")
    for lo, hi in [(0, 1800), (1800, 3600), (3600, 5400), (5400, 6600), (4200, 5400)]:
        r = sum(x[1] for x in rd if lo <= x[0] < hi) / (hi - lo)
        w = sum(x[1] for x in wr if lo <= x[0] < hi) / (hi - lo)
        print(f"   minutes {lo//60:3d}-{hi//60:3d}: reads {r/1e3:5.1f} K tok/s  writes {w/1e3:4.1f} K tok/s  sum {(r+w)/1e3:5.1f} K")
