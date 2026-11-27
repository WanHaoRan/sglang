"""Cleaning orders at the run's volume and at smaller ones, with the pass trigger rebuilt by a volume model
(a pass when the store is at or over 90 % of the volume at a 30 s tick, down to 80 %)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib, ssd_replay
from s01_validate import losses
arm = sys.argv[1] if len(sys.argv) > 1 else "ec_full"
D = ssd_replay.Data(34, arm); ssd_replay.first_sets(D); ssd_replay.uses_index(D)
base = ssd_replay.replay(D, delete=False)
n90 = sum(1 for q in D.reqs.values() if 5400 <= q["t_send"] < 6600)
FULL = 390100
for scale in (1.0, 0.5, 0.25):
    cap = dict(pages=int(FULL * scale), hi=0.90, lo=0.80)
    print(f"=== c34 {arm}, volume x{scale} ({cap['pages']} pages = {cap['pages']*64/1e6:.1f} M tokens)")
    for pol in ("today", "b_chain", "b_tail", "refresh", "touch_on_read", "b_tail+refresh", "b_tail+touch_on_read"):
        res = ssd_replay.replay(D, policy=pol, passes=cap, base=base)
        ps = [p for p in res["passes"] if p[0] < 6600]
        first = ps[0][0] if ps else float("nan")
        nl, tl, _ = losses(D, base, res, 0, 6600)
        n2, t2, _ = losses(D, base, res, 5400, 6600)
        print(f"   {pol:22s}: passes before 6,600 s {len(ps):2d} (first at minute {first/60:5.1f}); whole run to 6,600 s: {nl:4d} requests {tl/1e3:8.0f} K; minutes 90-110: {n2:4d} requests {t2/1e3:8.0f} K -> {t2*0.043e-3*128/n90:5.2f} s by N g", flush=True)
