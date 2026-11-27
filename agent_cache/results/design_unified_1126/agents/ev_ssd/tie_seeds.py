"""Spread over the order in which files of equal time are deleted (directory-scan order in the real cleaner).
Usage: python3 -I -B tie_seeds.py <arm> <scale>"""
import sys
sys.path.insert(0, "/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/ev_ssd")
from replay import *
arm, scale = sys.argv[1], float(sys.argv[2])
d = load_arm(arm); d["sharing"] = page_sharing(d); d["shared_since"] = page_shared_since(d); d["first_req"] = first_requests(d)
cap = int(390100 * scale)
ref = Sim(d, {}, passes="none").run()
POL = [("b_end_chain", dict(end_zero="chain")), ("b_end_tail", dict(end_zero="tail")),
       ("d_chain+refresh", dict(end_zero="chain", refresh="end")), ("d_tail+refresh", dict(end_zero="tail", refresh="end")),
       ("d_tier+refresh", dict(end_zero="tier", refresh="end")),
       ("d_tier_index+refresh+hit+recheck", dict(end_zero="tier_index", refresh="end", touch_on_hit=True, recheck=True))]
print(f"{arm} volume x{scale:g}: minutes 90-110, K tokens beyond the no-deletion replay (requests losing), five scan orders")
for name, pol in POL:
    cells = []
    for seed in (0, 101, 20261126, 7919, 123457):
        p = dict(pol); p["tie_seed"] = seed
        sim = Sim(d, p, passes="volume", cap=cap).run()
        l = summarize(sim, ref.out, 5400.0, 6600.0)
        cells.append(f"{l['extra_tokens']/1e3:7.1f} ({l['lost_requests']:3d})")
    print(f"  {name:34s} " + "  ".join(cells))
