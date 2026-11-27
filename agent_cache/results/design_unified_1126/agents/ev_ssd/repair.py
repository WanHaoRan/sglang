"""Refresh with repair: a refresh whose touch finds no file writes the page again from memory.
Usage: python3 -I -B repair.py <arm> <scale>"""
import sys
sys.path.insert(0, "/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/ev_ssd")
from replay import *
arm, scale = sys.argv[1], float(sys.argv[2])
d = load_arm(arm); d["sharing"] = page_sharing(d); d["shared_since"] = page_shared_since(d); d["first_req"] = first_requests(d)
cap = int(390100 * scale)
ref = Sim(d, {}, passes="none").run()
base = Sim(d, {}, passes="volume", cap=cap).run()
first = base.pass_log[0]["t_snap"]
POL = [("a_oldest_written", {}),
       ("c_refresh", dict(refresh="end")),
       ("c_refresh+repair", dict(refresh="end", repair=True)),
       ("d_tier+refresh", dict(end_zero="tier", refresh="end")),
       ("d_tier+refresh+repair", dict(end_zero="tier", refresh="end", repair=True)),
       ("d_tier+refresh_300s+repair", dict(end_zero="tier", refresh="end", refresh_limit=300.0, repair=True)),
       ("d_tier_index+refresh+hit+recheck", dict(end_zero="tier_index", refresh="end", touch_on_hit=True, recheck=True)),
       ("d_tier_index+refresh+hit+recheck+repair", dict(end_zero="tier_index", refresh="end", touch_on_hit=True, recheck=True, repair=True)),
       ("e_oracle_next_use", dict(order="oracle_use"))]
print(f"{arm} volume x{scale:g}: after the first pass ({first:.0f} s) to 6,600 s: requests losing / K tokens (first requests) | minutes 90-110: requests / K tokens / TTFT s | pages repaired, passes")
for name, pol in POL:
    sim = Sim(d, pol, passes="volume", cap=cap).run()
    a = summarize(sim, ref.out, first, 6600.0); l = summarize(sim, ref.out, 5400.0, 6600.0)
    k = a["by_kind"]
    print(f"  {name:40s} {a['lost_requests']:4d} /{a['extra_tokens']/1e3:8.1f} ({a['lost_first_requests']:3d} /{a['extra_tokens_first_requests']/1e3:7.1f}) | "
          f"{l['lost_requests']:4d} /{l['extra_tokens']/1e3:8.1f} /{l['ttft_s']:5.2f} s | repaired {sim.repairs:6d}, passes {len(sim.pass_log)}; "
          f"zero_head {k.get('zero_head',0)} zero_mid {k.get('zero_mid',0)} partial {k.get('partial',0)} in_flight {k.get('read_in_flight',0)}")
