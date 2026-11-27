"""Does a use count per stored head page (an index of stored pages) do better than reference counts over open
sessions' chains?  Usage: python3 -I -B tier_count.py <arm> <scale>"""
import sys
sys.path.insert(0, "/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/ev_ssd")
from replay import *
arm, scale = sys.argv[1], float(sys.argv[2])
d = load_arm(arm); d["sharing"] = page_sharing(d); d["shared_since"] = page_shared_since(d); d["first_req"] = first_requests(d)
d["user_times"] = page_user_times(d)
cap = int(390100 * scale)
ref = Sim(d, {}, passes="none").run()
base = Sim(d, {}, passes="volume", cap=cap).run()
first = base.pass_log[0]["t_snap"]
POL = [("a_oldest_written", {}),
       ("d_tier+refresh", dict(end_zero="tier", refresh="end")),
       ("d_tier_index+refresh+hit+recheck", dict(end_zero="tier_index", refresh="end", touch_on_hit=True, recheck=True)),
       ("d_tier_count+refresh+hit+recheck", dict(end_zero="tier_count", refresh="end", touch_on_hit=True, recheck=True)),
       ("e_oracle_next_use", dict(order="oracle_use"))]
print(f"{arm} volume x{scale:g}: requests losing / K tokens / TTFT s, after the first pass ({first:.0f} s) to 6,600 s | minutes 90-110 | first requests' share of the tokens (after the first pass)")
for name, pol in POL:
    sim = Sim(d, pol, passes="volume", cap=cap).run()
    a = summarize(sim, ref.out, first, 6600.0); l = summarize(sim, ref.out, 5400.0, 6600.0)
    print(f"  {name:36s} {a['lost_requests']:4d} /{a['extra_tokens']/1e3:8.1f} | {l['lost_requests']:4d} /{l['extra_tokens']/1e3:8.1f} /{l['ttft_s']:5.2f} s | "
          f"first requests {a['lost_first_requests']:4d} /{a['extra_tokens_first_requests']/1e3:8.1f}")
