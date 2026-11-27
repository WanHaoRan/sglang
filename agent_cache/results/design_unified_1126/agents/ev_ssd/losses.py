"""Who loses under a policy, and through which page.  Usage: python3 -I -B losses.py <arm> <scale> <hi> <lo> <policy> [n rows]"""
import sys, json, collections
sys.path.insert(0, "/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/ev_ssd")
from replay import *
import compare_policies as cp
arm, scale, hi, lo, name = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
nrows = int(sys.argv[6]) if len(sys.argv) > 6 else 15
d = load_arm(arm); d["sharing"] = page_sharing(d); d["shared_since"] = page_shared_since(d); d["first_req"] = first_requests(d)
cap = int(390100 * scale)
ref = Sim(d, {}, passes="none").run()
sim = Sim(d, cp.POLICIES[name], passes="volume", cap=cap, hi=hi, lo=lo).run()
first_send = {}
for r in d["reqs"]:
    first_send.setdefault(r.conv, r.t_send)
rows = []
agg = collections.Counter(); tok = collections.Counter()
for r in d["reqs"]:
    if r.rid not in sim.out or r.t_send >= 6600: continue
    c, fl = sim.out[r.rid]
    c0 = ref.out[r.rid][0]
    h = min(r.H, r.q)
    lost = (max(0, h - c) - max(0, h - c0)) * 64
    if lost < 64: continue
    if fl is None:
        agg[("other",)] += 1; tok[("other",)] += lost; continue
    pid = fl["pid"]
    sh = d["sharing"][pid] if pid < d["n_prompt_pages"] else 1
    w = fl["writer"]
    who = "own" if w == r.conv else "other"
    turn = "first request" if r.k == 0 else "returning"
    zeroed = "zeroed" if fl["del_mt"] == 0.0 else "aged"
    wend = fl["writer_end"]
    wstate = "writer ended" if (wend is not None and wend < fl["t"]) else "writer open"
    key = (fl["kind"], turn, "shared page" if sh >= 2 else "private page", "written by " + who, zeroed, wstate)
    agg[key] += 1; tok[key] += lost
    rows.append((round(r.t_send), r.rid, r.k, fl["kind"], fl["pos"], r.q, sh, who, zeroed, wstate, fl["del_pass"],
                 round(fl["del_mt"], 0), round(r.t_send - first_send[r.conv]), lost))
print(f"{arm} {name}: volume x{scale:g}, marks {hi}/{lo}; passes {[(round(p['t_snap']), p['deleted'], p['live_use']) for p in sim.pass_log]}")
print("losses by (kind, turn, page shared by >= 2 sessions, last writer, how the page left, writer state): requests, K tokens")
for k in sorted(agg, key=lambda k: -tok[k]):
    print(f"   {agg[k]:4d} {tok[k]/1e3:8.1f}  {k}")
print("rows: t_send, rid, k, kind, first missing page index, prompt pages, sessions sharing it, writer, left as, writer state, pass, file time at deletion, session age s, tokens lost")
for row in rows[:nrows]:
    print("   ", row)
