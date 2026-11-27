"""R1 (skip the SSD write of an ended session's last turn): how much of what it would skip do later sessions use?
Usage: python3 -I -B r1_exposure.py <arm>"""
import sys, math
sys.path.insert(0, "/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/ev_ssd")
from replay import *
arm = sys.argv[1]
d = load_arm(arm); first = first_requests(d)
sim = Sim(d, {}, passes="none").run()
# later use of a page by another session: first send time of a request of another session holding it after t
uses = {}
for r in d["reqs"]:
    for p in r.pages:
        uses.setdefault(p, []).append((r.t_send, r.conv))
tot_last = tot_last_first = 0; used_later = used_later_first = 0; n_single = n_end = 0
for r in d["reqs"]:
    if not r.ended: continue
    n_end += 1
    c = sim.out[r.rid][0]
    w = r.pages[c:r.q]
    single = r is first[r.conv]
    n_single += single
    tot_last += len(w) + r.n_out
    if single: tot_last_first += len(w) + r.n_out
    for p in w:
        if any(t > r.t_end and cv != r.conv for t, cv in uses[p]):
            used_later += 1
            if single: used_later_first += 1
print(f"{arm}: sessions with an end signal {n_end}, of them with a single request {n_single}")
print(f"   pages their last turn writes: {tot_last} ({tot_last*64/1e6:.2f} M tokens); of them a later request of another session holds {used_later} "
      f"({used_later*64/1e3:.0f} K tokens)")
print(f"   single-request sessions: last turn = first prompt: {tot_last_first} pages, later held by another session {used_later_first} ({used_later_first*64/1e3:.0f} K tokens)")
