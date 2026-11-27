"""Written pages by class (replay of order a, logged passes): outputs, last turns of ended sessions, first prompts.
Usage: python3 -I -B writes_by_class.py <arm>"""
import sys, collections
sys.path.insert(0, "/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/ev_ssd")
from replay import *
arm = sys.argv[1]
d = load_arm(arm); first = first_requests(d)
sim = Sim(d, {}, passes="logged").run()
tot_p = tot_o = last_p = last_o = first_p = 0
n_end = 0
for r in d["reqs"]:
    c = sim.out[r.rid][0]
    wp = r.q - c
    tot_p += wp; tot_o += r.n_out
    if r.ended:
        n_end += 1; last_p += wp; last_o += r.n_out
    if r is first[r.conv]:
        first_p += wp
tot = tot_p + tot_o
print(f"{arm}: pages written or touched {tot} ({tot*64/1e6:.2f} M tokens): prompt {tot_p} ({tot_p/tot:.1%}), output {tot_o} ({tot_o/tot:.1%})")
print(f"   last request of a session that sends the end signal ({n_end} sessions): prompt pages {last_p}, output pages {last_o} -> {(last_p+last_o)*64/1e6:.2f} M tokens, {(last_p+last_o)/tot:.2%} of all writes")
print(f"   first requests' own writes: {first_p} pages ({first_p*64/1e6:.2f} M tokens, {first_p/tot:.1%})")
# at each logged pass: stored pages by class (ended session's pages past the first prompt / ended sessions' first prompts / open)
