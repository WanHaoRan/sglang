import sys
sys.path.insert(0, "/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/ev_ssd")
from replay import *
for arm in ("ec_full", "jit_exact", "jit_p50"):
    d = load_arm(arm)
    a = sum(1 for r in d["reqs"] if r.ended); b = sum(1 for r in d["reqs"] if r.hint_ended)
    both = sum(1 for r in d["reqs"] if r.ended and r.hint_ended)
    convs = len(set(r.conv for r in d["reqs"]))
    print(f"{arm}: sessions {convs}; last request of the stream in the trace: {a}; hint.ended in the client record: {b}; both {both}")
