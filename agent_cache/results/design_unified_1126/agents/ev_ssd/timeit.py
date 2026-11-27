import sys, time, resource
sys.path.insert(0, "/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/ev_ssd")
from replay import *
t0 = time.time()
d = load_arm("ec_full")
print("load", round(time.time() - t0, 1), "s; maxrss MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
d["sharing"] = page_sharing(d)
t0 = time.time()
ref = Sim(d, {}, passes="none").run()
print("no-deletion run", round(time.time() - t0, 1), "s")
t0 = time.time()
a = Sim(d, {}, passes="volume").run()
print("volume run (a)", round(time.time() - t0, 1), "s; maxrss MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
for p in a.pass_log:
    print("  ", {k: (round(v, 1) if isinstance(v, float) else v) for k, v in p.items()})
print("logged:", [(round(p[0] - p[1], 1), p[2]) for p in d["passes"]])
first = d["passes"][0][0] - d["passes"][0][1]
print(summarize(a, ref.out, first, 6600))
print(summarize(a, ref.out, 5400, 6600))
b = Sim(d, {}, passes="logged").run()
print("logged passes:", summarize(b, ref.out, first, 6600))
