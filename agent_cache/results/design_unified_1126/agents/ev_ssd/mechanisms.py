"""Step 4: which mechanism each loss comes from, per cleaning order.
For every request (sent from the first pass of order a to 6,600 s) that recomputes more than it would with nothing
deleted, the first deleted page its lookups or reads met decides the mechanism:
  head_other   lookup found nothing; the missing page was last written by another session (a shared head)
  head_own     lookup found nothing; the missing page was written by the session itself (its own head aged out)
  partial      lookup found a run of pages, then a hole
  in_flight    lookup found the pages; a file was gone when the parked read ran
Usage: python3 -I -B mechanisms.py <arm> <scale> <hi> <lo> <policy> [<policy> ...]"""
import sys, json, collections
sys.path.insert(0, "/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/ev_ssd")
from replay import *
import compare_policies as cp
arm, scale, hi, lo = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
names = sys.argv[5:]
d = load_arm(arm); d["sharing"] = page_sharing(d); d["shared_since"] = page_shared_since(d); d["first_req"] = first_requests(d)
cap = int(390100 * scale)
ref = Sim(d, {}, passes="none").run()
base = Sim(d, {}, passes="volume", cap=cap, hi=hi, lo=lo).run()
first = base.pass_log[0]["t_snap"]
MECH = ("head_other", "head_own", "partial", "in_flight", "other")
out = {}
print(f"{arm}: volume x{scale:g} ({cap} pages), marks {hi:.0%}/{lo:.0%}; requests sent {first:.0f}-6,600 s; cells: requests / K tokens recomputed beyond the no-deletion replay")
print(f"{'order':34s} " + " ".join(f"{m:>16s}" for m in MECH) + f" {'total':>16s} | {'first requests':>16s} | {'shared page':>16s}")
for name in names:
    sim = Sim(d, cp.POLICIES[name], passes="volume", cap=cap, hi=hi, lo=lo).run()
    cnt = collections.Counter(); tok = collections.Counter()
    for r in d["reqs"]:
        if not (first <= r.t_send < 6600.0) or r.rid not in sim.out: continue
        c, fl = sim.out[r.rid]
        h = min(r.H, r.q)
        lost = (max(0, h - c) - max(0, h - ref.out[r.rid][0])) * 64
        if lost < 64: continue
        if fl is None: m = "other"
        elif fl["kind"] == "read_in_flight": m = "in_flight"
        elif fl["kind"] == "partial": m = "partial"
        else: m = "head_own" if fl["writer"] == r.conv else "head_other"
        cnt[m] += 1; tok[m] += lost
        cnt["total"] += 1; tok["total"] += lost
        if r.k == 0: cnt["first"] += 1; tok["first"] += lost
        if fl is not None and fl["pid"] < d["n_prompt_pages"] and d["sharing"][fl["pid"]] >= 2:
            cnt["shared"] += 1; tok["shared"] += lost
    out[name] = dict(count=dict(cnt), tokens=dict(tok))
    print(f"{name:34s} " + " ".join(f"{cnt[m]:6d} /{tok[m]/1e3:8.1f}" for m in MECH) + f" {cnt['total']:6d} /{tok['total']/1e3:8.1f} | "
          f"{cnt['first']:6d} /{tok['first']/1e3:8.1f} | {cnt['shared']:6d} /{tok['shared']/1e3:8.1f}")
json.dump(dict(arm=arm, scale=scale, hi=hi, lo=lo, first=first, table=out), open(f"out/mechanisms_{arm}_s{scale:g}_{int(hi*100)}_{int(lo*100)}.json", "w"))
