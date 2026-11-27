"""Step 1: does the replay reproduce what happened?  Policy (a) with the logged passes.
Usage: python3 -I -B validate.py <arm> [del_frac]"""
import sys, collections, json
sys.path.insert(0, "/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/ev_ssd")
from replay import *
arm = sys.argv[1]
del_frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.93
d = load_arm(arm)
reqs = d["reqs"]
print(f"{arm}: {len(reqs)} requests, {d['n_prompt_pages']} prompt pages, {d['n_ids'] - d['n_prompt_pages']} output pages, "
      f"lookups {sum(len(r.lookups) for r in reqs)} (+{d['n_unknown_lookups']} of requests with no client record)")
print("logged passes (end s, secs, groups, % before, % after):", d["passes"], "cleaner started at", round(d["cleaner_start"], 1))
sim = Sim(d, {}, passes="logged", del_frac=del_frac).run()
first = d["passes"][0][0] - d["passes"][0][1]
ends = [p[0] for p in d["passes"]]
# ---- lookups: predicted against logged
def cls(asked_hist, hit):
    if asked_hist <= 0: return "newtext"
    if hit == 0: return "zero"
    if hit < asked_hist: return "part"
    return "full"
win = collections.defaultdict(collections.Counter)
mism = []
uncompared = 0
tot = 0
for r, lk, pred in sim.lookup_cmp:
    tot += 1
    asked_hist = max(0, min(r.H if not lk.staged else lk.m + lk.n, lk.m + lk.n) - lk.m)
    w = "before pass 1" if lk.t < first else "pass windows" if any(e - 13 <= lk.t <= e + 0.5 for e in ends) else "after pass 1"
    a, b = cls(asked_hist, lk.hit), cls(asked_hist, pred)
    win[w][("logged " + a, "replay " + b)] += 1
    win[w]["exact" if pred == lk.hit else "differs"] += 1
    if pred != lk.hit:
        mism.append((w, r.rid, round(lk.t, 1), lk.m, lk.n, lk.hit, pred, asked_hist, lk.staged))
n_all = sum(len(r.lookups) for r in reqs)
print(f"\nlookups compared at the recorded start page: {tot} of {n_all}")
for w in ("before pass 1", "pass windows", "after pass 1"):
    c = win[w]
    print(f"  [{w}] exact {c['exact']}, differs {c['differs']}")
    for k in sorted(k for k in c if isinstance(k, tuple)):
        print(f"      {k[0]:16s} -> {k[1]:16s}: {c[k]}")
print("  mismatches (window, rid, t, start page, pages asked, logged hit, replay hit, history asked, staged):")
for m in mism[:60]:
    print("     ", m)
json.dump(mism, open(f"out/validate_mismatch_{arm}.json", "w"))
# ---- reads that failed on deleted files
logged_fail = [(r.rid, round(lk.t, 1)) for r in reqs for lk in r.lookups if lk.failed]
pred_fail = [(r.rid, round(lk.t, 1), start, hit, landed) for r, lk, start, hit, landed, t in sim.readfail_pred]
print(f"\nreads that failed on a deleted file: logged {len(logged_fail)}, replay {len(pred_fail)}")
lf = set(x[0] for x in logged_fail); pf = set(x[0] for x in pred_fail)
print(f"   requests: logged {len(lf)}, replay {len(pf)}, both {len(lf & pf)}; logged only {sorted(lf - pf)}; replay only {sorted(pf - lf)}")
# ---- final outcome: cached pages
same = diff = 0; dtok = 0; big = []
for r in reqs:
    if r.rid not in sim.out: continue
    c = sim.out[r.rid][0]
    if c == r.c_act: same += 1
    else:
        diff += 1; dtok += (r.c_act - c) * 64
        big.append((r.rid, round(r.t_send, 1), r.c_act, c, r.H, [(round(l.t, 1), l.m, l.n, l.hit, l.completed) for l in r.lookups]))
print(f"\nrequests whose replayed cached pages equal the recorded: {same}; differ: {diff} (recorded - replay = {dtok} tokens)")
for b in big[:25]:
    print("    ", b)
# ---- recomputed history per window: recorded vs replay
def rec_tokens(lo, hi, which):
    tot = 0; n = 0; lost = 0
    for r in reqs:
        if lo <= r.t_send < hi and r.rid in sim.out:
            c = r.c_act if which == "recorded" else sim.out[r.rid][0]
            x = max(0, min(r.H, r.q) - c) * 64
            tot += x; n += 1; lost += x >= 1024
    return tot, n, lost
print("\nhistory computed again (ideal - cached, page units), recorded | replay, and requests losing >= 1,024 tokens")
for nm, lo, hi in (("minute 60 to pass 1", 3600, first), ("pass 1 to 6,600 s", first, 6600), ("pass 1 to end", first, 7300)):
    a = rec_tokens(lo, hi, "recorded"); b = rec_tokens(lo, hi, "replay")
    print(f"  {nm:22s}: {a[0]/1e6:.3f} M ({a[2]} requests of {a[1]}) | {b[0]/1e6:.3f} M ({b[2]})")
print("\npasses in the replay:")
for p in sim.pass_log:
    print("  ", {k: (round(v, 1) if isinstance(v, float) else v) for k, v in p.items()})
# losses by kind after the first pass (replay's own attribution)
kinds = collections.Counter(); ktok = collections.Counter(); shared = collections.Counter()
for r in reqs:
    if r.t_send < first or r.rid not in sim.out: continue
    c, fl = sim.out[r.rid]
    if fl is None: continue
    lost = max(0, min(r.H, r.q) - c) * 64
    kinds[fl["kind"]] += 1; ktok[fl["kind"]] += lost
    shared[(fl["kind"], "written by another session" if fl["writer"] != r.conv else "written by the session itself")] += 1
print("\nreplay: requests after pass 1 whose lookup or read met a deleted page, by kind (count, tokens recomputed):")
for k in kinds: print(f"   {k:16s} {kinds[k]:4d}  {ktok[k]/1e6:.3f} M")
for k in sorted(shared): print("   ", k, shared[k])
