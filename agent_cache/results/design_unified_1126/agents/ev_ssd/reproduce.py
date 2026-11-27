"""Step 1 table: after each logged pass, how many history lookups found nothing or only part, how many reads failed on
a deleted file, and the history computed again - recorded against the replay (order a, logged passes).
Usage: python3 -I -B reproduce.py <arm>"""
import sys, collections, json
sys.path.insert(0, "/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/ev_ssd")
from replay import *
arm = sys.argv[1]
d = load_arm(arm); d["sharing"] = page_sharing(d)
reqs = d["reqs"]
sim = Sim(d, {}, passes="logged").run()
ref = Sim(d, {}, passes="none").run()
cuts = [0.0] + [p["t"] for p in sim.pass_log] + [1e9]
names = ["before pass 1"] + [f"pass {k+1} to {'pass ' + str(k+2) if k + 1 < len(sim.pass_log) else 'the end'}" for k in range(len(sim.pass_log))]
pred = {(r.rid, id(lk)): h for r, lk, h in sim.lookup_cmp}
rows = []
print(f"{arm}: logged passes end at {[round(p[0]) for p in d['passes']]} s, groups {[p[2] for p in d['passes']]}; replay deletes at "
      f"{[round(p['t'], 1) for p in sim.pass_log]} s, file times up to {[round(p['newest_deleted']) for p in sim.pass_log]} s")
print(f"{'interval':20s} | history lookups: logged zero/part/full | replay zero/part/full | differ | reads failed logged/replay | "
      f"requests | recomputed K: recorded / replay / if nothing were deleted | requests losing >= 1,024: recorded / replay")
tot = collections.Counter()
for i in range(len(cuts) - 1):
    lo, hi = cuts[i], cuts[i + 1]
    L = collections.Counter(); R = collections.Counter(); differ = 0; fl = fr = 0
    for r in reqs:
        for lk in r.lookups:
            if not (lo <= lk.t < hi): continue
            if lk.failed: fl += 1
            asked = max(0, min(r.H if not lk.staged else lk.m + lk.n, lk.m + lk.n) - lk.m)
            if asked <= 0: continue
            def c(h): return "zero" if h == 0 else "part" if h < asked else "full"
            L[c(lk.hit)] += 1
            ph = pred.get((r.rid, id(lk)))
            if ph is None: differ += 1; continue
            R[c(ph)] += 1
            differ += ph != lk.hit
    fr = sum(1 for (r, lk, start, hit, landed, t) in sim.readfail_pred if lo <= lk.t < hi)
    n = 0; rec_a = rec_r = rec_0 = 0; la = lr = 0
    for r in reqs:
        if not (lo <= r.t_send < min(hi, 6600.0)) or r.rid not in sim.out: continue
        n += 1
        h = min(r.H, r.q)
        a = max(0, h - r.c_act) * 64; b = max(0, h - sim.out[r.rid][0]) * 64; z = max(0, h - ref.out[r.rid][0]) * 64
        rec_a += a; rec_r += b; rec_0 += z; la += a >= 1024; lr += b >= 1024
    print(f"{names[i]:20s} | {L['zero']:5d} /{L['part']:4d} /{L['full']:5d}                  | {R['zero']:5d} /{R['part']:4d} /{R['full']:5d}     | {differ:4d}   | "
          f"{fl:3d} / {fr:3d}                 | {n:6d}   | {rec_a/1e3:9.1f} / {rec_r/1e3:9.1f} / {rec_0/1e3:9.1f}                      | {la:4d} / {lr:4d}")
    rows.append(dict(interval=names[i], logged=dict(L), replay=dict(R), differ=differ, reads_failed_logged=fl, reads_failed_replay=fr,
                     requests_sent_before_6600=n, recomputed_recorded=rec_a, recomputed_replay=rec_r, recomputed_no_deletion=rec_0,
                     losing_recorded=la, losing_replay=lr))
print("(requests and tokens count requests sent before 6,600 s; lookups and reads count the whole run)")
print("live pages deleted by each pass (a later request's prompt holds them), of them held by >= 2 sessions:",
      [(p["deleted"], p["live_use"], p["live_shared"]) for p in sim.pass_log])
json.dump(dict(arm=arm, rows=rows, passes=sim.pass_log), open(f"out/reproduce_{arm}.json", "w"))
