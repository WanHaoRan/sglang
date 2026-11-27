import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib, ssd_replay
arm = sys.argv[1] if len(sys.argv) > 1 else "jit_p50"
D = ssd_replay.Data(34, arm); ssd_replay.first_sets(D)
base = ssd_replay.replay(D, delete=False); res = ssd_replay.replay(D, policy="today")
p1 = D.passes[0][0]
lk = {}
for i, L in enumerate(D.lookups):
    lk.setdefault(L[1], []).append((round(L[0], 1), L[2], L[3], L[4], base["hits"][i], res["hits"][i], "staged" if L[0] < D.reqs[L[1]]["entry"] else "arr"))
rows = []
for rid, q in D.reqs.items():
    if not (p1 <= q["t_send"] < 6600): continue
    act = q["ideal"] - q["cached"]
    need = base["got"].get(rid, set()) & D.keyset(rid)
    mine = len(need - res["got"].get(rid, set())) * 64
    if act >= 1024 or mine >= 1024:
        rows.append((act - mine, rid, act, mine, q["k"], q["ideal"], q["cached"], q["sto"], lk.get(rid, [])[:4]))
rows.sort(reverse=True)
print("actual loss - replay loss, rid, actual, replay, k, ideal, cached, from storage, lookups (t, start page, asked, logged hit, no-deletion hit, replay hit, kind)")
for r in rows[:14]: print(r)
print("...")
for r in rows[-5:]: print(r)
print("sum actual", sum(r[2] for r in rows), "sum replay", sum(r[3] for r in rows))
