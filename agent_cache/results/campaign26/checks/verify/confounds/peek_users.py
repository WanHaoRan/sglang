"""Admission: are session arrivals the same schedule in every arm, and when does the 44-user cap bind?
  nice -n 19 python3 -I -B peek_users.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

D = {a: C.load(a) for a in C.RUNS}
ref = D["jit_exact"]["conv_start"]
print("client args that differ between runs (key: value per run)")
keys = sorted(set().union(*(set(D[a]["args"]) for a in D)))
for k in keys:
    vals = {a: D[a]["args"].get(k, "<absent>") for a in D}
    if k in ("output", "tag"):
        continue
    if len(set(map(str, vals.values()))) > 1:
        print(f"  {k}: " + ", ".join(f"{a}={vals[a]}" for a in D))
print("\nrun record (users, turns) and summary per run")
for a in D:
    s = D[a].get("summary") or {}
    print(f"  {a:14s} {D[a]['run_rec']} | " + " ".join(f"{k}={s.get(k)}" for k in ("elapsed_s", "users_started", "finished_users", "skipped_users", "turns", "errors", "retries", "capped_convs", "mean_ttft_s")))
print("\nper run: sessions started; arrival_s equal to jit_exact's for the same conv (n equal / n common, max abs diff); sessions queued for a slot (queued_s>0.5), mean and max queued_s; turns field equal")
for a in D:
    cs = D[a]["conv_start"]
    common = [c for c in cs if c in ref]
    dif = [abs(cs[c]["arrival_s"] - ref[c]["arrival_s"]) for c in common]
    q = [cs[c]["queued_s"] for c in cs]
    tq = sum(1 for c in common if cs[c]["turns"] == ref[c]["turns"] and cs[c].get("task_conv") == ref[c].get("task_conv") and cs[c].get("session_id") == ref[c].get("session_id"))
    print(f"  {a:14s} {len(cs):4d} | {sum(1 for x in dif if x < 0.002):4d}/{len(common):4d} max {max(dif):.3f} | queued {sum(1 for x in q if x > 0.5):4d} mean {C.mean(q):7.2f} max {max(q):7.1f} | same session {tq}/{len(common)}")
print("\nusers records per run: n, share with live==cap, share with waiting>0, mean live, mean waiting, mean in_flight; by 15 min: mean live / waiting / in_flight")
for a in D:
    u = D[a]["users"]
    if not u:
        print(f"  {a:14s} no users records")
        continue
    cells = []
    for m in range(0, 120, 15):
        x = [r for r in u if m * 60 <= r[0] < (m + 15) * 60]
        cells.append(f"{C.mean(r[2] for r in x):4.1f}/{C.mean(r[3] for r in x):4.1f}/{C.mean(r[4] for r in x):4.1f}" if x else "   -   ")
    print(f"  {a:14s} {len(u):4d} {100 * C.mean(1 if r[2] >= r[1] else 0 for r in u):5.1f} % {100 * C.mean(1 if r[3] > 0 else 0 for r in u):5.1f} % "
          f"{C.mean(r[2] for r in u):5.1f} {C.mean(r[3] for r in u):5.1f} {C.mean(r[4] for r in u):5.1f} | " + " ".join(cells))
