"""Read-only: paired TTFT difference on expired-plan turns at 128 users, with a turn bootstrap (within-run only)."""
import random
import sys
sys.argv = [sys.argv[0], sys.argv[1]]
exec(open(__file__.replace("_paired", "")).read().split("data = {a: client(a) for a in ARMS}")[0])
data = {a: client(a) for a in ARMS}
random.seed(0)
for p50_arm, exact_arm in (("old_p50", "oracle"), ("new_p50", "new_exact")):
    t0, turns = data[p50_arm]
    exp = expiries(p50_arm)
    d = []
    for key, (ts, staged) in exp.items():
        r = turns.get(key)
        if r is None or t0 + r["t_send"] < ts or r["t_send"] > 7200:
            continue
        e = data[exact_arm][1].get(key)
        if e is None:
            continue
        d.append(r["ttft"] - e["ttft"])
    d.sort()
    n = len(d)
    boots = []
    for _ in range(2000):
        s = [d[random.randrange(n)] for _ in range(n)]
        boots.append(sum(s) / n)
    boots.sort()
    print(f"{p50_arm} - {exact_arm}, first 2 h, {n} expired turns: mean {sum(d)/n:+.2f} s, median {d[n//2]:+.2f} s, "
          f"slower on {sum(1 for x in d if x > 0)}, turn bootstrap 95 % [{boots[50]:+.2f}, {boots[1949]:+.2f}] s")
