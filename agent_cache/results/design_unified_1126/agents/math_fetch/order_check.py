"""Check 3.4 (order on the device) on small instances by enumeration.

(1) all reads released: earliest due date minimises the maximum lateness (Jackson).
(2) with release times: preemptive earliest-due-date is a lower bound for every non-preemptive
    order (Horn); non-preemptive EDD can miss it, by at most the largest non-preemptible piece.
(3) every due date already passed: the order by c_j / size_j (largest first) minimises
    sum c_j * lateness (Smith).
(4) in between (some reads can still be on time) the cost is weighted tardiness: neither EDD nor
    the ratio order is optimal in general.
"""

import itertools

import numpy as np

rng = np.random.default_rng(4)


def completion(order, p, rel=None):
    t, C = 0.0, np.zeros(len(p))
    for j in order:
        if rel is not None:
            t = max(t, rel[j])
        t += p[j]
        C[j] = t
    return C


def preemptive_edd_lmax(p, d, rel):
    """Event simulation of preemptive EDD; returns the maximum lateness."""
    n = len(p)
    rem = p.astype(float).copy()
    t, done, Lmax = 0.0, 0, -np.inf
    events = sorted(set(rel))
    while done < n:
        avail = [j for j in range(n) if rel[j] <= t + 1e-12 and rem[j] > 1e-12]
        if not avail:
            t = min(r for r in events if r > t)
            continue
        j = min(avail, key=lambda k: d[k])
        nxt = [r for r in events if r > t + 1e-12]
        run = rem[j] if not nxt else min(rem[j], nxt[0] - t)
        t += run
        rem[j] -= run
        if rem[j] <= 1e-12:
            done += 1
            Lmax = max(Lmax, t - d[j])
    return Lmax


n, trials = 6, 300
bad1 = bad3 = 0
gap2, gap2_bound_viol, edd_np_worse = 0.0, 0, 0
edd_bad4 = smith_bad4 = 0
worst4 = (0.0, 0.0)
for _ in range(trials):
    p = rng.uniform(0.3, 3.0, n)
    d = rng.uniform(1.0, 10.0, n)
    c = rng.choice([1.0, 0.3, 0.05], n)
    perms = list(itertools.permutations(range(n)))
    # (1)
    best = min(np.max(completion(o, p) - d) for o in perms)
    edd = np.max(completion(np.argsort(d), p) - d)
    bad1 += edd > best + 1e-9
    # (2)
    rel = rng.uniform(0.0, 5.0, n)
    best_np = min(np.max(completion(o, p, rel) - d) for o in perms)
    pre = preemptive_edd_lmax(p, d, rel)
    gap2 = max(gap2, best_np - pre)
    gap2_bound_viol += (best_np - pre) > p.max() + 1e-9 or pre > best_np + 1e-9
    # non-preemptive EDD among released jobs
    t, left, L = 0.0, set(range(n)), -np.inf
    while left:
        av = [j for j in left if rel[j] <= t + 1e-12]
        if not av:
            t = min(rel[j] for j in left)
            continue
        j = min(av, key=lambda k: d[k])
        t += p[j]
        L = max(L, t - d[j])
        left.remove(j)
    edd_np_worse += L > best_np + 1e-9
    # (3) all due dates passed
    d_past = -rng.uniform(0.0, 5.0, n)
    cost = lambda o, dd: float(np.sum(c * np.maximum(completion(o, p) - dd, 0.0)))
    best3 = min(cost(o, d_past) for o in perms)
    smith = cost(np.argsort(-c / p), d_past)
    bad3 += smith > best3 + 1e-9
    # (4) mixed
    best4 = min(cost(o, d) for o in perms)
    e4, s4 = cost(np.argsort(d), d), cost(np.argsort(-c / p), d)
    edd_bad4 += e4 > best4 + 1e-9
    smith_bad4 += s4 > best4 + 1e-9
    worst4 = (max(worst4[0], e4 - best4), max(worst4[1], s4 - best4))
print(f"(1) {trials} instances of {n} reads, all released: EDD missed the minimum maximum lateness {bad1} times")
print(f"(2) with release times: preemptive EDD exceeded the best non-preemptive order {gap2_bound_viol} times (or the gap exceeded the largest read);"
      f" largest gap best-non-preemptive minus preemptive-EDD {gap2:.2f} s; non-preemptive EDD was not the best order {edd_np_worse} times")
print(f"(3) all due dates passed: the c/size order missed the minimum of sum c * lateness {bad3} times")
print(f"(4) mixed due dates, sum c * tardiness: EDD not optimal in {edd_bad4} of {trials}, c/size order not optimal in {smith_bad4};"
      f" worst excess {worst4[0]:.2f} s (EDD), {worst4[1]:.2f} s (c/size)")
