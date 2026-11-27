#!/usr/bin/env python3
"""Weighted tardiness on one device: how far each simple order is from the brute-force optimum.

CPU only, stdlib only.  Run:  nice -n 10 python3 -I -B sched_checks_wtard.py > sched_checks_wtard.out

Rules compared (all jobs released at 0, sizes p_j, due dates d_j, weights w_j = cost per second late):
  EDD     earliest due date first (Jackson).
  WSPT    largest w_j / p_j first (Smith's ratio rule; the draft's "largest cost per device-second first").
  hybrid  the draft, sections 3.4 and 5.2: EDD if EDD meets every due date, otherwise WSPT.
  MH      Moore-Hodgson style: add jobs in EDD order; whenever the job just added would finish late, move the job
          with the smallest w/p among those kept so far to a 'late' list; kept jobs run in EDD order, then the late
          list in WSPT order.
  ATC     apparent tardiness cost (Vepsalainen & Morton 1987), look-ahead k = 2.
"""
import itertools, random, math
random.seed(20261126)
EPS = 1e-9

def comp(order, p):
    t, C = 0.0, {}
    for j in order:
        t += p[j]; C[j] = t
    return C

def wt(order, p, d, w):
    C = comp(order, p)
    return sum(w[j] * max(0.0, C[j] - d[j]) for j in order)

def edd(n, d): return sorted(range(n), key=lambda j: d[j])
def wspt(n, p, w): return sorted(range(n), key=lambda j: -w[j] / p[j])

def mh(n, p, d, w):
    kept, late, t = [], [], 0.0
    for j in edd(n, d):
        kept.append(j); t += p[j]
        if t > d[j] + EPS:
            k = min(kept, key=lambda i: w[i] / p[i])
            kept.remove(k); late.append(k); t -= p[k]
    late.sort(key=lambda j: -w[j] / p[j])
    return kept + late

def atc(n, p, d, w, k=2.0):
    left, t, order, pbar = set(range(n)), 0.0, [], sum(p) / n
    while left:
        j = max(left, key=lambda j: (w[j] / p[j]) * math.exp(-max(0.0, d[j] - p[j] - t) / (k * pbar)))
        order.append(j); left.remove(j); t += p[j]
    return order

def run(trials=2000, n=7):
    names = ("EDD", "WSPT", "hybrid", "MH", "ATC")
    for label, slack in (("loose: due dates up to 2.0 x total work", 2.0), ("medium: up to 1.0 x", 1.0), ("mixed: up to 0.6 x", 0.6),
                         ("tight: up to 0.25 x", 0.25), ("all late at the start (d = 0)", 0.0)):
        st = {k: dict(sum=0.0, worst=0.0, opt=0, harm=0, exc=0.0) for k in names}
        tot_best = 0.0
        pos = 0
        for _ in range(trials):
            p = [random.uniform(0.2, 5) for _ in range(n)]
            w = [random.uniform(0.2, 3) for _ in range(n)]
            P = sum(p)
            d = [random.uniform(0, 1) * slack * P for _ in range(n)]
            best = min(wt(o, p, d, w) for o in itertools.permutations(range(n)))
            e = edd(n, d)
            feasible = all(c <= d[j] + EPS for j, c in comp(e, p).items())
            cost = {"EDD": wt(e, p, d, w), "WSPT": wt(wspt(n, p, w), p, d, w), "MH": wt(mh(n, p, d, w), p, d, w),
                    "ATC": wt(atc(n, p, d, w), p, d, w)}
            cost["hybrid"] = cost["EDD"] if feasible else cost["WSPT"]
            tot_best += best
            for k in names:
                st[k]["exc"] += cost[k] - best
            if best <= 1e-9:
                for k in names:
                    if cost[k] > 1e-9: st[k]["harm"] += 1      # late although an order with no lateness exists
                    else: st[k]["opt"] += 1
                continue
            pos += 1
            for k in names:
                r = cost[k] / best
                st[k]["sum"] += r; st[k]["worst"] = max(st[k]["worst"], r)
                if r <= 1 + 1e-9: st[k]["opt"] += 1
        print(f"regime {label}: {trials} instances of {n} jobs, {trials - pos} can meet every due date")
        for k in names:
            s = st[k]
            mean = s["sum"] / pos if pos else float("nan")
            print(f"   {k:6s} cost/optimum where some job must be late: mean {mean:7.3f}, worst {s['worst']:9.2f};"
                  f" optimal in {s['opt']:4d} of {trials}; late although avoidable in {s['harm']:4d};"
                  f" mean excess over optimum {s['exc'] / trials:7.3f} weighted s (optimum {tot_best / trials:7.3f})")

if __name__ == "__main__":
    run()
