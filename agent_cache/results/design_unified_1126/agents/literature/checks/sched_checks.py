#!/usr/bin/env python3
"""Brute-force checks of the single-machine scheduling results the draft cites (sections 3.4 and 5.2).

CPU only, stdlib only.  Run:  nice -n 10 python3 -I -B sched_checks.py > sched_checks.out

Notation: job j has size p_j (device seconds), due date d_j, release time r_j, weight w_j
(cost per second late).  C_j = completion time, lateness L_j = C_j - d_j, tardiness T_j = max(0, L_j).
"""
import itertools, random, math

random.seed(20261126)
EPS = 1e-9

def seq_completion(order, p, r=None):
    """Non-preemptive schedule of a fixed order, each job started as early as possible."""
    t, C = 0.0, {}
    for j in order:
        if r is not None:
            t = max(t, r[j])
        t += p[j]
        C[j] = t
    return C

def lmax(C, d):
    return max(C[j] - d[j] for j in C)

def wtard(C, d, w):
    return sum(w[j] * max(0.0, C[j] - d[j]) for j in C)

def edd_order(n, d):
    return sorted(range(n), key=lambda j: d[j])

# ---------------------------------------------------------------- (a) Jackson: EDD and 1||Lmax
def check_jackson(trials=3000, n=6):
    bad = 0
    for _ in range(trials):
        p = [random.uniform(0.1, 5) for _ in range(n)]
        d = [random.uniform(0, 15) for _ in range(n)]
        best = min(lmax(seq_completion(o, p), d) for o in itertools.permutations(range(n)))
        got = lmax(seq_completion(edd_order(n, d), p), d)
        if got > best + EPS:
            bad += 1
    return trials, bad

# ---------------------------------------------------------------- (b) Horn: preemptive EDD, release dates
def preemptive_edf(p, r, d, chunk=None):
    """Preemptive EDD.  chunk=None: preempt at any instant (decision points: releases, completions).
    chunk=g: a job is cut in pieces of at most g seconds and the device switches only between pieces."""
    n = len(p)
    rem = list(p)
    t = 0.0
    C = [None] * n
    done = 0
    while done < n:
        avail = [j for j in range(n) if rem[j] > EPS and r[j] <= t + EPS]
        if not avail:
            t = min(r[j] for j in range(n) if rem[j] > EPS)
            continue
        j = min(avail, key=lambda k: (d[k], k))
        if chunk is None:
            nxt = [r[k] for k in range(n) if rem[k] > EPS and r[k] > t + EPS]
            run = rem[j] if not nxt else min(rem[j], min(nxt) - t)
        else:
            run = min(rem[j], chunk)
        t += run
        rem[j] -= run
        if rem[j] <= EPS:
            C[j] = t
            done += 1
    return {j: C[j] for j in range(n)}

def horn_bound(p, r, d):
    """L*max = max over job subsets S of r(S) + p(S) - d(S)  (Lenstra & Shmoys, Theorem 3.5)."""
    n = len(p)
    best = -math.inf
    for mask in range(1, 1 << n):
        S = [j for j in range(n) if mask >> j & 1]
        best = max(best, min(r[j] for j in S) + sum(p[j] for j in S) - max(d[j] for j in S))
    return best

def check_horn(trials=2000, n=7):
    bad = 0
    for _ in range(trials):
        p = [random.uniform(0.1, 5) for _ in range(n)]
        r = [random.uniform(0, 10) for _ in range(n)]
        d = [r[j] + random.uniform(0, 12) for j in range(n)]
        lb = horn_bound(p, r, d)
        got = lmax(preemptive_edf(p, r, d), d)
        if abs(got - lb) > 1e-7:
            bad += 1
    return trials, bad

# ---------------------------------------------------------------- (c) no preemption + release dates
def nonidling_edd(p, r, d):
    """Non-preemptive, non-idling EDD: whenever the device frees, start the released job due first."""
    return preemptive_edf(p, r, d, chunk=math.inf)

def check_nonpreemptive(trials=2000, n=6):
    worse, worst_gap = 0, 0.0
    for _ in range(trials):
        p = [random.uniform(0.1, 5) for _ in range(n)]
        r = [random.uniform(0, 10) for _ in range(n)]
        d = [r[j] + random.uniform(0, 12) for j in range(n)]
        best = min(lmax(seq_completion(o, p, r), d) for o in itertools.permutations(range(n)))
        got = lmax(nonidling_edd(p, r, d), d)
        if got > best + 1e-7:
            worse += 1
            worst_gap = max(worst_gap, got - best)
    return trials, worse, worst_gap

# ---------------------------------------------------------------- (d) preemption only at chunk boundaries
def check_chunked(trials=3000, n=7):
    out = {}
    for g in (0.25, 1.0, 2.5):
        viol, max_excess, n_worse = 0, 0.0, 0
        for _ in range(trials):
            p = [random.uniform(0.1, 6) for _ in range(n)]
            r = [random.uniform(0, 10) for _ in range(n)]
            d = [r[j] + random.uniform(0, 12) for j in range(n)]
            opt = horn_bound(p, r, d)               # optimum with free preemption
            got = lmax(preemptive_edf(p, r, d, chunk=g), d)
            ex = got - opt
            if ex > 1e-7:
                n_worse += 1
            max_excess = max(max_excess, ex)
            if ex > g + 1e-7:
                viol += 1
        out[g] = (trials, n_worse, max_excess, viol)
    return out

# ---------------------------------------------------------------- (e) Smith: ratio rule and sum w C
def check_smith(trials=3000, n=6):
    bad = 0
    for _ in range(trials):
        p = [random.uniform(0.1, 5) for _ in range(n)]
        w = [random.uniform(0.1, 3) for _ in range(n)]
        zero = [0.0] * n
        best = min(wtard(seq_completion(o, p), zero, w) for o in itertools.permutations(range(n)))
        o = sorted(range(n), key=lambda j: -w[j] / p[j])
        if wtard(seq_completion(o, p), zero, w) > best + 1e-7:
            bad += 1
    return trials, bad

# ---------------------------------------------------------------- (f) weighted tardiness: EDD, WSPT, the draft's hybrid
def atc_order(n, p, d, w, k=2.0):
    """Apparent tardiness cost (Vepsalainen & Morton 1987): dynamic priority (w/p) exp(-max(0, d - p - t)/(k pbar))."""
    left = set(range(n))
    t, order = 0.0, []
    pbar = sum(p) / n
    while left:
        j = max(left, key=lambda j: (w[j] / p[j]) * math.exp(-max(0.0, d[j] - p[j] - t) / (k * pbar)))
        order.append(j)
        left.remove(j)
        t += p[j]
    return order

def check_wtard(trials=1500, n=7):
    res = {}
    for name, slack in (("loose (due dates far)", 2.0), ("mixed", 0.6), ("tight", 0.25), ("all late already (d=0)", 0.0)):
        stats = {k: [0.0, 0.0, 0] for k in ("EDD", "WSPT", "hybrid", "ATC")}   # sum ratio, worst ratio, times optimal
        used = 0
        for _ in range(trials):
            p = [random.uniform(0.2, 5) for _ in range(n)]
            w = [random.uniform(0.2, 3) for _ in range(n)]
            P = sum(p)
            d = [random.uniform(0, 1) * slack * P for _ in range(n)]
            best = min(wtard(seq_completion(o, p), d, w) for o in itertools.permutations(range(n)))
            e = edd_order(n, d)
            s = sorted(range(n), key=lambda j: -w[j] / p[j])
            ce, cs = wtard(seq_completion(e, p), d, w), wtard(seq_completion(s, p), d, w)
            feasible = lmax(seq_completion(e, p), d) <= EPS
            ch = ce if feasible else cs          # the draft: EDD if all due dates can be met, else cost per device-second
            ca = wtard(seq_completion(atc_order(n, p, d, w), p), d, w)
            if best <= 1e-9:
                # every rule that reaches zero is optimal; count, but ratios are undefined
                for k, c in (("EDD", ce), ("WSPT", cs), ("hybrid", ch), ("ATC", ca)):
                    if c <= 1e-9:
                        stats[k][2] += 1
                    else:
                        stats[k][1] = math.inf
                continue
            used += 1
            for k, c in (("EDD", ce), ("WSPT", cs), ("hybrid", ch), ("ATC", ca)):
                ratio = c / best
                stats[k][0] += ratio
                stats[k][1] = max(stats[k][1], ratio)
                if c <= best * (1 + 1e-9):
                    stats[k][2] += 1
        res[name] = (used, trials, stats)
    return res

def main():
    print("== (a) Jackson 1955: EDD order minimizes the maximum lateness, all jobs released at 0, no preemption needed")
    t, bad = check_jackson()
    print(f"   {t} random instances of 6 jobs, all 720 orders tried: EDD worse than the best order in {bad} instances")

    print("== (b) Horn 1974: with release dates and free preemption, preemptive EDD attains max_S r(S)+p(S)-d(S)")
    t, bad = check_horn()
    print(f"   {t} random instances of 7 jobs: preemptive EDD differs from the subset bound in {bad} instances")

    print("== (c) release dates, NO preemption: non-idling EDD is not optimal (the problem is strongly NP-hard)")
    p, r, d = [4.0, 1.0], [0.0, 1.0], [10.0, 2.5]
    C = nonidling_edd(p, r, d)
    Cb = seq_completion([1, 0], p, r)
    print(f"   counterexample: A (r=0, p=4, d=10), B (r=1, p=1, d=2.5).  Non-idling EDD starts A at 0: C_A={C[0]}, C_B={C[1]}, "
          f"Lmax={lmax(C, d)} (B is 2.5 late).")
    print(f"   Waiting for B: C_B={Cb[1]}, C_A={Cb[0]}, Lmax={lmax(Cb, d)} (both on time).  So 'if any order meets every due date, "
          f"EDD does' fails without preemption.")
    t, worse, gap = check_nonpreemptive()
    print(f"   {t} random instances of 6 jobs: non-idling EDD worse than the best non-preemptive order in {worse} "
          f"({100 * worse / t:.1f} %), largest excess {gap:.2f} s")

    print("== (d) preemption only at chunk boundaries (pieces of at most g seconds): Lmax <= optimum with free preemption + g")
    for g, (t, n_worse, mx, viol) in check_chunked().items():
        print(f"   g={g:4.2f} s: {t} instances; chunked EDD above the free-preemption optimum in {n_worse} ({100 * n_worse / t:.1f} %), "
              f"largest excess {mx:.3f} s (= {mx / g:.2f} g); excess above g in {viol} instances")

    print("== (e) Smith 1956: nonincreasing w/p minimizes sum w_j C_j")
    t, bad = check_smith()
    print(f"   {t} random instances of 6 jobs, all orders tried: ratio rule worse than the best order in {bad} instances")

    print("== (f) sum of weighted tardiness (the paper's stall term): cost of each rule over the brute-force optimum, 7 jobs")
    print("   rules: EDD; WSPT (largest w/p first = 'largest cost per device-second first'); hybrid = the draft's rule")
    print("   (EDD when EDD meets every due date, otherwise WSPT); ATC = apparent tardiness cost (Vepsalainen & Morton 1987)")
    for name, (used, trials, stats) in check_wtard().items():
        print(f"   regime {name}: {used} of {trials} instances have a positive optimum")
        for k in ("EDD", "WSPT", "hybrid", "ATC"):
            s = stats[k]
            mean = s[0] / used if used else float('nan')
            print(f"      {k:6s} mean ratio {mean:6.3f}  worst ratio {s[1]:8.3f}  optimal in {s[2]:5d} of {trials}")

    print("== (g) EDD does not minimize the SUM of tardiness: A (p=10, d=10), B, C, D (p=1; d=10.5, 10.6, 10.7), unit weights")
    p, d, w = [10, 1, 1, 1], [10, 10.5, 10.6, 10.7], [1, 1, 1, 1]
    e = edd_order(4, d)
    ce = wtard(seq_completion(e, p), d, w)
    best = min((wtard(seq_completion(o, p), d, w), o) for o in itertools.permutations(range(4)))
    print(f"   EDD order {e}: total tardiness {ce:.1f} s, max lateness {lmax(seq_completion(e, p), d):.1f} s;"
          f" best order {list(best[1])}: total {best[0]:.1f} s, max lateness {lmax(seq_completion(best[1], p), d):.1f} s")

if __name__ == "__main__":
    main()
