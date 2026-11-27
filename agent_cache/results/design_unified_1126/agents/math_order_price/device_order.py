#!/usr/bin/env python3
"""Order on the device (DESIGN.md 3.4, 5.2, result 4): what earliest-due-date gives, and what it does not.

Checks by exhaustive search on small random instances (stdlib only):
  D1  Jackson's rule: all jobs available, no preemption: EDD minimises the maximum lateness.
  D2  Horn's theorem: release dates and preemption: preemptive EDD attains the lower bound
          L* = max over (r, d) of [ r + (work of jobs released at or after r and due by d) - d ].
  D3  Preemption only at piece boundaries (page groups of at most p_max device-seconds): EDD over the
      released pieces is within p_max of L*; it is not always optimal.
  D4  With release dates and no preemption EDD among released jobs is not optimal (the problem is
      strongly NP-hard); gap against the non-preemptive optimum on random instances.
  D5  Weighted tardiness sum w_j (C_j - d_j)+ (the paper's objective on the device): EDD, Smith's ratio
      rule, the draft's "EDD, else largest cost per device-second", and the single index
      w_j / max(p_j, d_j - t) against the optimum (dynamic programme over subsets), in three regimes.
  D6  Reads that need room which a write must free first (precedence): EDD on due dates modified to
      d'_w = min(d_w, T_j - p_j) minimises the maximum lateness among precedence-feasible orders.
Output: device_order.out
"""
import itertools
import os
import random
import sys

OUT = []


def P(s=""):
    OUT.append(s)
    print(s, flush=True)


def lmax_of_order(order, p, d, r=None):
    t = 0.0
    worst = -1e18
    for j in order:
        if r is not None and r[j] > t:
            t = r[j]
        t += p[j]
        worst = max(worst, t - d[j])
    return worst


def preemptive_edd(p, d, r, piece=None):
    """Preemptive EDD.  With `piece`, a started piece of that length (or the job's remainder) runs to its end."""
    n = len(p)
    rem = list(p)
    t = min(r)
    comp = [None] * n
    events = sorted(set(r))
    while any(x > 1e-12 for x in rem):
        avail = [j for j in range(n) if r[j] <= t + 1e-12 and rem[j] > 1e-12]
        if not avail:
            t = min(r[j] for j in range(n) if rem[j] > 1e-12)
            continue
        j = min(avail, key=lambda q: (d[q], q))
        if piece is None:
            nxt = [e for e in events if e > t + 1e-12]
            run = rem[j] if not nxt else min(rem[j], nxt[0] - t)
        else:
            run = min(rem[j], piece)
        t += run
        rem[j] -= run
        if rem[j] <= 1e-12:
            comp[j] = t
    return max(comp[j] - d[j] for j in range(n))


def horn_bound(p, d, r):
    best = -1e18
    n = len(p)
    for ri in set(r):
        for di in set(d):
            work = sum(p[j] for j in range(n) if r[j] >= ri and d[j] <= di)
            if work > 0:
                best = max(best, ri + work - di)
    return best


def d1_jackson(rng, trials=300):
    bad = 0
    for _ in range(trials):
        n = rng.randint(2, 7)
        p = [rng.uniform(0.1, 2.0) for _ in range(n)]
        d = [rng.uniform(0.0, sum(p)) for _ in range(n)]
        edd = sorted(range(n), key=lambda j: d[j])
        best = min(lmax_of_order(o, p, d) for o in itertools.permutations(range(n)))
        if lmax_of_order(edd, p, d) > best + 1e-9:
            bad += 1
    P("D1. Jackson: EDD against all orders, %d instances (2-7 jobs): EDD worse than the best order in %d." % (trials, bad))


def d2_horn(rng, trials=2000):
    bad = 0
    worst = 0.0
    for _ in range(trials):
        n = rng.randint(2, 9)
        p = [rng.uniform(0.1, 2.0) for _ in range(n)]
        r = [rng.uniform(0.0, 4.0) for _ in range(n)]
        d = [r[j] + rng.uniform(0.0, 6.0) for j in range(n)]
        a = preemptive_edd(p, d, r)
        b = horn_bound(p, d, r)
        worst = max(worst, abs(a - b))
        if abs(a - b) > 1e-7:
            bad += 1
    P("D2. Horn: preemptive EDD against the lower bound L*, %d instances with release dates: "
      "mismatches %d (max |difference| %.1e)." % (trials, bad, worst))


def d3_pieces(rng, trials=3000):
    worst_gap = 0.0
    viol = 0
    nonopt = 0
    piece = 1.5
    for _ in range(trials):
        n = rng.randint(2, 9)
        p = [rng.uniform(0.1, 6.0) for _ in range(n)]
        r = [rng.uniform(0.0, 8.0) for _ in range(n)]
        d = [r[j] + rng.uniform(0.0, 12.0) for j in range(n)]
        a = preemptive_edd(p, d, r, piece=piece)
        b = horn_bound(p, d, r)
        gap = a - b
        worst_gap = max(worst_gap, gap)
        if gap > piece + 1e-9:
            viol += 1
        if gap > 1e-9:
            nonopt += 1
    P("D3. Pieces of at most p_max = %.1f device-seconds (a 512-page read window at 21.4 K tokens/s is 1.5 s):" % piece)
    P("    EDD over released pieces exceeds L* in %d of %d instances; largest excess %.3f s = %.2f p_max; "
      "excess above p_max: %d." % (nonopt, trials, worst_gap, worst_gap / piece, viol))


def d4_nonpreemptive(rng, trials=400):
    worse = 0
    gaps = []
    for _ in range(trials):
        n = rng.randint(3, 7)
        p = [rng.uniform(0.1, 3.0) for _ in range(n)]
        r = [rng.uniform(0.0, 4.0) for _ in range(n)]
        d = [r[j] + rng.uniform(0.0, 6.0) for j in range(n)]
        # EDD among released jobs, no preemption (Schrage)
        t = 0.0
        left = set(range(n))
        worst = -1e18
        while left:
            avail = [j for j in left if r[j] <= t + 1e-12]
            if not avail:
                t = min(r[j] for j in left)
                continue
            j = min(avail, key=lambda q: (d[q], q))
            t += p[j]
            worst = max(worst, t - d[j])
            left.remove(j)
        best = min(lmax_of_order(o, p, d, r) for o in itertools.permutations(range(n)))
        if worst > best + 1e-9:
            worse += 1
            gaps.append((worst - best) / max(p))
    P("D4. No preemption, release dates: EDD among released jobs is worse than the best order in %d of %d "
      "instances; excess at most %.2f of the longest job (theory: below 1)." % (
          worse, trials, max(gaps) if gaps else 0.0))


def wt_opt(p, d, w):
    """Minimum weighted tardiness, all jobs available at 0 (dynamic programme over subsets)."""
    n = len(p)
    full = 1 << n
    best = [1e18] * full
    best[0] = 0.0
    tsum = [0.0] * full
    for s in range(1, full):
        low = s & (-s)
        j = low.bit_length() - 1
        tsum[s] = tsum[s ^ low] + p[j]
    for s in range(1, full):
        c = tsum[s]
        b = 1e18
        m = s
        while m:
            low = m & (-m)
            j = low.bit_length() - 1
            v = best[s ^ low] + w[j] * max(0.0, c - d[j])
            if v < b:
                b = v
            m ^= low
        best[s] = b
    return best[full - 1]


def wt_of_order(order, p, d, w):
    t = 0.0
    tot = 0.0
    for j in order:
        t += p[j]
        tot += w[j] * max(0.0, t - d[j])
    return tot


def rule_edd(p, d, w):
    return sorted(range(len(p)), key=lambda j: d[j])


def rule_smith(p, d, w):
    return sorted(range(len(p)), key=lambda j: -w[j] / p[j])


def rule_draft(p, d, w):
    """EDD if EDD meets every due date; otherwise largest cost per device-second first (as section 5.2 reads)."""
    o = rule_edd(p, d, w)
    if lmax_of_order(o, p, d) <= 1e-12:
        return o
    return rule_smith(p, d, w)


def rule_draft_late(p, d, w):
    """A reading closer to 3.4: jobs that are already late (or cannot finish in time) go first by Smith's ratio,
    the others follow in EDD order; re-evaluated at every dispatch."""
    left = set(range(len(p)))
    t = 0.0
    order = []
    while left:
        late = [j for j in left if t + p[j] > d[j]]
        if late:
            j = max(late, key=lambda q: w[q] / p[q])
        else:
            j = min(left, key=lambda q: d[q])
        order.append(j)
        left.remove(j)
        t += p[j]
    return order


def rule_wmdd(p, d, w):
    """Weighted modified due date: at each dispatch take the job with the smallest max(p_j, d_j - t) / w_j."""
    left = set(range(len(p)))
    t = 0.0
    order = []
    while left:
        j = min(left, key=lambda q: max(p[q], d[q] - t) / w[q])
        order.append(j)
        left.remove(j)
        t += p[j]
    return order


def rule_edd_else_wmdd(p, d, w):
    """EDD while EDD meets every due date; otherwise the single index w / max(p, d - t)."""
    o = rule_edd(p, d, w)
    if lmax_of_order(o, p, d) <= 1e-12:
        return o
    return rule_wmdd(p, d, w)


def rule_atc(p, d, w, k=2.0):
    """Apparent tardiness cost (Vepsalainen and Morton 1987): (w/p) exp(-(d - p - t)+ / (k * mean p))."""
    import math
    left = set(range(len(p)))
    t = 0.0
    order = []
    pbar = sum(p) / len(p)
    while left:
        j = max(left, key=lambda q: (w[q] / p[q]) * math.exp(-max(d[q] - p[q] - t, 0.0) / (k * pbar)))
        order.append(j)
        left.remove(j)
        t += p[j]
    return order


def d5_tardiness(rng, trials=1500):
    P("D5. Weighted tardiness on one device, 8 jobs available at once (reads: weight about 1 per second, 0.3-3 s;")
    P("    writes: weight 0.05-3 per second, 0.05-0.5 s).  Mean cost over the optimum's cost (and share of")
    P("    instances solved optimally), by how tight the due dates are:")
    rules = (("EDD", rule_edd), ("Smith ratio", rule_smith), ("EDD else Smith (5.2)", rule_draft),
             ("late by Smith, rest EDD", rule_draft_late), ("w / max(p, d - t)", rule_wmdd),
             ("EDD else w/max(p,d-t)", rule_edd_else_wmdd), ("ATC, k = 2", rule_atc))
    P("    %-24s %s" % ("regime", "  ".join("%-22s" % nm for nm, _ in rules)))
    for label, lo, hi in (("all due dates can be met", 1.0, 2.5), ("some cannot (mixed)", 0.2, 1.1),
                          ("heavy overload", 0.0, 0.5), ("every job already late", -1.0, 0.0)):
        tot = {nm: 0.0 for nm, _ in rules}
        opt_cnt = {nm: 0 for nm, _ in rules}
        tot_opt = 0.0
        cnt = 0
        feas = 0
        for _ in range(trials):
            n = 8
            p, w = [], []
            for j in range(n):
                if rng.random() < 0.5:
                    p.append(rng.uniform(0.3, 3.0))
                    w.append(rng.uniform(0.5, 1.5))
                else:
                    p.append(rng.uniform(0.05, 0.5))
                    w.append(rng.uniform(0.05, 3.0))
            total = sum(p)
            d = [rng.uniform(lo, hi) * total for _ in range(n)]
            if label.startswith("all"):
                # make the instance feasible: due dates = completion times of a random order plus slack
                order = list(range(n))
                rng.shuffle(order)
                t = 0.0
                for j in order:
                    t += p[j]
                    d[j] = t + rng.uniform(0.0, 1.0)
            opt = wt_opt(p, d, w)
            cnt += 1
            tot_opt += opt
            if opt <= 1e-12:
                feas += 1
            for nm, fn in rules:
                c = wt_of_order(fn(p, d, w), p, d, w)
                tot[nm] += c
                if c <= opt + 1e-9:
                    opt_cnt[nm] += 1
        cells = []
        for nm, _ in rules:
            ratio = tot[nm] / tot_opt if tot_opt > 0 else float("nan")
            cells.append("%-22s" % ("%6.3f (%3.0f %% opt.)" % (ratio, 100.0 * opt_cnt[nm] / cnt)
                                    if tot_opt > 0 else "cost %.2f (%3.0f %% opt.)" % (tot[nm] / cnt, 100.0 * opt_cnt[nm] / cnt)))
        P("    %-24s %s" % (label, "  ".join(cells)))
    P("    Two counterexamples (weights 1):")
    p, d, w = [10.0, 1.0], [9.0, 9.5], [1.0, 1.0]
    P("      jobs (work, due) = (10, 9), (1, 9.5): EDD costs %.1f, the other order %.1f  -> EDD is not optimal "
      "once a job must be late." % (wt_of_order([0, 1], p, d, w), wt_of_order([1, 0], p, d, w)))
    p, d, w = [10.0, 1.0, 5.0], [10.0, 100.0, 12.0], [1.0, 5.0, 1.0]
    P("      jobs (work, due, weight) = (10, 10, 1), (1, 100, 5), (5, 12, 1): EDD costs %.1f, Smith's order %.1f, "
      "optimum %.1f  -> the fallback is not optimal either." % (
          wt_of_order(rule_edd(p, d, w), p, d, w), wt_of_order(rule_smith(p, d, w), p, d, w), wt_opt(p, d, w)))


def d6_precedence(rng, trials=400):
    bad = 0
    naive_bad = 0
    for _ in range(trials):
        nw = rng.randint(1, 3)
        nr = rng.randint(1, 3)
        n = nw + nr
        p = [rng.uniform(0.05, 0.5) for _ in range(nw)] + [rng.uniform(0.3, 2.0) for _ in range(nr)]
        d = [rng.uniform(0.0, 6.0) for _ in range(n)]
        # each read needs room freed by one write (chosen at random); a write may serve several reads
        pred = {nw + k: rng.randrange(nw) for k in range(nr)}

        def feasible(order):
            pos = {j: i for i, j in enumerate(order)}
            return all(pos[pred[j]] < pos[j] for j in pred)

        best = min(lmax_of_order(o, p, d) for o in itertools.permutations(range(n)) if feasible(o))
        dm = list(d)
        for j, wj in pred.items():
            dm[wj] = min(dm[wj], d[j] - p[j])
        order = sorted(range(n), key=lambda j: (dm[j], 0 if j < nw else 1))
        if not feasible(order) or lmax_of_order(order, p, d) > best + 1e-9:
            bad += 1
        # naive EDD on the raw due dates, repaired by moving a needed write just before its read
        raw = sorted(range(n), key=lambda j: d[j])
        fixed = []
        for j in raw:
            if j in pred and pred[j] not in fixed:
                fixed.append(pred[j])
            if j not in fixed:
                fixed.append(j)
        if lmax_of_order(fixed, p, d) > best + 1e-9:
            naive_bad += 1
    P("D6. Reads behind the writes that free their room, %d instances: EDD on modified due dates "
      "d'_w = min(d_w, T_j - p_j) misses the optimum in %d; EDD on raw due dates with the write pulled just "
      "before its read misses it in %d." % (trials, bad, naive_bad))


def main():
    rng = random.Random(1126)
    P("Order on the device: earliest due date and its limits")
    P("")
    d1_jackson(rng)
    d2_horn(rng)
    d3_pieces(rng)
    d4_nonpreemptive(rng)
    P("")
    d5_tardiness(rng)
    P("")
    d6_precedence(rng)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "device_order.out"), "w") as f:
        f.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    sys.exit(main())
