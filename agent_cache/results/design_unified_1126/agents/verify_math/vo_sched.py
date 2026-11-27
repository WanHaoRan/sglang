"""verify_math, part 6: device order (math_fetch R12, math_order_price 4), read or recompute (math_fetch R10),
the knapsack of Preempt (5), the room price (6), stability (9) and the many-session toy (math_retain R6, order_price 10).

Run: CUDA_VISIBLE_DEVICES= nice -n 10 <venv python> -I -B vo_sched.py > vo_sched.out
"""
from __future__ import annotations

import heapq
import itertools
import math
import sys

import numpy as np
from scipy import optimize

sys.path.insert(0, ".")
from vlib import TwoPiece  # noqa: E402

SEED = 55_1126


# ---------------------------------------------------------------- device order
def preemptive_edd(rel, p, due, piece=None):
    """Earliest due date among released jobs; preemption at any instant (piece=None) or only between pieces."""
    n = len(rel)
    left = list(p)
    t = 0.0
    comp = [None] * n
    order_rel = sorted(range(n), key=lambda j: rel[j])
    while any(c is None for c in comp):
        avail = [j for j in range(n) if comp[j] is None and rel[j] <= t + 1e-12]
        if not avail:
            t = min(rel[j] for j in range(n) if comp[j] is None)
            continue
        j = min(avail, key=lambda q: (due[q], q))
        future = [rel[q] for q in range(n) if comp[q] is None and rel[q] > t + 1e-12]
        if piece is None:
            run = left[j] if not future else min(left[j], min(future) - t)
        else:
            run = min(left[j], piece)
        t += run
        left[j] -= run
        if left[j] <= 1e-12:
            comp[j] = t
    return comp


def device_order():
    print("== Order on the device: Jackson, Horn, pieces, Smith, the mixed regime, precedence ==")
    rng = np.random.default_rng([SEED, 1])
    # Jackson
    miss = 0
    for trial in range(300):
        n = int(rng.integers(2, 8))
        p = rng.uniform(0.2, 3.0, size=n)
        d = rng.uniform(0.0, 8.0, size=n)
        best = min(max(np.cumsum(p[list(pm)]) - d[list(pm)]) for pm in itertools.permutations(range(n)))
        o = np.argsort(d, kind="stable")
        if max(np.cumsum(p[o]) - d[o]) > best + 1e-9:
            miss += 1
    print(f"   all reads released, 300 instances of 2-7 jobs: earliest due date misses the minimum of the maximum lateness {miss} times")
    # Horn + formula + non-preemptive lower bound + pieces
    gap_formula = 0.0
    below_nonpre = 0
    edd_np_notbest = 0
    worst_np = 0.0
    piece_excess = 0.0
    piece_over = 0
    piece_viol = 0
    pmax = 1.5
    longest_ratio = 0.0
    for trial in range(1500):
        n = int(rng.integers(2, 7))
        rel = rng.uniform(0.0, 6.0, size=n)
        p = rng.uniform(0.2, 3.0, size=n)
        due = rel + p + rng.uniform(-0.5, 5.0, size=n)
        comp = preemptive_edd(rel, p, due)
        Lpre = max(comp[j] - due[j] for j in range(n))
        Lstar = -np.inf
        for r0 in rel:
            for d0 in due:
                work = sum(p[j] for j in range(n) if rel[j] >= r0 - 1e-12 and due[j] <= d0 + 1e-12)
                if work > 0:
                    Lstar = max(Lstar, r0 + work - d0)
        gap_formula = max(gap_formula, abs(Lpre - Lstar))
        # best non-preemptive order
        bestnp = np.inf
        for pm in itertools.permutations(range(n)):
            t = 0.0
            L = -np.inf
            for j in pm:
                t = max(t, rel[j]) + p[j]
                L = max(L, t - due[j])
            bestnp = min(bestnp, L)
        if Lpre > bestnp + 1e-9:
            below_nonpre += 1
        # non-preemptive EDD among released
        done = [False] * n
        t = 0.0
        Lnp = -np.inf
        for _ in range(n):
            avail = [j for j in range(n) if not done[j] and rel[j] <= t + 1e-12]
            if not avail:
                t = min(rel[j] for j in range(n) if not done[j])
                avail = [j for j in range(n) if not done[j] and rel[j] <= t + 1e-12]
            j = min(avail, key=lambda q: (due[q], q))
            t += p[j]
            done[j] = True
            Lnp = max(Lnp, t - due[j])
        if Lnp > bestnp + 1e-9:
            edd_np_notbest += 1
            worst_np = max(worst_np, Lnp - bestnp)
            longest_ratio = max(longest_ratio, (Lnp - Lstar) / p.max())
        # pieces of at most pmax
        compp = preemptive_edd(rel, p, due, piece=pmax)
        Lp = max(compp[j] - due[j] for j in range(n))
        if Lp > Lstar + 1e-9:
            piece_over += 1
            piece_excess = max(piece_excess, Lp - Lstar)
        if Lp >= Lstar + pmax - 1e-12:
            piece_viol += 1
    print(f"   release times, 1,500 instances: preemptive earliest due date against the formula L* = max over (r, d) of [r + work released at or after r and due by d - d]: worst gap {gap_formula:.1e}; above the best non-preemptive order in {below_nonpre}")
    print(f"   non-preemptive earliest due date is not the best order in {edd_np_notbest} instances (worst excess {worst_np:.2f} s; excess over L* at most {longest_ratio:.2f} of the longest job)")
    print(f"   preemption only between pieces of {pmax} s: above L* in {piece_over} instances, largest excess {piece_excess:.3f} s; bound L* + p_max violated {piece_viol} times")
    # Smith when everything is late; mixed regime
    miss_smith = 0
    edd_bad = 0
    smith_bad = 0
    both_bad = 0
    tot = 0
    worst_e = worst_s = 0.0
    for trial in range(400):
        n = 6
        p = rng.uniform(0.2, 3.0, size=n)
        w = rng.choice([0.1, 1.0, 3.0], size=n)
        d_late = -rng.uniform(0.0, 5.0, size=n)
        best = min(float(np.sum(w[list(pm)] * np.maximum(np.cumsum(p[list(pm)]) - d_late[list(pm)], 0))) for pm in itertools.permutations(range(n)))
        o = np.argsort(-w / p, kind="stable")
        if float(np.sum(w[o] * np.maximum(np.cumsum(p[o]) - d_late[o], 0))) > best + 1e-9:
            miss_smith += 1
        d = rng.uniform(0.0, p.sum(), size=n)
        costs = {pm: float(np.sum(w[list(pm)] * np.maximum(np.cumsum(p[list(pm)]) - d[list(pm)], 0))) for pm in itertools.permutations(range(n))}
        best = min(costs.values())
        e = tuple(np.argsort(d, kind="stable"))
        s_ = tuple(np.argsort(-w / p, kind="stable"))
        tot += 1
        eb = costs[e] > best + 1e-9
        sb = costs[s_] > best + 1e-9
        edd_bad += eb
        smith_bad += sb
        both_bad += eb and sb
        worst_e = max(worst_e, costs[e] - best)
        worst_s = max(worst_s, costs[s_] - best)
    print(f"   every due date already passed, 400 instances of 6 jobs: largest w/p first misses the minimum weighted lateness {miss_smith} times")
    print(f"   mixed due dates (weighted tardiness): earliest due date not optimal in {edd_bad} of {tot}, largest w/p first not optimal in {smith_bad}, neither optimal in {both_bad} (worst excess {worst_e:.1f} and {worst_s:.1f} s)")
    c1 = max(10 - 9, 0) + max(11 - 9.5, 0)
    c2 = max(1 - 9.5, 0) + max(11 - 9, 0)
    print(f"   two jobs (work, due) = (10, 9), (1, 9.5): earliest due date costs {c1}, the other order {c2}")
    jobs = [(10, 10, 1), (1, 100, 5), (5, 12, 1)]
    allc = {}
    for pm in itertools.permutations(range(3)):
        t = 0
        cst = 0
        for j in pm:
            t += jobs[j][0]
            cst += jobs[j][2] * max(t - jobs[j][1], 0)
        allc[pm] = cst
    sm = tuple(sorted(range(3), key=lambda j: -jobs[j][2] / jobs[j][0]))
    print(f"   three jobs (work, due, weight) = (10, 10, 1), (1, 100, 5), (5, 12, 1): w/p order costs {allc[sm]}, the optimum {min(allc.values())}")
    # precedence: a write must finish before a read starts; modified due date d'_w = min(d_w, T_read - p_read)
    bad = 0
    for trial in range(400):
        n = int(rng.integers(3, 7))
        p = rng.uniform(0.2, 3.0, size=n)
        d = rng.uniform(0.0, 8.0, size=n)
        wj, rj = rng.choice(n, size=2, replace=False)
        best = np.inf
        for pm in itertools.permutations(range(n)):
            if pm.index(wj) > pm.index(rj):
                continue
            best = min(best, max(np.cumsum(p[list(pm)]) - d[list(pm)]))
        d2 = d.copy()
        d2[wj] = min(d[wj], d[rj] - p[rj])
        o = sorted(range(n), key=lambda j: (d2[j], 0 if j == wj else 1))
        if o.index(wj) > o.index(rj):
            bad += 1
            continue
        if max(np.cumsum(p[o]) - d[o]) > best + 1e-9:
            bad += 1
    print(f"   one write that must finish before one read starts: earliest due date on the modified date misses the constrained optimum {bad} times in 400")


# ---------------------------------------------------------------- read or recompute
def read_or_recompute():
    print("\n== Read or recompute (lambda = 0, linear recompute): crossover lengths ==")
    beta, lags = 21_400.0, 0.9
    Ls = np.arange(64, 65_536 + 1, 64)
    stated_bad = corrected_bad = cells = 0
    for gamma in (0.037e-3, 0.044e-3):
        for mu in (1, 2, 5, 128):
            for c in (1.0, 0.3):
                for W in (0.0, 2.0, 5.0):
                    for kappa in (0.0, 0.5, 2.0, 5.0):
                        read = c * np.maximum(lags + Ls / beta - W, 0.0) + kappa * Ls / beta
                        rec = mu * gamma * Ls
                        better_read = rec > read
                        D0 = lags - W
                        slope = mu * gamma - (c + kappa) / beta
                        if D0 >= 0:
                            stated = (Ls > c * D0 / slope) if slope > 0 else np.zeros(len(Ls), dtype=bool)
                            corrected = stated
                        elif slope >= 0:
                            stated = corrected = np.ones(len(Ls), dtype=bool)
                        else:
                            stated = Ls < c * (-D0) / (-slope)  # R10 as written
                            # corrected: while the read hides in the wait only the device price counts
                            corrected = stated if mu * gamma > kappa / beta else np.zeros(len(Ls), dtype=bool)
                        cells += 1
                        stated_bad += int(np.any(stated != better_read))
                        corrected_bad += int(np.any(corrected != better_read))
    print(f"   {cells} parameter cells (kappa = 0, 0.5, 2, 5), every length 64..65,536: R10's rule as written disagrees with a direct comparison in {stated_bad} cells (all with D0 < 0 and mu gamma <= kappa/beta); with the added branch 'recompute at every length when mu gamma <= kappa/beta' in {corrected_bad}")
    c_, k_, mu_, g_, W_ = 1.0, 2.0, 1, 0.04e-3, 5.0
    thr = c_ * (W_ - lags) / ((c_ + k_) / beta - mu_ * g_)
    print(f"   example: c = 1, kappa = 2, mu = 1, gamma = 0.04 ms, W = 5 s: the rule as written reads below {thr / 1000:.1f} K; at L = 20 K the read costs {k_ * 20000 / beta:.2f} s (device time only) against {mu_ * g_ * 20000:.2f} s to recompute")
    for c in (1.0, 0.3):
        row = []
        for mu in (1, 2, 5, 128):
            cell = []
            for gamma in (0.037e-3, 0.044e-3):
                slope = mu * gamma - c / beta
                cell.append("never" if slope <= 0 else f"{c * lags / slope / 1000:.2f} K")
            row.append(f"mu = {mu}: " + " / ".join(cell))
        print(f"   W = 0, kappa = 0, c = {c}: read wins above " + "; ".join(row) + "  (gamma 0.037 / 0.044 ms)")
    print(f"   gamma beta = {0.037e-3 * beta:.3f} to {0.044e-3 * beta:.3f}; mu_x = 1/(gamma beta) = {1 / (0.037e-3 * beta):.2f} to {1 / (0.044e-3 * beta):.2f} at c = 1, kappa = 0")
    L = 25_000
    print(f"   L = 25 K: K_rec = {0.037e-3 * L:.2f} to {0.044e-3 * L:.2f} s (mu = 1); K_read = {lags + L / beta:.2f} s (c = 1, W = 0), {0.3 * (lags + L / beta):.2f} (c = 0.3), {max(lags + L / beta - 2, 0):.2f} (W = 2 s)")
    # with the in-flight rent at lambda = 2e-5, mu = 2, c = 1: who wins up to 65 K?
    lam = 2e-5
    for gamma in (0.037e-3, 0.044e-3):
        read = (lags + Ls / beta) + lam * Ls * (lags + Ls / beta)
        rec = 2 * gamma * Ls
        print(f"   in-flight rent at lambda = 2e-5, mu = 2, c = 1, gamma = {gamma * 1e3:.3f} ms: read cheaper on {int(np.sum(read < rec))} of {len(Ls)} lengths up to 65,536")
    print(f"   at c = 0 the read loses only if kappa > mu gamma beta: {94 * 0.037e-3 * beta:.0f} to {128 * 0.044e-3 * beta:.0f} for mu = 94 to 128")


# ---------------------------------------------------------------- knapsack
def knapsack():
    print("\n== Preempt on host room: the covering knapsack ==")
    rng = np.random.default_rng([SEED, 2])
    worst_lp = 0.0
    add_viol = 0
    surplus_viol = 0
    worst_ratio = 1.0
    skip_worst = 1.0
    worst_frac = 0.0
    for trial in range(3000):
        n = int(rng.integers(2, 12))
        Lj = rng.integers(1, 40, size=n).astype(float)
        Cj = rng.uniform(0.0, 10.0, size=n) * rng.choice([0.2, 1.0, 5.0], size=n)
        m = float(rng.uniform(1.0, 0.9 * Lj.sum()))
        order = np.argsort(Cj / Lj, kind="stable")
        # divisible optimum by the sort, and by an LP solver
        need = m
        zlp = 0.0
        k = None
        for j in order:
            if Lj[j] >= need:
                zlp += Cj[j] * need / Lj[j]
                k = j
                break
            zlp += Cj[j]
            need -= Lj[j]
        if trial < 300:
            res = optimize.linprog(Cj, A_ub=-Lj[None, :], b_ub=[-m], bounds=(0, 1), method="highs")
            worst_lp = max(worst_lp, abs(res.fun - zlp))
        # 0-1 optimum by enumeration
        best = np.inf
        for mask in range(1 << n):
            tot = 0.0
            cst = 0.0
            for j in range(n):
                if (mask >> j) & 1:
                    tot += Lj[j]
                    cst += Cj[j]
            if tot >= m and cst < best:
                best = cst
        # greedy whole holders
        got = 0.0
        g = 0.0
        for j in order:
            got += Lj[j]
            g += Cj[j]
            if got >= m:
                break
        if g > zlp + Cj[k] + 1e-9 or g > best + Cj[k] + 1e-9:
            add_viol += 1
        if got - m >= Lj[k]:
            surplus_viol += 1
        if best > 1e-9:
            worst_ratio = max(worst_ratio, g / best)
        if Cj[k] > 1e-9:
            worst_frac = max(worst_frac, (g - best) / Cj[k])
        # skip variant
        taken = 0.0
        tc = 0.0
        cand = np.inf
        for j in order:
            if taken + Lj[j] >= m:
                cand = min(cand, tc + Cj[j])
            else:
                taken += Lj[j]
                tc += Cj[j]
        if best > 1e-9:
            skip_worst = max(skip_worst, cand / best)
        if cand > 2 * best + 1e-9:
            print("   !! skip variant above twice the optimum")
    print(f"   divisible problem: ratio greedy against an LP solver on 300 instances, worst difference {worst_lp:.1e}")
    print(f"   0-1 problem, 3,000 instances of 2-11 holders against enumeration: 'greedy <= optimum + one holder' violated {add_viol} times (largest (greedy - optimum)/(critical holder's cost) = {worst_frac:.2f}); surplus room >= critical holder's size {surplus_viol} times; largest greedy/optimum = {worst_ratio:.1f}")
    print(f"   skip variant: largest ratio to the optimum {skip_worst:.2f} (bound 2)")
    for M in (10, 100, 1000):
        print(f"   need 1 token; A: 1 token, cost 1; B: {M} tokens, cost {0.9 * M:g}: greedy takes B first, ratio {0.9 * M:g}", end=";")
    print()


# ---------------------------------------------------------------- price
def price():
    print("\n== The room price: instant problem and its multiplier ==")
    rng = np.random.default_rng([SEED, 3])
    worst = 0.0
    zero_ok = 0
    zero_cases = 0
    for trial in range(300):
        n = int(rng.integers(3, 40))
        Lj = rng.integers(64, 30_000, size=n).astype(float)
        nu = np.exp(rng.uniform(math.log(1e-8), math.log(1e-3), size=n))
        cap = float(rng.uniform(0.2, 1.3) * Lj.sum())
        res = optimize.linprog(-(nu * Lj), A_ub=Lj[None, :], b_ub=[cap], bounds=(0, 1), method="highs")
        dual = -res.ineqlin.marginals[0]
        order = np.argsort(-nu)
        cum = np.cumsum(Lj[order])
        if cum[-1] <= cap:
            zero_cases += 1
            zero_ok += int(abs(dual) < 1e-15)
            continue
        kcrit = order[int(np.argmax(cum > cap))]
        worst = max(worst, abs(dual - nu[kcrit]) / nu[kcrit])
    print(f"   fill the room in decreasing index (300 instances, LP solver): multiplier of the room constraint = index of the critical holder, worst relative gap {worst:.1e}; multiplier 0 in {zero_ok} of the {zero_cases} instances where everything fits")


# ---------------------------------------------------------------- stability
def stability():
    print("\n== Stability: the smallest model of the refused-read loop ==")
    g0, gamma, L, Dw = 0.46, 0.043e-3, 25_000.0, 1_900.0
    d0, d1, pmax = 100e3, 600e3, 0.55
    p = lambda d: min(pmax, max(0.0, pmax * (d - d0) / (d1 - d0)))
    x = lambda pp: 1.0 / (g0 + gamma * L * pp)
    F = lambda d: x(p(d)) * (Dw + L * p(d))
    amax = Dw / g0
    aloop = (Dw + L * pmax) / (g0 + gamma * L * pmax)
    print(f"   a_max = {amax:.0f} tokens/s; a_loop = {aloop:.0f} tokens/s; request rate in the loop {x(pmax):.2f} /s; share of rewrites in the loop {L * pmax / (Dw + L * pmax):.2f}")
    ds = np.linspace(0, 700e3, 7001)
    Fv = np.array([F(d) for d in ds])
    print(f"   F non-decreasing in d: {bool(np.all(np.diff(Fv) >= -1e-9))}; dF/dp > 0 iff g0 > gamma Dw: {g0:.2f} > {gamma * Dw:.3f}")
    for w in (4200.0, 5500.0, 6500.0, 10200.0, 13000.0, 16000.0):
        if not (amax < w < aloop):
            print(f"   w = {w:.0f}: no tipping point (w {'below a_max: the dirty set grows from 0' if w <= amax else 'above a_loop: only d = 0'})")
            continue
        pstar = (w * g0 - Dw) / (L * (1 - w * gamma))
        du = d0 + (d1 - d0) * pstar / pmax
        du_num = optimize.brentq(lambda d: F(d) - w, d0, d1)
        rate = (pmax / (d1 - d0)) * L * (g0 - gamma * Dw) * x(pstar) ** 2
        # integrate from 2 % above and below the tipping point
        def run(dstart, born_clean=False, T=3600.0, dt=0.05):
            d = dstart
            for _ in range(int(T / dt)):
                inflow = (Dw * x(p(d))) if born_clean else F(d)
                d = max(0.0, min(700e3, d + dt * (inflow - w)))
            return d
        up, down, bc = run(du * 1.02), run(du * 0.98), run(650e3, born_clean=True)
        # e-folding measured near the tipping point
        d = du + 500.0
        t = 0.0
        while d - du < 500.0 * math.e and t < 5000:
            d += 0.01 * (F(d) - w)
            t += 0.01
        print(f"   w = {w:.0f}: p* = {pstar:.4f}, tipping point {du / 1e3:.0f} K (root of F(d) = w: {du_num / 1e3:.0f} K); e-folding {1 / rate:.0f} s (measured {t:.0f} s); after one hour from 2 % above: {up / 1e3:.0f} K, from 2 % below: {down / 1e3:.0f} K; born clean from 650 K: {bc / 1e3:.0f} K")


# ---------------------------------------------------------------- many sessions toy
def index_table(law, ages):
    """(L/K) nu(a) on the grid `ages`: steepest descending chord of the curve (I(t), S(t)) to a later grid point."""
    S = law.sf(ages)
    I = float(law.M(0.0)) - law.M(ages)
    n = len(ages)
    out = np.zeros(n)
    stack = [n - 1]
    for j in range(n - 2, -1, -1):
        if S[j] <= 1e-13:
            stack = [j]
            continue
        while len(stack) >= 2:
            a, b = stack[-1], stack[-2]
            s1 = (S[a] - S[j]) / (I[a] - I[j])
            s2 = (S[b] - S[a]) / (I[b] - I[a]) if I[b] > I[a] else 0.0
            if s1 >= s2:
                stack.pop()
            else:
                break
        a = stack[-1]
        out[j] = (S[j] - S[a]) / (I[a] - I[j])
        stack.append(j)
    return out, S, I


def many_sessions():
    print("\n== Many sessions under a hard capacity (equal sizes, K = 1 per miss): index policy, LRU and the priced bound ==")
    CAP = 1200.0
    tool = TwoPiece(0.41, 1.69, 16.8, cap=CAP)
    human = TwoPiece(3.1, 96.0, 1040.0, cap=CAP)
    w_tool = 0.92
    ages = np.concatenate([[0.0], np.geomspace(1e-3, CAP * (1 - 1e-9), 4000), [CAP]])
    it, St, It = index_table(tool, ages)
    ih, Sh, Ih = index_table(human, ages)

    class Pool:
        def sf(self, t):
            return w_tool * tool.sf(t) + (1 - w_tool) * human.sf(t)

        def M(self, t):
            return w_tool * tool.M(t) + (1 - w_tool) * human.M(t)

    ip, Sp, Ip = index_table(Pool(), ages)
    meanG = float(w_tool * tool.M(0.0) + (1 - w_tool) * human.M(0.0))
    lams = np.geomspace(1e-5, 1.0, 4000)

    def bound(frac, known):
        best = (-1.0, None)
        for lam in lams:
            if known:
                v = w_tool * np.min(lam * It + St) + (1 - w_tool) * np.min(lam * Ih + Sh)
            else:
                v = np.min(lam * Ip + Sp)
            val = v - lam * frac * meanG
            if val > best[0]:
                best = (val, lam)
        return best

    def simulate(N, frac, policy, returns, seed):
        rng = np.random.default_rng(seed)
        C = max(1, int(round(frac * N)))
        in_mem = np.zeros(N, dtype=bool)
        idle0 = np.zeros(N)
        cls = np.zeros(N, dtype=int)
        nxt = np.zeros(N)

        def draw(i, t):
            c = 0 if rng.random() < w_tool else 1
            g = float((tool if c == 0 else human).sample(rng, 1)[0])
            cls[i] = c
            idle0[i] = t
            nxt[i] = t + g
            return t + g

        heap = [(draw(i, 0.0), i) for i in range(N)]
        heapq.heapify(heap)
        held = 0
        for i in range(min(C, N)):
            in_mem[i] = True
            held += 1
        misses = hits = 0
        seen = 0
        warm = returns // 6
        ev_idx = []
        lg = np.log(np.maximum(ages, 1e-6))
        while seen < returns + warm:
            t, i = heapq.heappop(heap)
            seen += 1
            if in_mem[i]:
                if seen > warm:
                    hits += 1
            else:
                if seen > warm:
                    misses += 1
                in_mem[i] = True
                held += 1
            heapq.heappush(heap, (draw(i, t), i))
            if held > C:
                idx = np.nonzero(in_mem)[0]
                age = t - idle0[idx]
                if policy == "lru":
                    v = idx[int(np.argmax(age))]
                elif policy == "belady":
                    v = idx[int(np.argmax(nxt[idx]))]
                else:
                    la = np.log(np.maximum(age, 1e-6))
                    if policy == "index":
                        val = np.where(cls[idx] == 0, np.interp(la, lg, it), np.interp(la, lg, ih))
                    else:  # pooled index: class not used
                        val = np.interp(la, lg, ip)
                    k = int(np.argmin(val))
                    v = idx[k]
                    if seen > warm:
                        ev_idx.append(float(val[k]))
                in_mem[v] = False
                held -= 1
        return misses / (misses + hits), ev_idx

    for frac in (0.25, 0.5):
        bk, lam_k = bound(frac, True)
        bp, lam_p = bound(frac, False)
        print(f"   room for {int(100 * frac)} % of the sessions: priced bound on the miss ratio {bk:.4f} (class known, lambda* = {lam_k:.4f} per slot-second), {bp:.4f} (class not known)")
        for N in (32, 128, 512):
            R = 200_000
            mi, ev = simulate(N, frac, "index", R, [SEED, 10, N])
            ml, _ = simulate(N, frac, "lru", R, [SEED, 11, N])
            mp, _ = simulate(N, frac, "pooled", R, [SEED, 12, N])
            mb, _ = simulate(N, frac, "belady", R, [SEED, 13, N])
            se = math.sqrt(mi * (1 - mi) / R)
            ev = np.array(ev)
            print(f"      N = {N}: class-aware index {mi:.4f} (+{100 * (mi / bk - 1):.1f} % over the bound), pooled index {mp:.4f}, LRU {ml:.4f}, clairvoyant {mb:.4f} (binomial se about {se:.4f}); index of the evicted prefix / lambda*: median {np.median(ev) / lam_k:.2f}, mean {ev.mean() / lam_k:.2f}")


if __name__ == "__main__":
    which = sys.argv[1:] or ["device_order", "read_or_recompute", "knapsack", "price", "stability", "many_sessions"]
    for w in which:
        globals()[w]()
