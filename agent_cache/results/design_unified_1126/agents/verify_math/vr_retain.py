"""verify_math, part 4: math_retain R1-R8 (the holding index), re-derived and tested with independent code.

Lattice laws: the gap takes values dt, 2 dt, ..., n dt with probabilities p_j, and never returns with
probability p_inf. For such laws the continuous formulas are exact with drop ages on the lattice.
Run: CUDA_VISIBLE_DEVICES= nice -n 10 <venv python> -I -B vr_retain.py > vr_retain.out
"""
from __future__ import annotations

import itertools
import math
import sys
from functools import lru_cache

import numpy as np
from scipy import optimize, special, stats

sys.path.insert(0, ".")
from vlib import Phibar, ScipyLaw, TwoPiece, phi, quad_sf, stoploss_vec  # noqa: E402

SEED = 77_1126


def lattice_from_sf(sf_vals, p_inf):
    """sf_vals[j] = P(G > j dt), j = 0..n, with sf_vals[0] = 1 and sf_vals[n] = p_inf."""
    S = np.asarray(sf_vals, dtype=float)
    S = np.maximum.accumulate(S[::-1])[::-1]
    S[0] = 1.0
    S[-1] = p_inf
    return S


def random_lattice(rng, n, kind):
    t = np.arange(n + 1) / n
    p_inf = float(rng.choice([0.0, 0.0, 0.05, 0.3]))
    if kind == "bimodal":
        a, b = sorted(rng.uniform(0.05, 0.95, size=2))
        w = rng.uniform(0.2, 0.8)
        cdf = w * stats.norm.cdf((t - a) / rng.uniform(0.01, 0.06)) + (1 - w) * stats.norm.cdf((t - b) / rng.uniform(0.01, 0.06))
    elif kind == "heavy":
        cdf = 1 - (1 + t * rng.uniform(5, 80)) ** (-rng.uniform(0.4, 2.5))
    elif kind == "sharp":
        cdf = stats.norm.cdf((t - rng.uniform(0.2, 0.8)) / rng.uniform(0.005, 0.03))
    elif kind == "uniform":
        lo = rng.uniform(0.0, 0.5)
        hi = rng.uniform(lo + 0.1, 1.0)
        cdf = np.clip((t - lo) / (hi - lo), 0, 1)
    elif kind == "weibull":
        k = rng.choice([0.4, 0.7, 1.0, 2.0, 4.0])
        cdf = 1 - np.exp(-((t / rng.uniform(0.1, 0.5)) ** k))
    else:  # random steps
        inc = rng.dirichlet(np.full(n, 0.05))
        cdf = np.concatenate([[0.0], np.cumsum(inc)])
    cdf = cdf - cdf[0]
    cdf = cdf / cdf[-1] if cdf[-1] > 0 else t
    S = p_inf + (1 - p_inf) * (1 - cdf)
    return lattice_from_sf(S, p_inf), p_inf


def index_bruteforce(S, dt):
    """(L/K) nu(j) = max over h >= 1 of (S[j] - S[j+h]) / (dt * sum_{i=j}^{j+h-1} S[i]); O(n^2)."""
    n = len(S) - 1
    I = np.concatenate([[0.0], np.cumsum(S[:-1]) * dt])
    out = np.zeros(n + 1)
    for j in range(n):
        if S[j] <= 0:
            continue
        num = S[j] - S[j + 1 :]
        den = dt * np.cumsum(S[j:-1])  # I[j+h] - I[j], summed from j to avoid cancellation
        out[j] = np.max(num / den)
    return out, I


def index_hull(S, dt):
    """The same by one pass from the right over the points (I(t), S(t)): slope of the first hull edge."""
    n = len(S) - 1
    I = np.concatenate([[0.0], np.cumsum(S[:-1]) * dt])
    out = np.zeros(n + 1)
    # ages with no mass left (S = 0) are one and the same point of the curve: start from the first of them
    dead = np.nonzero(S <= 1e-13)[0]
    top = int(dead[0]) if len(dead) else n
    stack = [top]  # indices of hull vertices of the suffix, leftmost last
    for j in range(top - 1, -1, -1):
        # pop while the edge j -> top is not steeper (more negative) than the edge top -> next
        while len(stack) >= 2:
            a, b = stack[-1], stack[-2]
            s1 = (S[a] - S[j]) / (I[a] - I[j])
            s2 = (S[b] - S[a]) / (I[b] - I[a]) if I[b] > I[a] else 0.0
            if s1 >= s2:  # j -> a is flatter than a -> b: a is above the chord j -> b, not a lower-hull vertex
                stack.pop()
            else:
                break
        a = stack[-1]
        out[j] = (S[j] - S[a]) / (I[a] - I[j])
        stack.append(j)
    return out


def r1_r3():
    print("== R1-R3: keep iff nu(a) > lambda; first passage; nested drop sets; the hull form ==")
    rng = np.random.default_rng([SEED, 1])
    n = 400
    dt = 0.25
    kinds = ["bimodal", "heavy", "sharp", "uniform", "weibull", "steps"]
    viol_test = viol_first = viol_nest = 0
    worst_val = 0.0
    worst_hull = 0.0
    cases = 0
    ties = 0
    for trial in range(120):
        S, p_inf = random_lattice(rng, n, kinds[trial % len(kinds)])
        nu_KL, I = index_bruteforce(S, dt)
        hull = index_hull(S, dt)
        alive = S > 1e-4  # below that, I(t) differences lose digits in double precision
        worst_hull = max(worst_hull, float(np.max(np.abs(hull[alive[: n + 1]] - nu_KL[alive[: n + 1]]) / np.maximum(nu_KL[alive[: n + 1]], 1e-12))) if alive.any() else 0.0)
        K, L = rng.uniform(0.2, 5.0), rng.choice([2000.0, 25000.0])
        nu = nu_KL * K / L
        prev_drop = None
        for lam in np.geomspace(1e-8, 1e-1, 29):
            lamL = lam * L
            # C(s) on the lattice and backward induction
            C = lamL * I + K * (S - p_inf)
            pi = np.where(S > 0, (S - p_inf) / np.maximum(S, 1e-300), 0.0)
            V = np.zeros(n + 1)
            V[n] = K * pi[n]
            stop = np.zeros(n + 1, dtype=bool)
            stop[n] = True
            for j in range(n - 1, -1, -1):
                if S[j] <= 0:
                    stop[j] = True
                    continue
                cont = lamL * dt + (S[j + 1] / S[j]) * V[j + 1]
                dropv = K * pi[j]
                if dropv <= cont + 1e-13:
                    V[j] = dropv
                    stop[j] = True
                else:
                    V[j] = cont
            # (iii): dropping optimal iff nu <= lambda; ties (|nu - lambda| tiny) skipped
            for j in range(0, n, 7):
                if S[j] <= 1e-9:
                    continue
                if abs(nu[j] - lam) <= 1e-9 * max(lam, 1e-12):
                    ties += 1
                    continue
                # cost of every drop age s >= j given alive at j, summed from j (no cancellation)
                seg = np.concatenate([[0.0], np.cumsum(S[j:-1])]) * dt
                costs = (lamL * seg + K * (S[j:] - p_inf)) / S[j]
                best_from_j = float(costs.min())
                drop_is_opt = costs[0] <= best_from_j + 1e-12 * max(1.0, costs[0])
                if drop_is_opt != (nu[j] <= lam):
                    viol_test += 1
                worst_val = max(worst_val, abs(V[j] - best_from_j) / max(1.0, best_from_j))
            # (iv) first passage from age 0 attains min C, and is the earliest minimiser
            first = int(np.argmax(nu <= lam)) if np.any(nu <= lam) else n
            if C[first] > C.min() + 1e-10 or np.any(C[:first] <= C.min() + 1e-13):
                # allow exact ties at earlier ages
                if C[first] > C.min() + 1e-10:
                    viol_first += 1
            drop_set = nu <= lam
            if prev_drop is not None and np.any(prev_drop & ~drop_set):
                viol_nest += 1
            prev_drop = drop_set
            cases += 1
    print(f"   {cases} (law, price) cases, 120 lattice laws of 400 points (six shapes; end mass 0, 5 or 30 %):")
    print(f"   'dropping is optimal iff nu <= lambda' violated {viol_test} times ({ties} exact ties skipped); first passage not a minimiser of C: {viol_first}; drop sets not nested in lambda: {viol_nest}")
    print(f"   backward induction against the direct minimum over drop ages: worst relative difference {worst_val:.2e}; hull index against the O(n^2) definition: worst relative difference {worst_hull:.2e}")

    # Monte Carlo of the rule on the tool entry (continuous law), K = 2 s, L = 25,000, lambda = 5e-6
    law = TwoPiece(0.41, 1.69, 16.8)
    K, L, lam = 2.0, 25_000.0, 5e-6
    lamL = lam * L
    ages = np.geomspace(1e-3, 3000.0, 6000)
    # index by brute force on a geometric grid of horizons
    def index_at(a):
        hs = np.geomspace(1e-4, 5e4, 4000)
        Sa = float(law.sf_raw(a))
        num = Sa - law.sf_raw(a + hs)
        den = float(law.M_raw(a)) - law.M_raw(a + hs)
        return float(np.max(num / den))
    idx = np.array([index_at(a) for a in ages[::20]])
    ag = ages[::20]
    below = (K / L) * idx <= lam
    first = ag[int(np.argmax(below))]
    Cfun = lambda s: lamL * (float(law.M_raw(0.0)) - law.M_raw(s)) + K * law.sf_raw(s)
    sg = np.geomspace(1e-3, 3000.0, 200_000)
    Cg = Cfun(sg)
    sb = sg[int(Cg.argmin())]
    rngm = np.random.default_rng([SEED, 11])
    G = law.sample(rngm, 1_000_000)
    cost = lamL * np.minimum(G, first) + K * (G > first)
    print(f"   tool entry, K = 2 s, L = 25,000, lambda = 5e-6: index rule drops at {first:.2f} s; best drop age by search {sb:.2f} s; cost formula {float(Cfun(first)):.4f}, simulated {cost.mean():.4f} (se {cost.std() / 1000:.4f}); dropping at once {K:.3f}; index at age 0 = {index_at(1e-9) :.4f} K/L per s while the hazard there is 0")
    # a myopic hazard rule (keep while (K/L) hazard >= lambda) evicts at the turn's end
    print(f"   hazard at 0.001 s = {float(law.hazard_raw(0.001)):.2e} /s: the rule 'keep while (K/L) x hazard >= lambda' drops at once and pays {K:.2f} against {float(Cg.min()):.4f}")


def r4_r5():
    print("\n== R4-R5: closed forms of the index ==")
    def sup_ratio(sf, intsf, a, hs):
        Sa = float(sf(a))
        num = Sa - sf(a + hs)
        den = np.array([intsf(a, a + h) for h in hs])
        r = num / den
        k = int(np.argmax(r))
        return float(r[k]), float(hs[k])

    hs = np.concatenate([np.geomspace(1e-6, 1.0, 200), np.geomspace(1.0, 1e5, 800)])
    # Weibull k <= 1: nu = hazard
    k, th = 0.5, 10.0
    d = stats.weibull_min(k, scale=th)
    law = ScipyLaw(d)
    for a in (0.5, 5.0, 50.0):
        v, h = sup_ratio(law.sf, law.int_sf, a, hs)
        print(f"   Weibull k = 0.5: age {a}: sup F/tau = {v:.6f} at h = {h:.2e}; hazard (k/theta)(a/theta)^(k-1) = {k / th * (a / th) ** (k - 1):.6f}")
    al, sc = 2.0, 2.0
    law = ScipyLaw(stats.lomax(al, scale=sc))
    for a in (2.0, 8.67, 68.67):
        v, h = sup_ratio(law.sf, law.int_sf, a, hs)
        print(f"   Lomax alpha 2, scale 2: age {a}: sup = {v:.6f}; alpha/(scale + a) = {al / (sc + a):.6f}")
    k, th = 3.0, 20.0
    law = ScipyLaw(stats.weibull_min(k, scale=th))
    for a in (0.0, 10.0, 25.0):
        v, h = sup_ratio(law.sf, law.int_sf, a, hs)
        x = (a / th) ** k
        mrl = math.exp(x) * th / k * special.gamma(1 / k) * special.gammaincc(1 / k, x)
        print(f"   Weibull k = 3: age {a}: sup = {v:.6f} at h = {h:.1e}; 1/mean residual life = {1 / mrl:.6f}")
    # lognormal: peak and tangent point
    for (m, s) in ((math.log(5.0), 1.5), (math.log(40.0), 0.5), (math.log(1.69), 1.792)):
        g = lambda z: float(phi(z) / Phibar(z)) - z - s
        zs = optimize.brentq(g, -15, 15)
        tstar = math.exp(m + s * zs)
        d = stats.lognorm(s, scale=math.exp(m))
        law = ScipyLaw(d)
        eta = lambda t: float(d.pdf(t) / d.sf(t))
        Ifun = lambda t: float(d.mean()) - float(stoploss_vec(d, np.array([t]))[0])
        for a in (0.05 * tstar, 0.5 * tstar, 2.0 * tstar):
            v, h = sup_ratio(law.sf, law.int_sf, a, hs)
            if a >= tstar:
                closed = eta(a)
            else:
                f = lambda b: eta(b) * (Ifun(b) - Ifun(a)) - (float(d.sf(a)) - float(d.sf(b)))
                bstar = optimize.brentq(f, tstar * (1 + 1e-9), tstar * 1e6)
                closed = eta(bstar)
            print(f"   lognormal median {math.exp(m):.2f} sigma {s}: hazard peak at {tstar:.4f} s; age {a:.4f}: sup F/tau = {v:.6f}; closed form = {closed:.6f}")
    # R5: sharp hint with an end probability
    for p in (0.1, 0.3):
        lo, hi = 15.0, 25.0
        sf = lambda t: p + (1 - p) * np.clip((hi - np.asarray(t, dtype=float)) / (hi - lo), 0, 1)
        intsf = lambda a, b: float(np.trapezoid(sf(np.linspace(a, b, 20001)), np.linspace(a, b, 20001)))
        hh = np.linspace(0.01, 400.0, 40000)
        num = 1.0 - sf(hh)
        den = np.cumsum(np.concatenate([[0.0], 0.5 * (sf(hh[1:]) + sf(hh[:-1])) * np.diff(hh)])) + intsf(0, hh[0])
        r = num / den
        kk = int(np.argmax(r))
        form = (1 - p) / ((1 - p) * 20.0 + p * 25.0)
        print(f"   U[15, 25] with P(ended) = {p}: sup F/tau = {r[kk]:.6f} at h = {hh[kk]:.2f} s; formula P(return)/E[min(R, b)] = {form:.6f}; the draft's P(R < inf)/E[R] = 0")
    # rising hazard, falling density, with an end probability
    p = 0.05
    d = stats.weibull_min(1.5, scale=20.0)
    sf = lambda t: p + (1 - p) * d.sf(t)
    hh = np.linspace(0.01, 600.0, 120000)
    den = np.cumsum(0.5 * (sf(hh) + sf(np.concatenate([[0.0], hh[:-1]]))) * np.diff(np.concatenate([[0.0], hh])))
    r = (1.0 - sf(hh)) / den
    kk = int(np.argmax(r))
    print(f"   Weibull k = 1.5, theta 20, P(ended) = 0.05: sup F/tau = {r[kk]:.5f} at h = {hh[kk]:.1f} s; (1 - p)/E[R | returns] = {(1 - p) / d.mean():.5f} overstates it")


def r6_relaxed():
    print("\n== R6 (i): the relaxed many-session problem is solved by one threshold on the index (LP check) ==")
    rng = np.random.default_rng([SEED, 6])
    n = 60
    dt = 1.0
    worst_gap = 0.0
    worst_lam = 0.0
    mixed_classes = []
    for trial in range(40):
        ncls = int(rng.integers(2, 6))
        Ss, pinfs, Ks, Ls, ws, nus, Is = [], [], [], [], [], [], []
        for c in range(ncls):
            S, p_inf = random_lattice(rng, n, ["bimodal", "heavy", "sharp", "uniform", "weibull", "steps"][int(rng.integers(0, 6))])
            Ss.append(S)
            pinfs.append(p_inf)
            Ks.append(rng.uniform(0.5, 3.0))
            Ls.append(float(rng.choice([1.0, 2.0, 5.0])))
            ws.append(rng.uniform(0.5, 2.0))
            nuKL, I = index_bruteforce(S, dt)
            nus.append(nuKL * Ks[-1] / Ls[-1])
            Is.append(I)
        cost = np.concatenate([ws[c] * Ks[c] * (Ss[c] - pinfs[c]) for c in range(ncls)])
        mem = np.concatenate([ws[c] * Ls[c] * Is[c] for c in range(ncls)])
        full = sum(ws[c] * Ls[c] * Is[c][-1] for c in range(ncls))
        # unconstrained demand at lambda = 0: hold while any chance of return
        cap = rng.uniform(0.1, 0.7) * sum(ws[c] * Ls[c] * Is[c][int(np.argmax(Ss[c] - pinfs[c] <= 1e-12)) if np.any(Ss[c] - pinfs[c] <= 1e-12) else n] for c in range(ncls))
        A_eq = np.zeros((ncls, len(cost)))
        for c in range(ncls):
            A_eq[c, c * (n + 1) : (c + 1) * (n + 1)] = 1.0
        res = optimize.linprog(cost, A_ub=mem[None, :], b_ub=[cap], A_eq=A_eq, b_eq=np.ones(ncls), bounds=(0, None), method="highs")
        lam_lp = -res.ineqlin.marginals[0]
        # dual function at lam_lp and threshold policy cost
        def session_val(c, lam):
            C = lam * Ls[c] * Is[c] + Ks[c] * (Ss[c] - pinfs[c])
            return ws[c] * C.min()
        dual = sum(session_val(c, lam_lp) for c in range(ncls)) - lam_lp * cap
        worst_gap = max(worst_gap, abs(dual - res.fun) / max(res.fun, 1e-9))
        # the LP solution: every class drops at a first-passage age of its index at lam_lp (or mixes two such ages)
        x = res.x.reshape(ncls, n + 1)
        nmix = 0
        ok = True
        for c in range(ncls):
            sup = np.nonzero(x[c] > 1e-7)[0]
            if len(sup) > 1:
                nmix += 1
            C = lam_lp * Ls[c] * Is[c] + Ks[c] * (Ss[c] - pinfs[c])
            if np.any(C[sup] > C.min() + 1e-7 * max(1.0, abs(C.min()))):
                ok = False
        mixed_classes.append(nmix)
        if not ok:
            print("   !! LP support not on the minimisers of the priced session cost", trial)
        # the smallest-memory threshold rule (drop at the first age with nu <= lam) is feasible, the largest is not below cap
        mem_lo = sum(ws[c] * Ls[c] * Is[c][int(np.argmax(nus[c] <= lam_lp * (1 + 1e-9)))] for c in range(ncls))
        mem_hi = sum(ws[c] * Ls[c] * Is[c][int(np.argmax(nus[c] < lam_lp * (1 - 1e-9))) if np.any(nus[c] < lam_lp * (1 - 1e-9)) else n] for c in range(ncls))
        worst_lam = max(worst_lam, max(0.0, mem_lo - cap) / cap, max(0.0, cap - mem_hi) / cap)
    print(f"   40 random instances (2-5 session classes, lattice laws, unequal K, L and weights): LP optimum equals sum_i v_i(lambda*) - lambda* C to {worst_gap:.1e} (relative)")
    print(f"   classes that randomise between two drop ages in the LP solution: at most {max(mixed_classes)} per instance; memory of the two tie-breaking threshold rules brackets the capacity (worst miss {worst_lam:.1e})")
    # (ii) counterexamples under the hard capacity
    print("   hard capacity, one slot: A (K = 1, back in 10 s) against B (K = 1.05, back in 11 s): indices 0.1000 and 0.0955; the index evicts B (pay 1.05), evicting A pays 1.00")
    nuA = max(0.5 * (h >= 1) / (0.5 * min(h, 1.0) + 0.5 * h) for h in np.linspace(0.01, 50, 5000))
    print(f"   A returns at 1 s with probability 1/2, else never; B returns at 10 s; equal K, L: nu_A = {nuA:.3f} K/L, nu_B = 0.100 K/L; the index keeps A and pays K (B's restore); keeping B pays K/2")


def r6_offline():
    print("\n== R6 (ii)-(iii): exact hints under a hard capacity, brute force ==")
    rng = np.random.default_rng([SEED, 7])

    def solve(seq, size, cost, cap):
        T = len(seq)
        nxt_use = [[None] * (max(seq) + 1) for _ in range(T + 1)]
        last = {}
        for pos in range(T - 1, -1, -1):
            for it in range(max(seq) + 1):
                nxt_use[pos][it] = last.get(it)
            last[seq[pos]] = pos
        items = sorted(set(seq))

        @lru_cache(maxsize=None)
        def opt(pos, cache):
            if pos == T:
                return 0.0
            it = seq[pos]
            base = 0.0 if it in cache else cost[it]
            cur = set(cache) | {it}
            best = float("inf")
            lst = sorted(cur)
            for r in range(len(lst) + 1):
                for drop in itertools.combinations(lst, r):
                    keep = frozenset(cur - set(drop))
                    if sum(size[i] for i in keep) <= cap:
                        best = min(best, base + opt(pos + 1, keep))
            return best

        def index_policy():
            cache = set()
            tot = 0.0
            for pos in range(T):
                it = seq[pos]
                if it not in cache:
                    tot += cost[it]
                cache.add(it)
                while sum(size[i] for i in cache) > cap:
                    def idx(i):
                        nu_ = nxt_use[pos][i]
                        if nu_ is None:
                            return -1.0
                        return cost[i] / (size[i] * (nu_ - pos))
                    victim = min(cache, key=lambda i: (idx(i), i))
                    cache.remove(victim)
            return tot

        return opt(0, frozenset()), index_policy()

    for label, sizes_set, costs_set in (("equal sizes and costs", [1], [1.0]), ("costs in {1, 3}", [1], [1.0, 3.0]), ("sizes in {1, 2}, cost = size", [1, 2], None), ("sizes in {1, 2}, equal costs", [1, 2], [1.0])):
        sub = 0
        worst = 1.0
        tot_o = tot_i = 0.0
        for trial in range(200):
            m = 5
            size = [int(rng.choice(sizes_set)) for _ in range(m)]
            cost = [float(size[i]) if costs_set is None else float(rng.choice(costs_set)) for i in range(m)]
            seq = [int(v) for v in rng.integers(0, m, size=12)]
            cap = 2 if sizes_set == [1] else 3
            o, ip = solve(seq, size, cost, cap)
            tot_o += o
            tot_i += ip
            if ip > o + 1e-9:
                sub += 1
                worst = max(worst, ip / o)
        print(f"   {label}: index order suboptimal in {sub} of 200 sequences (5 items, 12 requests); worst ratio {worst:.2f}; total cost +{100 * (tot_i / tot_o - 1):.2f} %")


def r7():
    print("\n== R7: price-free index with the queue wait and the restore's own hold ==")
    rng = np.random.default_rng([SEED, 8])
    laws = [
        ("lomax", ScipyLaw(stats.lomax(2.5, scale=7.0))),
        ("gamma5", ScipyLaw(stats.gamma(5.0, scale=4.0))),
        ("weib.7+end", ScipyLaw(stats.weibull_min(0.7, scale=8.0), p_inf=0.2)),
        ("tool", TwoPiece(0.41, 1.69, 16.8, cap=1200.0)),
        ("uniform", ScipyLaw(stats.uniform(18.0, 4.0))),
    ]
    hs = np.concatenate([np.geomspace(1e-4, 1.0, 120), np.geomspace(1.0, 3000.0, 500)])
    bad = 0
    tot = 0
    n = 400_000
    worst_z = 0.0
    for trial in range(60):
        name, law = laws[trial % len(laws)]
        a = float(rng.choice([0.0, 1.0, 6.0, 19.0]))
        Sa = float(law.sf(a))
        if Sa < 1e-6:
            continue
        L = float(rng.choice([2000.0, 25000.0, 60000.0]))
        r = rng.uniform(0.5, 6.0)
        wmean = rng.uniform(0.0, 30.0)
        c = float(rng.choice([0.0, 0.3, 1.0]))
        kdev = rng.uniform(0.02, 2.0)
        rho = rng.uniform(0.0, min(3.0, 0.9 * wmean)) if wmean > 0 else 0.0
        # index from the definition, by a search over horizons
        F = np.array([(Sa - float(law.sf(a + h))) / Sa for h in hs])
        tau = np.array([quad_sf(law, a, a + h) / Sa for h in hs])
        ok = F > 1e-12
        if not ok.any():
            continue
        g = float(np.min(tau[ok] / F[ok]))
        # simulate: R given R > a, W exponential with mean wmean (independent)
        G = law.sample(rng, n * 3)
        G = G[G > a][:n] - a
        W = rng.exponential(wmean, size=len(G)) if wmean > 0 else np.zeros(len(G))
        K0 = kdev + c * float(np.mean(np.maximum(r - W, 0.0)))
        nu = (K0 / L) / max(g + wmean - rho, 1e-12)
        ret = np.isfinite(G)
        for fac in (0.6, 1.6):
            lam = nu * fac
            lamL = lam * L
            drop_cost = np.where(ret, kdev + c * np.maximum(r - W, 0.0) + lamL * rho, 0.0)
            best_keep = float("inf")
            best_h = None
            # choose the horizon on the formula, then simulate that policy
            vals = lamL * (tau + F * (wmean - rho)) - F * K0
            kbest = int(np.argmin(np.where(ok, vals, np.inf)))
            h = hs[kbest]
            keep_cost = np.where(G <= h, lamL * (G + W), lamL * h + drop_cost)
            diff = drop_cost - keep_cost  # > 0: keeping is better
            est, se = float(diff.mean()), float(diff.std() / math.sqrt(len(diff)))
            pred_keep_better = lam < nu
            tot += 1
            if abs(est) > 3 * se and (est > 0) != pred_keep_better:
                bad += 1
            # also compare the simulated difference with the formula's value
            formula = F[kbest] * K0 - lamL * (tau[kbest] + F[kbest] * (wmean - rho))
            if se > 1e-9 * max(1.0, abs(est)) and abs(est - formula) > 1e-9 and F[kbest] * len(G) >= 200:
                worst_z = max(worst_z, abs(est - formula) / se)
    print(f"   {tot} simulated (law, age, prices) cases at 0.6 and 1.6 times the index: sign of 'drop now minus keep to the best horizon' contradicts the index in {bad}; worst |z| of the simulated difference against F K0 - lamL (tau + F (w - rho)) (cells with at least 200 returns inside the horizon): {worst_z:.2f}")
    # numbers quoted
    L, rho, beta = 25_000.0, 1.0, 21_400.0
    out = []
    for W in (2.0, 5.0, 10.0, 20.0, 40.0):
        out.append(f"W = {W:.0f} s: {(L / beta) / (L * (W - rho)):.2e} x kappa")
    print("   queued turn (re-read fits, K0 = kappa L/beta, rho = 1 s): nu_q = " + "; ".join(out))
    tool = TwoPiece(0.41, 1.69, 16.8)
    hs2 = np.geomspace(1e-5, 1e5, 6000)
    gs = []
    for a in (0.0, 2.0, 10.0, 60.0, 300.0):
        Sa = float(tool.sf_raw(a))
        F = (Sa - tool.sf_raw(a + hs2)) / Sa
        tau = (float(tool.M_raw(a)) - tool.M_raw(a + hs2)) / Sa
        okk = F > 1e-12
        gs.append(float(np.min(tau[okk] / F[okk])))
    print("   tool law: g = inf tau/F at idle ages 0, 2, 10, 60, 300 s: " + ", ".join(f"{v:.1f}" for v in gs) + " s")
    k1 = (1.0 * L / beta) / (L * (40.0 - 1.0))
    k2 = (1.0 * L / beta) / (L * (gs[1] + 5.0 - 1.0))
    print(f"   kappa = 1: queued turn 40 s from admission nu = {k1:.2e}; staged read idle 2 s on the tool law with w = 5 s: nu = {k2:.2e} -> the staged read does not yield first")


def r8():
    print("\n== R8: the index when a dropped prefix can be read back early ==")
    # brute force over (keep horizon, read start) on a lattice; W = 0, read time r held for its whole duration
    dt = 0.25
    n = 480  # 120 s
    t = np.arange(n + 1) * dt
    L = 25_000.0
    r = 2.0
    kr = int(round(r / dt))

    def lattice(cdf_fn, p_inf=0.0):
        cdf = cdf_fn(t)
        cdf = cdf / cdf[-1]
        S = p_inf + (1 - p_inf) * (1 - cdf)
        S[0] = 1.0
        return S

    laws = {
        "tool entry (falling hazard past 0.8 s)": lattice(lambda u: 1 - TwoPiece(0.41, 1.69, 16.8).sf_raw(u)),
        "lomax(1.5, 4) (falling hazard)": lattice(lambda u: stats.lomax(1.5, scale=4.0).cdf(u)),
        "uniform [40, 50]": lattice(lambda u: np.clip((u - 40) / 10, 0, 1)),
        "lognormal median 40 s sigma 0.1": lattice(lambda u: stats.lognorm(0.1, scale=40.0).cdf(np.maximum(u, 1e-9))),
        "point 40 s": lattice(lambda u: (u >= 40.0).astype(float)),
    }

    def phi_values(S, lamL, c, kdev):
        """Phi[j]: optimal Fetch cost from idle age j dt (alive), copy on the SSD only, held to admission after landing."""
        p = S[:-1] - S[1:]  # p[i] = P(G = (i+1) dt)
        Kp = c * r + kdev + lamL * r  # a read made at arrival
        # cost of 'release at lattice time s (>= j) unless arrived': per unit of total mass
        # arrivals at u <= s: Kp ; arrivals u in (s, s + r): c (r - (u - s)) + kdev + lamL r ; u >= s + r: lamL (u - s - r) + kdev + lamL r
        u = t[1:]
        best = np.full(n + 1, np.inf)
        suffix_mass = np.concatenate([np.cumsum(p[::-1])[::-1], [0.0]])  # mass of returns at index >= i
        # arrival policy from age j: Kp * P(return | alive at j)
        ret_mass = S - S[-1]
        arrival_cost = Kp * ret_mass  # unnormalised (per unit prior mass), for alive at j
        # start s (index si): total cost over prior mass = Kp * mass(j < u <= s) + sum_{u > s} cost(u)
        late_cost = np.zeros(n + 1)
        for si in range(n + 1):
            uu = u[si:]
            pp = p[si:]
            stall = np.maximum(r - (uu - t[si]), 0.0)
            hold = np.maximum(uu - t[si] - r, 0.0)
            late_cost[si] = float(np.sum(pp * (c * stall + lamL * hold + kdev + lamL * r)))
            # a session that never returns keeps the read's pages forever: only finite if S[-1] = 0 (proper laws here)
        cum_p = np.concatenate([[0.0], np.cumsum(p)])  # mass of returns at u <= t[i]
        for j in range(n + 1):
            si = np.arange(j, n + 1)
            tot = Kp * (cum_p[si] - cum_p[j]) + late_cost[si]
            best[j] = min(float(tot.min()), float(arrival_cost[j]))
        return best, arrival_cost  # both unnormalised by S[j]

    def keep_value(S, lamL, c, kdev, refetch):
        """min over keep horizons h >= 0 of the cost from age 0; returns (cost of dropping now, best cost with keeping > 0)."""
        I = np.concatenate([[0.0], np.cumsum(S[:-1]) * dt])
        phi_un, arr_un = phi_values(S, lamL, c, kdev)
        term = phi_un if refetch else arr_un
        tot = lamL * I + term  # keep until index h, then the SSD-only continuation
        return float(tot[0]), float(tot[1:].min())

    def index(S, c, kdev, refetch):
        lo, hi = 1e-12, 1e-1
        for _ in range(60):
            mid = math.sqrt(lo * hi)
            d0, k0 = keep_value(S, mid * L, c, kdev, refetch)
            if k0 < d0 - 1e-12:
                lo = mid
            else:
                hi = mid
        return math.sqrt(lo * hi)

    for name, S in laws.items():
        for (c, kdev) in ((1.0, 0.0), (1.0, 0.5), (0.0, 0.5)):
            if c == 0.0 and kdev == 0.0:
                continue
            gi = index(S, c, kdev, True)
            si = index(S, c, kdev, False)
            extra = ""
            if name.startswith("point"):
                R0 = 40.0
                extra = f"; exact-hint formula (kdev/L)/(R0 - r) = {(kdev / L) / (R0 - r):.3e}" if kdev > 0 else "; exact-hint formula: 0"
            print(f"   {name}: c = {c}, device cost {kdev} s per read: index with early re-fetch {gi:.3e}, with K paid at arrival {si:.3e}, ratio {gi / si:.3f}{extra}")


if __name__ == "__main__":
    which = sys.argv[1:] or ["r1_r3", "r4_r5", "r6_relaxed", "r6_offline", "r7", "r8"]
    for w in which:
        globals()[w]()
