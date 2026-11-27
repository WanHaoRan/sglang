"""verify_math, part 5: math_retain R9-R16 (the write, its due date and order, the tiers above and below).

Run: CUDA_VISIBLE_DEVICES= nice -n 10 <venv python> -I -B vr_write.py > vr_write.out
"""
from __future__ import annotations

import itertools
import math
import sys

import numpy as np
from scipy import optimize, stats

sys.path.insert(0, ".")
from vlib import TwoPiece  # noqa: E402
from vr_retain import random_lattice  # noqa: E402

SEED = 99_1126


def dp_single(S, p_inf, dt, lamL, Kc, Kd, w):
    n = len(S) - 1
    pi = np.where(S > 0, (S - p_inf) / np.maximum(S, 1e-300), 0.0)
    Vc = np.zeros(n + 1)
    Vd = np.zeros(n + 1)
    act_c = np.zeros(n + 1, dtype=int)  # 0 drop, 1 hold
    act_d = np.zeros(n + 1, dtype=int)  # 0 drop, 1 hold, 2 write
    strict_write_hold = False
    Vc[n] = Kc * pi[n]
    Vd[n] = min(Kd * pi[n], w + Vc[n])
    act_d[n] = 0 if Kd * pi[n] <= w + Vc[n] else 2
    for j in range(n - 1, -1, -1):
        if S[j] <= 0:
            continue
        surv = S[j + 1] / S[j]
        hold_c = lamL * dt + surv * Vc[j + 1]
        drop_c = Kc * pi[j]
        Vc[j] = min(drop_c, hold_c)
        act_c[j] = 0 if drop_c <= hold_c else 1
        hold_d = lamL * dt + surv * Vd[j + 1]
        drop_d = Kd * pi[j]
        wr = w + Vc[j]
        Vd[j] = min(drop_d, wr, hold_d)
        act_d[j] = int(np.argmin([drop_d, hold_d, wr]))
        tol = 1e-11 * max(1.0, Vd[j])
        if wr < min(drop_d, hold_d) - tol and hold_c < drop_c - tol:
            strict_write_hold = True
    return Vc, Vd, act_c, act_d, strict_write_hold


def r9_r10():
    print("== R9-R10: the value of a write; the two lives of a dirty page (single gap, lattice laws) ==")
    rng = np.random.default_rng([SEED, 1])
    n, dt = 300, 0.5
    kinds = ["bimodal", "heavy", "sharp", "uniform", "weibull", "steps"]
    mism = 0
    wrong_life = 0
    strict = 0
    crit_bad = 0
    nec_bad = suf_bad = 0
    id_worst = 0.0
    save_worst = 0.0
    cases = 0
    lifeW = 0
    for trial in range(400):
        S, p_inf = random_lattice(rng, n, kinds[trial % len(kinds)])
        I = np.concatenate([[0.0], np.cumsum(S[:-1]) * dt])
        L = 25_000.0
        lamL = float(np.exp(rng.uniform(math.log(1e-4), math.log(1.0))))
        Kc = rng.uniform(0.05, 2.0)
        Kd = Kc + rng.uniform(0.0, 20.0)
        w = rng.uniform(0.0, 3.0)
        Vc, Vd, act_c, act_d, sw = dp_single(S, p_inf, dt, lamL, Kc, Kd, w)
        CW = lamL * I + (Kc + w) * (S - p_inf) + w * p_inf
        CN = lamL * I + Kd * (S - p_inf)
        two = min(CW.min(), CN.min())
        cases += 1
        if abs(two - Vd[0]) > 1e-9 * max(1.0, two):
            mism += 1
        strict += int(sw)
        V0 = lambda K: float((lamL * I + K * (S - p_inf)).min())
        W_by_crit = V0(Kd) - V0(Kc + w) >= w * p_inf - 1e-12
        W_by_dp = CW.min() <= CN.min() + 1e-12
        if abs(CW.min() - CN.min()) > 1e-9 and W_by_crit != W_by_dp:
            crit_bad += 1
        lifeW += int(W_by_dp)
        # thresholds on pi at the two drop ages
        if Kd > Kc:
            theta = w / (Kd - Kc)
            a_w = int((lamL * I + (Kc + w) * (S - p_inf)).argmin())
            a_d = int(CN.argmin())
            pi = lambda j: (S[j] - p_inf) / S[j] if S[j] > 0 else 0.0
            if CW.min() < CN.min() - 1e-9 and pi(a_w) < theta - 1e-9:
                nec_bad += 1
            if pi(a_d) >= theta + 1e-9 and CW.min() > CN.min() + 1e-9:
                suf_bad += 1
        # R9 (i): V0(Kd) - V0(Kc) = integral of P_pay over K (V0 is concave piecewise linear in K)
        Ks = np.linspace(Kc, Kd, 4001)
        costs = lamL * I[None, :] + Ks[:, None] * (S - p_inf)[None, :]
        smin = costs.argmin(axis=1)
        ppay = (S - p_inf)[smin]
        integral = float(np.trapezoid(ppay, Ks))
        diff = V0(Kd) - V0(Kc)
        id_worst = max(id_worst, abs(integral - diff) / max(Kd - Kc, 1e-9))
        if np.any(np.diff(ppay) > 1e-12):
            print("   !! P_pay not non-increasing in K")
        # R9 (ii): 'hold h, then write if not back' saves exactly w F(h) against 'write now', h up to the clean drop age
        clean_drop = int(np.argmax(act_c == 0))
        for h in (clean_drop // 3, clean_drop):
            if h <= 0 or S[h] <= 0:
                continue
            now = w + Vc[0]
            later = lamL * I[h] + S[h] * (w + Vc[h])
            save_worst = max(save_worst, abs((now - later) - w * (1 - S[h])))
    print(f"   {cases} random cases (law, lamL, K_c, K_d, w): three-action backward induction against min(min C_W, min C_N): {mism} mismatches; 'write, then keep holding' strictly optimal at some age: {strict} cases")
    print(f"   life W chosen in {lifeW} cases; criterion V0(K_d) - V0(K_c + w) >= w p_inf disagrees with the direct comparison in {crit_bad}; necessary threshold violated {nec_bad}, sufficient threshold violated {suf_bad}")
    print(f"   identity V0(K_d) - V0(K_c) = integral of P_pay dK: worst error {id_worst:.1e} per unit of K_d - K_c (4,001-point trapezoid on a step function, so about 1/4,000 is expected); saving of the later write = w F(h): worst error {save_worst:.1e}")


def multi_gap():
    print("\n== R10 probe: several gaps (the page stays dirty across returns until written) ==")
    rng = np.random.default_rng([SEED, 2])
    n, dt = 300, 0.5
    kinds = ["bimodal", "heavy", "sharp", "uniform", "weibull", "steps"]
    strict = 0
    disagree = 0
    cases = 0
    examples = []
    dirs = {}
    for trial in range(300):
        S0, _ = random_lattice(rng, n, kinds[trial % len(kinds)])
        e = float(rng.choice([0.02, 0.1, 0.3]))
        # proper part of the law rescaled: P(G > t) = e + (1 - e) S_proper(t)
        Sp = (S0 - S0[-1]) / (1 - S0[-1])
        S = e + (1 - e) * Sp
        S[0] = 1.0
        S[-1] = e
        lamL = float(np.exp(rng.uniform(math.log(1e-4), math.log(1.0))))
        Kc = rng.uniform(0.05, 2.0)
        Kd = Kc + rng.uniform(0.0, 20.0)
        w = rng.uniform(0.0, 3.0)
        pi = (S - e) / S
        surv = S[1:] / S[:-1]
        qret = (S[:-1] - S[1:]) / S[:-1]
        C0 = D0 = 0.0
        for it in range(2000):
            Vc = np.zeros(n + 1)
            Vd = np.zeros(n + 1)
            Vc[n] = 0.0
            Vd[n] = 0.0
            sw = False
            first_act = None
            for j in range(n - 1, -1, -1):
                hold_c = lamL * dt + qret[j] * C0 + surv[j] * Vc[j + 1]
                drop_c = pi[j] * (Kc + C0)
                Vc[j] = min(drop_c, hold_c)
                hold_d = lamL * dt + qret[j] * D0 + surv[j] * Vd[j + 1]
                drop_d = pi[j] * (Kd + D0)
                wr = w + Vc[j]
                Vd[j] = min(drop_d, wr, hold_d)
                tol = 1e-10 * max(1.0, Vd[j])
                if wr < min(drop_d, hold_d) - tol and hold_c < drop_c - tol:
                    sw = True
            newC, newD = Vc[0], Vd[0]
            if abs(newC - C0) + abs(newD - D0) < 1e-12:
                break
            C0, D0 = newC, newD
        cases += 1
        strict += int(sw)
        # does the multi-gap optimum ever write (anywhere along the dirty path from age 0)?
        j = 0
        wrote = False
        while j < n:
            hold_d = lamL * dt + qret[j] * D0 + surv[j] * Vd[j + 1]
            drop_d = pi[j] * (Kd + D0)
            wr = w + Vc[j]
            m = min(drop_d, wr, hold_d)
            if wr <= m + 1e-12 and wr < drop_d - 1e-12:
                wrote = True
                break
            if drop_d <= m + 1e-12:
                break
            j += 1
        # single-gap criterion of R10 on the same gap law
        I = np.concatenate([[0.0], np.cumsum(S[:-1]) * dt])
        V0 = lambda K: float((lamL * I + K * (S - e)).min())
        single_W = V0(Kd) - V0(Kc + w) >= w * e
        margin = abs((V0(Kd) - V0(Kc + w)) - w * e)
        if margin > 1e-6 and single_W != wrote:
            disagree += 1
            dirs[(bool(single_W), bool(wrote))] = dirs.get((bool(single_W), bool(wrote)), 0) + 1
            if len(examples) < 3:
                examples.append((kinds[trial % len(kinds)], e, round(lamL, 4), round(Kc, 2), round(Kd, 2), round(w, 2), single_W, wrote))
    print(f"   {cases} random cases with a chance e in {{0.02, 0.1, 0.3}} that each turn is the last: 'write, then keep holding' strictly optimal: {strict}")
    print(f"   single-gap criterion of R10 against the several-gap optimum: different life in {disagree} cases ({dirs.get((False, True), 0)} where only the several-gap optimum writes, {dirs.get((True, False), 0)} the other way); examples (law, e, lamL, K_c, K_d, w, single-gap says write, several-gap optimum writes): {examples}")


def r11():
    print("\n== R11: due date of a write and the cost of missing it (tool law, K_c/L = 7e-5) ==")
    law = TwoPiece(0.41, 1.69, 16.8)
    KcL = 7e-5
    for lam in (5e-6, 2e-5):
        target = lam / KcL
        t = np.geomspace(0.8, 2000.0, 400_001)
        h = law.hazard_raw(t)
        k = int(np.argmax(h <= target))
        d = t[k]
        out = []
        for x in (1.0, 5.0, 20.0, 60.0, 300.0):
            y = np.linspace(0.0, x, 20_001)
            surv = law.sf_raw(d + y) / float(law.sf_raw(d))
            rate = lam - KcL * law.hazard_raw(d + y)
            cost = float(np.trapezoid(surv * rate, y))
            out.append(f"{x:.0f} s late: cost {cost / (lam * x):.3f}, rate {float(1 - KcL * law.hazard_raw(d + x) / lam):.2f}")
        print(f"   lambda = {lam:g}: due at {d:.2f} s of idle (hazard there {float(law.hazard_raw(d)):.4f} against lambda L / K_c = {target:.4f}); in units of lambda x lateness and of lambda: " + "; ".join(out))


def r12():
    print("\n== R12: order of writes on one device ==")
    rng = np.random.default_rng([SEED, 3])

    def edd_schedule(rel, due):
        m = len(rel)
        done = [False] * m
        comp = [0.0] * m
        t = 0
        left = m
        while left:
            avail = [j for j in range(m) if not done[j] and rel[j] <= t]
            if not avail:
                t += 1
                continue
            j = min(avail, key=lambda q: (due[q], q))
            comp[j] = t + 1
            done[j] = True
            left -= 1
            t += 1
        return comp

    def best_by_assignment(rel, due, f):
        m = len(rel)
        horizon = max(rel) + m + 1
        cost = np.full((m, horizon), 1e9)
        for j in range(m):
            for slot in range(rel[j], horizon):
                cost[j, slot] = f(j, slot + 1 - due[j])
        rows, cols = optimize.linear_sum_assignment(cost)
        return float(cost[rows, cols].sum())

    shapes = {
        "one convex shape (x^+)^2": lambda j, x: max(x, 0.0) ** 2,
        "one linear shape x^+": lambda j, x: max(x, 0.0),
        "one concave shape 1 - exp(-x^+/2)": lambda j, x: 1 - math.exp(-max(x, 0.0) / 2),
    }
    for name, f in shapes.items():
        bad = 0
        worst = 1.0
        for trial in range(500):
            m = 7
            rel = [int(v) for v in rng.integers(0, 6, size=m)]
            due = [float(rel[j] + rng.uniform(0.5, 6.0)) for j in range(m)]
            comp = edd_schedule(rel, due)
            c_edd = sum(f(j, comp[j] - due[j]) for j in range(m))
            c_opt = best_by_assignment(rel, due, f)
            if c_edd > c_opt + 1e-9:
                bad += 1
                worst = max(worst, c_edd / max(c_opt, 1e-12))
        print(f"   unit pages with release times, {name}: earliest due date not optimal in {bad} of 500 instances (exact optimum by assignment of pages to slots); worst ratio {worst:.2f}")
    # page-specific slopes
    bad = 0
    worst = 1.0
    for trial in range(500):
        m = 7
        rel = [int(v) for v in rng.integers(0, 6, size=m)]
        due = [float(rel[j] + rng.uniform(0.5, 6.0)) for j in range(m)]
        wj = [float(rng.choice([1.0, 5.0])) for _ in range(m)]
        f = lambda j, x: wj[j] * max(x, 0.0)
        comp = edd_schedule(rel, due)
        c_edd = sum(f(j, comp[j] - due[j]) for j in range(m))
        c_opt = best_by_assignment(rel, due, f)
        if c_edd > c_opt + 1e-9:
            bad += 1
            worst = max(worst, c_edd / max(c_opt, 1e-12))
    print(f"   page-specific slopes w_j in {{1, 5}}: earliest due date not optimal in {bad} of 500; worst ratio {worst:.2f}")
    # all overdue, sizes differ, constant cost rates L_j (lambda - nu_j): lowest clean index first
    bad = 0
    for trial in range(300):
        m = 6
        Lj = rng.choice([64, 256, 1024, 4096, 8192], size=m).astype(float)
        nu = rng.uniform(0.0, 1.0, size=m)  # in units of lambda
        beta = 21_400.0
        best = float("inf")
        for perm in itertools.permutations(range(m)):
            t = 0.0
            tot = 0.0
            for j in perm:
                t += Lj[j] / beta
                tot += Lj[j] * (1 - nu[j]) * t
            best = min(best, tot)
        order = np.argsort(nu, kind="stable")
        t = 0.0
        tot = 0.0
        for j in order:
            t += Lj[j] / beta
            tot += Lj[j] * (1 - nu[j]) * t
        if tot > best * (1 + 1e-12):
            bad += 1
    print(f"   all pages overdue, sizes 64 to 8,192 tokens, cost rate L_j (lambda - nu_j): 'lowest clean index first' not optimal in {bad} of 300 instances (all 720 orders)")
    print("   crossing index curves: A (index 5 lambda now, due in 1 s) and B (index 2 lambda, due in 10 s), 1 s per write: lowest index first finishes B at 1 s and A at 2 s (A 1 s late); due-date order finishes A at 1 s, B at 2 s (nothing late)")


def r13():
    print("\n== R13: share of pages whose write is ever needed, q = (1 - e) x / (x + e - x e) ==")
    rng = np.random.default_rng([SEED, 4])
    for x, e in ((0.046, 1 / 29.2), (0.067, 1 / 29.2), (0.103, 1 / 29.2), (0.3, 0.2)):
        npages = 400_000
        alive = np.ones(npages, dtype=bool)
        due = np.zeros(npages, dtype=bool)
        while alive.any():
            idx = np.nonzero(alive)[0]
            ended = rng.random(len(idx)) < e
            alive[idx[ended]] = False
            idx = idx[~ended]
            longgap = rng.random(len(idx)) < x
            due[idx[longgap]] = True
            alive[idx[longgap]] = False
        q = (1 - e) * x / (x + e - x * e)
        print(f"   x = {x}, e = {e:.4f}: simulated {due.mean():.4f}, formula {q:.4f}")
    mu, gamma, beta = 93, 0.043e-3, 21_400.0
    print(f"   mu gamma beta at mu = 93: {mu * gamma * beta:.1f}; eager wins once eps x pi >= kappa (1 - q)/(mu gamma beta - kappa): {1 * 0.35 / (mu * gamma * beta - 1):.4f} (kappa = 1), {10 * 0.35 / (mu * gamma * beta - 10):.4f} (kappa = 10)")
    # (v) two-state device price: idle (write free) / busy (write costs w)
    n, dt = 300, 0.5
    bad_idle = 0
    not_threshold = 0
    for trial in range(60):
        S, p_inf = random_lattice(rng, n, ["heavy", "weibull", "bimodal"][trial % 3])
        lamL = float(np.exp(rng.uniform(math.log(1e-3), math.log(0.5))))
        Kc = rng.uniform(0.05, 1.0)
        Kd = Kc + rng.uniform(1.0, 20.0)
        w = rng.uniform(0.1, 2.0)
        a, b = rng.uniform(0.02, 0.3), rng.uniform(0.02, 0.3)  # idle -> busy, busy -> idle per step
        pi = np.where(S > 0, (S - p_inf) / np.maximum(S, 1e-300), 0.0)
        Vc = np.zeros(n + 1)
        Vd = np.zeros((n + 1, 2))
        act_busy = np.zeros(n + 1, dtype=int)
        Vc[n] = Kc * pi[n]
        Vd[n, 0] = min(Kd * pi[n], Vc[n])
        Vd[n, 1] = min(Kd * pi[n], w + Vc[n])
        for j in range(n - 1, -1, -1):
            if S[j] <= 0:
                continue
            surv = S[j + 1] / S[j]
            Vc[j] = min(Kc * pi[j], lamL * dt + surv * Vc[j + 1])
            for dev in (0, 1):
                nxt = (1 - a) * Vd[j + 1, 0] + a * Vd[j + 1, 1] if dev == 0 else b * Vd[j + 1, 0] + (1 - b) * Vd[j + 1, 1]
                opts = [Kd * pi[j], lamL * dt + surv * nxt, (0.0 if dev == 0 else w) + Vc[j]]
                Vd[j, dev] = min(opts)
                if dev == 1:
                    act_busy[j] = int(np.argmin(opts))
            if abs(Vd[j, 0] - Vc[j]) > 1e-10 * max(1.0, Vc[j]):
                bad_idle += 1
    print(f"   two-state device price (idle: write free; busy: write costs w), 60 random cases: ages at which writing at once is not optimal while the device is idle: {bad_idle}")


def r15():
    print("\n== R15: the mirror (arithmetic on the brief's constants) ==")
    L = 24_900
    copy_s = L / (2304 / 6.5e-3)
    mu = 93
    print(f"   one GPU-to-host copy of {L:,} tokens at 2,304 tokens / 6.5 ms: {1000 * copy_s:.0f} ms; if it blocks the scheduler at mu = {mu}: {mu * copy_s:.1f} TTFT-seconds")
    print(f"   that exceeds the mirror's rent over an 8 s decode (lambda x L x 8) for lambda below {mu * copy_s / (L * 8):.1e}")
    print(f"   mirror 0.65 M of 1.63 M host tokens: host-only room 0.98 M -> factor {1.63 / 0.98:.2f} (with the brief's 1,627,648 - 650,000: {1627648 / (1627648 - 650000):.2f})")
    for lam in (5e-7, 2e-6):
        print(f"   mirror rent at lambda = {lam:g}: {lam * 650_000 / 2.2:.2f} s per request at 2.2 requests/s")


def r16():
    print("\n== R16: SSD capacity with chained pages ==")
    rng = np.random.default_rng([SEED, 5])
    # (i) one chain, root lookups: among all ways to delete m of n pages the tail is the unique optimum
    viol = 0
    for n in range(2, 12):
        for m in range(1, n):
            best = None
            arg = []
            for D in itertools.combinations(range(1, n + 1), m):
                got = min(D) - 1
                if best is None or got > best:
                    best, arg = got, [D]
                elif got == best:
                    arg.append(D)
            if arg != [tuple(range(n - m + 1, n + 1))] or n - best != m:
                viol += 1
    print(f"   chains of 2-11 pages, every deletion count: 'the tail is the unique optimum and loses exactly m pages' violated {viol} times")
    # exchange with anchors: deleted i kept j > i; swapping is never worse for lookups anchored at or below i or above j
    bad = 0
    checks = 0
    bad_inside = 0
    for trial in range(4000):
        n = int(rng.integers(3, 10))
        stored = rng.random(n + 1) < 0.7
        i, j = sorted(rng.choice(np.arange(1, n + 1), size=2, replace=False))
        stored[i] = False
        stored[j] = True
        swapped = stored.copy()
        swapped[i] = True
        swapped[j] = False

        def got(st, p):
            k = p
            while k <= n and st[k]:
                k += 1
            return k - p

        for p in range(1, n + 1):
            inside = i < p <= j
            if inside:
                if got(swapped, p) < got(stored, p):
                    bad_inside += 1
                continue
            checks += 1
            if got(swapped, p) < got(stored, p):
                bad += 1
    print(f"   exchange (delete the later page instead of the earlier): worse on a lookup anchored outside (i, j] in {bad} of {checks} checks; worse on {bad_inside} lookups anchored inside (i, j] (the stated exception)")
    n = 8
    print("   chain of 8, one page deleted: loss per lookup by deleted page; root lookups: " + " ".join(str(n - j + 1) for j in range(1, n + 1)) + "; anchor uniform on 1..8: " + " ".join(f"{(j / n) * (n - j + 1):.2f}" for j in range(1, n + 1)))
    # (iii) trees with root-anchored lookups: the threshold set {v : theta rho(v) > sigma} maximises value - sigma size
    viol = 0
    nest = 0
    for trial in range(150):
        nn = int(rng.integers(4, 10))
        parent = [-1] + [int(rng.integers(0, v)) for v in range(1, nn)]
        length = rng.integers(1, 6, size=nn).astype(float)
        leaves = [v for v in range(nn) if v not in parent]
        rate = {v: float(rng.uniform(0.1, 2.0)) for v in leaves}
        if rng.random() < 0.3:
            rate[leaves[0]] = 0.0  # an ended session
        chains = {}
        for lf in leaves:
            ch = []
            v = lf
            while v != -1:
                ch.append(v)
                v = parent[v]
            chains[lf] = ch[::-1]
        rho = np.zeros(nn)
        for lf, ch in chains.items():
            for v in ch:
                rho[v] += rate[lf]
        theta = 1.0

        def value(mask):
            tot = 0.0
            for lf, ch in chains.items():
                for v in ch:
                    if not (mask >> v) & 1:
                        break
                    tot += theta * rate[lf] * length[v]
            return tot

        vals = [value(mask) for mask in range(1 << nn)]
        sizes = [sum(length[v] for v in range(nn) if (mask >> v) & 1) for mask in range(1 << nn)]
        prev_set = None
        for sigma in sorted(set(np.concatenate([rho * 0.999, rho * 1.001, [0.0, 10.0]])), reverse=True):
            best = max(vals[m] - sigma * sizes[m] for m in range(1 << nn))
            thr = sum(1 << v for v in range(nn) if theta * rho[v] > sigma)
            if vals[thr] - sigma * sizes[thr] < best - 1e-9:
                viol += 1
            if prev_set is not None and (prev_set & ~thr):
                nest += 1
            prev_set = thr
        for v in range(1, nn):
            if rho[v] > rho[parent[v]] + 1e-12:
                viol += 1
    print(f"   150 random trees of 4-9 page groups, all subsets: threshold set not optimal for its price, or a child with a higher rate than its parent: {viol}; sets not nested in the price: {nest}")
    # (iv) shared head, sometimes resident
    worst = 0.0
    for trial in range(300):
        ns = int(rng.integers(1, 6))
        len_h = float(rng.integers(1, 10))
        n_s = rng.integers(0, 30, size=ns).astype(float)
        cold = rng.uniform(0.0, 1.0, size=ns)
        rate = rng.uniform(0.1, 2.0, size=ns)
        theta = float(rng.uniform(0.5, 2.0))
        with_h = sum(rate[s] * (cold[s] * (len_h + n_s[s]) + (1 - cold[s]) * n_s[s]) for s in range(ns)) * theta
        without = sum(rate[s] * (cold[s] * 0.0 + (1 - cold[s]) * n_s[s]) for s in range(ns)) * theta
        formula = theta * sum(rate[s] * cold[s] * (len_h + n_s[s]) for s in range(ns))
        worst = max(worst, abs((with_h - without) - formula))
    print(f"   value of a shared head's SSD copy = theta x sum of cold-lookup rates x (head + stored chain below): worst error {worst:.1e} on 300 random cases")
    ex = []
    for coldp in (1.0, 0.3, 0.05, 0.0):
        head = 5 * coldp * (6 + 20) / 6.0
        tail = 1.0  # a tail page of one session, per token, with the head stored: rate 1 x 1
        ex.append(f"{head / tail:.1f}" if coldp > 0 else "0")
    print("   5 sessions with 20 K chains below a 6 K head: head index / tail index at cold shares 100, 30, 5, 0 %: " + ", ".join(ex))


if __name__ == "__main__":
    which = sys.argv[1:] or ["r9_r10", "multi_gap", "r11", "r12", "r13", "r15", "r16"]
    for w in which:
        globals()[w]()
