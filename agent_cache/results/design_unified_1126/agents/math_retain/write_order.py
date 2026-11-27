"""Order of writes on one device: checks of DESIGN.md 4.2 "write in the order in which pages
will be evicted (lowest clean index first), and no later than d_w" and result 7.3.

Scheduling models used (stated with each check):
 M1. One device, n page writes of equal size (one time slot each), release time r_j (the page
     exists), due date d_j (its host copy would be given up), cost f(C_j - d_j) with ONE convex
     non-decreasing f, f = 0 before the due date. Claim: earliest due date among the released
     pages (EDD) minimises the total cost, whether or not the due dates can be met.
 M2. Same, but each page has its own lateness cost shape: EDD is no longer optimal in general.
 M3. All pages overdue, sizes L_j, constant cost rates L_j*(lam - nu_j): Smith's ratio is
     beta*(lam - nu_j), so the order is lowest clean index first, whatever the sizes.
 M4. Index curves that cross: "lowest clean index now" is not "earliest due date".
 M5. Pages whose session may return (the write is then not needed in this gap): stochastic
     dynamic program on two and three pages, EDD against the optimum.

Run: python -I -B write_order.py
"""

from __future__ import annotations

import functools
import itertools
import math

import numpy as np

RNG = np.random.default_rng(3)


def list_schedule(order: tuple[int, ...], release: list[int]) -> list[int]:
    """Unit jobs, one per slot, non-idling: at each slot serve the first released job in `order`."""
    n = len(order)
    done = [0] * n
    left = list(order)
    t = 0
    while left:
        pick = next((j for j in left if release[j] <= t), None)
        if pick is None:
            t = min(release[j] for j in left)
            continue
        left.remove(pick)
        t += 1
        done[pick] = t
    return done


def edd_schedule(release: list[int], due: list[float]) -> list[int]:
    n = len(release)
    done = [0] * n
    left = set(range(n))
    t = 0
    while left:
        avail = [j for j in left if release[j] <= t]
        if not avail:
            t = min(release[j] for j in left)
            continue
        pick = min(avail, key=lambda j: (due[j], j))
        left.remove(pick)
        t += 1
        done[pick] = t
    return done


def m1_m2() -> None:
    def study(label: str, cost_fns) -> None:
        bad = 0
        worst = 0.0
        trials = 400
        for _ in range(trials):
            n = 6
            release = [int(x) for x in RNG.integers(0, 6, n)]
            due = [r + float(RNG.integers(1, 8)) for r in release]
            fns = cost_fns(n)
            best = math.inf
            for order in itertools.permutations(range(n)):
                done = list_schedule(order, release)
                best = min(best, sum(fns[j](done[j] - due[j]) for j in range(n)))
            done = edd_schedule(release, due)
            edd = sum(fns[j](done[j] - due[j]) for j in range(n))
            if edd > best + 1e-9:
                bad += 1
                worst = max(worst, edd / best if best > 0 else math.inf)
        print(f"  {label:66s} EDD suboptimal in {bad:3d}/{trials}; worst ratio {worst if bad else 1.0:.3f}")

    quad = lambda x: max(0.0, x) ** 2  # noqa: E731
    lin = lambda x: max(0.0, x)  # noqa: E731
    sat = lambda x: 1.0 - math.exp(-max(0.0, x) / 2.0)  # concave: the session has probably returned  # noqa: E731
    study("M1 one convex shape f(x) = (x^+)^2 for every page", lambda n: [quad] * n)
    study("M1 one linear shape f(x) = x^+ (total tardiness, equal sizes)", lambda n: [lin] * n)
    study("   one CONCAVE shape f(x) = 1 - exp(-x^+/2) (outside the claim)", lambda n: [sat] * n)
    study(
        "M2 page-specific slopes f_j(x) = w_j x^+, w_j in {1, 5}",
        lambda n: [(lambda x, w=float(w): w * max(0.0, x)) for w in RNG.choice([1.0, 5.0], n)],
    )


def m3_smith() -> None:
    beta = 21_400.0
    lam = 2e-5
    bad = 0
    trials = 300
    for _ in range(trials):
        n = 6
        tokens = RNG.choice([64.0, 512.0, 2048.0, 8192.0], n)
        nu = RNG.uniform(0.0, lam, n)  # all overdue: clean index below the price
        rate = tokens * (lam - nu)
        proc = tokens / beta
        best = math.inf
        best_order = None
        for order in itertools.permutations(range(n)):
            t = 0.0
            cost = 0.0
            for j in order:
                t += proc[j]
                cost += rate[j] * t
            if cost < best - 1e-18:
                best, best_order = cost, order
        by_index = tuple(int(j) for j in np.argsort(nu, kind="stable"))
        t = 0.0
        cost = 0.0
        for j in by_index:
            t += proc[j]
            cost += rate[j] * t
        if cost > best * (1 + 1e-12):
            bad += 1
    print(f"  M3 overdue pages, sizes 64..8192 tokens, cost rate L_j(lam - nu_j): lowest-index-first suboptimal in {bad}/{trials}")


def m4_crossing() -> None:
    # two pages, one second of device time each; the price is lam = 1 (index units)
    # A: index 5 now, falls to 1 at t = 1 s (due at 1 s).  B: index 2 now, falls to 1 at t = 10 s.
    print("  M4 A: index 5 now, due at 1 s;  B: index 2 now, due at 10 s; each write takes 1 s.")
    print("     lowest index first: B done at 1 s, A done at 2 s -> A is 1 s late.  EDD: A at 1 s, B at 2 s -> nothing late.")


def m5_abandonment() -> None:
    """Pages of unit size; page j is late from slot d_j on and then costs (lam - k*h(age)) per
    slot while its session is still out; the session returns in a slot with hazard h(age) (the
    write is then no longer needed in this gap). One write per slot. h falls with age."""

    def hazard(age: int) -> float:
        return 0.6 / (1.0 + age)

    lam, k = 1.0, 2.5

    def rate(age: int) -> float:
        return max(0.0, lam - k * hazard(age))

    bad = 0
    worst = 1.0
    trials = 300
    horizon = 14
    for _ in range(trials):
        n = 3
        ends = [int(x) for x in RNG.integers(0, 5, n)]  # the turn of page j ended at -ends[j]: age = t + ends[j]

        @functools.lru_cache(maxsize=None)
        def opt(t: int, alive: tuple[bool, ...]) -> float:
            if t == horizon or not any(alive):
                return 0.0
            best = math.inf
            for serve in [j for j in range(n) if alive[j]]:
                best = min(best, step(t, alive, serve, opt))
            return best

        @functools.lru_cache(maxsize=None)
        def edd(t: int, alive: tuple[bool, ...]) -> float:
            if t == horizon or not any(alive):
                return 0.0
            serve = max((j for j in range(n) if alive[j]), key=lambda j: (ends[j], -j))  # oldest = earliest due
            return step(t, alive, serve, edd)

        def step(t, alive, serve, cont) -> float:
            # the served page is written during this slot; the others pay their rate, then may return
            cost = sum(rate(t + ends[j]) for j in range(n) if alive[j] and j != serve)
            others = [j for j in range(n) if alive[j] and j != serve]
            total = cost
            for stays in itertools.product([True, False], repeat=len(others)):
                prob = 1.0
                nxt = [False] * n
                for j, stay in zip(others, stays):
                    h = hazard(t + ends[j])
                    prob *= (1 - h) if stay else h
                    nxt[j] = stay
                total += prob * cont(t + 1, tuple(nxt))
            return total

        start = tuple([True] * n)
        a, b = opt(0, start), edd(0, start)
        if b > a * (1 + 1e-12) + 1e-15:
            bad += 1
            worst = max(worst, b / a)
    print(f"  M5 three unit pages, falling hazard, returns cancel the write: oldest-first (EDD) suboptimal in {bad}/{trials}; worst ratio {worst:.4f}")


def main() -> None:
    print("=" * 100)
    print("Write order on one device")
    m1_m2()
    m3_smith()
    m4_crossing()
    m5_abandonment()


if __name__ == "__main__":
    main()
