#!/usr/bin/env python3
"""Preempt on host room (DESIGN.md 5.1, result 5): the covering knapsack and the greedy order.

Problem:  minimise sum_j C_j x_j  subject to  sum_j L_j x_j >= m,  x_j in {0, 1}
(C_j = rise in holder j's priced cost when it is displaced, L_j = its tokens, m = tokens the read needs).

Checks:
  K1  Divisible problem (0 <= x_j <= 1): taking holders in increasing C_j / L_j, the last one in part, is
      optimal (against scipy's LP solver).
  K2  0-1 problem: greedy (whole holders in ratio order until m is covered) costs at most the optimum plus
      the cost of the last holder taken; it has no bounded ratio (worst case shown).
  K3  A two-line change gives a factor 2: when a holder would complete the cover, note "holders taken so
      far + this one" as a candidate and skip it; return the cheapest candidate.
  K4  Engine-sized instances (holders of about 25 K tokens in 64-token pages, m = 25 K): how far greedy is
      from the optimum, and how much room it frees beyond m.
Output: knapsack.out
"""
import itertools
import os
import random
import sys

import numpy as np
from scipy.optimize import linprog

OUT = []


def P(s=""):
    OUT.append(s)
    print(s, flush=True)


def lp_greedy(C, L, m):
    order = sorted(range(len(C)), key=lambda j: C[j] / L[j])
    need = m
    val = 0.0
    k = None
    for j in order:
        if need <= 0:
            break
        take = min(1.0, need / L[j])
        val += take * C[j]
        need -= take * L[j]
        k = j
    return val, k


def greedy01(C, L, m):
    order = sorted(range(len(C)), key=lambda j: C[j] / L[j])
    got = 0.0
    val = 0.0
    last = None
    for j in order:
        if got >= m:
            break
        got += L[j]
        val += C[j]
        last = j
    return val, got, last


def greedy_skip(C, L, m):
    """2-approximation: ratio order; an item that would complete the cover is a candidate and is skipped."""
    order = sorted(range(len(C)), key=lambda j: C[j] / L[j])
    got = 0.0
    val = 0.0
    best = float("inf")
    for j in order:
        if got + L[j] >= m:
            best = min(best, val + C[j])
        else:
            got += L[j]
            val += C[j]
    return best


def opt01(C, L, m):
    n = len(C)
    best = float("inf")
    for mask in range(1 << n):
        size = 0.0
        cost = 0.0
        for j in range(n):
            if mask >> j & 1:
                size += L[j]
                cost += C[j]
        if size >= m and cost < best:
            best = cost
    return best


def opt01_dp(C, Lp, mp):
    """Exact covering knapsack by dynamic programming over pages (integer sizes Lp, need mp)."""
    INF = float("inf")
    best = [INF] * (mp + 1)
    best[0] = 0.0
    for c, l in zip(C, Lp):
        for need in range(mp, 0, -1):
            prev = best[max(need - l, 0)] + c
            if prev < best[need]:
                best[need] = prev
    return best[mp]


def main():
    rng = random.Random(51)
    P("Preempt on host room: the covering knapsack")
    P("")
    worst = 0.0
    for _ in range(300):
        n = rng.randint(2, 12)
        L = [rng.uniform(1.0, 10.0) for _ in range(n)]
        C = [rng.uniform(0.0, 5.0) * l for l in L]
        m = rng.uniform(0.1, 0.9) * sum(L)
        val, _ = lp_greedy(C, L, m)
        res = linprog(C, A_ub=[[-l for l in L]], b_ub=[-m], bounds=[(0, 1)] * n, method="highs")
        worst = max(worst, abs(val - res.fun))
    P("K1. Divisible problem: ratio greedy against the LP optimum, 300 instances: max |difference| = %.1e." % worst)
    viol = 0
    max_ratio = 1.0
    max_ratio_skip = 1.0
    add_worst = 0.0
    for _ in range(3000):
        n = rng.randint(2, 12)
        L = [rng.uniform(1.0, 10.0) for _ in range(n)]
        C = [rng.uniform(0.01, 5.0) * l ** rng.choice([0.5, 1.0, 1.5]) for l in L]
        m = rng.uniform(0.05, 0.9) * sum(L)
        g, got, last = greedy01(C, L, m)
        o = opt01(C, L, m)
        lp, _ = lp_greedy(C, L, m)
        if g > o + C[last] + 1e-9 or g > lp + C[last] + 1e-9:
            viol += 1
        add_worst = max(add_worst, (g - o) / C[last])
        max_ratio = max(max_ratio, g / o)
        max_ratio_skip = max(max_ratio_skip, greedy_skip(C, L, m) / o)
    P("K2. 0-1 problem, 3,000 random instances (2-12 holders): greedy <= optimum + cost of the last holder "
      "taken: violations %d; largest (greedy - optimum) / (last holder's cost) = %.3f; largest greedy / optimum = %.2f."
      % (viol, add_worst, max_ratio))
    P("    Worst case for the ratio: need m = 1; holder A has 1 token and cost 1; holder B has M tokens and cost 0.9 M.")
    for M in (10, 100, 1000):
        C, L = [1.0, 0.9 * M], [1.0, float(M)]
        g, got, _ = greedy01(C, L, 1.0)
        P("      M = %5d: greedy takes B first (0.9 a token against 1): cost %.0f, optimum 1, ratio %.0f; "
          "the skip variant costs %.0f." % (M, g, g, greedy_skip(C, L, 1.0)))
    P("K3. Skip variant on the same 3,000 instances: largest cost / optimum = %.3f (theory: at most 2)." % max_ratio_skip)
    # adversarial search for the skip variant
    worst_skip = 1.0
    for _ in range(20000):
        n = rng.randint(2, 7)
        L = [rng.choice([1, 2, 3, 5, 8, 13, 21]) * 1.0 for _ in range(n)]
        C = [rng.choice([1, 2, 3, 5, 8, 13, 21]) * 1.0 for _ in range(n)]
        m = rng.randint(1, int(sum(L)))
        worst_skip = max(worst_skip, greedy_skip(C, L, m) / opt01(C, L, m))
    P("    Search over 20,000 small integer instances: largest cost / optimum = %.3f." % worst_skip)
    P("K4. Engine-sized instances: 60 holders, sizes lognormal around 25 K tokens (in 64-token pages), four cost")
    P("    classes (cost per token 0 for ended sessions, then about 1, 3 and 10 units with spread), need m = 25 K tokens:")
    ratios, over, ratios_skip = [], [], []
    for _ in range(400):
        n = 60
        Lp = [max(4, int(rng.lognormvariate(np.log(390.0), 0.6))) for _ in range(n)]   # pages
        cls = [rng.choice([0, 1, 1, 2, 2, 3]) for _ in range(n)]
        per_tok = [0.0 if c == 0 else [1.0, 3.0, 10.0][c - 1] * rng.lognormvariate(0.0, 0.5) for c in cls]
        C = [pt * l for pt, l in zip(per_tok, Lp)]
        if sum(1 for c in cls if c == 0) > 0:
            # drop the free holders first; the question is what happens once they are gone
            keep = [j for j in range(n) if cls[j] != 0]
            Lp = [Lp[j] for j in keep]
            C = [C[j] for j in keep]
        mp = 391
        g, got, last = greedy01(C, [float(l) for l in Lp], float(mp))
        o = opt01_dp(C, Lp, mp)
        ratios.append(g / o)
        ratios_skip.append(greedy_skip(C, [float(l) for l in Lp], float(mp)) / o)
        over.append(got / mp - 1.0)
    P("    greedy / optimum: mean %.3f, p90 %.3f, max %.2f;  skip variant: mean %.3f, max %.2f;  room freed beyond m: "
      "mean %.0f %% of m." % (float(np.mean(ratios)), float(np.quantile(ratios, 0.9)), max(ratios),
                             float(np.mean(ratios_skip)), max(ratios_skip), 100 * float(np.mean(over))))
    P("    (The exact optimum is a dynamic programme over pages: 60 holders x 391 pages, about 2e4 steps.)")
    worst = 0.0
    slack_zero = 0
    for _ in range(300):
        n = rng.randint(3, 15)
        Lh = np.array([rng.uniform(1.0, 10.0) for _ in range(n)])
        nu = np.array([rng.uniform(0.0, 5.0) for _ in range(n)])
        room = rng.uniform(0.1, 1.3) * Lh.sum()
        res = linprog(-(nu * Lh), A_ub=[Lh], b_ub=[room], bounds=[(0, 1)] * n, method="highs")
        price = -res.ineqlin.marginals[0]
        if room >= Lh.sum():
            slack_zero += 1
            worst = max(worst, abs(price))
            continue
        order = np.argsort(-nu)
        fill = np.cumsum(Lh[order])
        k = order[int(np.searchsorted(fill, room, side="left"))]
        worst = max(worst, abs(price - nu[k]))
    P("K5. Who stays (the other side of the same problem): maximise sum nu_j L_j x_j with sum L_j x_j <= C, 0 <= x <= 1.")
    P("    The solver's multiplier of the room constraint against the index of the marginal holder, 300 instances:")
    P("    max |difference| = %.1e; in the %d instances where everything fits the multiplier is 0." % (worst, slack_zero))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "knapsack.out"), "w") as f:
        f.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    sys.exit(main())
