"""Queued turns and staged reads: one index (DESIGN.md 4.1 bullet "a queued turn", 5.1 order).

Holder: a clean prefix of L tokens in host memory. Its session returns after a remaining gap R
(R = 0: the turn has arrived) and is then admitted after a queue wait W: the prefix is needed
at T = R + W. If the copy is dropped, the turn re-reads at its arrival (the recourse of Fetch
for falling-hazard gaps): it pays K0 = kappa*L/beta + c*E[(r - W)^+] and the re-read holds L
tokens for rho seconds before admission.

Claim checked here:  keeping until some horizon beats dropping now  iff
    lam <= nu = (K0/L) / (g + w - rho),    g = inf_h tau(h)/F(h),  w = E[W],
with g = 0 for an arrived turn, g = R0 for an exact hint, g = 1/hazard for falling hazards.

Run: python -I -B queued_staged.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hold_index import (  # noqa: E402
    held_time,
    index_hull,
    surv_lognormal,
    surv_point,
    surv_two_piece_lognormal,
    surv_uniform,
    with_end_mass,
)

RNG = np.random.default_rng(7)


def priced_costs(surv: np.ndarray, dt: float, lam: float, tokens: float, k0: float, wait: float, rho: float) -> np.ndarray:
    """Cost of 'keep until grid age k, then drop; a turn that arrives while kept is held to admission'."""
    held = held_time(surv, dt)
    ret_by = 1.0 - surv  # F(h)
    ret_ever = 1.0 - surv[-1]
    rent = lam * tokens * (held + ret_by * wait)
    restore = (ret_ever - ret_by) * (k0 + lam * tokens * rho)
    return rent + restore


def unified_index(surv: np.ndarray, dt: float, tokens: float, k0: float, wait: float, rho: float) -> float:
    held = held_time(surv, dt)
    base = index_hull(surv, held, 1.0)[0]  # sup_h F/tau at age 0
    g = math.inf if base <= 0 else 1.0 / base
    return (k0 / tokens) / (g + wait - rho)


def check_unified() -> None:
    dt, n = 0.05, 4000
    t = np.linspace(0.0, dt * n, n + 1)
    laws = {
        "tool LN 0.41/1.69/16.8": surv_two_piece_lognormal(t, 0.41, 1.69, 16.8),
        "LN(1.69,1.79)+5% ended": with_end_mass(surv_lognormal(t, 1.69, 1.79), 0.05),
        "uniform[15,25]": surv_uniform(t, 15.0, 25.0),
        "point 20 s": surv_point(t, 20.0),
        "random law": np.concatenate(([1.0], np.sort(RNG.random(n))[::-1] * 0.999)),
    }
    viol = 0
    tests = 0
    for name, surv in laws.items():
        for _ in range(200):
            tokens = float(RNG.choice([2_000, 8_000, 25_000, 60_000]))
            k0 = float(RNG.uniform(0.05, 5.0))
            wait = float(RNG.uniform(0.0, 40.0))
            rho = float(RNG.uniform(0.0, min(wait, 3.0)))
            nu = unified_index(surv, dt, tokens, k0, wait, rho)
            for factor in (0.5, 0.9, 1.1, 2.0):
                lam = nu * factor
                cost = priced_costs(surv, dt, lam, tokens, k0, wait, rho)
                keep_better = bool(np.min(cost[1:]) < cost[0] - 1e-12 * max(1.0, abs(cost[0])))
                tests += 1
                if keep_better != (factor < 1.0):
                    viol += 1
    print(f"  unified index nu = (K0/L)/(g + w - rho): {tests} random (law, L, K0, w, rho, lam) cases, violations {viol}")


def numbers() -> None:
    beta = 21_400.0  # tokens/s, measured device
    print("  Queued turn (g = 0), L = 25,000 tokens, read transfer L/beta = %.2f s, re-read hold rho = 1 s:" % (25_000 / beta))
    print("    wait W   kappa=0.1        kappa=1          kappa=10      (nu in TTFT-s per token-s; re-read fits, c*stall = 0)")
    for wait in (2.0, 5.0, 10.0, 20.0, 40.0):
        row = []
        for kappa in (0.1, 1.0, 10.0):
            k0 = kappa * 25_000 / beta
            row.append(f"{k0 / 25_000 / (wait - 1.0):.3e}")
        print(f"    {wait:5.1f}   " + "   ".join(row))
    print("  Staged read of a session not yet back, same L, expected wait after arrival w = 5 s, rho = 1 s:")
    dt, n = 0.05, 24_000
    t = np.linspace(0.0, dt * n, n + 1)
    tool = surv_two_piece_lognormal(t, 0.41, 1.69, 16.8)
    held = held_time(tool, dt)
    base = index_hull(tool, held, 1.0)
    for age in (0.0, 2.0, 10.0, 60.0, 300.0):
        k = int(round(age / dt))
        g = 1.0 / base[k]
        k0 = 1.0 * 25_000 / beta
        nu = k0 / 25_000 / (g + 5.0 - 1.0)
        print(f"    tool gap, idle {age:6.1f} s: g = {g:8.2f} s  time to GPU use g + w = {g + 5.0:8.2f} s  nu(kappa=1) = {nu:.3e}")
    print("  Order check (kappa = 1, equal L): a queued turn 40 s from admission has nu = %.3e," % (25_000 / beta / 25_000 / 39.0))
    k = int(round(2.0 / dt))
    print("  a staged read idle 2 s on the tool law with w = 5 s has nu = %.3e: the queued turn yields first." % (25_000 / beta / 25_000 / (1.0 / base[k] + 4.0)))


def exact_hint_refetch() -> None:
    # exact hint R0, queue wait W, re-read fits: keep costs lam*L*(R0+W); drop + just-in-time re-read costs
    # kappa*L/beta + lam*L*rho. Indifference price: (kappa/beta)/(R0 + W - rho).
    beta = 21_400.0
    viol = 0
    for _ in range(2000):
        r0 = RNG.uniform(0.5, 600.0)
        wait = RNG.uniform(0.0, 40.0)
        rho = RNG.uniform(0.0, 2.0)
        kappa = RNG.uniform(0.01, 20.0)
        tokens = RNG.uniform(1e3, 6e4)
        if r0 + wait <= rho:
            continue
        nu = (kappa / beta) / (r0 + wait - rho)
        for factor in (0.8, 1.25):
            lam = nu * factor
            keep = lam * tokens * (r0 + wait)
            drop = kappa * tokens / beta + lam * tokens * rho
            if (keep < drop) != (factor < 1.0):
                viol += 1
    print(f"  exact hint with a just-in-time re-read: nu = (kappa/beta)/(R0 + W - rho); violations {viol} in 4000 comparisons")


def main() -> None:
    print("=" * 100)
    print("Q1. Brute-force check of the unified index")
    check_unified()
    exact_hint_refetch()
    print("=" * 100)
    print("Q2. Numbers")
    numbers()


if __name__ == "__main__":
    main()
