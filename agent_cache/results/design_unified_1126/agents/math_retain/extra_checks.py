"""Small checks that back single sentences of the write-up.

a. A rising-hazard law mixed with an ended session: the sup over horizons is interior, and the
   draft's P(R < inf)/E[R] is zero.
b. The engine's two-piece lognormal entries: where the hazard rises (the index is not the hazard).
c. Index with the restore's own hold in the denominator: (K0/L)/(g - rho)^+.
d. Share of pages ever due under a TTL, Monte Carlo against (1 - e) x / (x + e - x e).
e. A late write against a late read: Smith ratios.

Run: python -I -B extra_checks.py
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
    surv_two_piece_lognormal,
    surv_weibull,
    with_end_mass,
)

RNG = np.random.default_rng(5)


def check_a() -> None:
    print("a. Rising hazard (Weibull k = 1.5, theta = 20) mixed with P(ended) = p, age 0, K/L = 1")
    dt, n = 0.02, 60_000
    t = np.linspace(0.0, dt * n, n + 1)
    for p in (0.0, 0.05, 0.3):
        surv = with_end_mass(surv_weibull(t, 20.0, 1.5), p)
        surv[-1] = p
        held = held_time(surv, dt)
        ratio = (1.0 - surv[1:]) / held[1:]
        k = int(np.argmax(ratio))
        mean_proper = 20.0 * math.gamma(1 + 1 / 1.5)
        print(
            f"   p = {p:4.2f}: sup F/tau = {ratio[k]:.5f} at h = {(k+1)*dt:8.2f} s;  (1-p)/E[R | returns] = {(1-p)/mean_proper:.5f};  "
            f"draft P(R<inf)/E[R] = {0.0 if p > 0 else 1/mean_proper:.5f}"
        )


def check_b() -> None:
    print("b. Two-piece lognormal entries (K/L = 1): index against hazard")
    dt = 0.01
    for name, (p10, p50, p90) in (("tool 0.41/1.69/16.8", (0.41, 1.69, 16.8)), ("human 3.1/96/1040", (3.1, 96.0, 1040.0))):
        n = 400_000 if p50 > 10 else 100_000
        t = np.linspace(0.0, dt * n, n + 1)
        surv = surv_two_piece_lognormal(t, p10, p50, p90)
        held = held_time(surv, dt)
        nu = index_hull(surv, held, 1.0)
        haz = np.zeros(n + 1)
        haz[:-1] = (surv[:-1] - surv[1:]) / (surv[:-1] * dt)
        peak = int(np.argmax(haz[: int(p50 / dt) - 1]))
        differ = np.where(nu[: n - 100] > haz[: n - 100] * 1.02)[0]
        spans = []
        if len(differ):
            start = prev = differ[0]
            for k in differ[1:]:
                if k != prev + 1:
                    spans.append((start * dt, prev * dt))
                    start = k
                prev = k
            spans.append((start * dt, prev * dt))
        spans = [(a, b) for a, b in spans if b - a > 0.05]
        z90 = 1.2815515655446004
        s_lo, s_hi = math.log(p50 / p10) / z90, math.log(p90 / p50) / z90
        print(f"   {name}: sigma_low {s_lo:.3f}, sigma_high {s_hi:.3f}; hazard peak below the median at {peak*dt:.2f} s; hazard jumps by x{s_lo/s_hi:.2f} at the median")
        print(f"      index at age 0 = {nu[0]:.4f}/s (hazard there is 0); ages where the index exceeds the hazard by > 2 %: "
              + ", ".join(f"{a:.2f}-{b:.2f} s" for a, b in spans))


def check_c() -> None:
    print("c. Restore = cost K0 plus L tokens held rho seconds: keep iff lam <= (K0/L)/(g - rho)^+")
    dt, n = 0.05, 4000
    t = np.linspace(0.0, dt * n, n + 1)
    surv = surv_two_piece_lognormal(t, 0.41, 1.69, 16.8)
    held = held_time(surv, dt)
    g = 1.0 / index_hull(surv, held, 1.0)[0]
    viol = 0
    for _ in range(2000):
        tokens = 25_000.0
        k0 = RNG.uniform(0.1, 3.0)
        rho = RNG.uniform(0.0, 5.0)
        claim = math.inf if g <= rho else (k0 / tokens) / (g - rho)
        lam = 10 ** RNG.uniform(-8, -2)
        cost = lam * tokens * held + (surv - surv[-1]) * (k0 + lam * tokens * rho)
        keep = bool(np.min(cost[1:]) < cost[0] * (1 - 1e-12))
        if abs(lam / claim - 1) < 1e-3:
            continue
        if keep != (lam < claim):
            viol += 1
    print(f"   tool law, g = {g:.3f} s: violations {viol} in 2000 random (K0, rho, lam)")


def check_d() -> None:
    print("d. Share of pages ever due under a TTL of d seconds (Monte Carlo, 200k pages)")
    e = 1.0 / 29.2
    for x in (0.046, 0.067, 0.103):
        due = 0
        n_pages = 200_000
        ends = RNG.random((n_pages, 400)) < e
        longs = RNG.random((n_pages, 400)) < x
        for i in range(n_pages):
            for j in range(400):
                if ends[i, j]:
                    break
                if longs[i, j]:
                    due += 1
                    break
        formula = (1 - e) * x / (x + e - x * e)
        print(f"   x = {x:.3f}: simulated {due/n_pages:.4f}; (1-e) x/(x + e - x e) = {formula:.4f}; for pages of a non-final turn x/(x+e-xe) = {x/(x+e-x*e):.4f}")


def check_e() -> None:
    print("e. Device priority of a late write against a late read (cost per device-second)")
    beta = 21_400.0
    for lam, nu_page, c, l_read in ((1e-6, 0.0, 1.0, 25_000.0), (1e-6, 0.0, 0.01, 25_000.0), (5e-5, 1e-5, 1.0, 25_000.0), (1e-6, 5e-7, 0.0, 25_000.0)):
        write = beta * max(0.0, lam - nu_page)
        read = c * beta / l_read
        print(
            f"   lam {lam:.0e}, page index {nu_page:.0e}, c {c}, read of {l_read:.0f} tokens: write {write:.3e}, read {read:.3e} -> "
            f"{'write first' if write > read else 'read first'}  (write first iff (lam - nu)*L_read > c: {(lam-nu_page)*l_read:.3f} vs {c})"
        )


def main() -> None:
    check_a()
    check_b()
    check_c()
    check_d()
    check_e()


if __name__ == "__main__":
    main()
