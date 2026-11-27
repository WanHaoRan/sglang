"""Retain when a dropped prefix can be fetched back before the session returns.

Question (task item 1, last part): what changes in the index when "K at the return" is replaced
by the optimal Fetch cost?

Model (one gap, proper law on a grid, no queue wait): the prefix (L tokens) is kept until age h,
then dropped; a read of r seconds is started at age s >= h, or at the arrival if that comes
first. Costs: rent lam*L per second held (kept copy, or the read from its start to the arrival),
stall c per second from the arrival to the landing, kdev (device time) for every read made.

  cost(h, s) = E[ R <= h:      lam*L*R
                  h < R <= s:  lam*L*h + c*r + kdev + lam*L*r                (read at arrival)
                  R > s:       lam*L*h + kdev + lam*L*(max(R, s+r) - s) + c*(s + r - R)^+ ]

Checked:
 1. the set of prices at which keeping is optimal is an interval [0, nu_hat] (indexability
    survives on these laws);
 2. falling hazard: nu_hat equals the simple index with K0 = c*r + kdev and the read's own
    hold in the denominator: (K0/L) sup_h F/(tau - r F);
 3. sharp hints: nu_hat is far below the simple index, and close to kdev/(L*(E[R] - r)):
    with a free device a far prefix is evicted and read back just in time.

Run: python -I -B refetch.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hold_index import held_time, surv_lognormal, surv_point, surv_two_piece_lognormal, surv_uniform  # noqa: E402


def cost_table(surv: np.ndarray, dt: float, lam_l: float, c: float, kdev: float, r_steps: int) -> np.ndarray:
    """cost[h, s] for 0 <= h <= s <= n on the grid (s = n means: never before arrival)."""
    n = len(surv) - 1
    prob = surv[:-1] - surv[1:]  # P(R = t_j), j = 1..n
    tj = np.arange(1, n + 1) * dt
    r = r_steps * dt
    held = held_time(surv, dt)
    ret_by = 1.0 - surv
    # read at arrival: K_arr = c*r + kdev + lam_l*r
    k_arr = c * r + kdev + lam_l * r
    # for a start s (grid index): outcomes R > t_s
    stage_tail = np.zeros(n + 1)  # E[(kdev + lam_l*(max(R, s+r) - s) + c*(s+r-R)^+) ; R > t_s]
    for s in range(n + 1):
        ts = s * dt
        later = tj > ts + 1e-12
        if not np.any(later):
            continue
        rr = tj[later]
        val = kdev + lam_l * (np.maximum(rr, ts + r) - ts) + c * np.maximum(0.0, ts + r - rr)
        stage_tail[s] = float(np.sum(prob[later] * val))
    cost = np.full((n + 1, n + 1), np.inf)
    for h in range(n + 1):
        # rent while kept: lam_l * E[min(R, t_h)] ; returns in (t_h, t_s] read at arrival
        s_idx = np.arange(h, n + 1)
        arrive_mid = ret_by[s_idx] - ret_by[h]
        cost[h, h:] = lam_l * held[h] + arrive_mid * k_arr + stage_tail[s_idx]
    return cost


def generalized_index(surv: np.ndarray, dt: float, tokens: float, c: float, kdev: float, r_steps: int):
    """Largest price at which keeping (some h > 0) is at least as good as dropping now, and a
    scan of prices to see whether the keep set is an interval."""
    lams = np.geomspace(1e-9, 1e-1, 161)
    keep = []
    for lam in lams:
        cost = cost_table(surv, dt, lam * tokens, c, kdev, r_steps)
        drop_now = float(np.min(cost[0, :]))
        keep_best = float(np.min(cost[1:, :]))
        keep.append(keep_best < drop_now - 1e-12 * max(1.0, drop_now))
    keep = np.array(keep)
    switches = int(np.sum(keep[1:] != keep[:-1]))
    if not keep.any():
        return 0.0, switches
    lo = lams[np.where(keep)[0][-1]]
    hi = lams[min(len(lams) - 1, np.where(keep)[0][-1] + 1)]
    for _ in range(40):
        mid = math.sqrt(lo * hi)
        cost = cost_table(surv, dt, mid * tokens, c, kdev, r_steps)
        if float(np.min(cost[1:, :])) < float(np.min(cost[0, :])) - 1e-13:
            lo = mid
        else:
            hi = mid
    return lo, switches


def simple_index(surv: np.ndarray, dt: float, tokens: float, k0: float, r: float) -> float:
    held = held_time(surv, dt)
    ret_by = 1.0 - surv
    den = held[1:] - r * ret_by[1:]
    val = np.where(den > 0, ret_by[1:] / np.maximum(den, 1e-300), np.inf)
    return (k0 / tokens) * float(np.max(val))


def main() -> None:
    dt, n = 0.25, 480  # 120 s
    t = np.linspace(0.0, dt * n, n + 1)
    tokens = 25_000.0
    r_steps = 8  # a 2 s read
    r = r_steps * dt
    laws = {
        "tool LN 0.41/1.69/16.8 (falling hazard after 0.8 s)": surv_two_piece_lognormal(t, 0.41, 1.69, 16.8),
        "LN(median 5, sigma 1.5)": surv_lognormal(t, 5.0, 1.5),
        "uniform[40,50] (sharp hint)": surv_uniform(t, 40.0, 50.0),
        "LN(median 40, sigma 0.1) (sharp hint)": surv_lognormal(t, 40.0, 0.1),
        "point 40 s (exact hint)": surv_point(t, 40.0),
    }
    print("=" * 100)
    print("Retain with early re-fetch: generalized index nu_hat against the simple index (L = 25,000, read r = 2 s)")
    for c, kdev in ((1.0, 0.0), (1.0, 0.5), (0.0, 0.5)):
        print(f"  stall weight c = {c}, device cost per read = {kdev} s")
        for name, surv in laws.items():
            surv = surv.copy()
            surv[-1] = 0.0  # proper law on the grid
            nu_hat, switches = generalized_index(surv, dt, tokens, c, kdev, r_steps)
            k0 = c * r + kdev
            nu_simple = simple_index(surv, dt, tokens, k0, r)
            held = held_time(surv, dt)
            jit = kdev / (tokens * max(held[-1] - r, 1e-9))
            print(
                f"    {name:52s} nu_hat {nu_hat:.3e}  simple (K at arrival) {nu_simple:.3e}  ratio {nu_hat/nu_simple if nu_simple>0 else float('nan'):.3f}  "
                f"kdev/(L(E[R]-r)) {jit:.3e}  keep-set switches {switches}"
            )


if __name__ == "__main__":
    main()
